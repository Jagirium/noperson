#include "compat.hpp"

// mask_postprocess.cu — CrossSwap occluder and XSeg output semantics.

extern "C" __global__
void occluder_threshold_kernel(
    float* __restrict__ mask,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    mask[idx] = mask[idx] > 0.0f ? 1.0f : 0.0f;
}

extern "C" __global__
void xseg_postprocess_kernel(
    float* __restrict__ mask,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float value = fminf(fmaxf(mask[idx], 0.0f), 1.0f);
    mask[idx] = value < 0.1f ? 0.0f : value;
}

extern "C" __global__
void imagenet_normalize_copy_kernel(
    const float* __restrict__ source,
    float* __restrict__ destination,
    const unsigned int plane,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int channel = idx / plane;
    const float means[3] = {0.485f, 0.456f, 0.406f};
    const float stds[3] = {0.229f, 0.224f, 0.225f};
    destination[idx] = (source[idx] * (1.0f / 255.0f) - means[channel]) / stds[channel];
}

extern "C" __global__
void landmark_normalize_kernel(
    float* __restrict__ image,
    const unsigned int pixels,
    const unsigned int mode
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = pixels * 3;
    if (idx >= total) return;
    unsigned int channel = idx / pixels;
    if (mode == 0) {
        const float means[3] = {104.0f, 117.0f, 123.0f};
        image[idx] -= means[channel];
    } else if (mode == 1) {
        image[idx] *= 1.0f / 255.0f;
    }
}

extern "C" __global__
void parser_argmax_kernel(
    const float* __restrict__ logits, // [19,H,W]
    unsigned char* __restrict__ classes,
    const unsigned int pixels
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    float best = logits[idx];
    unsigned char best_class = 0;
    for (unsigned int class_id = 1; class_id < 19; ++class_id) {
        float value = logits[class_id * pixels + idx];
        if (value > best) {
            best = value;
            best_class = (unsigned char)class_id;
        }
    }
    classes[idx] = best_class;
}

extern "C" __global__
void parser_class_mask_kernel(
    const unsigned char* __restrict__ classes,
    float* __restrict__ mask,
    const unsigned int pixels,
    const unsigned int class_id,
    const unsigned int foreground_mode
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    unsigned int actual = (unsigned int)classes[idx];
    if (foreground_mode != 0) {
        bool background = actual == 0 || actual == 14 || actual == 15
            || actual == 16 || actual == 17 || actual == 18;
        mask[idx] = background ? 0.0f : 1.0f;
    } else {
        mask[idx] = actual == class_id ? 1.0f : 0.0f;
    }
}

extern "C" __global__
void parser_makeup_kernel(
    float* __restrict__ image,
    const unsigned char* __restrict__ classes,
    const unsigned int pixels,
    const unsigned int hair_enabled,
    const float hair_red,
    const float hair_green,
    const float hair_blue,
    const float hair_blend,
    const unsigned int lips_enabled,
    const float lips_red,
    const float lips_green,
    const float lips_blue,
    const float lips_blend
) {
    unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    unsigned int actual = (unsigned int)classes[pixel];
    bool hair = hair_enabled != 0 && actual == 17;
    bool lips = lips_enabled != 0 && (actual == 12 || actual == 13);
    if (!hair && !lips) return;
    float blend = hair ? hair_blend : lips_blend;
    float colors[3] = {
        hair ? hair_red : lips_red,
        hair ? hair_green : lips_green,
        hair ? hair_blue : lips_blue
    };
    for (unsigned int channel = 0; channel < 3; ++channel) {
        unsigned int idx = channel * pixels + pixel;
        image[idx] = image[idx] * (1.0f - blend) + colors[channel] * blend;
    }
}

extern "C" __global__
void mask_invert_kernel(
    float* __restrict__ mask,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    mask[idx] = 1.0f - mask[idx];
}

__device__ __forceinline__ int morphology_clamp_index(const int index, const int length) {
    return index < 0 ? 0 : (index >= length ? length - 1 : index);
}

// A radius-N 3x3 max-pool sequence is a separable square max filter. The
// horizontal pass also complements erosion inputs so both signs share max logic.
extern "C" __global__
void morphology_mask_horizontal_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int height,
    const unsigned int width,
    const unsigned int radius,
    const unsigned int negative
) {
    extern __shared__ float tile[];
    const unsigned int tile_width = blockDim.x + 2 * radius;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int origin_x = (int)(blockIdx.x * blockDim.x) - (int)radius;
    for (unsigned int local_x = threadIdx.x; local_x < tile_width; local_x += blockDim.x) {
        const int source_x = morphology_clamp_index(origin_x + (int)local_x, (int)width);
        float value = y < height ? src[y * width + (unsigned int)source_x] : 0.0f;
        tile[threadIdx.y * tile_width + local_x] = negative != 0 ? 1.0f - value : value;
    }
    __syncthreads();
    if (x >= width || y >= height) return;
    float maximum = 0.0f;
    for (unsigned int offset = 0; offset <= 2 * radius; ++offset) {
        maximum = fmaxf(maximum, tile[threadIdx.y * tile_width + threadIdx.x + offset]);
    }
    dst[y * width + x] = maximum;
}

extern "C" __global__
void morphology_mask_vertical_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int height,
    const unsigned int width,
    const unsigned int radius,
    const unsigned int negative
) {
    extern __shared__ float tile[];
    const unsigned int tile_height = blockDim.y + 2 * radius;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int origin_y = (int)(blockIdx.y * blockDim.y) - (int)radius;
    const unsigned int threads = blockDim.x * blockDim.y;
    const unsigned int lane = threadIdx.y * blockDim.x + threadIdx.x;
    for (unsigned int local = lane; local < tile_height * blockDim.x; local += threads) {
        const unsigned int local_y = local / blockDim.x;
        const unsigned int local_x = local % blockDim.x;
        const unsigned int source_x = blockIdx.x * blockDim.x + local_x;
        const int source_y = morphology_clamp_index(origin_y + (int)local_y, (int)height);
        tile[local] = source_x < width ? src[(unsigned int)source_y * width + source_x] : 0.0f;
    }
    __syncthreads();
    if (x >= width || y >= height) return;
    float maximum = 0.0f;
    for (unsigned int offset = 0; offset <= 2 * radius; ++offset) {
        maximum = fmaxf(maximum, tile[(threadIdx.y + offset) * blockDim.x + threadIdx.x]);
    }
    dst[y * width + x] = negative != 0 ? 1.0f - maximum : maximum;
}

extern "C" __global__
void restore_ellipse_mask_kernel(
    float* __restrict__ mask,
    const unsigned int width,
    const unsigned int height,
    const int center_x,
    const int center_y,
    const int radius_x,
    const int radius_y,
    const float blend,
    const unsigned int feather
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixels = width * height;
    if (idx >= pixels || radius_x <= 0 || radius_y <= 0 || feather == 0) return;
    int x = (int)(idx % width);
    int y = (int)(idx / width);
    if (x < center_x - radius_x || x >= center_x + radius_x
        || y < center_y - radius_y || y >= center_y + radius_y) return;
    float dx = (float)(x - center_x) / (float)radius_x;
    float dy = (float)(y - center_y) / (float)radius_y;
    float distance = sqrtf(dx * dx + dy * dy);
    float soft = fminf(fmaxf(
        (1.0f - distance) * (float)radius_x / (float)feather,
        0.0f
    ), 1.0f);
    mask[idx] *= 1.0f - soft * (1.0f - blend);
}

extern "C" __global__
void fake_diff_mask_kernel(
    const float* __restrict__ swapped,
    const float* __restrict__ original,
    float* __restrict__ mask,
    const unsigned int pixels,
    const float threshold
) {
    unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    bool changed = false;
    for (unsigned int channel = 0; channel < 3; ++channel) {
        unsigned int idx = channel * pixels + pixel;
        changed = changed || fabsf(swapped[idx] - original[idx]) >= threshold;
    }
    mask[pixel] = changed ? 1.0f : 0.0f;
}

extern "C" __global__
void fake_diff_composite_kernel(
    float* __restrict__ swapped,
    const float* __restrict__ original,
    const float* __restrict__ mask,
    const unsigned int pixels,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float keep_swap = mask[idx % pixels];
    swapped[idx] = swapped[idx] * keep_swap + original[idx] * (1.0f - keep_swap);
}

extern "C" __global__
void fake_diff_composite_direct_kernel(
    float* __restrict__ swapped,
    const float* __restrict__ original,
    const unsigned int pixels,
    const float threshold
) {
    unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    bool changed = false;
    for (unsigned int channel = 0; channel < 3; ++channel) {
        unsigned int idx = channel * pixels + pixel;
        changed = changed || fabsf(swapped[idx] - original[idx]) >= threshold;
    }
    float keep_swap = changed ? 1.0f : 0.0f;
    for (unsigned int channel = 0; channel < 3; ++channel) {
        unsigned int idx = channel * pixels + pixel;
        swapped[idx] = swapped[idx] * keep_swap + original[idx] * (1.0f - keep_swap);
    }
}

extern "C" __global__
void semantic_region_mask_stage1_kernel(
    const unsigned char* __restrict__ classes,
    float* __restrict__ mask,
    unsigned int* __restrict__ partials,
    const unsigned int pixels,
    const unsigned int region
) {
    unsigned int value = 0u;
    for (unsigned long long idx = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
             + (unsigned long long)threadIdx.x;
         idx < (unsigned long long)pixels;
         idx += (unsigned long long)blockDim.x * (unsigned long long)gridDim.x) {
        unsigned int actual = (unsigned int)classes[idx];
        bool selected = region == 0
            ? (actual == 4 || actual == 5)
            : (actual == 11 || actual == 12 || actual == 13);
        mask[idx] = selected ? 1.0f : 0.0f;
        value += selected ? 1u : 0u;
    }

    const unsigned int lane = noperson_lane_id();
    const unsigned int subgroup = noperson_subgroup_id();
    const unsigned int subgroup_count = noperson_subgroups_in_block();
    __shared__ unsigned int subgroup_totals[NOPERSON_MAX_SUBGROUPS_PER_BLOCK];
    value = noperson_subgroup_reduce_sum(value);
    if (lane == 0) subgroup_totals[subgroup] = value;
    __syncthreads();
    if (subgroup != 0) return;
    value = lane < subgroup_count ? subgroup_totals[lane] : 0u;
    value = noperson_subgroup_reduce_sum(value);
    if (lane == 0) partials[blockIdx.x] = value;
}

extern "C" __global__
void semantic_region_mask_stage2_kernel(
    const unsigned int* __restrict__ partials,
    unsigned int* __restrict__ count,
    const unsigned int blocks
) {
    if (threadIdx.x != 0) return;
    unsigned int total = 0u;
    for (unsigned int block = 0; block < blocks; ++block) total += partials[block];
    *count = total;
}

extern "C" __global__
void semantic_temporal_mask_kernel(
    float* __restrict__ current,
    float* __restrict__ previous,
    const unsigned int* __restrict__ count,
    const unsigned int* __restrict__ valid,
    const unsigned int pixels,
    const float alpha
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    if (*count < 10u) {
        current[idx] = 0.0f;
        return;
    }
    float value = current[idx];
    if (*valid != 0u) value = alpha * value + (1.0f - alpha) * previous[idx];
    current[idx] = value;
    previous[idx] = value;
}

extern "C" __global__
void semantic_mark_valid_kernel(
    unsigned int* __restrict__ valid,
    const unsigned int* __restrict__ count
) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && *count >= 10u) *valid = 1u;
}

extern "C" __global__
void semantic_region_stats_stage1_kernel(
    const float* __restrict__ swapped,
    const float* __restrict__ original,
    const float* __restrict__ mask,
    float* __restrict__ partials,
    const unsigned int pixels
) {
    float values[8] = {0.0f};
    for (unsigned long long idx = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
             + (unsigned long long)threadIdx.x;
         idx < (unsigned long long)pixels;
         idx += (unsigned long long)blockDim.x * (unsigned long long)gridDim.x) {
        float region = mask[idx];
        float surround = fminf(fmaxf(1.0f - region, 0.0f), 1.0f);
        values[0] += surround;
        values[1] += region;
        for (unsigned int channel = 0; channel < 3; ++channel) {
            unsigned long long offset = (unsigned long long)channel * (unsigned long long)pixels + idx;
            values[2 + channel] += swapped[offset] * surround;
            values[5 + channel] += original[offset] * region;
        }
    }

    const unsigned int lane = noperson_lane_id();
    const unsigned int subgroup = noperson_subgroup_id();
    const unsigned int subgroup_count = noperson_subgroups_in_block();
    __shared__ float subgroup_totals[NOPERSON_MAX_SUBGROUPS_PER_BLOCK][8];
    for (unsigned int field = 0; field < 8; ++field) {
        values[field] = noperson_subgroup_reduce_sum(values[field]);
    }
    if (lane == 0) {
        for (unsigned int field = 0; field < 8; ++field) {
            subgroup_totals[subgroup][field] = values[field];
        }
    }
    __syncthreads();
    if (subgroup != 0) return;
    for (unsigned int field = 0; field < 8; ++field) {
        float total = lane < subgroup_count ? subgroup_totals[lane][field] : 0.0f;
        total = noperson_subgroup_reduce_sum(total);
        if (lane == 0) partials[blockIdx.x * 8 + field] = total;
    }
}

extern "C" __global__
void semantic_region_stats_stage2_kernel(
    const float* __restrict__ partials,
    float* __restrict__ stats,
    const unsigned int blocks
) {
    const unsigned int field = threadIdx.x;
    if (field >= 8) return;
    float total = 0.0f;
    for (unsigned int block = 0; block < blocks; ++block) total += partials[block * 8 + field];
    stats[field] = total;
}

extern "C" __global__
void semantic_composite_kernel(
    float* __restrict__ swapped,
    const float* __restrict__ original,
    const float* __restrict__ mask,
    const float* __restrict__ stats,
    const unsigned int* __restrict__ count,
    const unsigned int pixels,
    const float blend,
    const float luminance_factor,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total || *count < 10u) return;
    unsigned int channel = idx / pixels;
    unsigned int pixel = idx % pixels;
    float surround_sum = fmaxf(stats[0], 1.0f);
    float region_sum = fmaxf(stats[1], 1.0f);
    float surround_mean = stats[2 + channel] / surround_sum;
    float original_mean = stats[5 + channel] / region_sum;
    float shift = (surround_mean - original_mean) * luminance_factor;
    float matched = fminf(fmaxf(original[idx] + shift, 0.0f), 255.0f);
    float effective = mask[pixel] * blend;
    swapped[idx] = effective * matched + (1.0f - effective) * swapped[idx];
}
