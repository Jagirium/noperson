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
void imagenet_normalize_kernel(
    float* __restrict__ image,
    const unsigned int plane,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int channel = idx / plane;
    const float means[3] = {0.485f, 0.456f, 0.406f};
    const float stds[3] = {0.229f, 0.224f, 0.225f};
    image[idx] = (image[idx] * (1.0f / 255.0f) - means[channel]) / stds[channel];
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
void mask_invert_kernel(
    float* __restrict__ mask,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    mask[idx] = 1.0f - mask[idx];
}

extern "C" __global__
void semantic_region_mask_kernel(
    const unsigned char* __restrict__ classes,
    float* __restrict__ mask,
    unsigned int* __restrict__ count,
    const unsigned int pixels,
    const unsigned int region
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    unsigned int actual = (unsigned int)classes[idx];
    bool selected = region == 0
        ? (actual == 4 || actual == 5)
        : (actual == 11 || actual == 12 || actual == 13);
    mask[idx] = selected ? 1.0f : 0.0f;
    if (selected) atomicAdd(count, 1u);
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
void semantic_region_stats_kernel(
    const float* __restrict__ swapped,
    const float* __restrict__ original,
    const float* __restrict__ mask,
    float* __restrict__ stats,
    const unsigned int pixels
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    float region = mask[idx];
    float surround = fminf(fmaxf(1.0f - region, 0.0f), 1.0f);
    atomicAdd(&stats[0], surround);
    atomicAdd(&stats[1], region);
    for (unsigned int channel = 0; channel < 3; ++channel) {
        unsigned int offset = channel * pixels + idx;
        atomicAdd(&stats[2 + channel], swapped[offset] * surround);
        atomicAdd(&stats[5 + channel], original[offset] * region);
    }
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
