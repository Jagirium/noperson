__device__ __forceinline__ float srgb_to_linear(float value) {
    return value > 0.04045f
        ? powf((value + 0.055f) / 1.055f, 2.4f)
        : value / 12.92f;
}

__device__ __forceinline__ void rgb_to_lab(const float rgb[3], float lab[3]) {
    float r = srgb_to_linear(rgb[0]);
    float g = srgb_to_linear(rgb[1]);
    float b = srgb_to_linear(rgb[2]);
    float xyz[3] = {
        (0.412453f * r + 0.357580f * g + 0.180423f * b) / 0.95047f,
        0.212671f * r + 0.715160f * g + 0.072169f * b,
        (0.019334f * r + 0.119193f * g + 0.950227f * b) / 1.08883f,
    };
    float f[3];
    for (int channel = 0; channel < 3; ++channel) {
        float value = xyz[channel];
        f[channel] = value > 0.008856f
            ? powf(fmaxf(value, 0.008856f), 1.0f / 3.0f)
            : 7.787f * value + 4.0f / 29.0f;
    }
    lab[0] = 116.0f * f[1] - 16.0f;
    lab[1] = 500.0f * (f[0] - f[1]);
    lab[2] = 200.0f * (f[1] - f[2]);
}

__device__ __forceinline__ float linear_to_srgb(float value) {
    return value > 0.0031308f
        ? 1.055f * powf(fmaxf(value, 0.0031308f), 1.0f / 2.4f) - 0.055f
        : 12.92f * value;
}

__device__ __forceinline__ void lab_to_rgb(const float lab[3], float rgb[3]) {
    float fy = (lab[0] + 16.0f) / 116.0f;
    float f[3] = {
        lab[1] / 500.0f + fy,
        fy,
        fmaxf(fy - lab[2] / 200.0f, 0.0f),
    };
    float normalized[3];
    for (int channel = 0; channel < 3; ++channel) {
        float value = f[channel];
        normalized[channel] = value > 0.2068966f
            ? value * value * value
            : (value - 4.0f / 29.0f) / 7.787f;
    }
    float x = normalized[0] * 0.95047f;
    float y = normalized[1];
    float z = normalized[2] * 1.08883f;
    float linear[3] = {
        3.2404813432005266f * x - 1.5371515162713185f * y - 0.4985363261688878f * z,
        -0.9692549499965682f * x + 1.8759900014898907f * y + 0.0415559265582928f * z,
        0.0556466391351772f * x - 0.2040413383665112f * y + 1.0573110696453443f * z,
    };
    for (int channel = 0; channel < 3; ++channel) {
        rgb[channel] = fminf(fmaxf(linear_to_srgb(linear[channel]), 0.0f), 1.0f);
    }
}

template <int Fields>
__device__ __forceinline__ void reduce_float_fields(float values[Fields], float* partials) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5u;
    __shared__ float warp_totals[8][Fields];

    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        for (int field = 0; field < Fields; ++field) {
            values[field] += __shfl_down_sync(0xffffffffu, values[field], offset);
        }
    }
    if (lane == 0) {
        for (int field = 0; field < Fields; ++field) {
            warp_totals[warp][field] = values[field];
        }
    }
    __syncthreads();

    if (warp != 0) return;
    for (int field = 0; field < Fields; ++field) {
        float total = lane < 8 ? warp_totals[lane][field] : 0.0f;
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0) partials[blockIdx.x * Fields + field] = total;
    }
}

template <int Fields>
__device__ __forceinline__ void finalize_float_fields(
    const float* partials,
    float* stats,
    const unsigned int blocks
) {
    const unsigned int field = threadIdx.x;
    if (field >= Fields) return;
    float total = 0.0f;
    for (unsigned int block = 0; block < blocks; ++block) {
        total += partials[block * Fields + field];
    }
    stats[field] = total;
}

extern "C" __global__
void dfm_rct_stats_stage1_kernel(
    const float* __restrict__ source,
    const float* __restrict__ like,
    const float* __restrict__ mask,
    float* __restrict__ partials,
    const unsigned int pixels,
    const float cutoff
) {
    float values[12] = {0.0f};
    for (unsigned long long pixel = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
             + (unsigned long long)threadIdx.x;
         pixel < (unsigned long long)pixels;
         pixel += (unsigned long long)blockDim.x * (unsigned long long)gridDim.x) {
        if (mask[pixel] < cutoff) continue;
        float source_rgb[3];
        float like_rgb[3];
        for (int channel = 0; channel < 3; ++channel) {
            source_rgb[channel] = source[pixel * 3ull + (unsigned long long)channel];
            like_rgb[channel] = like[pixel * 3ull + (unsigned long long)channel];
        }
        float source_lab[3];
        float like_lab[3];
        rgb_to_lab(source_rgb, source_lab);
        rgb_to_lab(like_rgb, like_lab);
        for (int channel = 0; channel < 3; ++channel) {
            values[channel] += source_lab[channel];
            values[3 + channel] += source_lab[channel] * source_lab[channel];
            values[6 + channel] += like_lab[channel];
            values[9 + channel] += like_lab[channel] * like_lab[channel];
        }
    }
    reduce_float_fields<12>(values, partials);
}

extern "C" __global__
void dfm_rct_stats_stage2_kernel(
    const float* __restrict__ partials,
    float* __restrict__ stats,
    const unsigned int blocks
) {
    finalize_float_fields<12>(partials, stats, blocks);
}

extern "C" __global__
void dfm_rct_apply_kernel(
    float* __restrict__ source,
    const float* __restrict__ stats,
    const unsigned int pixels
) {
    unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    float rgb[3] = {
        source[pixel * 3],
        source[pixel * 3 + 1],
        source[pixel * 3 + 2],
    };
    float lab[3];
    rgb_to_lab(rgb, lab);
    float count = (float)pixels;
    float correction = fmaxf(count - 1.0f, 1.0f);
    for (int channel = 0; channel < 3; ++channel) {
        float source_mean = stats[channel] / count;
        float like_mean = stats[6 + channel] / count;
        float source_var = fmaxf(
            (stats[3 + channel] - count * source_mean * source_mean) / correction,
            0.0f
        );
        float like_var = fmaxf(
            (stats[9 + channel] - count * like_mean * like_mean) / correction,
            0.0f
        );
        lab[channel] = (lab[channel] - source_mean)
            * (sqrtf(like_var) / (sqrtf(source_var) + 1e-6f))
            + like_mean;
    }
    lab[0] = fminf(fmaxf(lab[0], 0.0f), 100.0f);
    lab[1] = fminf(fmaxf(lab[1], -127.0f), 127.0f);
    lab[2] = fminf(fmaxf(lab[2], -127.0f), 127.0f);
    lab_to_rgb(lab, rgb);
    for (int channel = 0; channel < 3; ++channel) {
        source[pixel * 3 + channel] = rgb[channel];
    }
}

extern "C" __global__
void auto_color_dfl_stats_stage1_kernel(
    const float* __restrict__ original_chw,
    const float* __restrict__ swapped_chw,
    const float* __restrict__ mask,
    float* __restrict__ partials,
    const unsigned int pixels,
    const unsigned int use_mask
) {
    float values[13] = {0.0f};
    for (unsigned long long pixel = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
             + (unsigned long long)threadIdx.x;
         pixel < (unsigned long long)pixels;
         pixel += (unsigned long long)blockDim.x * (unsigned long long)gridDim.x) {
        if (use_mask && mask[pixel] < 0.2f) continue;
        float original_rgb[3];
        float swapped_rgb[3];
        for (int channel = 0; channel < 3; ++channel) {
            unsigned long long offset = (unsigned long long)channel * (unsigned long long)pixels + pixel;
            original_rgb[channel] = original_chw[offset] / 255.0f;
            swapped_rgb[channel] = swapped_chw[offset] / 255.0f;
        }
        float original_lab[3];
        float swapped_lab[3];
        rgb_to_lab(original_rgb, original_lab);
        rgb_to_lab(swapped_rgb, swapped_lab);
        for (int channel = 0; channel < 3; ++channel) {
            values[channel] += original_lab[channel];
            values[3 + channel] += original_lab[channel] * original_lab[channel];
            values[6 + channel] += swapped_lab[channel];
            values[9 + channel] += swapped_lab[channel] * swapped_lab[channel];
        }
        values[12] += 1.0f;
    }
    reduce_float_fields<13>(values, partials);
}

extern "C" __global__
void auto_color_dfl_stats_stage2_kernel(
    const float* __restrict__ partials,
    float* __restrict__ stats,
    const unsigned int blocks
) {
    finalize_float_fields<13>(partials, stats, blocks);
}

extern "C" __global__
void auto_color_dfl_apply_kernel(
    float* __restrict__ swapped_chw,
    const float* __restrict__ stats,
    const unsigned int pixels,
    const float blend
) {
    unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels || stats[12] < 1.0f) return;

    float target_rgb[3];
    for (int channel = 0; channel < 3; ++channel) {
        target_rgb[channel] = swapped_chw[channel * pixels + pixel] / 255.0f;
    }
    float lab[3];
    rgb_to_lab(target_rgb, lab);
    float count = stats[12];
    float correction = fmaxf(count - 1.0f, 1.0f);
    for (int channel = 0; channel < 3; ++channel) {
        float original_mean = stats[channel] / count;
        float swapped_mean = stats[6 + channel] / count;
        float original_var = fmaxf(
            (stats[3 + channel] - count * original_mean * original_mean) / correction,
            0.0f
        );
        float swapped_var = fmaxf(
            (stats[9 + channel] - count * swapped_mean * swapped_mean) / correction,
            0.0f
        );
        lab[channel] = (lab[channel] - swapped_mean)
            * ((sqrtf(original_var) + 1e-6f) / (sqrtf(swapped_var) + 1e-6f))
            + original_mean;
    }
    lab[0] = fminf(fmaxf(lab[0], 0.0f), 100.0f);
    lab[1] = fminf(fmaxf(lab[1], -127.0f), 127.0f);
    lab[2] = fminf(fmaxf(lab[2], -127.0f), 127.0f);
    float matched_rgb[3];
    lab_to_rgb(lab, matched_rgb);
    for (int channel = 0; channel < 3; ++channel) {
        float value = ((1.0f - blend) * target_rgb[channel] + blend * matched_rgb[channel]) * 255.0f;
        swapped_chw[channel * pixels + pixel] = fminf(fmaxf(value, 0.0f), 255.0f);
    }
}
