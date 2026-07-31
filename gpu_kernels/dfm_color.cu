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

extern "C" __global__
void dfm_rct_stats_kernel(
    const float* __restrict__ source,
    const float* __restrict__ like,
    const float* __restrict__ mask,
    float* __restrict__ stats,
    const unsigned int pixels,
    const float cutoff
) {
    unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels || mask[pixel] < cutoff) return;
    float source_rgb[3];
    float like_rgb[3];
    for (int channel = 0; channel < 3; ++channel) {
        source_rgb[channel] = source[pixel * 3 + channel];
        like_rgb[channel] = like[pixel * 3 + channel];
    }
    float source_lab[3];
    float like_lab[3];
    rgb_to_lab(source_rgb, source_lab);
    rgb_to_lab(like_rgb, like_lab);
    for (int channel = 0; channel < 3; ++channel) {
        atomicAdd(&stats[channel], source_lab[channel]);
        atomicAdd(&stats[3 + channel], source_lab[channel] * source_lab[channel]);
        atomicAdd(&stats[6 + channel], like_lab[channel]);
        atomicAdd(&stats[9 + channel], like_lab[channel] * like_lab[channel]);
    }
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
