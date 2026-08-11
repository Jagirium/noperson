__device__ __forceinline__ float clamp_u8(float value) {
    return truncf(fminf(fmaxf(value, 0.0f), 255.0f));
}

__device__ __forceinline__ float grayscale(float r, float g, float b) {
    return r * 0.2989f + g * 0.587f + b * 0.114f;
}

extern "C" __global__
void color_adjust_prep_stage1_kernel(
    float* image,
    unsigned int* partials,
    const unsigned int pixels,
    const float gamma,
    const float red,
    const float green,
    const float blue,
    const float brightness
) {
    unsigned int value = 0u;
    for (unsigned long long pixel = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
             + (unsigned long long)threadIdx.x;
         pixel < (unsigned long long)pixels;
         pixel += (unsigned long long)blockDim.x * (unsigned long long)gridDim.x) {
        const float offsets[3] = {red, green, blue};
        float rgb[3];
        for (int channel = 0; channel < 3; ++channel) {
            unsigned long long offset = (unsigned long long)channel * (unsigned long long)pixels + pixel;
            float adjusted = clamp_u8(powf(image[offset], gamma) + offsets[channel]);
            rgb[channel] = clamp_u8(adjusted * brightness);
            image[offset] = rgb[channel];
        }
        value += (unsigned int)floorf(grayscale(rgb[0], rgb[1], rgb[2]));
    }

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5u;
    __shared__ unsigned int warp_totals[8];
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) warp_totals[warp] = value;
    __syncthreads();
    if (warp != 0) return;
    value = lane < 8 ? warp_totals[lane] : 0u;
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) partials[blockIdx.x] = value;
}

extern "C" __global__
void color_adjust_prep_stage2_kernel(
    const unsigned int* partials,
    unsigned int* gray_sum,
    const unsigned int blocks
) {
    if (threadIdx.x != 0) return;
    unsigned int total = 0u;
    for (unsigned int block = 0; block < blocks; ++block) total += partials[block];
    *gray_sum = total;
}

extern "C" __global__
void color_contrast_saturation_kernel(
    float* image,
    const unsigned int* gray_sum,
    const unsigned int pixels,
    const float contrast,
    const float saturation
) {
    unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    float mean = (float)(*gray_sum) / (float)pixels;
    float rgb[3];
    for (int channel = 0; channel < 3; ++channel) {
        rgb[channel] = clamp_u8(
            image[channel * pixels + pixel] * contrast + mean * (1.0f - contrast)
        );
    }
    float gray = floorf(grayscale(rgb[0], rgb[1], rgb[2]));
    for (int channel = 0; channel < 3; ++channel) {
        image[channel * pixels + pixel] = clamp_u8(
            rgb[channel] * saturation + gray * (1.0f - saturation)
        );
    }
}

__device__ __forceinline__ unsigned int hash_u32(unsigned int value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

__device__ __forceinline__ float gaussian_noise(unsigned int key) {
    float u1 = ((float)hash_u32(key) + 1.0f) / 4294967296.0f;
    float u2 = ((float)hash_u32(key ^ 0x9e3779b9u) + 1.0f) / 4294967296.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

__device__ __forceinline__ void hue_shift(float rgb[3], float hue) {
    float r = rgb[0] / 255.0f;
    float g = rgb[1] / 255.0f;
    float b = rgb[2] / 255.0f;
    float maximum = fmaxf(r, fmaxf(g, b));
    float minimum = fminf(r, fminf(g, b));
    float range = maximum - minimum;
    if (range == 0.0f) return;

    float saturation = range / maximum;
    float rc = (maximum - r) / range;
    float gc = (maximum - g) / range;
    float bc = (maximum - b) / range;
    float h = maximum == r ? bc - gc : (maximum == g ? 2.0f + rc - bc : 4.0f + gc - rc);
    h = fmodf(h / 6.0f + 1.0f + hue, 1.0f);
    if (h < 0.0f) h += 1.0f;
    float h6 = h * 6.0f;
    int sector = ((int)floorf(h6)) % 6;
    float fraction = h6 - floorf(h6);
    float p = (1.0f - saturation) * maximum;
    float q = (1.0f - saturation * fraction) * maximum;
    float t = (1.0f - saturation * (1.0f - fraction)) * maximum;
    float out[3];
    if (sector == 0) { out[0] = maximum; out[1] = t; out[2] = p; }
    else if (sector == 1) { out[0] = q; out[1] = maximum; out[2] = p; }
    else if (sector == 2) { out[0] = p; out[1] = maximum; out[2] = t; }
    else if (sector == 3) { out[0] = p; out[1] = q; out[2] = maximum; }
    else if (sector == 4) { out[0] = t; out[1] = p; out[2] = maximum; }
    else { out[0] = maximum; out[1] = p; out[2] = q; }
    for (int channel = 0; channel < 3; ++channel) {
        rgb[channel] = truncf(fminf(fmaxf(out[channel], 0.0f), 1.0f) * 255.999f);
    }
}

extern "C" __global__
void color_sharpness_hue_noise_kernel(
    const float* source,
    float* destination,
    const unsigned int width,
    const unsigned int height,
    const float sharpness,
    const float hue,
    const float noise,
    const unsigned int seed
) {
    unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixels = width * height;
    if (pixel >= pixels) return;
    unsigned int x = pixel % width;
    unsigned int y = pixel / width;
    float rgb[3];
    for (int channel = 0; channel < 3; ++channel) {
        float value = source[channel * pixels + pixel];
        if (x > 0 && x + 1 < width && y > 0 && y + 1 < height) {
            float weighted = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    float weight = (dx == 0 && dy == 0) ? 5.0f : 1.0f;
                    unsigned int neighbor = (y + dy) * width + (x + dx);
                    weighted += source[channel * pixels + neighbor] * weight;
                }
            }
            float blurred = rintf(weighted / 13.0f);
            value = clamp_u8(value * sharpness + blurred * (1.0f - sharpness));
        }
        rgb[channel] = value;
    }
    hue_shift(rgb, hue);
    for (int channel = 0; channel < 3; ++channel) {
        float value = rgb[channel];
        if (noise > 0.0f) {
            value += noise * gaussian_noise(seed ^ pixel ^ (unsigned int)(channel * 0x85ebca6bu));
        }
        destination[channel * pixels + pixel] = fminf(fmaxf(value, 0.0f), 255.0f);
    }
}
