// alpha_blend.cu — Alpha compositing on GPU.
//
// For each pixel: out[c] = src[c] * mask + dst[c] * (1.0 - mask)
// mask is single-channel, src/dst are 3-channel.

extern "C" __global__
void alpha_blend_kernel(
    const float* __restrict__ src,   // [C, H, W]
    const float* __restrict__ dst,   // [C, H, W]
    const float* __restrict__ mask,  // [1, H, W]
    float* __restrict__ out,         // [C, H, W]
    const unsigned int n_pixels,     // H * W
    const unsigned int C             // number of channels (3)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels * C) return;

    unsigned int pixel = i % n_pixels;
    float m = mask[pixel];
    out[i] = src[i] * m + dst[i] * (1.0f - m);
}

// Constant-alpha blend used by the face restorer. `dst` is updated in place,
// so no third 512x512 allocation or host round-trip is needed.
extern "C" __global__
void scalar_blend_inplace_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int n,
    const float alpha
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i] * alpha + dst[i] * (1.0f - alpha);
}
