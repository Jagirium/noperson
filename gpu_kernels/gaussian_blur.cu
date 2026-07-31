// gaussian_blur.cu — separable 1D Gaussian blur for single-channel masks.
//
// Two-pass:
//   horizontal: src [H, W] → tmp [H, W]  (1D kernel along x)
//   vertical:   tmp [H, W] → dst [H, W]  (1D kernel along y)
//
// Kernel weights are uploaded as a small f32 array (max 65 taps).

__device__ __forceinline__ int gaussian_border_index(
    int index,
    const int length,
    const unsigned int border_mode
) {
    if (index >= 0 && index < length) return index;
    if (border_mode == 1) return -1; // zero padding
    if (length <= 1) return 0;
    while (index < 0 || index >= length) {
        index = index < 0 ? -index : 2 * length - 2 - index;
    }
    return index; // torchvision reflect padding (edge is not repeated)
}

extern "C" __global__
void gaussian_blur_h_kernel(
    const float* __restrict__ src,   // [H, W]
    float* __restrict__ dst,         // [H, W]
    const float* __restrict__ kernel,// [ks]
    const unsigned int H,
    const unsigned int W,
    const unsigned int ks,
    const unsigned int border_mode
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = H * W;
    if (idx >= total) return;

    unsigned int y = idx / W;
    unsigned int x = idx % W;
    int half = (int)ks / 2;

    float val = 0.0f;
    for (int k = 0; k < (int)ks; ++k) {
        int sx = (int)x + k - half;
        sx = gaussian_border_index(sx, (int)W, border_mode);
        if (sx >= 0) val += src[y * W + sx] * kernel[k];
    }
    dst[y * W + x] = val;
}

extern "C" __global__
void gaussian_blur_v_kernel(
    const float* __restrict__ src,   // [H, W]
    float* __restrict__ dst,         // [H, W]
    const float* __restrict__ kernel,// [ks]
    const unsigned int H,
    const unsigned int W,
    const unsigned int ks,
    const unsigned int border_mode
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = H * W;
    if (idx >= total) return;

    unsigned int y = idx / W;
    unsigned int x = idx % W;
    int half = (int)ks / 2;

    float val = 0.0f;
    for (int k = 0; k < (int)ks; ++k) {
        int sy = (int)y + k - half;
        sy = gaussian_border_index(sy, (int)H, border_mode);
        if (sy >= 0) val += src[sy * W + x] * kernel[k];
    }
    dst[y * W + x] = val;
}

// Resize single-channel mask [src_h, src_w] → [dst_h, dst_w] via bilinear.
extern "C" __global__
void mask_resize_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int src_h,
    const unsigned int src_w,
    const unsigned int dst_h,
    const unsigned int dst_w
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = dst_h * dst_w;
    if (idx >= total) return;

    unsigned int dy = idx / dst_w;
    unsigned int dx = idx % dst_w;

    float sy = (float)dy * ((float)src_h / (float)dst_h);
    float sx = (float)dx * ((float)src_w / (float)dst_w);

    unsigned int y0 = (unsigned int)sy;
    unsigned int x0 = (unsigned int)sx;
    if (y0 >= src_h - 1) y0 = src_h - 2;
    if (x0 >= src_w - 1) x0 = src_w - 2;
    unsigned int y1 = y0 + 1;
    unsigned int x1 = x0 + 1;
    float wy = sy - (float)y0;
    float wx = sx - (float)x0;

    float v = src[y0 * src_w + x0] * (1.0f - wx) * (1.0f - wy)
            + src[y0 * src_w + x1] * wx * (1.0f - wy)
            + src[y1 * src_w + x0] * (1.0f - wx) * wy
            + src[y1 * src_w + x1] * wx * wy;
    dst[idx] = v;
}

// Element-wise multiply two same-sized buffers in place: a[i] *= b[i].
extern "C" __global__
void mask_mul_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    const unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = a[idx] * b[idx];
}
