// warp_affine.cu — Bilinear affine warp for CHW float32 images.
//
// Maps each destination pixel to a source pixel via inverse affine,
// then bilinear samples. Replaces NPP warp_affine for CHW layout
// (NPP operates on HWC interleaved, CHW is more natural for us).

extern "C" __global__
void warp_affine_chw_kernel(
    const float* __restrict__ src,   // [3, src_h, src_w]
    float* __restrict__ dst,         // [3, dst_h, dst_w]
    const unsigned int src_h,
    const unsigned int src_w,
    const unsigned int dst_h,
    const unsigned int dst_w,
    // Inverse affine coefficients (dst → src): 6 floats
    const float inv00, const float inv01, const float inv02,
    const float inv10, const float inv11, const float inv12
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = dst_h * dst_w;
    if (idx >= total) return;

    unsigned int dy = idx / dst_w;
    unsigned int dx = idx % dst_w;

    // Inverse affine: dst pixel → src pixel
    float sx = inv00 * (float)dx + inv01 * (float)dy + inv02;
    float sy = inv10 * (float)dx + inv11 * (float)dy + inv12;

    // Out of bounds → zero
    if (sx < 0.0f || sy < 0.0f || sx >= (float)(src_w - 1) || sy >= (float)(src_h - 1)) {
        for (unsigned int c = 0; c < 3; c++) {
            dst[c * total + idx] = 0.0f;
        }
        return;
    }

    // Bilinear interpolation
    unsigned int x0 = (unsigned int)sx;
    unsigned int y0 = (unsigned int)sy;
    unsigned int x1 = x0 + 1;
    unsigned int y1 = y0 + 1;
    float wx = sx - (float)x0;
    float wy = sy - (float)y0;

    float w00 = (1.0f - wx) * (1.0f - wy);
    float w10 = wx * (1.0f - wy);
    float w01 = (1.0f - wx) * wy;
    float w11 = wx * wy;

    unsigned int src_pixels = src_h * src_w;
    for (unsigned int c = 0; c < 3; c++) {
        float v = src[c * src_pixels + y0 * src_w + x0] * w00
                + src[c * src_pixels + y0 * src_w + x1] * w10
                + src[c * src_pixels + y1 * src_w + x0] * w01
                + src[c * src_pixels + y1 * src_w + x1] * w11;
        dst[c * total + idx] = v;
    }
}

// Resize variant — same math but computes scale-based mapping instead of affine.
extern "C" __global__
void resize_chw_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int src_h,
    const unsigned int src_w,
    const unsigned int dst_h,
    const unsigned int dst_w,
    const unsigned int channels
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = dst_h * dst_w;
    if (idx >= total) return;

    unsigned int dy = idx / dst_w;
    unsigned int dx = idx % dst_w;

    float sy = (float)dy * (float)src_h / (float)dst_h;
    float sx = (float)dx * (float)src_w / (float)dst_w;

    unsigned int y0 = min((unsigned int)sy, src_h - 1);
    unsigned int y1 = min(y0 + 1, src_h - 1);
    unsigned int x0 = min((unsigned int)sx, src_w - 1);
    unsigned int x1 = min(x0 + 1, src_w - 1);
    float wy = sy - (float)y0;
    float wx = sx - (float)x0;

    float w00 = (1.0f - wx) * (1.0f - wy);
    float w10 = wx * (1.0f - wy);
    float w01 = (1.0f - wx) * wy;
    float w11 = wx * wy;

    unsigned int src_pixels = src_h * src_w;
    for (unsigned int c = 0; c < channels; c++) {
        float v = src[c * src_pixels + y0 * src_w + x0] * w00
                + src[c * src_pixels + y0 * src_w + x1] * w10
                + src[c * src_pixels + y1 * src_w + x0] * w01
                + src[c * src_pixels + y1 * src_w + x1] * w11;
        dst[c * total + idx] = v;
    }
}
