#include "compat.hpp"

// Bilinear CHW resize used by the frame enhancer.
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
