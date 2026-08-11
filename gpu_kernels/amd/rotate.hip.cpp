#include "compat.hpp"

extern "C" __global__ void rotate_quadrants_chw_kernel(
    const float* src,
    float* dst,
    unsigned int src_h,
    unsigned int src_w,
    unsigned int quarter_turns
) {
    const unsigned int out_h = (quarter_turns & 1U) ? src_w : src_h;
    const unsigned int out_w = (quarter_turns & 1U) ? src_h : src_w;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixels = out_h * out_w;
    if (index >= 3U * pixels) return;

    const unsigned int channel = index / pixels;
    const unsigned int pixel = index % pixels;
    const unsigned int y = pixel / out_w;
    const unsigned int x = pixel % out_w;
    unsigned int src_y;
    unsigned int src_x;
    switch (quarter_turns & 3U) {
        case 1U:
            src_y = x;
            src_x = src_w - 1U - y;
            break;
        case 2U:
            src_y = src_h - 1U - y;
            src_x = src_w - 1U - x;
            break;
        case 3U:
            src_y = src_h - 1U - x;
            src_x = y;
            break;
        default:
            src_y = y;
            src_x = x;
            break;
    }
    dst[index] = src[channel * src_h * src_w + src_y * src_w + src_x];
}
