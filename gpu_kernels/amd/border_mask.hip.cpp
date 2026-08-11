#include "compat.hpp"

// border_mask.cu — Generate border mask + soft oval on GPU.
//
// Combined kernel: generates border mask with zeroed edges and soft oval,
// all in one pass. Replaces generate_border_mask() + soft_oval_mask().

extern "C" __global__
void border_oval_mask_kernel(
    float* __restrict__ mask,       // [size, size] output
    const unsigned int size,
    const unsigned int border_top,
    const unsigned int border_bottom,   // size - bottom_slider
    const unsigned int border_left,
    const unsigned int border_right,    // size - right_slider
    // Oval parameters
    const float oval_cx,
    const float oval_cy,
    const float oval_rx,
    const float oval_ry,
    const float oval_feather,
    const unsigned int use_oval
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size * size) return;

    unsigned int y = idx / size;
    unsigned int x = idx % size;

    // Border: 0 outside, 1 inside
    float border = 1.0f;
    if (y < border_top || y >= border_bottom || x < border_left || x >= border_right) {
        border = 0.0f;
    }

    // Soft oval
    float dx = ((float)x - oval_cx) / oval_rx;
    float dy = ((float)y - oval_cy) / oval_ry;
    float dist = sqrtf(dx * dx + dy * dy);
    float scale = oval_rx / fmaxf(oval_feather, 1.0f);
    float oval = use_oval
        ? fminf(fmaxf((1.0f - dist) * scale, 0.0f), 1.0f)
        : 1.0f;

    mask[idx] = border * oval;
}
