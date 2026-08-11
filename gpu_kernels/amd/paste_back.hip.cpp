#include "compat.hpp"

// paste_back.cu — Inverse-mapping paste with bilinear sampling + alpha blend.
//
// For each frame pixel in the bounding box:
//   1. Apply forward affine to get face coordinates
//   2. Bilinear sample from swapped face
//   3. Bilinear sample from mask
//   4. Blend: frame = swap * mask + frame * (1 - mask)
//
// This replaces the CPU paste_back loop entirely.

extern "C" __global__
void paste_back_kernel(
    float* __restrict__ frame,        // [3, frame_h, frame_w] — modified in place
    const float* __restrict__ swap,   // [3, face_size, face_size]
    const float* __restrict__ mask,   // [face_size, face_size]
    const unsigned int frame_h,
    const unsigned int frame_w,
    const unsigned int face_size,
    // Bounding box in frame space (inclusive)
    const unsigned int left,
    const unsigned int top,
    const unsigned int right,
    const unsigned int bottom,
    // Forward affine coefficients (frame → face): 6 floats
    const float a00, const float a01, const float a02,
    const float a10, const float a11, const float a12
) {
    unsigned int bbox_w = right - left;
    unsigned int bbox_h = bottom - top;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = bbox_w * bbox_h;
    if (idx >= total) return;

    unsigned int dy = idx / bbox_w;
    unsigned int dx = idx % bbox_w;
    unsigned int fx = left + dx;
    unsigned int fy = top + dy;

    // Forward affine: frame pixel → face coordinates
    float sx = a00 * (float)fx + a01 * (float)fy + a02;
    float sy = a10 * (float)fx + a11 * (float)fy + a12;

    // Bounds check in face space
    float fs = (float)face_size;
    if (sx < 0.0f || sy < 0.0f || sx >= fs - 1.0f || sy >= fs - 1.0f) return;

    // Bilinear interpolation coordinates
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

    // Sample mask
    unsigned int fs2 = face_size;
    float m = mask[y0 * fs2 + x0] * w00
            + mask[y0 * fs2 + x1] * w10
            + mask[y1 * fs2 + x0] * w01
            + mask[y1 * fs2 + x1] * w11;

    if (m < 0.001f) return;

    // Blend each channel
    unsigned int frame_pixels = frame_h * frame_w;
    unsigned int face_pixels = face_size * face_size;

    for (unsigned int c = 0; c < 3; c++) {
        float sv = swap[c * face_pixels + y0 * fs2 + x0] * w00
                 + swap[c * face_pixels + y0 * fs2 + x1] * w10
                 + swap[c * face_pixels + y1 * fs2 + x0] * w01
                 + swap[c * face_pixels + y1 * fs2 + x1] * w11;

        unsigned int fi = c * frame_pixels + fy * frame_w + fx;
        frame[fi] = sv * m + frame[fi] * (1.0f - m);
    }
}
