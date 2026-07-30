// layout_convert.cu — HWC ↔ CHW layout conversion on GPU.
//
// Input frames arrive as HWC (height, width, channels), but ONNX models
// expect CHW (channels, height, width).

extern "C" __global__
void hwc_to_chw_kernel(
    const float* __restrict__ hwc,  // [H, W, C]
    float* __restrict__ chw,        // [C, H, W]
    const unsigned int H,
    const unsigned int W,
    const unsigned int C
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = H * W * C;
    if (idx >= total) return;

    // idx in CHW layout: c * H * W + y * W + x
    unsigned int c = idx / (H * W);
    unsigned int rem = idx % (H * W);
    unsigned int y = rem / W;
    unsigned int x = rem % W;

    // Read from HWC: y * W * C + x * C + c
    unsigned int hwc_idx = y * W * C + x * C + c;
    chw[idx] = hwc[hwc_idx];
}

extern "C" __global__
void chw_to_hwc_kernel(
    const float* __restrict__ chw,  // [C, H, W]
    float* __restrict__ hwc,        // [H, W, C]
    const unsigned int H,
    const unsigned int W,
    const unsigned int C
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = H * W * C;
    if (idx >= total) return;

    // idx in HWC layout: y * W * C + x * C + c
    unsigned int c = idx % C;
    unsigned int rem = idx / C;
    unsigned int x = rem % W;
    unsigned int y = rem / W;

    // Read from CHW: c * H * W + y * W + x
    unsigned int chw_idx = c * H * W + y * W + x;
    hwc[idx] = chw[chw_idx];
}
