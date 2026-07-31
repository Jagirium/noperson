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

extern "C" __global__
void chw_rgb_to_nhwc_bgr_unit_kernel(
    const float* __restrict__ chw,
    float* __restrict__ nhwc,
    const unsigned int pixels
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = pixels * 3;
    if (idx >= total) return;
    unsigned int pixel = idx / 3;
    unsigned int bgr_channel = idx % 3;
    unsigned int rgb_channel = 2 - bgr_channel;
    nhwc[idx] = chw[rgb_channel * pixels + pixel] * (1.0f / 255.0f);
}

extern "C" __global__
void nhwc_bgr_unit_to_chw_rgb_kernel(
    const float* __restrict__ nhwc,
    float* __restrict__ chw,
    const unsigned int pixels
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = pixels * 3;
    if (idx >= total) return;
    unsigned int rgb_channel = idx / pixels;
    unsigned int pixel = idx % pixels;
    unsigned int bgr_channel = 2 - rgb_channel;
    chw[idx] = fminf(fmaxf(nhwc[pixel * 3 + bgr_channel] * 255.0f, 0.0f), 255.0f);
}
