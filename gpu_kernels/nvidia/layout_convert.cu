// CrossSwap DFM tensors use NHWC BGR in [0, 1]; the rest of the pipeline
// stays CHW RGB in [0, 255].

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
