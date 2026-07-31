// mask_postprocess.cu — CrossSwap occluder and XSeg output semantics.

extern "C" __global__
void occluder_threshold_kernel(
    float* __restrict__ mask,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    mask[idx] = mask[idx] > 0.0f ? 1.0f : 0.0f;
}

extern "C" __global__
void xseg_postprocess_kernel(
    float* __restrict__ mask,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float value = fminf(fmaxf(mask[idx], 0.0f), 1.0f);
    mask[idx] = value < 0.1f ? 0.0f : value;
}
