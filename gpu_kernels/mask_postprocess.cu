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

extern "C" __global__
void imagenet_normalize_kernel(
    float* __restrict__ image,
    const unsigned int plane,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int channel = idx / plane;
    const float means[3] = {0.485f, 0.456f, 0.406f};
    const float stds[3] = {0.229f, 0.224f, 0.225f};
    image[idx] = (image[idx] * (1.0f / 255.0f) - means[channel]) / stds[channel];
}

extern "C" __global__
void parser_argmax_kernel(
    const float* __restrict__ logits, // [19,H,W]
    unsigned char* __restrict__ classes,
    const unsigned int pixels
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    float best = logits[idx];
    unsigned char best_class = 0;
    for (unsigned int class_id = 1; class_id < 19; ++class_id) {
        float value = logits[class_id * pixels + idx];
        if (value > best) {
            best = value;
            best_class = (unsigned char)class_id;
        }
    }
    classes[idx] = best_class;
}

extern "C" __global__
void parser_class_mask_kernel(
    const unsigned char* __restrict__ classes,
    float* __restrict__ mask,
    const unsigned int pixels,
    const unsigned int class_id,
    const unsigned int foreground_mode
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    unsigned int actual = (unsigned int)classes[idx];
    if (foreground_mode != 0) {
        bool background = actual == 0 || actual == 14 || actual == 15
            || actual == 16 || actual == 17 || actual == 18;
        mask[idx] = background ? 0.0f : 1.0f;
    } else {
        mask[idx] = actual == class_id ? 1.0f : 0.0f;
    }
}

extern "C" __global__
void mask_invert_kernel(
    float* __restrict__ mask,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    mask[idx] = 1.0f - mask[idx];
}
