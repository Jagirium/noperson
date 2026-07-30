// normalize.cu — Normalize/denormalize pixel values on GPU.
//
// normalize_kernel:   out[i] = in[i] * (1.0f / 255.0f)
// denormalize_kernel: out[i] = clamp(in[i] * 255.0f, 0.0f, 255.0f)

extern "C" __global__
void normalize_kernel(float* __restrict__ data, const unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * (1.0f / 255.0f);
    }
}

extern "C" __global__
void denormalize_kernel(float* __restrict__ data, const unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = data[i] * 255.0f;
        data[i] = fminf(fmaxf(v, 0.0f), 255.0f);
    }
}
