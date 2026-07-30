// cosine_sim.cu — Cosine similarity between 512-dim embeddings on GPU.
//
// Uses shared memory reduction for dot product and norms.
// Single block, 512 threads (one per dimension).

extern "C" __global__
void cosine_sim_kernel(
    const float* __restrict__ a,    // [512]
    const float* __restrict__ b,    // [512]
    float* __restrict__ result,     // [1] output similarity
    const unsigned int dim          // 512
) {
    __shared__ float s_dot[512];
    __shared__ float s_norm_a[512];
    __shared__ float s_norm_b[512];

    unsigned int i = threadIdx.x;
    if (i < dim) {
        float ai = a[i];
        float bi = b[i];
        s_dot[i] = ai * bi;
        s_norm_a[i] = ai * ai;
        s_norm_b[i] = bi * bi;
    } else {
        s_dot[i] = 0.0f;
        s_norm_a[i] = 0.0f;
        s_norm_b[i] = 0.0f;
    }
    __syncthreads();

    // Parallel reduction
    for (unsigned int stride = 256; stride > 0; stride >>= 1) {
        if (i < stride) {
            s_dot[i] += s_dot[i + stride];
            s_norm_a[i] += s_norm_a[i + stride];
            s_norm_b[i] += s_norm_b[i + stride];
        }
        __syncthreads();
    }

    if (i == 0) {
        float denom = sqrtf(s_norm_a[0]) * sqrtf(s_norm_b[0]);
        result[0] = (denom > 1e-8f) ? (s_dot[0] / denom) : 0.0f;
    }
}
