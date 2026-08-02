// Fused latent projection for Inswapper.
//
// Computes L2norm(L2norm(embedding) @ emap) in one 512-thread block while
// preserving the original row-wise accumulation and reduction order.

extern "C" __global__
void calc_latent_512_kernel(
    float* __restrict__ embedding,
    const float* __restrict__ emap,
    float* __restrict__ output
) {
    __shared__ float normalized[512];
    __shared__ float reduction[512];

    const unsigned int col = threadIdx.x;
    const float embedding_value = embedding[col];
    reduction[col] = embedding_value * embedding_value;
    __syncthreads();

    for (unsigned int stride = 256; stride > 0; stride >>= 1) {
        if (col < stride) reduction[col] += reduction[col + stride];
        __syncthreads();
    }

    const float embedding_norm = sqrtf(reduction[0]);
    const float normalized_value = embedding_norm > 1e-10f
        ? embedding_value / embedding_norm
        : embedding_value;
    normalized[col] = normalized_value;
    embedding[col] = normalized_value;
    __syncthreads();

    float sum = 0.0f;
    for (unsigned int row = 0; row < 512; ++row) {
        sum += normalized[row] * emap[row * 512 + col];
    }
    output[col] = sum;
    reduction[col] = sum * sum;
    __syncthreads();

    for (unsigned int stride = 256; stride > 0; stride >>= 1) {
        if (col < stride) reduction[col] += reduction[col + stride];
        __syncthreads();
    }

    const float output_norm = sqrtf(reduction[0]);
    if (output_norm > 1e-10f) output[col] = sum / output_norm;
}
