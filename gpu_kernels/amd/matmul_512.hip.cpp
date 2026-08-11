#include "compat.hpp"

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

// CrossSwap assignment matcher. Sources are deliberately visited in their
// configured order: this is a first-match operation, never an argmax.
extern "C" __global__
void select_first_embedding_source_kernel(
    const float* __restrict__ query,
    const float* __restrict__ target_bank,
    const float* __restrict__ thresholds,
    const unsigned int* __restrict__ target_present,
    unsigned int source_count,
    unsigned int* __restrict__ selected_index
) {
    __shared__ float dot_reduction[512];
    __shared__ float norm_reduction[512];
    __shared__ float query_norm;
    __shared__ unsigned int matched;

    const unsigned int lane = threadIdx.x;
    const float query_value = query[lane];
    dot_reduction[lane] = query_value * query_value;
    if (lane == 0) {
        *selected_index = 0xffffffffu;
        matched = 0;
    }
    __syncthreads();

    for (unsigned int stride = 256; stride > 0; stride >>= 1) {
        if (lane < stride) dot_reduction[lane] += dot_reduction[lane + stride];
        __syncthreads();
    }
    if (lane == 0) query_norm = sqrtf(dot_reduction[0]);
    __syncthreads();

    for (unsigned int source = 0; source < source_count; ++source) {
        // `None` is an unconditional match at this exact source position.
        if (target_present[source] == 0) {
            if (lane == 0) *selected_index = source;
            return;
        }

        const unsigned long long target_offset =
            static_cast<unsigned long long>(source) * 512ull;
        const float target_value = target_bank[target_offset + lane];
        dot_reduction[lane] = query_value * target_value;
        norm_reduction[lane] = target_value * target_value;
        __syncthreads();

        for (unsigned int stride = 256; stride > 0; stride >>= 1) {
            if (lane < stride) {
                dot_reduction[lane] += dot_reduction[lane + stride];
                norm_reduction[lane] += norm_reduction[lane + stride];
            }
            __syncthreads();
        }

        if (lane == 0) {
            const float denominator = query_norm * sqrtf(norm_reduction[0]);
            const float cosine = denominator > 1e-8f
                ? dot_reduction[0] / denominator
                : 0.0f;
            const float similarity = (1.0f + cosine) * 0.5f;
            if (similarity >= thresholds[source]) {
                *selected_index = source;
                matched = 1;
            }
        }
        __syncthreads();
        if (matched != 0) return;
    }
}
