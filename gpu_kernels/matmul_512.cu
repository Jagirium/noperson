// matmul_512.cu — Matrix-vector multiply for calc_inswapper_latent.
//
// Computes: latent[j] = sum_i(embedding[i] * emap[i][j]) for j in 0..512
// This is: latent = embedding @ emap  (row-vector × matrix → row-vector)
// Then L2-normalize the result.
//
// emap is [512, 512] row-major (emap[i*512 + j] = element at row i, col j).
// embedding is [512], output (latent) is [512].

extern "C" __global__
void matmul_512_kernel(
    const float* __restrict__ embedding,  // [512]
    const float* __restrict__ emap,       // [512, 512] row-major
    float* __restrict__ output,           // [512]
    const unsigned int dim                // 512
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= dim) return;

    float sum = 0.0f;
    // latent[col] = sum_row(embedding[row] * emap[row*dim + col])
    for (unsigned int row = 0; row < dim; row++) {
        sum += embedding[row] * emap[row * dim + col];
    }
    output[col] = sum;
}

// L2 normalize a vector in-place.
extern "C" __global__
void l2_normalize_kernel(
    float* __restrict__ vec,
    const unsigned int dim
) {
    __shared__ float s_sum[512];

    unsigned int i = threadIdx.x;
    if (i < dim) {
        s_sum[i] = vec[i] * vec[i];
    } else {
        s_sum[i] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = 256; stride > 0; stride >>= 1) {
        if (i < stride) {
            s_sum[i] += s_sum[i + stride];
        }
        __syncthreads();
    }

    if (i < dim) {
        float norm = sqrtf(s_sum[0]);
        if (norm > 1e-10f) {
            vec[i] /= norm;
        }
    }
}
