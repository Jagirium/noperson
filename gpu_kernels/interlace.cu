// interlace.cu — Extract/scatter interlaced tiles for batched Inswapper.
//
// For dim=2, a 256×256 face is split into 4 interlaced 128×128 tiles:
//   tile[0] = face[:, 0::2, 0::2]
//   tile[1] = face[:, 0::2, 1::2]
//   tile[2] = face[:, 1::2, 0::2]
//   tile[3] = face[:, 1::2, 1::2]
//
// Layout: face is [C, H, W] where H=W=dim*tile_size, tiles is [dim*dim, C, tile_size, tile_size]

extern "C" __global__
void interlace_extract_kernel(
    const float* __restrict__ face,   // [C, dim*T, dim*T]
    float* __restrict__ tiles,        // [dim*dim, C, T, T]
    const unsigned int dim,
    const unsigned int C,
    const unsigned int T,             // tile_size (128)
    const unsigned int total          // dim*dim * C * T * T
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int face_size = dim * T;     // e.g. 256
    unsigned int tile_pixels = T * T;
    unsigned int tile_elements = C * tile_pixels;

    // Decode flat index → (tile_idx, c, ty, tx)
    unsigned int tile_idx = idx / tile_elements;
    unsigned int rem = idx % tile_elements;
    unsigned int c = rem / tile_pixels;
    unsigned int pixel = rem % tile_pixels;
    unsigned int ty = pixel / T;
    unsigned int tx = pixel % T;

    // tile_idx → (tj, ti) where tile_idx = tj * dim + ti
    unsigned int tj = tile_idx / dim;
    unsigned int ti = tile_idx % dim;

    // Map to face coordinates
    unsigned int fy = ty * dim + tj;
    unsigned int fx = tx * dim + ti;

    // Read from face [C, face_size, face_size]
    unsigned int face_idx = c * face_size * face_size + fy * face_size + fx;
    tiles[idx] = face[face_idx];
}

extern "C" __global__
void interlace_scatter_kernel(
    const float* __restrict__ tiles,  // [dim*dim, C, T, T]
    float* __restrict__ face,         // [C, dim*T, dim*T]
    const unsigned int dim,
    const unsigned int C,
    const unsigned int T,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int face_size = dim * T;
    unsigned int tile_pixels = T * T;
    unsigned int tile_elements = C * tile_pixels;

    unsigned int tile_idx = idx / tile_elements;
    unsigned int rem = idx % tile_elements;
    unsigned int c = rem / tile_pixels;
    unsigned int pixel = rem % tile_pixels;
    unsigned int ty = pixel / T;
    unsigned int tx = pixel % T;

    unsigned int tj = tile_idx / dim;
    unsigned int ti = tile_idx % dim;

    unsigned int fy = ty * dim + tj;
    unsigned int fx = tx * dim + ti;

    unsigned int face_idx = c * face_size * face_size + fy * face_size + fx;
    face[face_idx] = tiles[idx];
}
