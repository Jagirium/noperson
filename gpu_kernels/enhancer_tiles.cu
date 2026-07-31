// enhancer_tiles.cu — Batch contiguous frame tiles and crop enhanced output.

extern "C" __global__
void enhancer_pack_tiles_kernel(
    const float* __restrict__ frame,  // [C, H, W], raw pixel range
    float* __restrict__ tiles,        // [N, C, T, T]
    const unsigned int frame_h,
    const unsigned int frame_w,
    const unsigned int tiles_x,
    const unsigned int tile_size,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const unsigned int tile_pixels = tile_size * tile_size;
    const unsigned int tile_elements = 3 * tile_pixels;
    const unsigned int tile_idx = idx / tile_elements;
    const unsigned int rem = idx % tile_elements;
    const unsigned int channel = rem / tile_pixels;
    const unsigned int pixel = rem % tile_pixels;
    const unsigned int local_y = pixel / tile_size;
    const unsigned int local_x = pixel % tile_size;
    const unsigned int tile_y = tile_idx / tiles_x;
    const unsigned int tile_x = tile_idx % tiles_x;
    const unsigned int frame_y = tile_y * tile_size + local_y;
    const unsigned int frame_x = tile_x * tile_size + local_x;

    if (frame_y < frame_h && frame_x < frame_w) {
        const unsigned int frame_idx =
            channel * frame_h * frame_w + frame_y * frame_w + frame_x;
        tiles[idx] = frame[frame_idx] * (1.0f / 255.0f);
    } else {
        tiles[idx] = 0.0f;
    }
}

extern "C" __global__
void enhancer_scatter_tiles_kernel(
    const float* __restrict__ tiles,  // [N, C, output_tile, output_tile]
    float* __restrict__ frame,        // [C, output_h, output_w]
    const unsigned int output_h,
    const unsigned int output_w,
    const unsigned int tiles_x,
    const unsigned int output_tile,
    const unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const unsigned int frame_pixels = output_h * output_w;
    const unsigned int channel = idx / frame_pixels;
    const unsigned int pixel = idx % frame_pixels;
    const unsigned int frame_y = pixel / output_w;
    const unsigned int frame_x = pixel % output_w;
    const unsigned int tile_y = frame_y / output_tile;
    const unsigned int tile_x = frame_x / output_tile;
    const unsigned int local_y = frame_y % output_tile;
    const unsigned int local_x = frame_x % output_tile;
    const unsigned int tile_idx = tile_y * tiles_x + tile_x;
    const unsigned int tile_pixels = output_tile * output_tile;
    const unsigned int tile_elements = 3 * tile_pixels;
    const unsigned int tile_offset =
        tile_idx * tile_elements + channel * tile_pixels + local_y * output_tile + local_x;
    const float value = tiles[tile_offset] * 255.0f;
    frame[idx] = fminf(fmaxf(value, 0.0f), 255.0f);
}
