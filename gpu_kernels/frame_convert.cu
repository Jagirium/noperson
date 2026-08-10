// frame_convert.cu — u8 ↔ f32 layout conversion for camera frames.
//
// Webcam delivers HWC u8 RGB bytes. ONNX pipeline wants CHW f32 in [0, 255].
// Display wants HWC u8 again for QImage.
//
// These kernels move the conversion off the CPU's nested scalar loops.

extern "C" __global__
void hwc_u8_to_chw_f32_kernel(
    const unsigned char* __restrict__ hwc,  // [H, W, 3] u8
    float* __restrict__ chw,                // [3, H, W] f32
    const unsigned int H,
    const unsigned int W
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = H * W;
    if (idx >= total) return;

    unsigned int y = idx / W;
    unsigned int x = idx % W;

    unsigned int src = (y * W + x) * 3;
    unsigned int plane = H * W;

    chw[0 * plane + y * W + x] = (float)hwc[src + 0];
    chw[1 * plane + y * W + x] = (float)hwc[src + 1];
    chw[2 * plane + y * W + x] = (float)hwc[src + 2];
}

extern "C" __global__
void chw_f32_to_hwc_u8_kernel(
    const float* __restrict__ chw,          // [3, H, W] f32 in [0, 255]
    unsigned char* __restrict__ hwc,        // [H, W, 3] u8
    const unsigned int H,
    const unsigned int W
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = H * W;
    if (idx >= total) return;

    unsigned int y = idx / W;
    unsigned int x = idx % W;
    unsigned int plane = H * W;

    float r = chw[0 * plane + y * W + x];
    float g = chw[1 * plane + y * W + x];
    float b = chw[2 * plane + y * W + x];

    r = fminf(fmaxf(r, 0.0f), 255.0f);
    g = fminf(fmaxf(g, 0.0f), 255.0f);
    b = fminf(fmaxf(b, 0.0f), 255.0f);

    unsigned int dst = (y * W + x) * 3;
    hwc[dst + 0] = (unsigned char)r;
    hwc[dst + 1] = (unsigned char)g;
    hwc[dst + 2] = (unsigned char)b;
}

extern "C" __global__
void chw_f32_to_rgba_u8_pitched_kernel(
    const float* __restrict__ chw,
    unsigned char* __restrict__ rgba,
    const unsigned int H,
    const unsigned int W,
    const unsigned int row_bytes
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = H * W;
    if (idx >= total) return;

    unsigned int y = idx / W;
    unsigned int x = idx % W;
    unsigned int plane = H * W;
    unsigned int dst = y * row_bytes + x * 4;
    float r = fminf(fmaxf(chw[0 * plane + idx], 0.0f), 255.0f);
    float g = fminf(fmaxf(chw[1 * plane + idx], 0.0f), 255.0f);
    float b = fminf(fmaxf(chw[2 * plane + idx], 0.0f), 255.0f);
    rgba[dst + 0] = (unsigned char)r;
    rgba[dst + 1] = (unsigned char)g;
    rgba[dst + 2] = (unsigned char)b;
    rgba[dst + 3] = 255;
}

__device__ __forceinline__
float sample_chw_bilinear(
    const float* __restrict__ chw,
    const unsigned int channel,
    const unsigned int src_h,
    const unsigned int src_w,
    const unsigned int dst_y,
    const unsigned int dst_x,
    const unsigned int dst_h,
    const unsigned int dst_w
) {
    float sx = ((float)dst_x + 0.5f) * ((float)src_w / (float)dst_w) - 0.5f;
    float sy = ((float)dst_y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    sx = fminf(fmaxf(sx, 0.0f), (float)(src_w - 1));
    sy = fminf(fmaxf(sy, 0.0f), (float)(src_h - 1));

    unsigned int x0 = (unsigned int)sx;
    unsigned int y0 = (unsigned int)sy;
    unsigned int x1 = min(x0 + 1, src_w - 1);
    unsigned int y1 = min(y0 + 1, src_h - 1);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    unsigned int plane = src_h * src_w;
    const float* src = chw + channel * plane;

    float top = src[y0 * src_w + x0] * (1.0f - fx)
              + src[y0 * src_w + x1] * fx;
    float bottom = src[y1 * src_w + x0] * (1.0f - fx)
                 + src[y1 * src_w + x1] * fx;
    return top * (1.0f - fy) + bottom * fy;
}

__device__ __forceinline__
unsigned char round_clamp_u8(float value) {
    return (unsigned char)__float2int_rn(fminf(fmaxf(value, 0.0f), 255.0f));
}

// Pitched external surfaces are byte-addressed. Do not cast row starts to
// unsigned short*: an odd byte pitch makes every second row unaligned.
__device__ __forceinline__
void store_p010_sample(
    unsigned char* row,
    const unsigned int sample_index,
    const float code_10
) {
    const unsigned short packed = ((unsigned short)(code_10 + 0.5f)) << 6;
    const unsigned int byte_offset = sample_index * 2;
    row[byte_offset] = (unsigned char)(packed & 0xffu);
    row[byte_offset + 1] = (unsigned char)(packed >> 8);
}

// Pitch-aware NV12/P010 surface -> CHW f32 RGB [0,255]. The source pointer
// may be owned by NVDEC; no wrapper allocation or device copy is required.
extern "C" __global__
void nv12_to_chw_f32_kernel(
    const unsigned char* __restrict__ source,
    const unsigned int pitch,
    float* __restrict__ chw,
    const unsigned int H,
    const unsigned int W,
    const unsigned int p010,
    const float y_offset,
    const float y_scale,
    const float rv,
    const float gu,
    const float gv,
    const float bu
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = H * W;
    if (idx >= total) return;

    unsigned int y = idx / W;
    unsigned int x = idx % W;
    float y_code;
    float u_code;
    float v_code;
    if (p010 != 0) {
        const unsigned short* y_plane = (const unsigned short*)source;
        const unsigned short* uv_plane =
            (const unsigned short*)(source + (unsigned long long)pitch * H);
        unsigned int pitch_samples = pitch / 2;
        y_code = (float)(y_plane[y * pitch_samples + x] >> 6) * 0.25f;
        unsigned int uv = (y / 2) * pitch_samples + (x & ~1u);
        u_code = (float)(uv_plane[uv] >> 6) * 0.25f;
        v_code = (float)(uv_plane[uv + 1] >> 6) * 0.25f;
    } else {
        const unsigned char* uv_plane = source + (unsigned long long)pitch * H;
        y_code = (float)source[y * pitch + x];
        unsigned int uv = (y / 2) * pitch + (x & ~1u);
        u_code = (float)uv_plane[uv];
        v_code = (float)uv_plane[uv + 1];
    }

    float luma = (y_code - y_offset) * y_scale;
    float cb = u_code - 128.0f;
    float cr = v_code - 128.0f;
    float r = luma + rv * cr;
    float g = luma + gu * cb + gv * cr;
    float b = luma + bu * cb;
    unsigned int plane = H * W;
    chw[idx] = fminf(fmaxf(r, 0.0f), 255.0f);
    chw[plane + idx] = fminf(fmaxf(g, 0.0f), 255.0f);
    chw[2 * plane + idx] = fminf(fmaxf(b, 0.0f), 255.0f);
}

// CHW f32 RGB [3, src_h, src_w] in [0, 255] -> scaled NV12.
// One thread writes one 2x2 luma macroblock and its interleaved UV sample.
extern "C" __global__
void chw_f32_to_nv12_scaled_kernel(
    const float* __restrict__ chw,
    unsigned char* __restrict__ nv12,
    const unsigned int pitch,
    const unsigned int src_h,
    const unsigned int src_w,
    const unsigned int dst_h,
    const unsigned int dst_w,
    const unsigned int p010,
    const float yr,
    const float yg,
    const float yb,
    const float y_offset,
    const float ur,
    const float ug,
    const float ub,
    const float vr,
    const float vg,
    const float vb,
    const float uv_offset
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (dst_h / 2) * (dst_w / 2);
    if (idx >= total) return;

    unsigned int macro_index = idx;
    unsigned int macro_w = dst_w / 2;
    unsigned int x = (macro_index % macro_w) * 2;
    unsigned int y = (macro_index / macro_w) * 2;

    // Keep the sampling and accumulation order byte-identical to the former
    // even-coordinate path: 00, 01, 10, then 11. Each RGB triplet is sampled
    // once and retained for both luma and chroma.
    float r00 = sample_chw_bilinear(chw, 0, src_h, src_w, y, x, dst_h, dst_w);
    float g00 = sample_chw_bilinear(chw, 1, src_h, src_w, y, x, dst_h, dst_w);
    float b00 = sample_chw_bilinear(chw, 2, src_h, src_w, y, x, dst_h, dst_w);
    float r01 = sample_chw_bilinear(chw, 0, src_h, src_w, y, x + 1, dst_h, dst_w);
    float g01 = sample_chw_bilinear(chw, 1, src_h, src_w, y, x + 1, dst_h, dst_w);
    float b01 = sample_chw_bilinear(chw, 2, src_h, src_w, y, x + 1, dst_h, dst_w);
    float r10 = sample_chw_bilinear(chw, 0, src_h, src_w, y + 1, x, dst_h, dst_w);
    float g10 = sample_chw_bilinear(chw, 1, src_h, src_w, y + 1, x, dst_h, dst_w);
    float b10 = sample_chw_bilinear(chw, 2, src_h, src_w, y + 1, x, dst_h, dst_w);
    float r11 = sample_chw_bilinear(chw, 0, src_h, src_w, y + 1, x + 1, dst_h, dst_w);
    float g11 = sample_chw_bilinear(chw, 1, src_h, src_w, y + 1, x + 1, dst_h, dst_w);
    float b11 = sample_chw_bilinear(chw, 2, src_h, src_w, y + 1, x + 1, dst_h, dst_w);

    float y00 = yr * r00 + yg * g00 + yb * b00 + y_offset;
    float y01 = yr * r01 + yg * g01 + yb * b01 + y_offset;
    float y10 = yr * r10 + yg * g10 + yb * b10 + y_offset;
    float y11 = yr * r11 + yg * g11 + yb * b11 + y_offset;
    if (p010 != 0) {
        unsigned char* p010_output = nv12 + (size_t)y * pitch;
        float y00_10 = fminf(fmaxf(y00 * 4.0f, 0.0f), 1023.0f);
        float y01_10 = fminf(fmaxf(y01 * 4.0f, 0.0f), 1023.0f);
        float y10_10 = fminf(fmaxf(y10 * 4.0f, 0.0f), 1023.0f);
        float y11_10 = fminf(fmaxf(y11 * 4.0f, 0.0f), 1023.0f);
        store_p010_sample(p010_output, x, y00_10);
        store_p010_sample(p010_output, x + 1, y01_10);
        unsigned char* p010_next_row = nv12 + (size_t)(y + 1) * pitch;
        store_p010_sample(p010_next_row, x, y10_10);
        store_p010_sample(p010_next_row, x + 1, y11_10);
    } else {
        unsigned char* y_row = nv12 + (size_t)y * pitch;
        unsigned char* y_next_row = nv12 + (size_t)(y + 1) * pitch;
        y_row[x] = round_clamp_u8(y00);
        y_row[x + 1] = round_clamp_u8(y01);
        y_next_row[x] = round_clamp_u8(y10);
        y_next_row[x + 1] = round_clamp_u8(y11);
    }

    // NV12 chroma represents a 2x2 luma block. Average the four resampled RGB
    // pixels before applying the selected YUV matrix and range transform.
    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;
    sum_r += r00;
    sum_g += g00;
    sum_b += b00;
    sum_r += r01;
    sum_g += g01;
    sum_b += b01;
    sum_r += r10;
    sum_g += g10;
    sum_b += b10;
    sum_r += r11;
    sum_g += g11;
    sum_b += b11;
    float r = sum_r * 0.25f;
    float g = sum_g * 0.25f;
    float b = sum_b * 0.25f;
    float u_code = ur * r + ug * g + ub * b + uv_offset;
    float v_code = vr * r + vg * g + vb * b + uv_offset;
    unsigned char* uv_plane = nv12 + (size_t)pitch * dst_h;
    if (p010 != 0) {
        unsigned char* p010_output = uv_plane + (size_t)(y / 2) * pitch;
        float u10 = fminf(fmaxf(u_code * 4.0f, 0.0f), 1023.0f);
        float v10 = fminf(fmaxf(v_code * 4.0f, 0.0f), 1023.0f);
        store_p010_sample(p010_output, x, u10);
        store_p010_sample(p010_output, x + 1, v10);
    } else {
        unsigned char* uv_row = uv_plane + (size_t)(y / 2) * pitch;
        uv_row[x] = round_clamp_u8(u_code);
        uv_row[x + 1] = round_clamp_u8(v_code);
    }
}

// Letterbox resize + normalize: src [3, src_h, src_w] f32 in [0, 255] →
// dst [3, target, target] f32 in [0, 1] (or scaled via mul/add).
//
// Matches letterbox_resize_normalize() CPU reference:
//   - Preserve aspect ratio, resize so that max dim = target
//   - Padding on right/bottom filled with `add`
//   - output = pixel * mul + add
//
// One thread per destination pixel per channel.
extern "C" __global__
void letterbox_resize_kernel(
    const float* __restrict__ src,  // [3, src_h, src_w]
    float* __restrict__ dst,        // [3, target, target]
    const unsigned int src_h,
    const unsigned int src_w,
    const unsigned int target,
    const unsigned int new_h,
    const unsigned int new_w,
    const float mul,
    const float add
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = 3 * target * target;
    if (idx >= total) return;

    unsigned int c = idx / (target * target);
    unsigned int rem = idx % (target * target);
    unsigned int y = rem / target;
    unsigned int x = rem % target;

    if (y >= new_h || x >= new_w) {
        // Padding area
        dst[idx] = add;
        return;
    }

    float sx = (float)x * ((float)src_w / (float)new_w);
    float sy = (float)y * ((float)src_h / (float)new_h);

    unsigned int x0 = (unsigned int)sx;
    unsigned int y0 = (unsigned int)sy;
    if (x0 >= src_w - 1) x0 = src_w - 2;
    if (y0 >= src_h - 1) y0 = src_h - 2;
    unsigned int x1 = x0 + 1;
    unsigned int y1 = y0 + 1;

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    unsigned int plane = src_h * src_w;
    float v00 = src[c * plane + y0 * src_w + x0];
    float v01 = src[c * plane + y0 * src_w + x1];
    float v10 = src[c * plane + y1 * src_w + x0];
    float v11 = src[c * plane + y1 * src_w + x1];

    float v = v00 * (1.0f - fx) * (1.0f - fy)
            + v01 * fx * (1.0f - fy)
            + v10 * (1.0f - fx) * fy
            + v11 * fx * fy;

    dst[idx] = v * mul + add;
}

// Element-wise normalize in-place: data[i] = data[i] * mul + add.
extern "C" __global__
void affine_scale_kernel(
    float* __restrict__ data,
    const unsigned int n,
    const float mul,
    const float add
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = data[idx] * mul + add;
}

// Out-of-place variant used when model input staging would otherwise require
// a full device copy followed by affine_scale_kernel.
extern "C" __global__
void affine_scale_copy_kernel(
    const float* __restrict__ source,
    float* __restrict__ destination,
    const unsigned int n,
    const float mul,
    const float add
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    destination[idx] = source[idx] * mul + add;
}
