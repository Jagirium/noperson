// Deterministic face-output compaction. One 256-thread block scans anchors in
// fixed chunks; warp ballots retain exact model index order.

__device__ __forceinline__ unsigned int ordered_chunk_offset(
    const unsigned int selected,
    unsigned int* warp_counts,
    unsigned int* warp_offsets,
    unsigned int* chunk_count
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int selected_mask = __ballot_sync(0xffffffffu, selected != 0u);
    if (lane == 31u) warp_counts[warp] = __popc(selected_mask);
    __syncthreads();

    if (warp == 0u && lane < 8u) {
        unsigned int offset = 0;
        for (unsigned int index = 0; index < lane; ++index) offset += warp_counts[index];
        warp_offsets[lane] = offset;
        if (lane == 7u) *chunk_count = offset + warp_counts[7];
    }
    __syncthreads();

    const unsigned int lower_mask = lane == 0u ? 0u : (1u << lane) - 1u;
    return warp_offsets[warp] + __popc(selected_mask & lower_mask);
}

extern "C" __global__
void yolo_face_compact_kernel(
    const float* __restrict__ output,
    float* __restrict__ candidates,
    unsigned int* __restrict__ count,
    const unsigned int anchors,
    const float threshold,
    const float scale
) {
    __shared__ unsigned int warp_counts[8];
    __shared__ unsigned int warp_offsets[8];
    __shared__ unsigned int chunk_count;
    __shared__ unsigned int found;
    const unsigned int lane = threadIdx.x;
    if (lane == 0) found = 0;
    __syncthreads();

    for (unsigned int base = 0; base < anchors; base += 256) {
        const unsigned int i = base + lane;
        const float score = i < anchors ? output[4 * anchors + i] : 0.0f;
        const unsigned int selected = i < anchors && score > threshold;
        const unsigned int exclusive =
            ordered_chunk_offset(selected, warp_counts, warp_offsets, &chunk_count);
        const unsigned int chunk_total = chunk_count;

        if (selected) {
            const float cx = output[i];
            const float cy = output[anchors + i];
            const float width = output[2 * anchors + i];
            const float height = output[3 * anchors + i];
            float* candidate = candidates + (found + exclusive) * 15;
            candidate[0] = (cx - width / 2.0f) / scale;
            candidate[1] = (cy - height / 2.0f) / scale;
            candidate[2] = (cx + width / 2.0f) / scale;
            candidate[3] = (cy + height / 2.0f) / scale;
            for (unsigned int keypoint = 0; keypoint < 5; ++keypoint) {
                candidate[4 + keypoint * 2] =
                    output[(5 + keypoint * 3) * anchors + i] / scale;
                candidate[5 + keypoint * 2] =
                    output[(6 + keypoint * 3) * anchors + i] / scale;
            }
            candidate[14] = score;
        }
        __syncthreads();
        if (lane == 0) found += chunk_total;
        __syncthreads();
    }
    if (lane == 0) *count = found;
}

extern "C" __global__
void anchor_face_compact_kernel(
    const float* __restrict__ output,
    float* __restrict__ candidates,
    unsigned int* __restrict__ count,
    const float threshold,
    const float scale
) {
    __shared__ unsigned int warp_counts[8];
    __shared__ unsigned int warp_offsets[8];
    __shared__ unsigned int chunk_count;
    __shared__ unsigned int found;
    const unsigned int lane = threadIdx.x;
    const unsigned int strides[3] = {8, 16, 32};
    const unsigned int score_offsets[3] = {0, 12800, 16000};
    const unsigned int bbox_offsets[3] = {16800, 68000, 80800};
    const unsigned int kps_offsets[3] = {84000, 212000, 244000};
    if (lane == 0) found = 0;
    __syncthreads();

    for (unsigned int level = 0; level < 3; ++level) {
        const unsigned int stride = strides[level];
        const unsigned int feature = 512 / stride;
        const unsigned int anchors = feature * feature * 2;
        const float s = (float)stride;
        for (unsigned int base = 0; base < anchors; base += 256) {
            const unsigned int i = base + lane;
            const float score = i < anchors ? output[score_offsets[level] + i] : 0.0f;
            const unsigned int selected = i < anchors && score >= threshold;
            const unsigned int exclusive =
                ordered_chunk_offset(selected, warp_counts, warp_offsets, &chunk_count);
            const unsigned int chunk_total = chunk_count;

            if (selected) {
                const unsigned int location = i / 2;
                const float ax = (float)(location % feature) * s;
                const float ay = (float)(location / feature) * s;
                const float* bbox = output + bbox_offsets[level] + i * 4;
                const float* keypoints = output + kps_offsets[level] + i * 10;
                float* candidate = candidates + (found + exclusive) * 15;
                candidate[0] = (ax - bbox[0] * s) / scale;
                candidate[1] = (ay - bbox[1] * s) / scale;
                candidate[2] = (ax + bbox[2] * s) / scale;
                candidate[3] = (ay + bbox[3] * s) / scale;
                for (unsigned int keypoint = 0; keypoint < 5; ++keypoint) {
                    candidate[4 + keypoint * 2] =
                        (ax + keypoints[keypoint * 2] * s) / scale;
                    candidate[5 + keypoint * 2] =
                        (ay + keypoints[keypoint * 2 + 1] * s) / scale;
                }
                candidate[14] = score;
            }
            __syncthreads();
            if (lane == 0) found += chunk_total;
            __syncthreads();
        }
    }
    if (lane == 0) *count = found;
}
