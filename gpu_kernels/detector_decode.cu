// Deterministic YOLO-face output compaction. A single GPU thread preserves
// model index order exactly while avoiding a 672 KiB device-to-host transfer.

extern "C" __global__
void yolo_face_compact_kernel(
    const float* __restrict__ output, // [20, 8400]
    float* __restrict__ candidates,   // [8400, 15]: bbox4, kps10, score
    unsigned int* __restrict__ count,
    const unsigned int anchors,
    const float threshold,
    const float scale
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int found = 0;
    for (unsigned int i = 0; i < anchors; ++i) {
        const float score = output[4 * anchors + i];
        if (score <= threshold) continue;

        const float cx = output[i];
        const float cy = output[anchors + i];
        const float width = output[2 * anchors + i];
        const float height = output[3 * anchors + i];
        float* candidate = candidates + found * 15;
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
        ++found;
    }
    *count = found;
}
