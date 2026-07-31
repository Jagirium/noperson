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

extern "C" __global__
void anchor_face_compact_kernel(
    const float* __restrict__ output, // nine packed RetinaFace/SCRFD tensors
    float* __restrict__ candidates,
    unsigned int* __restrict__ count,
    const float threshold,
    const float scale
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const unsigned int strides[3] = {8, 16, 32};
    const unsigned int score_offsets[3] = {0, 12800, 16000};
    const unsigned int bbox_offsets[3] = {16800, 68000, 80800};
    const unsigned int kps_offsets[3] = {84000, 212000, 244000};
    unsigned int found = 0;

    for (unsigned int level = 0; level < 3; ++level) {
        const unsigned int stride = strides[level];
        const unsigned int feature = 512 / stride;
        const unsigned int anchors = feature * feature * 2;
        const float s = (float)stride;
        for (unsigned int i = 0; i < anchors; ++i) {
            const float score = output[score_offsets[level] + i];
            if (score < threshold) continue;
            const unsigned int location = i / 2;
            const float ax = (float)(location % feature) * s;
            const float ay = (float)(location / feature) * s;
            const float* bbox = output + bbox_offsets[level] + i * 4;
            const float* keypoints = output + kps_offsets[level] + i * 10;
            float* candidate = candidates + found * 15;
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
            ++found;
        }
    }
    *count = found;
}
