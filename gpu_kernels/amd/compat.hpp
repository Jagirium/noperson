#pragma once

#include <hip/hip_runtime.h>

constexpr unsigned int NOPERSON_MAX_SUBGROUPS_PER_BLOCK = 32;

__device__ __forceinline__ unsigned int noperson_lane_id() {
    return threadIdx.x % warpSize;
}

__device__ __forceinline__ unsigned int noperson_subgroup_id() {
    return threadIdx.x / warpSize;
}

__device__ __forceinline__ unsigned int noperson_subgroups_in_block() {
    return (blockDim.x + warpSize - 1u) / warpSize;
}

template <typename T>
__device__ __forceinline__ T noperson_subgroup_reduce_sum(T value) {
    for (unsigned int offset = warpSize / 2u; offset > 0u; offset >>= 1u) {
        value += __shfl_down(value, offset, warpSize);
    }
    return value;
}

__device__ __forceinline__ unsigned long long noperson_subgroup_ballot(bool selected) {
    return __ballot(selected);
}

__device__ __forceinline__ unsigned int noperson_popcount(unsigned long long mask) {
    return __popcll(mask);
}

__device__ __forceinline__ unsigned long long noperson_lower_lane_mask(
    unsigned int lane
) {
    return lane == 0u ? 0ull : (1ull << lane) - 1ull;
}
