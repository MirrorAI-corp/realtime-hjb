// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

/// Templated value-postprocessor kernels used by all GPU solvers.
///
/// `apply_value_postproc_t` implements (analogous to hj_reachability's
/// `value_postprocessor`):
///
///     V_new = obstacle_only:  max(V, obstacle_sdf)
///             target_only:    min(V, target_sdf)
///             reach_avoid:    min(target_sdf, max(V, obstacle_sdf))
///
/// Reach-avoid yields the Mitchell-Tomlin reach-while-avoid value function.
/// Callers pass `obstacle == nullptr` or `target == nullptr` for the
/// degenerate cases.

#include <cuda_runtime.h>

namespace hjb {
namespace gpu {

template <typename T>
__global__ void apply_value_postproc_t(T* d, const T* obstacle, const T* target, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    T v = d[i];
    if (obstacle) v = fmax(v, obstacle[i]);
    if (target)   v = fmin(v, target[i]);
    d[i] = v;
}

} // namespace gpu
} // namespace hjb
