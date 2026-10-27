// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

#ifdef HJB_USE_CUDA

#include "core/grid.hpp"
#include "core/array_nd.hpp"
#include "options.hpp"

namespace hjb {
namespace gpu {

/// Vehicle in a 2D flow field ("drifter"): x' = w(x) + a_max u + b_max d,
/// |u| <= 1 (control), |d| <= 1 (disturbance).
///     H(x, p) = w(x) . p - (a_max - b_max) |p|
/// flow_x and flow_y hold w at every grid cell. Starting from the target as
/// the zero sublevel set of initial_data, SolveResult::ttr is the guaranteed
/// time for a vehicle starting at each cell to reach the target, despite the
/// disturbance. To get the region a vehicle starting at the target can reach
/// instead (forward reachability), pass the negated flow.
///
/// Scheme: WENO5 + TVD-RK3 (VeryHigh/High) or first-order upwind + forward
/// Euler (Low/Medium); Lax-Friedrichs bounds max|w_i| + a_max + b_max;
/// FP32 or FP64. Uses obstacle_sdf / target_sdf and initial_labels.
SolveResult solveGPU_Drifter(const Grid& grid, const ArrayND& initial_data,
                             const SolveOptions& opts,
                             const ArrayND& flow_x, const ArrayND& flow_y,
                             double a_max, double b_max);

/// solveGPU_Drifter with opts.accuracy forced to Low.
SolveResult solveGPU_DrifterLow(const Grid& grid, const ArrayND& initial_data,
                                const SolveOptions& opts,
                                const ArrayND& flow_x, const ArrayND& flow_y,
                                double a_max, double b_max);

} // namespace gpu
} // namespace hjb

#endif // HJB_USE_CUDA
