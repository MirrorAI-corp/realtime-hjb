// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

#include "core/grid.hpp"
#include "core/array_nd.hpp"
#include "options.hpp"

#ifdef HJB_USE_CUDA

namespace hjb {
namespace gpu {

/// Air3D collision-avoidance game (Mitchell, Bayen and Tomlin, IEEE TAC 2005).
///
/// State (x, y, psi): the pursuer's position and heading relative to the
/// evader. Grid dimension 2 (psi) should be periodic on [0, 2 pi).
/// The evader (control) turns at |omega_e| <= omega_max to avoid collision,
/// the pursuer (disturbance) at |omega_p| <= omega_max to cause it:
///     x'   = -v_a + v_b cos(psi) + omega_e y
///     y'   =        v_b sin(psi) - omega_e x
///     psi' = omega_p - omega_e
///     H(x, p) = (-v_a + v_b cos psi) p0 + v_b sin(psi) p1
///               + omega_max |y p0 - x p1 - p2| - omega_max |p2|
/// With the collision set as the zero sublevel set of initial_data, the
/// result's {V <= 0} is the backward reachable tube: states from which the
/// pursuer can force a collision within tMax.
///
/// Scheme: WENO5 + TVD-RK3 with per-cell Lax-Friedrichs bounds; FP32 or FP64.
/// Uses obstacle_sdf / target_sdf when given.
SolveResult solveGPU_Air3D(const Grid& grid, const ArrayND& initial_data,
                           const SolveOptions& opts,
                           double v_a, double v_b, double omega_max);

} // namespace gpu
} // namespace hjb

#endif
