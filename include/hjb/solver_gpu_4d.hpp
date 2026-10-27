// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

#ifdef HJB_USE_CUDA

#include "core/grid.hpp"
#include "core/array_nd.hpp"
#include "options.hpp"

#include <vector>

namespace hjb {
namespace gpu {

/// Pursuit-evasion between two agents in the plane.
///
/// State (x_ego, y_ego, x_obs, y_obs). Our agent, the pursuer (control), moves
/// at speed up to v_ego; the other agent, the evader (disturbance), at speed up
/// to v_obs:
///     H(x, p) = -v_ego |(p0, p1)| + v_obs |(p2, p3)|
/// With the capture set (for example, the two agents closer than r) as the
/// zero sublevel set of initial_data, the result's {V <= 0} is the set of joint
/// states from which our agent can capture the evader within tMax, whatever
/// the evader does.
///
/// wall_mask (optional, grid.total_points entries): 1.0 marks states that stay
/// frozen at their initial value (for example, either agent inside a wall).
/// ego_speed_map (optional, N0 * N1 entries): our agent's speed limit as a
/// function of its own position, overriding v_ego there; values must not
/// exceed v_ego.
///
/// Scheme: WENO5 + TVD-RK3, Lax-Friedrichs coefficients from the local
/// gradient direction; FP32 or FP64. Uses value_clip_lo / value_clip_hi.
SolveResult solveGPU_PursuitEvasion4D(const Grid& grid,
                                      const ArrayND& initial_data,
                                      const SolveOptions& opts,
                                      double v_ego,
                                      double v_obs,
                                      const std::vector<double>& wall_mask = {},
                                      const std::vector<double>& ego_speed_map = {});

} // namespace gpu
} // namespace hjb

#endif
