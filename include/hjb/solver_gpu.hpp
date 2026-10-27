// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

#ifdef HJB_USE_CUDA

#include "core/grid.hpp"
#include "core/array_nd.hpp"
#include "options.hpp"

namespace hjb {
namespace gpu {

/// Holonomic vehicle in 2D: x' = u, |u| <= inputBound.
/// Hamiltonian H(x, p) = inputBound * |p|. The level set of the target grows at
/// speed inputBound, so SolveResult::ttr is the minimum time to reach it.
/// Scheme: WENO5 + TVD-RK3 (VeryHigh/High), ENO2 (Medium) or first-order
/// upwind (Low); global Lax-Friedrichs. FP64 only.
SolveResult solveGPU_HolonomicL2(const Grid& grid, const ArrayND& initial_data,
                                 const SolveOptions& opts, double inputBound);

/// Double integrator in 2D: state (position, velocity), x1' = x2, x2' = u,
/// |u| <= inputBound. Hamiltonian H(x, p) = x2 * p1 + inputBound * |p2|
/// (the control drives the state into the target). WENO5 + TVD-RK3 with
/// per-cell Lax-Friedrichs bounds (|x2|, inputBound). FP64 only.
SolveResult solveGPU_DoubleIntegrator2D(const Grid& grid, const ArrayND& initial_data,
                                        const SolveOptions& opts, double inputBound);

} // namespace gpu
} // namespace hjb

#endif // HJB_USE_CUDA
