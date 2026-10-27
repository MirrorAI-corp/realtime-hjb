// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

#include <cstddef>
#include <limits>
#include <cmath>

// Host/device annotation macros
#ifdef __CUDACC__
#define HJB_HD __host__ __device__
#define HJB_D  __device__
#else
#define HJB_HD
#define HJB_D
#endif

namespace hjb {

/// Maximum supported grid dimensions
constexpr int kMaxDim = 4;

/// Scheme order. The GPU solvers implement two of these: Low (first-order
/// upwind, forward Euler) and VeryHigh (fifth-order WENO, third-order TVD
/// Runge-Kutta). High is treated as VeryHigh; Medium is second-order ENO for
/// the holonomic solver and first order elsewhere.
enum class AccuracyLevel {
    Low,
    Medium,
    High,
    VeryHigh
};

/// Lax-Friedrichs dissipation type (reserved; see SolveOptions::dissipation)
enum class DissipationType {
    Global,     ///< Global Lax-Friedrichs (GLF)
    Local,      ///< Local Lax-Friedrichs (LLF)
    LocalLocal  ///< Local-Local Lax-Friedrichs (LLLF)
};

/// Boundary condition type per dimension
enum class BoundaryType {
    Extrapolate, ///< Linear extrapolation from 2 interior points
    Periodic     ///< Periodic wrapping
};

/// Floating-point precision of the GPU solve. FP32 is the real-time mode:
/// on consumer GPUs, where FP64 runs at 1/64 of the FP32 rate, it is 20 to 30
/// times faster, and the reachable set differs from FP64 by at most a thin
/// band of cells along its boundary.
enum class Precision {
    FP64,  ///< Double precision (default)
    FP32   ///< Single precision
};

/// Time-to-reach value for cells the front never reaches
constexpr double kInfinity = std::numeric_limits<double>::infinity();

} // namespace hjb
