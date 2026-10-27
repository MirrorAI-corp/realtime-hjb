// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

// Backward reachable tube for the Air3D collision-avoidance game
// (Mitchell, Bayen and Tomlin, IEEE TAC 2005).
//
// Two aircraft fly at 5 m/s with turn rates of at most 1 rad/s. The state is
// the pursuer's position and heading relative to the evader. The evader steers
// to avoid collision (distance below 5 m) and the pursuer steers to cause it.
// The BRT at horizon T is every relative state from which the pursuer can force
// a collision within T seconds, whatever the evader does.
//
// Usage: air3d_brt [N] [fp64]      (default N = 51, FP32)
// Output: air3d_V.npy, the value function at t = T (shape N x N x N).

#include <hjb/hjb.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 51;
    const bool fp64 = argc > 2 && std::strcmp(argv[2], "fp64") == 0;

    // Relative state (x, y, psi): x and y in metres, psi the relative heading.
    // psi wraps around, so its boundary is periodic.
    using hjb::BoundaryType;
    hjb::Grid grid = hjb::Grid::create(
        3, {-20.0, -20.0, 0.0}, {20.0, 20.0, 2.0 * M_PI}, {N, N, N},
        {BoundaryType::Extrapolate, BoundaryType::Extrapolate, BoundaryType::Periodic});

    // Collision set: within 5 m in (x, y), for every heading. V0 < 0 inside.
    hjb::ArrayND v0 = hjb::shapeCylinder(grid, {false, false, true}, {0.0, 0.0, 0.0}, 5.0);

    hjb::SolveOptions opts;
    opts.accuracy = hjb::AccuracyLevel::VeryHigh;  // WENO5 + TVD-RK3
    opts.precision = fp64 ? hjb::Precision::FP64 : hjb::Precision::FP32;
    opts.tMax = 2.0;                                // horizon T, seconds
    opts.computeTTR = false;

    const double v_a = 5.0, v_b = 5.0, omega_max = 1.0;

    // The first call also creates the CUDA context; time a second call.
    hjb::gpu::solveGPU_Air3D(grid, v0, opts, v_a, v_b, omega_max);
    auto t0 = std::chrono::steady_clock::now();
    hjb::SolveResult r = hjb::gpu::solveGPU_Air3D(grid, v0, opts, v_a, v_b, omega_max);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    int inside = 0;
    for (int i = 0; i < grid.total_points; ++i) inside += r.final_data[i] <= 0.0;

    std::printf("Air3D BRT, N=%d, %s: %d of %d cells in the tube, solved in %.2f ms\n",
                N, fp64 ? "FP64" : "FP32", inside, grid.total_points, ms);
    hjb::writeNpy("air3d_V.npy", r.final_data.data(), {N, N, N});
    std::printf("wrote air3d_V.npy\n");
    return 0;
}
