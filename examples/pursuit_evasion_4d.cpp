// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

// Four-dimensional pursuit-evasion in the plane.
//
// The state is (x_ego, y_ego, x_obs, y_obs): the positions of a pursuer (speed
// up to v_ego = 1) and an evader (speed up to v_obs = 0.7) in [-1, 1]^2.
// Capture happens when they are closer than 0.2. The tube at horizon T is every
// joint state from which the pursuer can capture the evader within T, whatever
// the evader does.
//
// Usage: pursuit_evasion_4d [N] [fp64]    (default N = 31, FP32)

#include <hjb/hjb.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 31;
    const bool fp64 = argc > 2 && std::strcmp(argv[2], "fp64") == 0;

    using hjb::BoundaryType;
    const BoundaryType E = BoundaryType::Extrapolate;
    hjb::Grid grid = hjb::Grid::create(4, {-1, -1, -1, -1}, {1, 1, 1, 1}, {N, N, N, N}, {E, E, E, E});

    // Capture set: distance between the two agents below 0.2. Initial values
    // can be any function that is negative exactly on the target set; here it
    // is the signed distance, written out as a loop over grid cells.
    hjb::ArrayND v0(grid);
    int idx[hjb::kMaxDim];
    for (int i = 0; i < grid.total_points; ++i) {
        grid.unflatIndex(i, idx);
        double dx = grid.coord(0, idx[0]) - grid.coord(2, idx[2]);
        double dy = grid.coord(1, idx[1]) - grid.coord(3, idx[3]);
        v0[i] = std::sqrt(dx * dx + dy * dy) - 0.2;
    }

    hjb::SolveOptions opts;
    opts.accuracy = hjb::AccuracyLevel::VeryHigh;
    opts.precision = fp64 ? hjb::Precision::FP64 : hjb::Precision::FP32;
    opts.tMax = 0.8;
    opts.computeTTR = false;

    hjb::gpu::solveGPU_PursuitEvasion4D(grid, v0, opts, 1.0, 0.7);  // warm-up (CUDA context)
    auto t0 = std::chrono::steady_clock::now();
    hjb::SolveResult r = hjb::gpu::solveGPU_PursuitEvasion4D(grid, v0, opts, /*v_ego_max=*/1.0, /*v_obs=*/0.7);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    int inside = 0;
    for (int i = 0; i < grid.total_points; ++i) inside += r.final_data[i] <= 0.0;
    std::printf("4D pursuit-evasion BRT, N=%d, %s: %d of %d cells, solved in %.1f ms\n",
                N, fp64 ? "FP64" : "FP32", inside, grid.total_points, ms);
    return 0;
}
