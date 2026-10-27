// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

// Minimum time-to-reach for a vehicle in a wind field.
//
// A vehicle moves at up to a_max = 1 in any direction, carried by a wind
// w(x) = (0.4, 0) and pushed by a bounded disturbance of size b_max = 0.25.
// The solver propagates the level set of the target (a disk of radius 0.1 at
// the origin) and records, for every cell, the first time the front reaches
// it: the guaranteed time-to-reach (TTR) map.
//
// Usage: time_to_reach_2d [N]       (default N = 200)
// Output: ttr.npy (shape N x N, +inf where the target is not reachable within T).

#include <hjb/hjb.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 200;

    using hjb::BoundaryType;
    hjb::Grid grid = hjb::Grid::create(
        2, {-1.0, -1.0}, {1.0, 1.0}, {N, N},
        {BoundaryType::Extrapolate, BoundaryType::Extrapolate});

    hjb::ArrayND target = hjb::shapeSphere(grid, {0.0, 0.0}, 0.1);

    // The wind is a per-cell field; here it is constant.
    hjb::ArrayND wind_x(grid), wind_y(grid);
    wind_x.fill(0.4);
    wind_y.fill(0.0);

    hjb::SolveOptions opts;
    opts.accuracy = hjb::AccuracyLevel::VeryHigh;  // WENO5 + TVD-RK3
    opts.precision = hjb::Precision::FP32;
    opts.tMax = 1.5;
    opts.computeTTR = true;

    hjb::SolveResult r = hjb::gpu::solveGPU_Drifter(grid, target, opts, wind_x, wind_y,
                                                   /*a_max=*/1.0, /*b_max=*/0.25);

    // Report the time-to-reach from two points the same distance from the
    // target, one upwind and one downwind.
    auto ttr_at = [&](double x, double y) {
        int i = (int)std::lround((x - grid.min[0]) / grid.dx[0]);
        int j = (int)std::lround((y - grid.min[1]) / grid.dx[1]);
        return r.ttr[grid.flatIndex2D(i, j)];
    };
    int reached = 0;
    for (int i = 0; i < grid.total_points; ++i) reached += std::isfinite(r.ttr[i]);

    std::printf("TTR from (-0.6, 0): %.3f   from (+0.6, 0): %.3f\n", ttr_at(-0.6, 0.0), ttr_at(0.6, 0.0));
    std::printf("%d of %d cells reach the target within T = %.1f\n", reached, grid.total_points, opts.tMax);
    hjb::writeNpy("ttr.npy", r.ttr.data(), {N, N});
    std::printf("wrote ttr.npy\n");
    return 0;
}
