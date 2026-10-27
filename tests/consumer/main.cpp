// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

// Links against an installed RealtimeHJB. Building it is the test; running it
// needs a GPU.

#include <hjb/hjb.hpp>

#include <cstdio>

int main() {
    using hjb::BoundaryType;
    hjb::Grid grid = hjb::Grid::create(2, {-1.0, -1.0}, {1.0, 1.0}, {51, 51},
                                       {BoundaryType::Extrapolate, BoundaryType::Extrapolate});
    hjb::ArrayND v0 = hjb::shapeSphere(grid, {0.0, 0.0}, 0.2);
    hjb::SolveOptions opts;
    opts.tMax = 0.5;
    hjb::SolveResult r = hjb::gpu::solveGPU_HolonomicL2(grid, v0, opts, 1.0);
    std::printf("RealtimeHJB %s: final time %.3f\n", HJB_VERSION_STRING, r.final_time);
    return 0;
}
