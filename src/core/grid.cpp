// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#include "hjb/core/grid.hpp"

namespace hjb {

Grid Grid::create(int dim,
                  const std::array<double, kMaxDim>& gmin,
                  const std::array<double, kMaxDim>& gmax,
                  const std::array<int, kMaxDim>& gN,
                  const std::array<BoundaryType, kMaxDim>& gbdry) {
    Grid g;
    g.dim = dim;
    g.min = gmin;
    g.max = gmax;
    g.N = gN;
    g.bdry = gbdry;

    g.total_points = 1;
    for (int d = 0; d < dim; d++) {
        // Periodic dims tile the domain with `gN[d]` cells of equal width:
        // grid points lie at gmin + i*dx for i=0..gN-1; gmax is identified
        // with gmin (one full period later). Non-periodic dims include both
        // endpoints as distinct points.
        if (gbdry[d] == BoundaryType::Periodic) {
            g.dx[d] = (gmax[d] - gmin[d]) / gN[d];
        } else {
            g.dx[d] = (gmax[d] - gmin[d]) / (gN[d] - 1);
        }
        g.total_points *= gN[d];
    }

    // Build coordinate arrays
    g.xs.resize(dim);
    for (int d = 0; d < dim; d++) {
        g.xs[d].resize(g.total_points);
    }

    // Fill coordinate arrays using unflatIndex
    for (int flat = 0; flat < g.total_points; flat++) {
        int idx[kMaxDim];
        g.unflatIndex(flat, idx);
        for (int d = 0; d < dim; d++) {
            g.xs[d][flat] = gmin[d] + idx[d] * g.dx[d];
        }
    }

    return g;
}

int Grid::flatIndex(const int* idx) const {
    int flat = 0;
    int stride = 1;
    for (int d = dim - 1; d >= 0; d--) {
        flat += idx[d] * stride;
        stride *= N[d];
    }
    return flat;
}

void Grid::unflatIndex(int flat, int* idx) const {
    for (int d = dim - 1; d >= 0; d--) {
        idx[d] = flat % N[d];
        flat /= N[d];
    }
}

} // namespace hjb
