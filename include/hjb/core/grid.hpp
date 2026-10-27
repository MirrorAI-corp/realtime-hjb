// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

#include "types.hpp"
#include <array>
#include <vector>
#include <cassert>

namespace hjb {

/// N-dimensional rectilinear grid
struct Grid {
    int dim = 0;                              ///< Number of dimensions
    std::array<int, kMaxDim> N{};             ///< Points per dimension
    std::array<double, kMaxDim> dx{};         ///< Grid spacing
    std::array<double, kMaxDim> min{};        ///< Domain lower bounds
    std::array<double, kMaxDim> max{};        ///< Domain upper bounds
    std::array<BoundaryType, kMaxDim> bdry{}; ///< Boundary type per dim
    int total_points = 0;                     ///< Product of N[0..dim-1]

    /// Coordinate arrays: xs[d] has total_points entries
    std::vector<std::vector<double>> xs;

    /// Create a grid
    static Grid create(int dim,
                       const std::array<double, kMaxDim>& gmin,
                       const std::array<double, kMaxDim>& gmax,
                       const std::array<int, kMaxDim>& gN,
                       const std::array<BoundaryType, kMaxDim>& gbdry);

    /// Flat index from multi-index (row-major: last dimension varies fastest)
    HJB_HD int flatIndex(const int* idx) const;

    /// Multi-index from flat index
    HJB_HD void unflatIndex(int flat, int* idx) const;

    /// 2D specialization
    HJB_HD int flatIndex2D(int i, int j) const { return i * N[1] + j; }
    HJB_HD void unflatIndex2D(int flat, int& i, int& j) const {
        i = flat / N[1];
        j = flat % N[1];
    }

    /// 3D specialization
    HJB_HD int flatIndex3D(int i, int j, int k) const {
        return (i * N[1] + j) * N[2] + k;
    }

    /// Coordinate value along dimension d at grid index i
    HJB_HD double coord(int d, int i) const {
        return min[d] + i * dx[d];
    }
};

} // namespace hjb
