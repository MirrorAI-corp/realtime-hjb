// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#include "hjb/level_set/shapes.hpp"

#include <algorithm>
#include <cmath>

namespace hjb {

ArrayND shapeSphere(const Grid& grid,
                    const std::array<double, kMaxDim>& center,
                    double radius) {
    std::array<bool, kMaxDim> ignore{};
    return shapeCylinder(grid, ignore, center, radius);
}

ArrayND shapeCylinder(const Grid& grid,
                      const std::array<bool, kMaxDim>& ignore,
                      const std::array<double, kMaxDim>& center,
                      double radius) {
    ArrayND result(grid);
    int idx[kMaxDim] = {0};
    for (int i = 0; i < grid.total_points; ++i) {
        grid.unflatIndex(i, idx);
        double dist_sq = 0.0;
        for (int d = 0; d < grid.dim; ++d) {
            if (ignore[d]) continue;
            double diff = grid.coord(d, idx[d]) - center[d];
            dist_sq += diff * diff;
        }
        result[i] = std::sqrt(dist_sq) - radius;
    }
    return result;
}

ArrayND shapeRectangle(const Grid& grid,
                       const std::array<double, kMaxDim>& lower,
                       const std::array<double, kMaxDim>& upper) {
    ArrayND result(grid);
    int idx[kMaxDim] = {0};
    for (int i = 0; i < grid.total_points; ++i) {
        grid.unflatIndex(i, idx);
        double v = -1e300;
        for (int d = 0; d < grid.dim; ++d) {
            double x = grid.coord(d, idx[d]);
            v = std::max(v, std::max(lower[d] - x, x - upper[d]));
        }
        result[i] = v;
    }
    return result;
}

} // namespace hjb
