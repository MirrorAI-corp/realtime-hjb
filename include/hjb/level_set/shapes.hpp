// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

#include "../core/grid.hpp"
#include "../core/array_nd.hpp"
#include <array>

namespace hjb {

/// Signed distance to a ball of the given radius: negative inside, positive
/// outside. In 2D this is a disk, in 3D a sphere. Uses every grid dimension.
ArrayND shapeSphere(const Grid& grid,
                    const std::array<double, kMaxDim>& center,
                    double radius);

/// Signed distance to a cylinder: a ball in the dimensions where
/// `ignore[d]` is false, extended along the dimensions where it is true.
/// Example: the Air3D collision set is a disk in (x, y) for every heading,
/// `shapeCylinder(grid, {false, false, true}, {0, 0, 0}, 5.0)`.
ArrayND shapeCylinder(const Grid& grid,
                      const std::array<bool, kMaxDim>& ignore,
                      const std::array<double, kMaxDim>& center,
                      double radius);

/// Implicit surface of an axis-aligned box: negative inside, positive
/// outside. Exact inside the box and along the face normals; outside a
/// corner it returns the largest face distance (a lower bound on the true
/// distance), as in ToolboxLS's shapeRectangleByCorners.
ArrayND shapeRectangle(const Grid& grid,
                       const std::array<double, kMaxDim>& lower,
                       const std::array<double, kMaxDim>& upper);

} // namespace hjb
