// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hjb {

/// Write a row-major float64 N-D array to NumPy .npy v1.0 format.
/// Shape is given as a vector of dimension sizes; data length must equal
/// the product of the shape entries.
void writeNpy(const std::string& path,
              const double* data,
              const std::vector<int>& shape);

/// Convenience overload for 2D arrays (row-major, shape (Nx, Ny)).
inline void writeNpy2D(const std::string& path, const double* data, int Nx, int Ny) {
    writeNpy(path, data, {Nx, Ny});
}

} // namespace hjb
