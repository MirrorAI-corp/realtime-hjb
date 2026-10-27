// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

#include "grid.hpp"
#include <vector>
#include <cstring>

namespace hjb {

/// Host/device array with N-D indexing via a Grid
class ArrayND {
public:
    ArrayND() = default;
    explicit ArrayND(int size);
    explicit ArrayND(const Grid& grid);

    /// Access host data
    double& operator[](int i) { return host_data_[i]; }
    double  operator[](int i) const { return host_data_[i]; }

    double* data() { return host_data_.data(); }
    const double* data() const { return host_data_.data(); }

    int size() const { return static_cast<int>(host_data_.size()); }

    /// Resize (host only)
    void resize(int n, double val = 0.0);

    /// Fill with a value
    void fill(double val);

    /// Copy from another array
    void copyFrom(const ArrayND& other);

    std::vector<double>& hostVec() { return host_data_; }
    const std::vector<double>& hostVec() const { return host_data_; }

#ifdef HJB_USE_CUDA
    /// Allocate device memory
    void allocateDevice();

    /// Copy host -> device
    void toDevice();

    /// Copy device -> host
    void toHost();

    /// Free device memory
    void freeDevice();

    /// Raw device pointer
    double* devicePtr() { return device_data_; }
    const double* devicePtr() const { return device_data_; }
#endif

private:
    std::vector<double> host_data_;

#ifdef HJB_USE_CUDA
    double* device_data_ = nullptr;
    int device_size_ = 0;
#endif
};

} // namespace hjb
