// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#include "hjb/core/array_nd.hpp"
#include <algorithm>

namespace hjb {

ArrayND::ArrayND(int size) : host_data_(size, 0.0) {}

ArrayND::ArrayND(const Grid& grid) : host_data_(grid.total_points, 0.0) {}

void ArrayND::resize(int n, double val) {
    host_data_.resize(n, val);
}

void ArrayND::fill(double val) {
    std::fill(host_data_.begin(), host_data_.end(), val);
}

void ArrayND::copyFrom(const ArrayND& other) {
    host_data_ = other.host_data_;
}

} // namespace hjb
