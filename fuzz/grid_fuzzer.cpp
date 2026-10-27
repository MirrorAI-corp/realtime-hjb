// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

// libFuzzer target for the host-side code: grid construction and indexing,
// the level-set shapes and the .npy writer. Invalid grids must be rejected
// with std::invalid_argument; valid ones must index consistently.

#include "hjb/core/grid.hpp"
#include "hjb/core/array_nd.hpp"
#include "hjb/io/npy_writer.hpp"
#include "hjb/level_set/shapes.hpp"

#include <fuzzer/FuzzedDataProvider.h>

#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

void require(bool ok) {
    if (!ok) __builtin_trap();
}

// Grids above this size are valid but only cost time, so they are skipped.
constexpr long long kMaxFuzzPoints = 1 << 15;

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider in(data, size);
    using hjb::kMaxDim;

    const int dim = in.ConsumeIntegralInRange<int>(-1, kMaxDim + 2);
    std::array<double, kMaxDim> lo{}, hi{}, center{}, upper{};
    std::array<int, kMaxDim> n{};
    std::array<hjb::BoundaryType, kMaxDim> bdry{};
    std::array<bool, kMaxDim> ignore{};
    long double points = 1;
    for (int d = 0; d < kMaxDim; ++d) {
        lo[d] = in.ConsumeFloatingPoint<double>();
        hi[d] = in.ConsumeBool() ? lo[d] + in.ConsumeFloatingPointInRange<double>(0.0, 100.0)
                                 : in.ConsumeFloatingPoint<double>();
        n[d] = in.ConsumeBool() ? in.ConsumeIntegralInRange<int>(-2, 40)
                                : in.ConsumeIntegral<int>();
        bdry[d] = in.ConsumeBool() ? hjb::BoundaryType::Periodic : hjb::BoundaryType::Extrapolate;
        center[d] = in.ConsumeFloatingPoint<double>();
        upper[d] = in.ConsumeFloatingPoint<double>();
        ignore[d] = in.ConsumeBool();
        if (d < dim) points *= n[d];
    }
    const double radius = in.ConsumeFloatingPoint<double>();
    // Valid but large grids are skipped; grids over INT_MAX points must throw.
    if (points > kMaxFuzzPoints && points <= INT_MAX) return 0;

    hjb::Grid g;
    try {
        g = hjb::Grid::create(dim, lo, hi, n, bdry);
    } catch (const std::invalid_argument&) {
        return 0;
    }
    require(g.dim == dim && g.total_points >= 2 && g.total_points <= kMaxFuzzPoints);

    int idx[kMaxDim];
    for (int f = 0; f < g.total_points; f += 1 + g.total_points / 64) {
        g.unflatIndex(f, idx);
        for (int d = 0; d < dim; ++d) require(idx[d] >= 0 && idx[d] < g.N[d]);
        require(g.flatIndex(idx) == f);
    }

    hjb::ArrayND sphere = hjb::shapeSphere(g, center, radius);
    hjb::ArrayND cylinder = hjb::shapeCylinder(g, ignore, center, radius);
    hjb::ArrayND box = hjb::shapeRectangle(g, center, upper);
    require(sphere.size() == g.total_points && cylinder.size() == g.total_points &&
            box.size() == g.total_points);

    std::vector<int> shape(g.N.begin(), g.N.begin() + dim);
    hjb::writeNpy("/dev/null", sphere.data(), shape);

    // The writer must reject negative dimensions before touching the data.
    std::vector<int> bad = shape;
    bad[in.ConsumeIntegralInRange<int>(0, dim - 1)] = -in.ConsumeIntegralInRange<int>(1, INT_MAX);
    try {
        hjb::writeNpy("/dev/null", nullptr, bad);
        require(false);
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
