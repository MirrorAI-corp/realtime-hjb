// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

// Regression and analytic tests for the GPU solvers. Run with `ctest` or
// directly: ./hjb_tests [name-filter]. Each test prints PASS or FAIL with the
// numbers it compared; the exit status is the number of failures.
//
// Analytic checks use problems with closed-form time-to-reach. Regression
// checks use BRT cell counts verified against hj_reachability (Air3D) and
// allow 0.2 % so that other GPU architectures pass.

#include <hjb/hjb.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(const char* test, bool ok, const std::string& detail) {
    std::printf("%s  %-34s %s\n", ok ? "PASS" : "FAIL", test, detail.c_str());
    if (!ok) ++g_failures;
}

std::string fmt(const char* f, double a, double b) {
    char buf[160];
    std::snprintf(buf, sizeof buf, f, a, b);
    return buf;
}

int cellsInside(const hjb::ArrayND& v) {
    int n = 0;
    for (int i = 0; i < v.size(); ++i) n += v[i] <= 0.0;
    return n;
}

bool near(double a, double b, double rel) { return std::fabs(a - b) <= rel * std::fabs(b); }

using hjb::BoundaryType;
const BoundaryType E = BoundaryType::Extrapolate;
const BoundaryType P = BoundaryType::Periodic;

double ttrAt(const hjb::Grid& g, const hjb::SolveResult& r, double x, double y) {
    int i = (int)std::lround((x - g.min[0]) / g.dx[0]);
    int j = (int)std::lround((y - g.min[1]) / g.dx[1]);
    return r.ttr[g.flatIndex2D(i, j)];
}

// ---- grid and shapes ---------------------------------------------------------------

void testGrid() {
    hjb::Grid g = hjb::Grid::create(3, {-1, -2, 0}, {1, 2, 2 * M_PI}, {11, 21, 16}, {E, E, P});
    bool ok = g.total_points == 11 * 21 * 16;
    int idx[hjb::kMaxDim], back[hjb::kMaxDim];
    for (int f = 0; f < g.total_points && ok; f += 97) {
        g.unflatIndex(f, idx);
        std::memcpy(back, idx, sizeof idx);
        ok = g.flatIndex(back) == f;
    }
    ok = ok && std::fabs(g.coord(0, 10) - 1.0) < 1e-12 && std::fabs(g.dx[2] - 2 * M_PI / 16) < 1e-12;
    check("grid indexing and spacing", ok, "periodic dx = span / N, others span / (N - 1)");
}

template <typename Fn>
bool throwsInvalid(Fn fn) {
    try { fn(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

void testInvalidInput() {
    auto grid = [](int dim, double lo, double hi, int n) {
        return [=] { hjb::Grid::create(dim, {lo, lo, lo, lo}, {hi, hi, hi, hi}, {n, n, n, n}, {E, E, E, E}); };
    };
    bool ok = throwsInvalid(grid(0, -1, 1, 11)) && throwsInvalid(grid(5, -1, 1, 11)) &&
              throwsInvalid(grid(2, 1, -1, 11)) && throwsInvalid(grid(2, -1, 1, 1)) &&
              throwsInvalid(grid(2, -1, 1, 0)) && throwsInvalid(grid(4, -1, 1, 300)) &&
              throwsInvalid(grid(2, -1, INFINITY, 11)) &&
              throwsInvalid([] { hjb::writeNpy("/dev/null", nullptr, {3, -1}); });
    check("invalid input rejected", ok, "bad grid arguments and negative .npy shapes throw");
}

void testShapes() {
    hjb::Grid g = hjb::Grid::create(2, {-1, -1}, {1, 1}, {21, 21}, {E, E});
    hjb::ArrayND s = hjb::shapeSphere(g, {0, 0}, 0.5);
    hjb::ArrayND r = hjb::shapeRectangle(g, {-0.5, -0.5}, {0.5, 0.5});
    int c = g.flatIndex2D(10, 10), edge = g.flatIndex2D(20, 10);
    bool ok = std::fabs(s[c] + 0.5) < 1e-12 && std::fabs(s[edge] - 0.5) < 1e-12 &&
              std::fabs(r[c] + 0.5) < 1e-12 && std::fabs(r[edge] - 0.5) < 1e-12;
    check("shapeSphere / shapeRectangle", ok, "values at centre and edge");
}

// ---- analytic time-to-reach ----------------------------------------------------------

// Holonomic vehicle, speed 1, disk target radius 0.1: TTR(x) = |x| - 0.1.
void testHolonomicTTR() {
    hjb::Grid g = hjb::Grid::create(2, {-1, -1}, {1, 1}, {101, 101}, {E, E});
    hjb::ArrayND v0 = hjb::shapeSphere(g, {0, 0}, 0.1);
    hjb::SolveOptions o;
    o.accuracy = hjb::AccuracyLevel::VeryHigh;
    o.tMax = 1.0;
    hjb::SolveResult r = hjb::gpu::solveGPU_HolonomicL2(g, v0, o, 1.0);
    double t1 = ttrAt(g, r, 0.6, 0.0), t2 = ttrAt(g, r, -0.42, 0.42);
    double e1 = 0.5, e2 = std::hypot(0.42, 0.42) - 0.1;
    check("holonomic TTR", near(t1, e1, 0.03) && near(t2, e2, 0.03),
          fmt("TTR %.4f, %.4f", t1, t2) + fmt(" (exact %.4f, %.4f)", e1, e2));
}

// Same vehicle in a constant wind (0.4, 0) with a disturbance bound 0.25:
// effective speed 1.15 toward the target from upwind, 0.35 from downwind.
void testDrifterTTR() {
    hjb::Grid g = hjb::Grid::create(2, {-1, -1}, {1, 1}, {200, 200}, {E, E});
    hjb::ArrayND v0 = hjb::shapeSphere(g, {0, 0}, 0.1);
    hjb::ArrayND wx(g), wy(g);
    wx.fill(0.4);
    wy.fill(0.0);
    hjb::SolveOptions o;
    o.accuracy = hjb::AccuracyLevel::VeryHigh;
    o.precision = hjb::Precision::FP32;
    o.tMax = 1.5;
    hjb::SolveResult r = hjb::gpu::solveGPU_Drifter(g, v0, o, wx, wy, 1.0, 0.25);
    double up = ttrAt(g, r, -0.6, 0.0), down = ttrAt(g, r, 0.6, 0.0);
    check("drifter TTR in constant wind", near(up, 0.5 / 1.15, 0.03) && near(down, 0.5 / 0.35, 0.03),
          fmt("TTR %.4f, %.4f", up, down) + fmt(" (exact %.4f, %.4f)", 0.5 / 1.15, 0.5 / 0.35));
}

// ---- reachable sets ---------------------------------------------------------------------

// Air3D at N = 51: 11197 cells (hj_reachability: 11201), FP32 and FP64.
void testAir3D() {
    const int N = 51;
    hjb::Grid g = hjb::Grid::create(3, {-20, -20, 0}, {20, 20, 2 * M_PI}, {N, N, N}, {E, E, P});
    hjb::ArrayND v0 = hjb::shapeCylinder(g, {false, false, true}, {0, 0, 0}, 5.0);
    hjb::SolveOptions o;
    o.accuracy = hjb::AccuracyLevel::VeryHigh;
    o.tMax = 2.0;
    o.computeTTR = false;
    o.precision = hjb::Precision::FP64;
    int c64 = cellsInside(hjb::gpu::solveGPU_Air3D(g, v0, o, 5.0, 5.0, 1.0).final_data);
    o.precision = hjb::Precision::FP32;
    int c32 = cellsInside(hjb::gpu::solveGPU_Air3D(g, v0, o, 5.0, 5.0, 1.0).final_data);
    check("Air3D BRT, N=51", near(c64, 11197, 0.002) && near(c32, 11197, 0.002),
          fmt("cells FP64 %.0f, FP32 %.0f (expect 11197)", c64, c32));
}

// 4D pursuit-evasion at N = 25: FP32 and FP64 tubes agree to 0.2 %.
void testPursuitEvasion4D() {
    const int N = 25;
    hjb::Grid g = hjb::Grid::create(4, {-1, -1, -1, -1}, {1, 1, 1, 1}, {N, N, N, N}, {E, E, E, E});
    hjb::ArrayND v0(g);
    int idx[hjb::kMaxDim];
    for (int i = 0; i < g.total_points; ++i) {
        g.unflatIndex(i, idx);
        v0[i] = std::hypot(g.coord(0, idx[0]) - g.coord(2, idx[2]),
                           g.coord(1, idx[1]) - g.coord(3, idx[3])) - 0.2;
    }
    hjb::SolveOptions o;
    o.accuracy = hjb::AccuracyLevel::VeryHigh;
    o.tMax = 0.8;
    o.computeTTR = false;
    o.precision = hjb::Precision::FP64;
    int c64 = cellsInside(hjb::gpu::solveGPU_PursuitEvasion4D(g, v0, o, 1.0, 0.7).final_data);
    o.precision = hjb::Precision::FP32;
    int c32 = cellsInside(hjb::gpu::solveGPU_PursuitEvasion4D(g, v0, o, 1.0, 0.7).final_data);
    bool grew = c64 > cellsInside(v0);
    check("4D pursuit-evasion FP32 vs FP64", grew && near(c32, c64, 0.002),
          fmt("cells FP64 %.0f, FP32 %.0f", c64, c32));
}

// Double integrator: the reachable set is point-symmetric, V(x, v) = V(-x, -v).
void testDoubleIntegratorSymmetry() {
    const int N = 81;
    hjb::Grid g = hjb::Grid::create(2, {-1, -1}, {1, 1}, {N, N}, {E, E});
    hjb::ArrayND v0 = hjb::shapeSphere(g, {0, 0}, 0.1);
    hjb::SolveOptions o;
    o.accuracy = hjb::AccuracyLevel::VeryHigh;
    o.tMax = 1.0;
    o.computeTTR = false;
    hjb::SolveResult r = hjb::gpu::solveGPU_DoubleIntegrator2D(g, v0, o, 1.0);
    double worst = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            worst = std::fmax(worst, std::fabs(r.final_data[g.flatIndex2D(i, j)] -
                                               r.final_data[g.flatIndex2D(N - 1 - i, N - 1 - j)]));
    check("double integrator symmetry", worst < 1e-9 && cellsInside(r.final_data) > cellsInside(v0),
          fmt("max |V(x) - V(-x)| = %.2e (tube grew %.0f cells)", worst,
              cellsInside(r.final_data) - cellsInside(v0)));
}

}  // namespace

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : "";
    struct T { const char* name; std::function<void()> fn; };
    std::vector<T> tests = {
        {"grid", testGrid},
        {"invalid_input", testInvalidInput},
        {"shapes", testShapes},
        {"holonomic", testHolonomicTTR},
        {"drifter", testDrifterTTR},
        {"air3d", testAir3D},
        {"pe4d", testPursuitEvasion4D},
        {"double_integrator", testDoubleIntegratorSymmetry},
    };
    std::printf("RealtimeHJB %s tests\n", HJB_VERSION_STRING);
    for (const auto& t : tests)
        if (std::strstr(t.name, filter)) t.fn();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures;
}
