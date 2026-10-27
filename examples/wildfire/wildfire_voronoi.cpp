// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

/// Voronoi capture-basin tessellation for a homogeneous fleet of aerial
/// response agents in a 2D spatial wind field.
///
/// Uses the label propagation of the drifter solver: one combined
/// time-to-reach solve from all agents, with labels, gives the K-agent
/// partition (which agent reaches each cell first). The program also runs
/// the K separate solves and takes the argmin, and reports the time of both
/// and how many cells get the same label.
///
/// Usage: wildfire_voronoi [out_dir]   (default: ./wildfire_voronoi)
/// Output (.npy arrays and text files in out_dir):
///   <out_dir>/<scenario>_value.npy           final value field
///   <out_dir>/<scenario>_ttr.npy             time-to-reach
///   <out_dir>/<scenario>_labels.npy          combined-solve labels
///   <out_dir>/<scenario>_ref_labels.npy      K-separate argmin labels
///   <out_dir>/<scenario>_flowx.npy           wind field x-component
///   <out_dir>/<scenario>_flowy.npy           wind field y-component
///   <out_dir>/agents.txt                     agent positions
///   <out_dir>/timing.txt                     wall-clock comparison

#include "hjb/core/grid.hpp"
#include "hjb/core/array_nd.hpp"
#include "hjb/options.hpp"
#include "hjb/solver_gpu_drifter.hpp"
#include "hjb/io/npy_writer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

namespace fs = std::filesystem;

struct Agent {
    double x;
    double y;
};

/// Build a spatially-varying wind field. Two scenarios:
///   "calm"  - background NW wind ~8 m/s, smooth
///   "gusty" - background NE wind ~12 m/s with a counter-clockwise rotor
///             centered just south of the domain (simulating a wildfire
///             plume's convective updraft pulling surface wind inward).
static void buildWindField(const hjb::Grid& g, const std::string& scenario,
                            hjb::ArrayND& flow_x, hjb::ArrayND& flow_y) {
    const int np = g.total_points;
    const double Lx = g.max[0] - g.min[0];
    const double Ly = g.max[1] - g.min[1];
    for (int i = 0; i < np; ++i) {
        double x = g.xs[0][i];
        double y = g.xs[1][i];
        if (scenario == "calm") {
            flow_x[i] =  6.0;  // NE breeze
            flow_y[i] =  5.0;
        } else {
            // Background SE wind + a counter-clockwise rotor centered at
            // (cx, cy) some way south of the domain center.
            double cx = g.min[0] + 0.55 * Lx;
            double cy = g.min[1] - 0.10 * Ly;
            double rx = x - cx;
            double ry = y - cy;
            double r2 = rx * rx + ry * ry;
            double R0 = 0.25 * std::max(Lx, Ly);
            double envelope = std::exp(-r2 / (2.0 * R0 * R0));
            double Omega = 22.0; // rotor circulation magnitude (m/s at R0)
            double rotor_x = -ry / R0 * Omega * envelope;
            double rotor_y =  rx / R0 * Omega * envelope;
            flow_x[i] =  9.0 + rotor_x;
            flow_y[i] = -3.0 + rotor_y;
        }
    }
}

/// Build the combined initial level set as the min over agents:
///   phi(x) = min_i (||x - p_i|| - r_disc)
/// Cells inside any agent's disc start at phi <= 0 and carry that agent's
/// label; everything else starts at phi > 0 and label = invalid.
static void buildCombinedInitial(const hjb::Grid& g,
                                  const std::vector<Agent>& agents,
                                  double r_disc,
                                  hjb::ArrayND& phi,
                                  std::vector<std::uint16_t>& labels) {
    const int np = g.total_points;
    labels.assign(np, hjb::kVoronoiLabelInvalid);
    for (int i = 0; i < np; ++i) {
        double x = g.xs[0][i];
        double y = g.xs[1][i];
        double best = 1e30;
        int best_id = -1;
        for (size_t k = 0; k < agents.size(); ++k) {
            double dx = x - agents[k].x;
            double dy = y - agents[k].y;
            double d = std::sqrt(dx * dx + dy * dy) - r_disc;
            if (d < best) { best = d; best_id = static_cast<int>(k); }
        }
        phi[i] = best;
        if (best <= 0.0 && best_id >= 0) {
            labels[i] = static_cast<std::uint16_t>(best_id);
        }
    }
}

/// Build a single-agent initial level set: phi = ||x - p|| - r.
static void buildSingleInitial(const hjb::Grid& g, const Agent& a,
                                double r_disc, hjb::ArrayND& phi) {
    const int np = g.total_points;
    for (int i = 0; i < np; ++i) {
        double x = g.xs[0][i];
        double y = g.xs[1][i];
        double dx = x - a.x;
        double dy = y - a.y;
        phi[i] = std::sqrt(dx * dx + dy * dy) - r_disc;
    }
}

/// Write a double array as .npy.
static void writeF64Npy(const std::string& path, const double* data,
                         int Nx, int Ny) {
    hjb::writeNpy2D(path, data, Nx, Ny);
}

/// Cast a uint16 array to double and write as .npy (Python casts back).
static void writeU16AsF64Npy(const std::string& path,
                              const std::uint16_t* data, int Nx, int Ny) {
    std::vector<double> buf(static_cast<size_t>(Nx) * Ny);
    for (int i = 0; i < Nx * Ny; ++i) {
        buf[i] = (data[i] == hjb::kVoronoiLabelInvalid) ? -1.0 : double(data[i]);
    }
    hjb::writeNpy2D(path, buf.data(), Nx, Ny);
}

static void writeArrayNDNpy(const std::string& path,
                             const hjb::ArrayND& a, int Nx, int Ny) {
    writeF64Npy(path, a.data(), Nx, Ny);
}

struct Timing {
    double combined_s;
    double per_agent_s;
    double sum_separate_s;
};

static Timing runScenario(const hjb::Grid& g, const std::string& scenario,
                           const std::vector<Agent>& agents,
                           double r_disc, double cruise_mps,
                           double tMax,
                           const fs::path& out_dir) {
    const int Nx = g.N[0], Ny = g.N[1];
    const int np = g.total_points;

    hjb::ArrayND flow_x(g), flow_y(g);
    buildWindField(g, scenario, flow_x, flow_y);

    // Solver consumes -wind to recover forward TTR semantics: the drifter
    // Hamiltonian solves a Mitchell backward-reach problem, so to make the
    // basin extend DOWNWIND of each agent we pass -wind. Matches the
    // negation convention in fleet_sim.cpp.
    hjb::ArrayND flow_x_solver(g), flow_y_solver(g);
    for (int i = 0; i < np; ++i) {
        flow_x_solver[i] = -flow_x[i];
        flow_y_solver[i] = -flow_y[i];
    }

    // --- Combined solve with label propagation ---
    hjb::ArrayND phi_combined(g);
    std::vector<std::uint16_t> init_labels;
    buildCombinedInitial(g, agents, r_disc, phi_combined, init_labels);

    hjb::SolveOptions opts;
    opts.accuracy = hjb::AccuracyLevel::Low;
    opts.factorCFL = 0.75;
    opts.tMax = tMax;
    opts.computeTTR = true;
    opts.restrictUpdate = true;
    opts.initial_labels = init_labels;

    auto t0 = Clock::now();
    auto combined = hjb::gpu::solveGPU_Drifter(
        g, phi_combined, opts, flow_x_solver, flow_y_solver,
        /*a_max=*/cruise_mps, /*b_max=*/0.0);
    auto t1 = Clock::now();
    double combined_s = std::chrono::duration<double>(t1 - t0).count();

    // --- Reference: K independent single-agent solves ---
    std::vector<hjb::ArrayND> per_agent_ttr;
    per_agent_ttr.reserve(agents.size());
    hjb::SolveOptions opts_ref = opts;
    opts_ref.initial_labels.clear();

    auto t2 = Clock::now();
    double per_agent_s = 0.0;
    for (size_t k = 0; k < agents.size(); ++k) {
        hjb::ArrayND phi_k(g);
        buildSingleInitial(g, agents[k], r_disc, phi_k);
        auto ta = Clock::now();
        auto r_k = hjb::gpu::solveGPU_Drifter(
            g, phi_k, opts_ref, flow_x_solver, flow_y_solver,
            /*a_max=*/cruise_mps, /*b_max=*/0.0);
        auto tb = Clock::now();
        per_agent_s = std::max(per_agent_s,
                               std::chrono::duration<double>(tb - ta).count());
        per_agent_ttr.push_back(std::move(r_k.ttr));
    }
    auto t3 = Clock::now();
    double sum_separate_s = std::chrono::duration<double>(t3 - t2).count();

    // Reference argmin labels.
    std::vector<std::uint16_t> ref_labels(np, hjb::kVoronoiLabelInvalid);
    for (int i = 0; i < np; ++i) {
        double best = std::numeric_limits<double>::infinity();
        int best_k = -1;
        for (size_t k = 0; k < agents.size(); ++k) {
            double v = per_agent_ttr[k][i];
            if (std::isfinite(v) && v < best) { best = v; best_k = int(k); }
        }
        if (best_k >= 0) ref_labels[i] = static_cast<std::uint16_t>(best_k);
    }

    // --- Outputs ---
    fs::create_directories(out_dir);
    writeArrayNDNpy((out_dir / (scenario + "_value.npy")).string(),
                    combined.final_data, Nx, Ny);
    writeArrayNDNpy((out_dir / (scenario + "_ttr.npy")).string(),
                    combined.ttr, Nx, Ny);
    writeArrayNDNpy((out_dir / (scenario + "_flowx.npy")).string(),
                    flow_x, Nx, Ny);
    writeArrayNDNpy((out_dir / (scenario + "_flowy.npy")).string(),
                    flow_y, Nx, Ny);
    writeU16AsF64Npy((out_dir / (scenario + "_labels.npy")).string(),
                     combined.labels.data(), Nx, Ny);
    writeU16AsF64Npy((out_dir / (scenario + "_ref_labels.npy")).string(),
                     ref_labels.data(), Nx, Ny);

    // Label-agreement audit: fraction of cells with V <= 0 (front has
    // passed) where combined.labels matches ref_labels.
    int agree = 0, total = 0;
    for (int i = 0; i < np; ++i) {
        if (combined.final_data[i] <= 0.0
            && combined.labels[i] != hjb::kVoronoiLabelInvalid
            && ref_labels[i] != hjb::kVoronoiLabelInvalid) {
            total++;
            if (combined.labels[i] == ref_labels[i]) agree++;
        }
    }
    double agreement = total > 0 ? 100.0 * agree / total : 0.0;
    std::printf("  %-6s combined %.3f s  separates %.3f s (max %.3f)  "
                "labels agree %.2f%% (%d/%d)\n",
                scenario.c_str(), combined_s, sum_separate_s, per_agent_s,
                agreement, agree, total);

    return {combined_s, per_agent_s, sum_separate_s};
}

int main(int argc, char** argv) {
    // --- Domain: 20 km x 20 km at 100 m resolution -> N = 200 ---
    const int Nx = 200, Ny = 200;
    const double Lx_m = 20000.0, Ly_m = 20000.0;
    auto grid = hjb::Grid::create(
        2,
        {0.0, 0.0}, {Lx_m, Ly_m}, {Nx, Ny},
        {hjb::BoundaryType::Extrapolate, hjb::BoundaryType::Extrapolate});

    std::printf("Wildfire Voronoi demo: %dx%d grid over %.0fx%.0f km, "
                "dx=%.0f m\n",
                Nx, Ny, Lx_m / 1000.0, Ly_m / 1000.0, grid.dx[0]);

    // --- Homogeneous fleet of K=6 aerial response agents.  Mixed
    // perimeter + interior staging so the Voronoi tessellation is
    // visually informative. ---
    std::vector<Agent> agents = {
        {  2500.0, 17500.0},  // NW
        { 17500.0, 17500.0},  // NE
        {  2500.0,  2500.0},  // SW
        { 17500.0,  2500.0},  // SE
        { 10000.0, 12000.0},  // central
        { 10000.0,  6000.0},  // south of center
    };
    const double r_disc   = 250.0;       // 250 m agent footprint
    const double cruise   = 50.0;        // 50 m/s ground speed (180 km/h)
    const double tMax     = 600.0;       // 10 minutes (front fills 20 km @50 m/s)

    // Output directory: first argument, default ./wildfire_voronoi.
    fs::path out_dir = (argc > 1) ? fs::path(argv[1]) : fs::path("wildfire_voronoi");
    fs::create_directories(out_dir);

    // Agent dump for the figure script.
    {
        std::ofstream of((out_dir / "agents.txt").string());
        for (const auto& a : agents) {
            of << a.x << "  " << a.y << "\n";
        }
    }

    // GPU warm-up: amortize CUDA context init / kernel JIT cost out of
    // the reported timings. Tiny grid, short horizon.
    {
        auto warm_grid = hjb::Grid::create(
            2, {0.0, 0.0}, {1000.0, 1000.0}, {32, 32},
            {hjb::BoundaryType::Extrapolate, hjb::BoundaryType::Extrapolate});
        hjb::ArrayND fx(warm_grid), fy(warm_grid), phi(warm_grid);
        for (int i = 0; i < warm_grid.total_points; ++i) {
            fx[i] = 1.0; fy[i] = 0.0;
            double x = warm_grid.xs[0][i] - 500.0;
            double y = warm_grid.xs[1][i] - 500.0;
            phi[i] = std::sqrt(x * x + y * y) - 100.0;
        }
        hjb::SolveOptions wopts;
        wopts.accuracy = hjb::AccuracyLevel::Low;
        wopts.tMax = 10.0;
        wopts.computeTTR = true;
        std::vector<std::uint16_t> wlab(warm_grid.total_points, hjb::kVoronoiLabelInvalid);
        for (int i = 0; i < warm_grid.total_points; ++i)
            if (phi[i] <= 0.0) wlab[i] = 0;
        wopts.initial_labels = wlab;
        (void)hjb::gpu::solveGPU_Drifter(warm_grid, phi, wopts, fx, fy, 5.0, 0.0);
    }

    auto t_calm  = runScenario(grid, "calm",  agents, r_disc, cruise, tMax, out_dir);
    auto t_gusty = runScenario(grid, "gusty", agents, r_disc, cruise, tMax, out_dir);

    // Timing summary.
    {
        std::ofstream of((out_dir / "timing.txt").string());
        of << "scenario combined_s sum_separate_s speedup\n";
        of << "calm  "  << t_calm.combined_s  << "  " << t_calm.sum_separate_s
           << "  " << (t_calm.sum_separate_s / t_calm.combined_s) << "\n";
        of << "gusty " << t_gusty.combined_s << "  " << t_gusty.sum_separate_s
           << "  " << (t_gusty.sum_separate_s / t_gusty.combined_s) << "\n";
    }

    std::printf("\n  K = %zu agents, combined-vs-separate speedup: "
                "calm %.1fx  gusty %.1fx\n",
                agents.size(),
                t_calm.sum_separate_s  / t_calm.combined_s,
                t_gusty.sum_separate_s / t_gusty.combined_s);
    std::printf("  Output: %s\n", out_dir.string().c_str());
    return 0;
}
