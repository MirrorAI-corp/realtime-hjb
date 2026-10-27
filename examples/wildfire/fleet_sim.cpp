// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

/// Closed-loop wildfire-response fleet simulation.
///
/// Six quadcopters with holonomic 2D dynamics patrol a 20 km x 20 km region
/// in a time-varying, non-uniform wind. At every 10 Hz control tick the
/// program recomputes the time-to-reach field and the capture-basin
/// partition from the drones' current positions, in one labelled solve of
/// the drifter solver. A drone in transit therefore carries its basin with
/// it, and a new fire goes to the drone that can reach it first under the
/// current wind. Fires that no drone can reach within the 10-minute horizon
/// wait until one can.
///
/// Usage: fleet_sim [out_dir]        (default: ./wildfire_fleet)
/// Output: one binary frame per simulated second and run_summary.json in
/// out_dir.

#include "hjb/core/grid.hpp"
#include "hjb/core/array_nd.hpp"
#include "hjb/options.hpp"
#include "hjb/solver_gpu_drifter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// --------------------------------------------------------------------------
// Sim parameters
// --------------------------------------------------------------------------
static constexpr int    NX = 200;
static constexpr int    NY = 200;
static constexpr double LX_M = 20000.0;
static constexpr double LY_M = 20000.0;

static constexpr double DT_TICK     = 0.10;     // 10 Hz outer loop
static constexpr double T_SIM       = 900.0;    // 15 minutes of sim time
static constexpr int    SAVE_STRIDE = 10;       // 900 saved frames -> 30 s @ 30 fps (30x realtime)
static constexpr double T_HJ        = 600.0;    // 10-min tactical dispatch horizon

static constexpr double DRONE_CRUISE_MPS = 20.0;       // 72 km/h ground (no wind)
static constexpr double DISC_R_M         = 250.0;      // drone footprint
static constexpr double SUPPRESS_RADIUS  = 300.0;      // arrive-at-fire threshold
static constexpr double SUPPRESS_DURATION = 8.0;       // seconds on station

enum class Status : uint8_t {
    PATROL      = 0,  // station-keeping at home
    ENROUTE     = 1,  // flying to assigned fire
    SUPPRESSING = 2,  // on top of fire
    RETURNING   = 3,  // heading back to home after suppression
};

enum class FireState : uint8_t {
    PENDING     = 0,  // not yet appeared
    BURNING     = 1,  // appeared, awaiting suppression
    SUPPRESSED  = 2,
};

struct Drone {
    double x, y;
    double home_x, home_y;
    Status status = Status::PATROL;
    int    assigned_fire = -1;
    double t_suppress_end = 0.0;
    double predicted_arrival_t = -1.0;  // absolute sim time of arrival at fire
};

struct Fire {
    double x, y;
    double t_appear;
    FireState state = FireState::PENDING;
    int  assigned_drone = -1;
    double t_dispatched = -1.0;
    double t_suppressed = -1.0;
};

// --------------------------------------------------------------------------
// Time-varying non-uniform wind field
//
// Background wind direction rotates slowly over 240 s (synoptic shift).
// A plume rotor of variable strength oscillates in the south; the
// rotor's center walks slowly NE as the fire complex evolves.
// --------------------------------------------------------------------------
static void buildWindField(double t, hjb::ArrayND& fx, hjb::ArrayND& fy,
                            const hjb::Grid& g) {
    const double Lx = g.max[0] - g.min[0];
    const double Ly = g.max[1] - g.min[1];

    // Synoptic wind direction rotates slowly over the 15-min sim:
    // from NE (-45 deg) to NW (-105 deg), i.e., a 60-deg shift in 15 min
    // (4 deg/min). That is gentle, on the slow end of realistic
    // synoptic-front passage timescales.
    double s = t / T_SIM;
    s = std::clamp(s, 0.0, 1.0);
    double theta = (-45.0 + 60.0 * s) * M_PI / 180.0;
    // Background ~12 m/s (~27 mph), within the documented
    // "active fire spread" range (8.3 to 27.7 m/s).  Strong enough that
    // ratio to 20 m/s cruise is 0.6, producing visibly anisotropic
    // basins everywhere, but drones still beat the wind in calm phases.
    double Vbg = 12.0;
    double bgx = -Vbg * std::sin(theta);
    double bgy =  Vbg * std::cos(theta);

    // Rotor: counter-clockwise, center walks slowly SE -> NE over the
    // full 15-min sim.  Tangential speed: base ~10 m/s, a sustained
    // Santa-Ana gust pulse around t=500 s (std=80 s wide) adds up to
    // +30 m/s peak (peak tangential ≈ 40 m/s = 90 mph, the documented
    // extreme-gust regime), plus a long slow oscillation ±6 m/s on a
    // 4-minute period.  Combined with 12 m/s background, peak wind
    // during the gust reaches ~50 m/s in the rotor core, well above
    // the 20 m/s drone cruise, so the windward side of the rotor
    // becomes outright unreachable in the 2-min dispatch window.
    double cx = g.min[0] + 0.30 * Lx + 0.40 * Lx * s;
    double cy = g.min[1] + 0.15 * Ly + 0.40 * Ly * s;
    double R0 = 0.22 * std::max(Lx, Ly);
    double gust = std::exp(-((t - 500.0) * (t - 500.0)) / (2.0 * 80.0 * 80.0));
    double Omega = 10.0 + 30.0 * gust + 6.0 * std::sin(2.0 * M_PI * t / 240.0);

    const int np = g.total_points;
    for (int i = 0; i < np; ++i) {
        double x = g.xs[0][i];
        double y = g.xs[1][i];
        double rx = x - cx;
        double ry = y - cy;
        double r2 = rx * rx + ry * ry;
        double env = std::exp(-r2 / (2.0 * R0 * R0));
        double rotor_x = -ry / R0 * Omega * env;
        double rotor_y =  rx / R0 * Omega * env;
        fx[i] = bgx + rotor_x;
        fy[i] = bgy + rotor_y;
    }
}

// --------------------------------------------------------------------------
// Build the combined initial level set from drone positions.
// --------------------------------------------------------------------------
static void buildInitialFromDrones(const std::vector<Drone>& drones,
                                    double r_disc, const hjb::Grid& g,
                                    hjb::ArrayND& phi,
                                    std::vector<std::uint16_t>& labels) {
    const int np = g.total_points;
    labels.assign(np, hjb::kVoronoiLabelInvalid);
    for (int i = 0; i < np; ++i) {
        double x = g.xs[0][i];
        double y = g.xs[1][i];
        double best = 1e30;
        int best_id = -1;
        for (size_t k = 0; k < drones.size(); ++k) {
            double dx = x - drones[k].x;
            double dy = y - drones[k].y;
            double d = std::sqrt(dx * dx + dy * dy) - r_disc;
            if (d < best) { best = d; best_id = int(k); }
        }
        phi[i] = best;
        if (best <= 0.0 && best_id >= 0) {
            labels[i] = std::uint16_t(best_id);
        }
    }
}

// --------------------------------------------------------------------------
// Look up the label at a world position via nearest-cell on the grid.
// --------------------------------------------------------------------------
static int labelAt(const hjb::Grid& g, const std::vector<std::uint16_t>& labels,
                   double x, double y) {
    int i = int(std::round((x - g.min[0]) / g.dx[0]));
    int j = int(std::round((y - g.min[1]) / g.dx[1]));
    i = std::clamp(i, 0, g.N[0] - 1);
    j = std::clamp(j, 0, g.N[1] - 1);
    int idx = i * g.N[1] + j;
    auto v = labels[idx];
    return (v == hjb::kVoronoiLabelInvalid) ? -1 : int(v);
}

// --------------------------------------------------------------------------
// Update one drone's state: pick a target based on status, command an
// airspeed vector toward it, integrate ground-velocity = airspeed + wind.
// --------------------------------------------------------------------------
static void advanceDrone(Drone& d, double dt, const hjb::Grid& g,
                          const hjb::ArrayND& fx, const hjb::ArrayND& fy,
                          const std::vector<Fire>& fires) {
    double tx = d.x, ty = d.y;
    bool   has_target = false;

    switch (d.status) {
        case Status::ENROUTE:
        case Status::SUPPRESSING:
            if (d.assigned_fire >= 0
                && fires[d.assigned_fire].state == FireState::BURNING) {
                tx = fires[d.assigned_fire].x;
                ty = fires[d.assigned_fire].y;
                has_target = true;
            }
            break;
        case Status::RETURNING:
            tx = d.home_x;  ty = d.home_y;
            has_target = true;
            break;
        case Status::PATROL:
            tx = d.home_x;  ty = d.home_y;
            // Patrol = station-keep at home, small dead-band so wind drift
            // doesn't keep us chasing. If within 200 m of home, hold.
            {
                double dx = tx - d.x, dy = ty - d.y;
                if (std::sqrt(dx * dx + dy * dy) < 200.0) has_target = false;
                else has_target = true;
            }
            break;
    }

    double cmd_vx = 0.0, cmd_vy = 0.0;
    if (has_target) {
        double dx = tx - d.x, dy = ty - d.y;
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist > 1e-6) {
            cmd_vx = DRONE_CRUISE_MPS * dx / dist;
            cmd_vy = DRONE_CRUISE_MPS * dy / dist;
        }
    }

    // Sample wind at drone's nearest cell.
    int i = int(std::round((d.x - g.min[0]) / g.dx[0]));
    int j = int(std::round((d.y - g.min[1]) / g.dx[1]));
    i = std::clamp(i, 0, g.N[0] - 1);
    j = std::clamp(j, 0, g.N[1] - 1);
    int idx = i * g.N[1] + j;
    double wx = fx[idx];
    double wy = fy[idx];

    d.x += (cmd_vx + wx) * dt;
    d.y += (cmd_vy + wy) * dt;
    d.x = std::clamp(d.x, g.min[0] + 50.0, g.max[0] - 50.0);
    d.y = std::clamp(d.y, g.min[1] + 50.0, g.max[1] - 50.0);
}

// --------------------------------------------------------------------------
// 3x3 majority filter on labels.  Smooths out 1-cell stripe artifacts
// along Voronoi bisectors that arise from monotone-upwind label
// propagation under FP-noise at the bisector.  Preserves the
// 0xFF "unreachable" marker exactly (only neighbors with the same
// label vote; unreachable cells stay unreachable).
// --------------------------------------------------------------------------
static std::vector<std::uint16_t> smoothLabels3x3(
        const std::vector<std::uint16_t>& in, int Nx, int Ny) {
    std::vector<std::uint16_t> out(in.size());
    // A 3x3 stencil contains at most 9 cells, so at most 9 distinct labels
    // can participate in any single vote. Enumerate them on the fly rather
    // than indexing into a fixed-size counts[] array, so the filter is
    // correct for any uint16_t label (the kVoronoiLabelInvalid sentinel
    // aside).
    std::uint16_t labels[9];
    int counts[9];
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int idx = i * Ny + j;
            std::uint16_t self = in[idx];
            if (self == hjb::kVoronoiLabelInvalid) {
                out[idx] = self;
                continue;
            }
            int n_distinct = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int ii = i + di, jj = j + dj;
                    if (ii < 0 || ii >= Nx || jj < 0 || jj >= Ny) continue;
                    std::uint16_t v = in[ii * Ny + jj];
                    if (v == hjb::kVoronoiLabelInvalid) continue;
                    int slot = -1;
                    for (int s = 0; s < n_distinct; ++s) {
                        if (labels[s] == v) { slot = s; break; }
                    }
                    if (slot < 0) {
                        labels[n_distinct] = v;
                        counts[n_distinct] = 1;
                        ++n_distinct;
                    } else {
                        ++counts[slot];
                    }
                }
            }
            std::uint16_t best = self;
            int best_n = 0;
            for (int s = 0; s < n_distinct; ++s) {
                if (labels[s] == self) { best_n = counts[s]; break; }
            }
            for (int s = 0; s < n_distinct; ++s) {
                if (counts[s] > best_n) { best_n = counts[s]; best = labels[s]; }
            }
            out[idx] = best;
        }
    }
    return out;
}

// --------------------------------------------------------------------------
// Binary frame writer. Layout (little-endian throughout):
//
//   header:
//     uint32 magic   = 0x57464C54 ("WFLT")
//     uint32 version = 1
//     int32  Nx, Ny
//     int32  K_drones, K_fires
//     float64 t
//   labels: uint8[Nx*Ny]  (drone id, or 0xFF for unlabeled)
//   wind:   float32[Nx*Ny] for fx, then float32[Nx*Ny] for fy
//   drones (v1): per-drone {float64 x, y, home_x, home_y; uint8 status; int8 fire}
//   drones (v2): per-drone {float64 x, y, home_x, home_y, predicted_arrival_t; uint8 status; int8 fire}
//   fires:        per-fire {float64 x, y, t_appear, t_dispatched, t_suppressed; uint8 state; int8 drone}
// --------------------------------------------------------------------------
static void writeFrame(const fs::path& path, double t,
                       int Nx, int Ny,
                       const std::vector<std::uint16_t>& labels,
                       const hjb::ArrayND& fx, const hjb::ArrayND& fy,
                       const std::vector<Drone>& drones,
                       const std::vector<Fire>& fires) {
    std::ofstream of(path, std::ios::binary);
    std::uint32_t magic = 0x57464C54u, version = 2u;
    std::int32_t  iNx = Nx, iNy = Ny;
    std::int32_t  K = int(drones.size()), F = int(fires.size());
    of.write(reinterpret_cast<const char*>(&magic),   4);
    of.write(reinterpret_cast<const char*>(&version), 4);
    of.write(reinterpret_cast<const char*>(&iNx), 4);
    of.write(reinterpret_cast<const char*>(&iNy), 4);
    of.write(reinterpret_cast<const char*>(&K),   4);
    of.write(reinterpret_cast<const char*>(&F),   4);
    of.write(reinterpret_cast<const char*>(&t),   8);

    // Labels as uint8 (max 254 agents, 0xFF = unlabeled).
    std::vector<std::uint8_t> lab8(Nx * Ny);
    for (int i = 0; i < Nx * Ny; ++i) {
        lab8[i] = (labels[i] == hjb::kVoronoiLabelInvalid)
                  ? 0xFFu : std::uint8_t(labels[i]);
    }
    of.write(reinterpret_cast<const char*>(lab8.data()), Nx * Ny);

    std::vector<float> fx32(Nx * Ny), fy32(Nx * Ny);
    for (int i = 0; i < Nx * Ny; ++i) {
        fx32[i] = float(fx[i]);
        fy32[i] = float(fy[i]);
    }
    of.write(reinterpret_cast<const char*>(fx32.data()), Nx * Ny * 4);
    of.write(reinterpret_cast<const char*>(fy32.data()), Nx * Ny * 4);

    for (const auto& d : drones) {
        of.write(reinterpret_cast<const char*>(&d.x), 8);
        of.write(reinterpret_cast<const char*>(&d.y), 8);
        of.write(reinterpret_cast<const char*>(&d.home_x), 8);
        of.write(reinterpret_cast<const char*>(&d.home_y), 8);
        of.write(reinterpret_cast<const char*>(&d.predicted_arrival_t), 8);
        std::uint8_t st = std::uint8_t(d.status);
        std::int8_t  af = std::int8_t(std::max(-1, std::min(127, d.assigned_fire)));
        of.write(reinterpret_cast<const char*>(&st), 1);
        of.write(reinterpret_cast<const char*>(&af), 1);
    }
    for (const auto& f : fires) {
        of.write(reinterpret_cast<const char*>(&f.x), 8);
        of.write(reinterpret_cast<const char*>(&f.y), 8);
        of.write(reinterpret_cast<const char*>(&f.t_appear), 8);
        of.write(reinterpret_cast<const char*>(&f.t_dispatched), 8);
        of.write(reinterpret_cast<const char*>(&f.t_suppressed), 8);
        std::uint8_t st = std::uint8_t(f.state);
        std::int8_t  ad = std::int8_t(std::max(-1, std::min(127, f.assigned_drone)));
        of.write(reinterpret_cast<const char*>(&st), 1);
        of.write(reinterpret_cast<const char*>(&ad), 1);
    }
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    auto grid = hjb::Grid::create(
        2, {0.0, 0.0}, {LX_M, LY_M}, {NX, NY},
        {hjb::BoundaryType::Extrapolate, hjb::BoundaryType::Extrapolate});

    // K = 6 drones at perimeter + interior staging.
    std::vector<Drone> drones = {
        { 2500.0, 17500.0, 2500.0, 17500.0},  // 0  NW
        {17500.0, 17500.0,17500.0, 17500.0},  // 1  NE
        { 2500.0,  2500.0, 2500.0,  2500.0},  // 2  SW
        {17500.0,  2500.0,17500.0,  2500.0},  // 3  SE
        {10000.0, 12000.0,10000.0, 12000.0},  // 4  central
        {10000.0,  6000.0,10000.0,  6000.0},  // 5  south-center
    };

    // Scripted fire schedule.  Three fires, well spaced in time so each
    // suppress-return cycle has room to complete at 20 m/s cruise:
    //   - t=  60 s  NE area: baseline static-Voronoi-style dispatch.
    //   - t= 360 s  SW area: appears while the first responder is still
    //               returning; dispatch goes to the live Voronoi owner.
    //   - t= 660 s  central-east: appears during/after the rotor gust,
    //               so the live wind-shaped Voronoi differs strongly
    //               from the home-station Voronoi.
    std::vector<Fire> fires = {
        {15000.0, 15500.0,  60.0, FireState::PENDING, -1, -1, -1},
        { 4500.0,  5500.0, 360.0, FireState::PENDING, -1, -1, -1},
        {13000.0, 11500.0, 660.0, FireState::PENDING, -1, -1, -1},
    };

    // Output directory: first argument, default ./wildfire_fleet.
    fs::path out_dir = (argc > 1) ? fs::path(argv[1]) : fs::path("wildfire_fleet");
    fs::create_directories(out_dir);

    // Fleet/schedule metadata for the renderer.
    {
        std::ofstream of((out_dir / "meta.txt").string());
        of << "Lx_m " << LX_M << "  Ly_m " << LY_M << "\n";
        of << "Nx "   << NX   << "  Ny "  << NY   << "\n";
        of << "dt "   << DT_TICK << "  T_sim " << T_SIM
           << "  stride " << SAVE_STRIDE << "\n";
        of << "K_drones " << drones.size()
           << "  K_fires " << fires.size() << "\n";
        of << "cruise_mps " << DRONE_CRUISE_MPS << "\n";
        for (size_t k = 0; k < drones.size(); ++k) {
            of << "drone " << k << "  home  "
               << drones[k].home_x << " " << drones[k].home_y << "\n";
        }
        for (size_t k = 0; k < fires.size(); ++k) {
            of << "fire  " << k << "  at "
               << fires[k].x << " " << fires[k].y
               << "  t_appear " << fires[k].t_appear << "\n";
        }
    }

    // Two wind fields: `fx, fy` is the PHYSICAL wind (for saving + viz +
    // drone integration). `fx_solver, fy_solver` is the negated copy
    // passed to the solver.  The drifter Hamiltonian solves a Mitchell
    // backward-reach problem; to recover forward TTR semantics (drone at
    // p reaches cell x fastest, basin extends DOWNWIND) we pass -wind.
    hjb::ArrayND fx(grid), fy(grid), phi(grid);
    hjb::ArrayND fx_solver(grid), fy_solver(grid);
    std::vector<std::uint16_t> init_labels;

    hjb::SolveOptions opts;
    opts.accuracy = hjb::AccuracyLevel::Low;
    opts.precision = hjb::Precision::FP32;  // real-time deployment precision
    opts.factorCFL = 0.75;
    opts.tMax = T_HJ;
    opts.computeTTR = true;        // for predicted-arrival display
    opts.restrictUpdate = true;

    // GPU warm-up: drop CUDA init out of the per-tick timing.
    {
        auto warm_g = hjb::Grid::create(
            2, {0.0, 0.0}, {1000.0, 1000.0}, {32, 32},
            {hjb::BoundaryType::Extrapolate, hjb::BoundaryType::Extrapolate});
        hjb::ArrayND wfx(warm_g), wfy(warm_g), wphi(warm_g);
        for (int i = 0; i < warm_g.total_points; ++i) {
            wfx[i] = 1.0; wfy[i] = 0.0;
            double x = warm_g.xs[0][i] - 500.0;
            double y = warm_g.xs[1][i] - 500.0;
            wphi[i] = std::sqrt(x * x + y * y) - 100.0;
        }
        hjb::SolveOptions wo;
        wo.accuracy = hjb::AccuracyLevel::Low;
        wo.precision = hjb::Precision::FP32;
        wo.tMax = 10.0;
        wo.computeTTR = false;
        std::vector<std::uint16_t> wlab(warm_g.total_points, hjb::kVoronoiLabelInvalid);
        for (int i = 0; i < warm_g.total_points; ++i)
            if (wphi[i] <= 0.0) wlab[i] = 0;
        wo.initial_labels = wlab;
        (void)hjb::gpu::solveGPU_Drifter(warm_g, wphi, wo, wfx, wfy, 5.0, 0.0);
    }

    const int Nticks = int(T_SIM / DT_TICK);
    int saved = 0;
    double solve_s_total = 0.0;
    int solve_count = 0;
    auto t0_wall = Clock::now();

    for (int step = 0; step < Nticks; ++step) {
        double t = step * DT_TICK;

        // 1) Wind field (physical).
        buildWindField(t, fx, fy, grid);
        // Solver consumes -wind to recover forward TTR semantics.
        const int np_local = grid.total_points;
        for (int i = 0; i < np_local; ++i) {
            fx_solver[i] = -fx[i];
            fy_solver[i] = -fy[i];
        }

        // 2) Combined TTR + Voronoi solve (label propagation enabled).
        buildInitialFromDrones(drones, DISC_R_M, grid, phi, init_labels);
        opts.initial_labels = init_labels;
        auto ts = Clock::now();
        auto result = hjb::gpu::solveGPU_Drifter(
            grid, phi, opts, fx_solver, fy_solver,
            /*a_max=*/DRONE_CRUISE_MPS, /*b_max=*/0.0);
        auto te = Clock::now();
        solve_s_total += std::chrono::duration<double>(te - ts).count();
        solve_count++;
        const auto& labels = result.labels;

        // 3) Dispatch any newly-appeared fires using the LIVE Voronoi at
        // the fire's location.
        //
        // Policy: if the fire's cell is UNREACHABLE in the current
        // dispatch horizon (label == invalid), DO NOT dispatch.  The
        // fire remains pending; we will retry every tick.  This is the
        // honest operational answer: "no drone can reach this fire in
        // the response window right now."  The fire becomes assignable
        // once wind shifts or a drone repositions such that some
        // basin covers it.
        //
        // If the cell IS covered but the owner is currently busy with
        // another fire, also leave it pending and wait for the owner to
        // free up rather than dispatching a drone whose 2-min basin
        // doesn't actually reach this cell.
        for (size_t f = 0; f < fires.size(); ++f) {
            auto& fire = fires[f];
            if (fire.state != FireState::PENDING) continue;
            if (t < fire.t_appear) continue;

            int owner = labelAt(grid, labels, fire.x, fire.y);
            if (owner < 0) continue;        // fire is in an unreachable cell
            if (drones[owner].status != Status::PATROL
                && drones[owner].status != Status::RETURNING) continue;  // owner busy

            // Sample predicted TTR at the fire's cell from the solver's
            // combined TTR field (== TTR for whichever drone owns the
            // cell, which is `owner`).
            int fi = int(std::round((fire.x - grid.min[0]) / grid.dx[0]));
            int fj = int(std::round((fire.y - grid.min[1]) / grid.dx[1]));
            fi = std::clamp(fi, 0, grid.N[0] - 1);
            fj = std::clamp(fj, 0, grid.N[1] - 1);
            double ttr_at_fire = result.ttr[fi * grid.N[1] + fj];

            fire.state = FireState::BURNING;
            fire.assigned_drone = owner;
            fire.t_dispatched = t;
            drones[owner].status = Status::ENROUTE;
            drones[owner].assigned_fire = int(f);
            drones[owner].predicted_arrival_t =
                std::isfinite(ttr_at_fire) ? (t + ttr_at_fire) : -1.0;
            std::printf("  t=%6.2f s  fire %zu (%.0f, %.0f) -> drone %d "
                        "(TTR %.1f s)\n",
                        t, f, fire.x, fire.y, owner, ttr_at_fire);
        }

        // 4) Per-drone state machine update + position integrate.
        for (size_t k = 0; k < drones.size(); ++k) {
            auto& d = drones[k];

            if (d.status == Status::ENROUTE && d.assigned_fire >= 0) {
                const auto& fire = fires[d.assigned_fire];
                double dx = fire.x - d.x, dy = fire.y - d.y;
                if (std::sqrt(dx * dx + dy * dy) < SUPPRESS_RADIUS) {
                    d.status = Status::SUPPRESSING;
                    d.t_suppress_end = t + SUPPRESS_DURATION;
                    d.predicted_arrival_t = -1.0;
                }
            } else if (d.status == Status::SUPPRESSING) {
                if (t >= d.t_suppress_end) {
                    if (d.assigned_fire >= 0) {
                        fires[d.assigned_fire].state = FireState::SUPPRESSED;
                        fires[d.assigned_fire].t_suppressed = t;
                    }
                    d.assigned_fire = -1;
                    d.status = Status::RETURNING;
                }
            } else if (d.status == Status::RETURNING) {
                double dx = d.home_x - d.x, dy = d.home_y - d.y;
                if (std::sqrt(dx * dx + dy * dy) < 200.0) {
                    d.status = Status::PATROL;
                }
            }

            advanceDrone(d, DT_TICK, grid, fx, fy, fires);
        }

        // 5) Save frame.
        if (step % SAVE_STRIDE == 0) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "frame_%05d.bin", saved);
            auto smoothed = smoothLabels3x3(labels, NX, NY);
            writeFrame(out_dir / buf, t, NX, NY, smoothed, fx, fy, drones, fires);
            saved++;
        }
    }
    auto t1_wall = Clock::now();
    double wall_s = std::chrono::duration<double>(t1_wall - t0_wall).count();

    std::printf("\nFleet sim done.\n");
    std::printf("  Ticks: %d  saved frames: %d (every %d ticks)\n",
                Nticks, saved, SAVE_STRIDE);
    std::printf("  Per-tick solve mean: %.2f ms  (%d solves)\n",
                1000.0 * solve_s_total / std::max(1, solve_count), solve_count);
    std::printf("  Total sim wall: %.2f s   (sim time %.0f s -> %.1fx realtime)\n",
                wall_s, T_SIM, T_SIM / wall_s);
    std::printf("  Frames at %s\n", out_dir.string().c_str());

    int suppressed = 0;
    for (const auto& f : fires) if (f.state == FireState::SUPPRESSED) suppressed++;
    std::printf("  Fires: %d/%d suppressed by end of sim\n",
                suppressed, int(fires.size()));

    // Machine-readable summary.
    {
        double per_tick_s = solve_s_total / std::max(1, solve_count);
        std::ofstream js((out_dir / "run_summary.json").string());
        js << "{\n"
           << "  \"case\": \"fleet_voronoi\",\n"
           << "  \"N\": " << NX << ",\n"
           << "  \"precision\": \"fp32\",\n"
           << "  \"accuracy\": \"low\",\n"
           << "  \"ticks\": " << Nticks << ",\n"
           << "  \"saved_frames\": " << saved << ",\n"
           << "  \"solve_count\": " << solve_count << ",\n"
           << "  \"per_tick_solve_mean_s\": " << per_tick_s << ",\n"
           << "  \"per_tick_solve_hz\": " << (per_tick_s > 0 ? 1.0 / per_tick_s : 0.0) << ",\n"
           << "  \"wall_total_s\": " << wall_s << ",\n"
           << "  \"realtime_factor\": " << (T_SIM / wall_s) << ",\n"
           << "  \"fires_total\": " << int(fires.size()) << ",\n"
           << "  \"fires_suppressed\": " << suppressed << "\n"
           << "}\n";
        std::printf("  Wrote %s\n", (out_dir / "run_summary.json").string().c_str());
    }
    return 0;
}
