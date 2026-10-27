// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

#include "core/types.hpp"
#include "core/array_nd.hpp"

#include <cstdint>
#include <vector>

namespace hjb {

/// Label value for cells that no agent reaches (see SolveOptions::initial_labels).
constexpr std::uint16_t kVoronoiLabelInvalid = 0xFFFF;

/// Options for one solve. Every solver reads tMax, factorCFL, computeTTR and
/// restrictUpdate; docs/solvers.md lists which of the other fields each
/// solver uses.
struct SolveOptions {
    /// Spatial and temporal scheme. VeryHigh (and High) select WENO5 with
    /// TVD-RK3; Low selects first-order upwind with forward Euler. Air3D and
    /// the 4D solver always use WENO5 with TVD-RK3.
    AccuracyLevel accuracy = AccuracyLevel::Low;

    /// Reserved. Each solver uses the Lax-Friedrichs variant documented in
    /// docs/solvers.md, whatever this is set to.
    DissipationType dissipation = DissipationType::Global;

    /// Arithmetic precision on the GPU (Air3D, 4D and drifter solvers).
    /// FP32 is 20 to 30 times faster than FP64 on consumer GPUs.
    Precision precision = Precision::FP64;

    /// CFL number: dt = factorCFL / (sum over dimensions of max |dH/dp_i| / dx_i).
    double factorCFL = 0.75;

    /// Horizon: the solver integrates from time 0 to tMax.
    double tMax = 1.0;

    /// Record, for every cell, the first time V becomes <= 0 (SolveResult::ttr).
    bool computeTTR = true;

    /// Keep V non-increasing over time, V <- min(V_new, V_old) after each
    /// step. With it the result is the backward reachable tube (all states
    /// that reach the target at any time up to tMax); without it, the
    /// backward reachable set at exactly tMax.
    bool restrictUpdate = true;

    /// Print the grid size and step progress to stdout.
    bool verbose = false;

    /// Optional per-cell obstacle term, applied after each step as
    /// V <- max(V, obstacle_sdf). Pass the negated signed distance to the
    /// obstacle (positive inside it) to keep states inside the obstacle out of
    /// the reachable set. Empty means unused; otherwise the length must equal
    /// grid.total_points.
    std::vector<double> obstacle_sdf;

    /// Optional per-cell target term, applied after each step as
    /// V <- min(V, target_sdf). Together with obstacle_sdf this gives the
    /// reach-avoid value function min(target_sdf, max(V, obstacle_sdf)).
    std::vector<double> target_sdf;

    /// Value clamp for the 4D pursuit-evasion solver, applied after each RK3
    /// stage as a guard against runaway values far from the front. The
    /// default suits domains a few units across; widen it for larger domains
    /// where V legitimately exceeds +-50.
    double value_clip_lo = -50.0;
    double value_clip_hi = 50.0;

    /// Optional agent labels for capture-basin (Voronoi) partitioning, drifter
    /// solver only. Give each cell inside an agent's initial set (V0 <= 0) that
    /// agent's id and every other cell kVoronoiLabelInvalid. The solver carries
    /// the labels with the front, so one solve from the union of all agents'
    /// initial sets returns which agent reaches each cell first
    /// (SolveResult::labels). Empty disables labelling.
    std::vector<std::uint16_t> initial_labels;
};

/// Output of one solve.
struct SolveResult {
    /// Time-to-reach per cell: the first time V <= 0 there, +inf if never
    /// within tMax. Filled when SolveOptions::computeTTR is set.
    ArrayND ttr;
    /// Value function at the final time; the reachable set is {V <= 0}.
    ArrayND final_data;
    /// Time actually reached (equals tMax unless the solve stopped early).
    double final_time = 0.0;
    /// Agent label per cell (drifter solver with initial_labels only).
    /// Cells no agent reaches keep kVoronoiLabelInvalid.
    std::vector<std::uint16_t> labels;
};

} // namespace hjb
