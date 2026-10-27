// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

/// 3D WENO5+RK3+LLF Air3D pursuit-evasion solver.
///
/// State: (x, y, psi). Periodic boundary on psi via `grid.boundaries[2]`.
/// Each RK3 stage is ONE fused kernel (`air3d_stage_kernel<Real, STAGE>`):
/// WENO5 + LLF + Hamiltonian + the RK stage combine + (stage 3 only)
/// restrict/postproc/TTR. Templated on `Real` in {float, double}.
///
///   - The wavespeed field alpha(x,y,psi) is time-invariant, so the CFL max
///     is computed once on the host.
///   - The stage combine happens in registers inside the stencil kernel.
///     Stage 3 writes V in place (the stencil source is y2; V is only read
///     at the thread's own cell).
///   - cos/sin(psi) come from precomputed 1D tables.
///   - threadIdx.x maps to the unit-stride grid axis (psi) so pointwise
///     reads/writes are warp-coalesced.
///
/// Step loop (HJB_AIR3D_STEP_MODE, default `replay`):
///   replay      the step graph is replayed back to back with no host
///               round-trip; the host only counts steps (dt is constant
///               until the final, clamped step). The last few steps run in
///               `graph_sync` mode so the horizon ends exactly as before.
///   graph_sync  one graph launch per step, then a 2x4-byte dt/t read-back
///               and cudaStreamSynchronize.
///   sync        same as graph_sync but with individual stream launches.
/// All three produce bit-identical V.

#include "hjb/solver_gpu_3d.hpp"
#include "hjb/core/types.hpp"
#include "detail/weno5.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <vector>

namespace hjb {
namespace gpu {

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ----- Fused per-RK3-stage Air3D kernel: WENO5 (direct fetch) + LLF + RHS
// + stage combine in one launch.
//   STAGE 1:  out = vold + dt*rhs(src)              (src == vold == V)
//   STAGE 2:  out = 3/4 vold + 1/4 (src + dt*rhs(src))   (src == y1)
//   STAGE 3:  out = 1/3 vold + 2/3 (src + dt*rhs(src))   (src == y2)
//             then restrict (min with vold), postproc, TTR. out may alias
//             vold (in-place): vold is only read at the thread's own cell.
// `dt_ptr`/`t_accum` are device-resident so the whole step can be captured
// into a CUDA graph.
//
// Stencil values come straight from global memory via weno5_axis (fetchB in
// detail/weno5.cuh handles the periodic-psi wrap and the linear ghost
// extrapolation inline). Up to 101^3 the FP32 field (4 MB) fits in L2, so a
// shared-memory tile would only add halo loads and synchronization.
template <typename T, int STAGE>
__global__ void air3d_stage_kernel(
    const T* __restrict__ src,
    const T* vold,     // no __restrict__: aliases `out` in the in-place stage 3
    T* out,
    const T* __restrict__ dt_ptr,
    const T* __restrict__ cp_tab,
    const T* __restrict__ sp_tab,
    const T* __restrict__ obstacle,
    const T* __restrict__ target,
    double* ttr,
    const T* __restrict__ t_accum,
    int restrict_update,
    int N0, int N1, int N2,
    T x0, T dx0,
    T y0, T dx1,
    T inv_dx0, T inv_dx1, T inv_dx2,
    T v_a, T v_b, T omega_max,
    int periodic0, int periodic1, int periodic2, int np)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= np) return;
    int gz = gid % N2; int rem = gid / N2;
    int gy = rem % N1; int gx = rem / N1;
    int s1 = N2, s0 = N1 * N2;

    T dL0, dR0, dL1, dR1, dL2, dR2;
    weno5_axis<T>(src, gx, N0, gid - gx * s0, s0, periodic0, inv_dx0, dL0, dR0);
    weno5_axis<T>(src, gy, N1, gid - gy * s1, s1, periodic1, inv_dx1, dL1, dR1);
    weno5_axis<T>(src, gz, N2, gid - gz,      1,  periodic2, inv_dx2, dL2, dR2);

    T center = src[gid];

    T dc0 = T(0.5) * (dL0 + dR0);
    T dc1 = T(0.5) * (dL1 + dR1);
    T dc2 = T(0.5) * (dL2 + dR2);

    T x   = x0 + T(gx) * dx0;
    T y   = y0 + T(gy) * dx1;
    T cp = cp_tab[gz], sp = sp_tab[gz];
    T f0 = -v_a + v_b * cp;
    T f1 =        v_b * sp;
    T q_ctrl = y * dc0 - x * dc1 - dc2;

    T H_std = f0 * dc0 + f1 * dc1
            + omega_max * fabs(q_ctrl)
            - omega_max * fabs(dc2);
    T H = -H_std;

    T a0 = fabs(f0) + omega_max * fabs(y);
    T a1 = fabs(f1) + omega_max * fabs(x);
    T a2 = T(2) * omega_max;
    T diss = T(0.5) * (a0 * (dR0 - dL0) + a1 * (dR1 - dL1) + a2 * (dR2 - dL2));

    T r = -H + diss;
    T dt = *dt_ptr;

    T v;
    if (STAGE == 1) {
        v = center + dt * r;                              // vold == src == V
    } else if (STAGE == 2) {
        v = T(0.75) * vold[gid] + T(0.25) * (center + dt * r);
    } else {
        T vo = vold[gid];
        v = T(1.0/3.0) * vo + T(2.0/3.0) * (center + dt * r);
        if (restrict_update) v = fmin(v, vo);
        if (obstacle) v = fmax(v, obstacle[gid]);
        if (target)   v = fmin(v, target[gid]);
        if (ttr && isinf(ttr[gid]) && v <= T(0)) {
            double time = double(*t_accum - dt);
            ttr[gid] = (vo > T(0))
                ? time + double(vo) / double(vo - v) * double(dt)
                : time;
        }
    }
    out[gid] = v;
}

// dt = clamp(factorCFL / max_alpha, 0, tMax - t_accum); t_accum += dt.
// max_alpha is a host-computed constant (Air3D wavespeeds are time-invariant);
// only the time clamp needs device state, so this stays a 1-thread kernel
// inside the captured graph.
template <typename T> __global__ static void compute_dt_3d_const(
    T max_alpha, T* dt_out, T* t_accum,
    T factorCFL, T tMax)
{
    T dt = (max_alpha > T(0)) ? (factorCFL / max_alpha) : T(1e10);
    T remaining = tMax - *t_accum;
    if (dt > remaining) dt = remaining;
    if (dt < T(0)) dt = T(0);
    *dt_out = dt;
    *t_accum += dt;
}

enum class Air3DStepMode { Replay, GraphSync, Sync };

static Air3DStepMode air3d_step_mode() {
    const char* s = std::getenv("HJB_AIR3D_STEP_MODE");
    if (!s || !*s || !std::strcmp(s, "replay")) return Air3DStepMode::Replay;
    if (!std::strcmp(s, "graph_sync")) return Air3DStepMode::GraphSync;
    if (!std::strcmp(s, "sync"))       return Air3DStepMode::Sync;
    fprintf(stderr, "HJB_AIR3D_STEP_MODE='%s' (want replay|graph_sync|sync)\n", s);
    exit(1);
}

// ----- Templated solve impl -----
template <typename T>
static SolveResult solve_air3d_impl(
    const Grid& grid, const ArrayND& initial_data,
    const SolveOptions& opts,
    double v_a, double v_b, double omega_max)
{
    int N0=grid.N[0], N1=grid.N[1], N2=grid.N[2];
    int np = grid.total_points;
    double dx[3] = {grid.dx[0], grid.dx[1], grid.dx[2]};
    double lo[3] = {grid.min[0], grid.min[1], grid.min[2]};
    int periodic[3] = {
        grid.bdry[0] == BoundaryType::Periodic ? 1 : 0,
        grid.bdry[1] == BoundaryType::Periodic ? 1 : 0,
        grid.bdry[2] == BoundaryType::Periodic ? 1 : 0,
    };

    const char* dtype = (sizeof(T) == 4) ? "fp32" : "fp64";
    if (opts.verbose) printf("  Air3D WENO5+RK3+LLF (fused-stage %s): %dx%dx%d = %d pts (periodic=[%d,%d,%d])\n",
           dtype, N0,N1,N2,np, periodic[0],periodic[1],periodic[2]);

    T *d_data,*d_y1,*d_y2;
    CK(cudaMalloc(&d_data,  np * sizeof(T)));
    CK(cudaMalloc(&d_y1,    np * sizeof(T)));
    CK(cudaMalloc(&d_y2,    np * sizeof(T)));

    // cos/sin(psi) lookup tables (psi varies along dim 2 only).
    T *d_cp_tab, *d_sp_tab;
    CK(cudaMalloc(&d_cp_tab, N2 * sizeof(T)));
    CK(cudaMalloc(&d_sp_tab, N2 * sizeof(T)));
    {
        std::vector<T> cp(N2), sp(N2);
        for (int k = 0; k < N2; ++k) {
            double psi = lo[2] + k * dx[2];
            cp[k] = T(std::cos(psi));
            sp[k] = T(std::sin(psi));
        }
        CK(cudaMemcpy(d_cp_tab, cp.data(), N2 * sizeof(T), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sp_tab, sp.data(), N2 * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Value-postprocessor buffers (static obstacle / reach-avoid target).
    T *d_obstacle = nullptr, *d_target = nullptr;
    if ((int)opts.obstacle_sdf.size() == np) {
        CK(cudaMalloc(&d_obstacle, np * sizeof(T)));
        std::vector<T> buf(np);
        for (int i = 0; i < np; ++i) buf[i] = T(opts.obstacle_sdf[i]);
        CK(cudaMemcpy(d_obstacle, buf.data(), np * sizeof(T), cudaMemcpyHostToDevice));
    }
    if ((int)opts.target_sdf.size() == np) {
        CK(cudaMalloc(&d_target, np * sizeof(T)));
        std::vector<T> buf(np);
        for (int i = 0; i < np; ++i) buf[i] = T(opts.target_sdf[i]);
        CK(cudaMemcpy(d_target, buf.data(), np * sizeof(T), cudaMemcpyHostToDevice));
    }

    double* d_ttr = nullptr;
    if (opts.computeTTR) {
        CK(cudaMalloc(&d_ttr, np * sizeof(double)));
        std::vector<double> ttr_init(np, INFINITY);
        for (int i=0; i<np; i++) if (initial_data[i] <= 0.0) ttr_init[i] = 0.0;
        CK(cudaMemcpy(d_ttr, ttr_init.data(), np * sizeof(double), cudaMemcpyHostToDevice));
    }

    T* d_dt;  CK(cudaMalloc(&d_dt,  sizeof(T)));   // device-resident step size
    T* d_t_accum; CK(cudaMalloc(&d_t_accum, sizeof(T)));
    {
        T zero = T(0);
        CK(cudaMemcpy(d_t_accum, &zero, sizeof(T), cudaMemcpyHostToDevice));
    }

    // Pinned host slot for the dt + t-accumulator read-back per step.
    T* h_dt_pinned = nullptr;
    CK(cudaMallocHost(&h_dt_pinned, sizeof(T)));
    T* h_t_pinned  = nullptr;
    CK(cudaMallocHost(&h_t_pinned, sizeof(T)));

    // Host -> device with implicit conversion.
    std::vector<T> host_buf(np);
    for (int i = 0; i < np; ++i) host_buf[i] = T(initial_data[i]);
    CK(cudaMemcpy(d_data, host_buf.data(), np * sizeof(T), cudaMemcpyHostToDevice));

    // Global CFL wavespeed, computed ONCE on the host. Air3D's per-cell
    //   alpha = (|f0(psi)| + w|y|)/dx0 + (|f1(psi)| + w|x|)/dx1 + 2w/dx2
    // is additively separable in psi, |y|, |x|, so its grid max is
    //   max_psi[|f0|/dx0 + |f1|/dx1] + w*max|y|/dx0 + w*max|x|/dx1 + 2w/dx2
    // (the psi terms share the same psi, hence the joint max over the table).
    // This replaces one full-grid alpha write + CUB reduction per RK stage.
    double max_alpha;
    {
        double psi_term = 0.0;
        for (int k = 0; k < N2; ++k) {
            double psi = lo[2] + k * dx[2];
            double f0m = std::fabs(-v_a + v_b * std::cos(psi));
            double f1m = std::fabs(v_b * std::sin(psi));
            psi_term = std::max(psi_term, f0m / dx[0] + f1m / dx[1]);
        }
        double max_abs_x = std::max(std::fabs(lo[0]), std::fabs(lo[0] + (N0-1)*dx[0]));
        double max_abs_y = std::max(std::fabs(lo[1]), std::fabs(lo[1] + (N1-1)*dx[1]));
        max_alpha = psi_term
                  + omega_max * max_abs_y / dx[0]
                  + omega_max * max_abs_x / dx[1]
                  + 2.0 * omega_max / dx[2];
    }

    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    int restrict_flag = opts.restrictUpdate ? 1 : 0;
    int block_sz = 256;
    int grid_sz = (np + block_sz - 1) / block_sz;

    // One fused launch per RK3 stage. STAGE is a template arg; src/vold/out
    // wire the SSP-RK3 dataflow:  V -> y1 -> y2 -> V (in place).
    auto launch_stage = [&](int stage, const T* src, const T* vold, T* out) {
        auto go = [&](auto stage_c) {
            constexpr int Sc = decltype(stage_c)::value;
            air3d_stage_kernel<T, Sc><<<grid_sz, block_sz, 0, stream>>>(
                src, vold, out, d_dt, d_cp_tab, d_sp_tab,
                (Sc == 3) ? d_obstacle : nullptr,
                (Sc == 3) ? d_target   : nullptr,
                (Sc == 3) ? d_ttr      : nullptr,
                d_t_accum, restrict_flag,
                N0, N1, N2,
                T(lo[0]), T(dx[0]), T(lo[1]), T(dx[1]),
                T(1.0/dx[0]), T(1.0/dx[1]), T(1.0/dx[2]),
                T(v_a), T(v_b), T(omega_max),
                periodic[0], periodic[1], periodic[2], np);
        };
        if (stage == 1)      go(std::integral_constant<int,1>{});
        else if (stage == 2) go(std::integral_constant<int,2>{});
        else                 go(std::integral_constant<int,3>{});
    };

    // One full RK3 step. With `readback`, the step ends with the 4-byte dt and
    // t_accum copies the host loop polls.
    auto enqueue_step = [&](bool readback) {
        compute_dt_3d_const<T><<<1, 1, 0, stream>>>(T(max_alpha), d_dt, d_t_accum,
                                                     T(opts.factorCFL), T(opts.tMax));
        launch_stage(1, d_data, d_data, d_y1);
        launch_stage(2, d_y1,   d_data, d_y2);
        launch_stage(3, d_y2,   d_data, d_data);   // in-place final combine
        if (readback) {
            CK(cudaMemcpyAsync(h_dt_pinned, d_dt,      sizeof(T), cudaMemcpyDeviceToHost, stream));
            CK(cudaMemcpyAsync(h_t_pinned,  d_t_accum, sizeof(T), cudaMemcpyDeviceToHost, stream));
        }
    };
    // Capture a step into a CUDA graph. Every per-step kernel argument is a
    // device pointer or call-constant, so the same recording replays every step.
    // Launch errors (e.g. no kernel image for this GPU) surface here instead of
    // being swallowed by the capture.
    auto capture = [&](bool readback) {
        CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        enqueue_step(readback);
        cudaError_t launch_err = cudaGetLastError();
        cudaGraph_t g;
        CK(cudaStreamEndCapture(stream, &g));
        CK(launch_err);
        cudaGraphExec_t ge;
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        CK(cudaGraphDestroy(g));
        return ge;
    };

    const Air3DStepMode mode = air3d_step_mode();
    cudaGraphExec_t graph_exec = nullptr, lean_exec = nullptr;
    if (mode != Air3DStepMode::Sync) graph_exec = capture(true);

    double t = 0; int steps = 0;
    const T t_end = T(opts.tMax);   // the device clamps t_accum to this value

    if (mode == Air3DStepMode::Replay) {
        // Count the steps on the host by mirroring compute_dt_3d_const in T,
        // then replay all but the last two without any host round-trip. The
        // device computes the actual dt; the 2-step margin absorbs any
        // host/device rounding difference in dt, and the synced loop below
        // takes the final (clamped) steps exactly as graph_sync does.
        T th = T(0);
        const T dtc = (max_alpha > 0.0) ? T(opts.factorCFL) / T(max_alpha) : T(1e10);
        long n_est = 0;
        while (th < t_end && n_est < (1L << 30)) {
            T dt = std::min(dtc, t_end - th);
            if (!(dt > T(0))) break;
            th += dt; ++n_est;
        }
        long n_replay = n_est - 2;
        if (n_replay > 0) {
            lean_exec = capture(false);
            for (long k = 0; k < n_replay; ++k) CK(cudaGraphLaunch(lean_exec, stream));
            CK(cudaGetLastError());
            CK(cudaMemcpyAsync(h_t_pinned, d_t_accum, sizeof(T), cudaMemcpyDeviceToHost, stream));
            CK(cudaStreamSynchronize(stream));
            t = double(*h_t_pinned);
            steps = int(n_replay);
        }
    }

    // Synced loop: all steps in graph_sync/sync modes, the tail in replay mode.
    // Stops once t_accum reaches T(tMax): when T(tMax) rounds below tMax the
    // relative-tolerance test alone never fires (the device clamps dt to 0).
    while (t < opts.tMax - 1e-12 * opts.tMax && T(t) < t_end) {
        if (mode == Air3DStepMode::Sync) enqueue_step(true);
        else CK(cudaGraphLaunch(graph_exec, stream));
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(stream));
        double t_new = double(*h_t_pinned);
        if (!(t_new > t)) {
            fprintf(stderr, "Air3D: time stopped advancing at t=%.9g (tMax=%.9g)\n", t, opts.tMax);
            exit(1);
        }
        t = t_new;
        steps++;
        if (opts.verbose && steps % 200 == 0) printf("  step %d, t=%.3f/%.3f\n", steps, t, opts.tMax);
    }
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (lean_exec)  cudaGraphExecDestroy(lean_exec);
    cudaStreamDestroy(stream);
    if (opts.verbose) printf("  Air3D done (%s): %d steps, final t=%.3f\n", dtype, steps, t);

    SolveResult result;
    if (opts.computeTTR) {
        result.ttr.resize(np);
        CK(cudaMemcpy(result.ttr.data(), d_ttr, np * sizeof(double), cudaMemcpyDeviceToHost));
    }
    result.final_data.resize(np);
    // Device -> host with implicit widen-to-double.
    CK(cudaMemcpy(host_buf.data(), d_data, np * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < np; ++i) result.final_data[i] = double(host_buf[i]);
    result.final_time = t;

    cudaFree(d_data); cudaFree(d_y1); cudaFree(d_y2);
    cudaFree(d_cp_tab); cudaFree(d_sp_tab);
    if (d_ttr) cudaFree(d_ttr);
    if (d_obstacle) cudaFree(d_obstacle);
    if (d_target)   cudaFree(d_target);
    cudaFree(d_dt); cudaFree(d_t_accum);
    cudaFreeHost(h_dt_pinned);
    cudaFreeHost(h_t_pinned);
    return result;
}

SolveResult solveGPU_Air3D(
    const Grid& grid, const ArrayND& initial_data,
    const SolveOptions& opts,
    double v_a, double v_b, double omega_max)
{
    if (opts.precision == Precision::FP32) {
        return solve_air3d_impl<float>(grid, initial_data, opts, v_a, v_b, omega_max);
    }
    return solve_air3d_impl<double>(grid, initial_data, opts, v_a, v_b, omega_max);
}

} // namespace gpu
} // namespace hjb
