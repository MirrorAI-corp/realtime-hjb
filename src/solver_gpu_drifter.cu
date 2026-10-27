// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#include "hjb/solver_gpu_drifter.hpp"
#include "detail/value_postproc.cuh"
#include "detail/weno5.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace hjb {
namespace gpu {

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ----- Per-dimension 1D ghost fill (templated) -----
template <typename T>
__global__ void pad0_drifter_kernel_t(const T* data, T* padded,
                                       int Nx, int Ny, int gw, int bdry) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pNx = Nx + 2 * gw;
    if (tid >= pNx * Ny) return;
    int pi = tid / Ny; int j = tid % Ny;
    int ii = pi - gw;
    if (ii >= 0 && ii < Nx) {
        padded[pi * Ny + j] = data[ii * Ny + j];
    } else if (ii < 0) {
        int g = -ii - 1;
        if (bdry == 0) {
            T v0 = data[0 * Ny + j];
            T v1 = data[1 * Ny + j];
            padded[pi * Ny + j] = v0 - T(g + 1) * (v1 - v0);
        } else {
            int src = ((Nx - 1 - g) % Nx + Nx) % Nx;
            padded[pi * Ny + j] = data[src * Ny + j];
        }
    } else {
        int g = ii - Nx;
        if (bdry == 0) {
            T vn1 = data[(Nx - 1) * Ny + j];
            T vn2 = data[(Nx - 2) * Ny + j];
            padded[pi * Ny + j] = vn1 + T(g + 1) * (vn1 - vn2);
        } else {
            int src = g % Nx;
            padded[pi * Ny + j] = data[src * Ny + j];
        }
    }
}

template <typename T>
__global__ void pad1_drifter_kernel_t(const T* data, T* padded,
                                       int Nx, int Ny, int gw, int bdry) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pNy = Ny + 2 * gw;
    if (tid >= Nx * pNy) return;
    int i = tid / pNy; int pj = tid % pNy;
    int jj = pj - gw;
    if (jj >= 0 && jj < Ny) {
        padded[i * pNy + pj] = data[i * Ny + jj];
    } else if (jj < 0) {
        int g = -jj - 1;
        if (bdry == 0) {
            T v0 = data[i * Ny + 0];
            T v1 = data[i * Ny + 1];
            padded[i * pNy + pj] = v0 - T(g + 1) * (v1 - v0);
        } else {
            int src = ((Ny - 1 - g) % Ny + Ny) % Ny;
            padded[i * pNy + pj] = data[i * Ny + src];
        }
    } else {
        int g = jj - Ny;
        if (bdry == 0) {
            T vn1 = data[i * Ny + Ny - 1];
            T vn2 = data[i * Ny + Ny - 2];
            padded[i * pNy + pj] = vn1 + T(g + 1) * (vn1 - vn2);
        } else {
            int src = g % Ny;
            padded[i * pNy + pj] = data[i * Ny + src];
        }
    }
}

// weno5_recon lives in detail/weno5.cuh.

template <typename T>
__global__ void lf_drifter_low_kernel_t(
    const T* pad0, const T* pad1,
    const T* flow_x, const T* flow_y,
    T* rhs,
    int Nx, int Ny, int gw,
    T inv_dx0, T inv_dx1,
    T alpha0, T alpha1,
    T a_max, T b_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int i = idx / Ny, j = idx % Ny;

    int p0 = (i + gw) * Ny + j;
    T dL0 = (pad0[p0] - pad0[p0 - Ny]) * inv_dx0;
    T dR0 = (pad0[p0 + Ny] - pad0[p0]) * inv_dx0;

    int pNy = Ny + 2 * gw;
    int p1 = i * pNy + (j + gw);
    T dL1 = (pad1[p1] - pad1[p1 - 1]) * inv_dx1;
    T dR1 = (pad1[p1 + 1] - pad1[p1]) * inv_dx1;

    T dc0 = T(0.5) * (dL0 + dR0);
    T dc1 = T(0.5) * (dL1 + dR1);

    T wx = flow_x[idx];
    T wy = flow_y[idx];
    T grad_norm = sqrt(dc0 * dc0 + dc1 * dc1 + T(1e-30));
    T H_std = wx * dc0 + wy * dc1 + (b_max - a_max) * grad_norm;
    T H = -H_std;
    T diss = T(0.5) * (alpha0 * (dR0 - dL0) + alpha1 * (dR1 - dL1));
    rhs[idx] = -H + diss;
}

// Fused first-order (Low-accuracy) drifter step: direct-fetch upwind
// derivatives (no pad buffers) + drifter Hamiltonian + LLF dissipation +
// the RK1 update + restrict + obstacle/target postproc + TTR, all in one
// launch reading `src` and writing `dst` (ping-pong). One launch per step
// keeps launch overhead low on small grids, where a solve runs hundreds of
// steps. fetchB lives in detail/weno5.cuh.
template <typename T>
__global__ void lf_drifter_low_fused_t(
    const T* __restrict__ src, T* __restrict__ dst,
    const T* __restrict__ flow_x, const T* __restrict__ flow_y,
    const T* __restrict__ obstacle, const T* __restrict__ target,
    double* ttr,
    int Nx, int Ny, int bdry0, int bdry1,
    T inv_dx0, T inv_dx1, T alpha0, T alpha1, T bma,
    T dt, int restrict_update, double t_now)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int i = idx / Ny, j = idx % Ny;

    T center = src[idx];
    T xm = fetchB<T>(src, i - 1, Nx, j, Ny, bdry0);
    T xp = fetchB<T>(src, i + 1, Nx, j, Ny, bdry0);
    T ym = fetchB<T>(src, j - 1, Ny, i * Ny, 1, bdry1);
    T yp = fetchB<T>(src, j + 1, Ny, i * Ny, 1, bdry1);

    T dL0 = (center - xm) * inv_dx0, dR0 = (xp - center) * inv_dx0;
    T dL1 = (center - ym) * inv_dx1, dR1 = (yp - center) * inv_dx1;
    T dc0 = T(0.5) * (dL0 + dR0);
    T dc1 = T(0.5) * (dL1 + dR1);

    T wx = flow_x[idx], wy = flow_y[idx];
    T grad_norm = sqrt(dc0 * dc0 + dc1 * dc1 + T(1e-30));
    T H_std = wx * dc0 + wy * dc1 + bma * grad_norm;
    T diss = T(0.5) * (alpha0 * (dR0 - dL0) + alpha1 * (dR1 - dL1));
    T v = center + dt * (H_std + diss);            // RK1: rhs = -(-H_std)+diss

    if (restrict_update) v = fmin(v, center);      // min with start-of-step V
    if (obstacle) v = fmax(v, obstacle[idx]);
    if (target)   v = fmin(v, target[idx]);
    if (ttr && isinf(ttr[idx]) && v <= T(0)) {
        ttr[idx] = (center > T(0))
            ? t_now + double(center) / double(center - v) * double(dt)
            : t_now;
    }
    dst[idx] = v;
}

template <typename T>
__global__ void lf_drifter_weno5_kernel_t(
    const T* pad0, const T* pad1,
    const T* flow_x, const T* flow_y,
    T* rhs,
    int Nx, int Ny, int gw,
    T inv_dx0, T inv_dx1,
    T alpha0, T alpha1,
    T a_max, T b_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int i = idx / Ny, j = idx % Ny;

    int s0 = Ny;
    int p0 = (i + gw) * s0 + j;
    T v1 = (pad0[p0 - 2*s0] - pad0[p0 - 3*s0]) * inv_dx0;
    T v2 = (pad0[p0 - s0]   - pad0[p0 - 2*s0]) * inv_dx0;
    T v3 = (pad0[p0]        - pad0[p0 - s0])    * inv_dx0;
    T v4 = (pad0[p0 + s0]   - pad0[p0])         * inv_dx0;
    T v5 = (pad0[p0 + 2*s0] - pad0[p0 + s0])    * inv_dx0;
    T dL0 = weno5_recon<T>(v1, v2, v3, v4, v5);
    T w1 = (pad0[p0 + 3*s0] - pad0[p0 + 2*s0]) * inv_dx0;
    T dR0 = weno5_recon<T>(w1, v5, v4, v3, v2);

    int pNy = Ny + 2 * gw;
    int p1 = i * pNy + (j + gw);
    v1 = (pad1[p1-2] - pad1[p1-3]) * inv_dx1;
    v2 = (pad1[p1-1] - pad1[p1-2]) * inv_dx1;
    v3 = (pad1[p1]   - pad1[p1-1]) * inv_dx1;
    v4 = (pad1[p1+1] - pad1[p1])   * inv_dx1;
    v5 = (pad1[p1+2] - pad1[p1+1]) * inv_dx1;
    T dL1 = weno5_recon<T>(v1, v2, v3, v4, v5);
    w1 = (pad1[p1+3] - pad1[p1+2]) * inv_dx1;
    T dR1 = weno5_recon<T>(w1, v5, v4, v3, v2);

    T dc0 = T(0.5) * (dL0 + dR0);
    T dc1 = T(0.5) * (dL1 + dR1);

    T wx = flow_x[idx];
    T wy = flow_y[idx];
    T grad_norm = sqrt(dc0 * dc0 + dc1 * dc1 + T(1e-30));
    T H_std = wx * dc0 + wy * dc1 + (b_max - a_max) * grad_norm;
    T H = -H_std;
    T diss = T(0.5) * (alpha0 * (dR0 - dL0) + alpha1 * (dR1 - dL1));
    rhs[idx] = -H + diss;
}

// Fused per-RK3-stage drifter WENO5 kernel: direct-fetch stencil (no pads)
// + Hamiltonian + LLF dissipation + RK stage combine + restrict/postproc/TTR
// (stage 3), one launch per stage. fetchB/weno5_axis/rk3_stage_write live in
// detail/weno5.cuh. Stage 3 writes V in place (stencil source is y2).
template <typename T, int STAGE>
__global__ void lf_drifter_weno5_stage_t(
    const T* __restrict__ src, const T* vold, T* out,
    const T* __restrict__ flow_x, const T* __restrict__ flow_y,
    const T* __restrict__ obstacle, const T* __restrict__ target,
    double* ttr,
    int Nx, int Ny, int bdry0, int bdry1,
    T inv_dx0, T inv_dx1,
    T alpha0, T alpha1, T bma,        // bma = b_max - a_max
    T dt, int restrict_update, double t_now)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int i = idx / Ny, j = idx % Ny;

    T dL0, dR0, dL1, dR1;
    weno5_axis<T>(src, i, Nx, j, Ny, bdry0, inv_dx0, dL0, dR0);     // dim0: stride Ny
    weno5_axis<T>(src, j, Ny, i * Ny, 1, bdry1, inv_dx1, dL1, dR1); // dim1: stride 1
    T dc0 = T(0.5) * (dL0 + dR0);
    T dc1 = T(0.5) * (dL1 + dR1);

    T wx = flow_x[idx];
    T wy = flow_y[idx];
    T grad_norm = sqrt(dc0 * dc0 + dc1 * dc1 + T(1e-30));
    T H_std = wx * dc0 + wy * dc1 + bma * grad_norm;
    T diss = T(0.5) * (alpha0 * (dR0 - dL0) + alpha1 * (dR1 - dL1));
    T r = H_std + diss;   // rhs = -(-H_std) + diss

    rk3_stage_write<T, STAGE>(idx, src[idx], r, dt, vold, out,
                              obstacle, target, ttr, restrict_update, t_now);
}

template <typename T>
__global__ void euler_drifter_t(T* d, const T* o, const T* r, T dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) d[i] = o[i] + dt * r[i];
}
template <typename T>
__global__ void rk3_s2_drifter_t(T* y2, const T* o, const T* y1, const T* r, T dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y2[i] = T(0.75)*o[i] + T(0.25)*(y1[i] + dt*r[i]);
}
template <typename T>
__global__ void rk3_s3_drifter_t(T* d, const T* o, const T* y2, const T* r, T dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) d[i] = (T(1)/T(3))*o[i] + (T(2)/T(3))*(y2[i] + dt*r[i]);
}
template <typename T>
__global__ void restrict_drifter_t(T* d, const T* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) d[i] = fmin(d[i], o[i]);
}
template <typename T>
__global__ void ttr_drifter_t(double* ttr, const T* o, const T* c, double t, double dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && isinf(ttr[i]) && c[i] <= T(0)) {
        ttr[i] = (o[i] > T(0)) ? t + double(o[i]) / double(o[i] - c[i]) * dt : t;
    }
}
template <typename T>
__global__ void abs_kernel_t(const T* src, T* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = fabs(src[i]);
}

// Voronoi label propagation. A cell gets a label ONLY when the front has
// actually arrived (V <= 0).  Cells outside the front stay unlabeled so
// "unreachable in the solve horizon" is preserved as label == 0xFFFF.
//
// Behavior:
//   - V > 0  : keep label = INVALID. A cell that was labeled but is now
//              > 0 was pushed back by a clamp; that cannot happen with
//              restrictUpdate, but if it does the label drops to INVALID.
//   - V <= 0 with already-set label : sticky (label locked at arrival).
//   - V <= 0 with no label yet      : inherit from a 4-neighbor that
//                                     also has V <= 0 and the smallest V
//                                     (the cell the front came from).
//
// Reads from labels_old, writes to labels; caller must memcpy labels_old
// before each call to avoid intra-step races.
template <typename T>
__global__ void update_voronoi_labels_t(
    std::uint16_t* labels,
    const std::uint16_t* labels_old,
    const T* values,
    int Nx, int Ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = Nx * Ny;
    if (idx >= n) return;
    int i = idx / Ny, j = idx % Ny;

    std::uint16_t my_label = labels_old[idx];
    T v_self = values[idx];

    // Front has not arrived here yet: stay unlabeled.
    if (v_self > T(0)) {
        labels[idx] = std::uint16_t(0xFFFF);
        return;
    }

    // Front is here. If already labeled, lock the label.
    if (my_label != std::uint16_t(0xFFFF)) {
        labels[idx] = my_label;
        return;
    }

    // First arrival this step: inherit from the 4-neighbor with the
    // smallest V that also has a label (the upwind cell the front
    // arrived from).
    T v_min = v_self;
    std::uint16_t pick = std::uint16_t(0xFFFF);
    if (i > 0) {
        int nb = (i - 1) * Ny + j;
        T vn = values[nb];
        std::uint16_t ln = labels_old[nb];
        if (ln != std::uint16_t(0xFFFF) && vn <= T(0) && vn < v_min) {
            v_min = vn; pick = ln;
        }
    }
    if (i < Nx - 1) {
        int nb = (i + 1) * Ny + j;
        T vn = values[nb];
        std::uint16_t ln = labels_old[nb];
        if (ln != std::uint16_t(0xFFFF) && vn <= T(0) && vn < v_min) {
            v_min = vn; pick = ln;
        }
    }
    if (j > 0) {
        int nb = idx - 1;
        T vn = values[nb];
        std::uint16_t ln = labels_old[nb];
        if (ln != std::uint16_t(0xFFFF) && vn <= T(0) && vn < v_min) {
            v_min = vn; pick = ln;
        }
    }
    if (j < Ny - 1) {
        int nb = idx + 1;
        T vn = values[nb];
        std::uint16_t ln = labels_old[nb];
        if (ln != std::uint16_t(0xFFFF) && vn <= T(0) && vn < v_min) {
            v_min = vn; pick = ln;
        }
    }
    labels[idx] = pick;
}

// Forward declaration of dispatching entry.
SolveResult solveGPU_Drifter(const Grid& grid, const ArrayND& initial_data,
                              const SolveOptions& opts,
                              const ArrayND& flow_x, const ArrayND& flow_y,
                              double a_max, double b_max);

SolveResult solveGPU_DrifterLow(const Grid& grid, const ArrayND& initial_data,
                                const SolveOptions& opts,
                                const ArrayND& flow_x, const ArrayND& flow_y,
                                double a_max, double b_max)
{
    SolveOptions o = opts;
    o.accuracy = AccuracyLevel::Low;
    return solveGPU_Drifter(grid, initial_data, o, flow_x, flow_y, a_max, b_max);
}

template <typename T>
static SolveResult solve_drifter_impl(
    const Grid& grid, const ArrayND& initial_data,
    const SolveOptions& opts,
    const ArrayND& flow_x, const ArrayND& flow_y,
    double a_max, double b_max)
{
    int Nx = grid.N[0], Ny = grid.N[1];
    int np = Nx * Ny;
    bool use_weno5 = (opts.accuracy == AccuracyLevel::VeryHigh ||
                       opts.accuracy == AccuracyLevel::High);
    T inv_dx0 = T(1.0 / grid.dx[0]);
    T inv_dx1 = T(1.0 / grid.dx[1]);
    int bdry0 = (grid.bdry[0] == BoundaryType::Periodic) ? 1 : 0;
    int bdry1 = (grid.bdry[1] == BoundaryType::Periodic) ? 1 : 0;

    T *d_data, *d_data_old = nullptr, *d_rhs = nullptr, *d_pad0 = nullptr, *d_pad1 = nullptr;
    T *d_flow_x, *d_flow_y;
    T *d_y1 = nullptr, *d_y2 = nullptr;
    T *d_obstacle = nullptr, *d_target = nullptr;
    double *d_ttr = nullptr;
    std::uint16_t *d_labels = nullptr, *d_labels_old = nullptr;
    const bool use_labels = ((int)opts.initial_labels.size() == np);
    CUDA_CHECK(cudaMalloc(&d_data,     np * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_flow_x,   np * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_flow_y,   np * sizeof(T)));
    if (use_weno5) {
        // Fused per-stage path: no pads, no rhs buffer, no V_old copy.
        CUDA_CHECK(cudaMalloc(&d_y1, np * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_y2, np * sizeof(T)));
    } else {
        // Fused first-order path: one ping-pong buffer (d_y1), no pads/rhs/old.
        CUDA_CHECK(cudaMalloc(&d_y1, np * sizeof(T)));
    }
    if (opts.computeTTR) {
        CUDA_CHECK(cudaMalloc(&d_ttr, np * sizeof(double)));
    }
    if (use_labels) {
        CUDA_CHECK(cudaMalloc(&d_labels,     np * sizeof(std::uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_labels_old, np * sizeof(std::uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_labels, opts.initial_labels.data(),
                              np * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    }
    if ((int)opts.obstacle_sdf.size() == np) {
        CUDA_CHECK(cudaMalloc(&d_obstacle, np * sizeof(T)));
        std::vector<T> buf(np);
        for (int i = 0; i < np; ++i) buf[i] = T(opts.obstacle_sdf[i]);
        CUDA_CHECK(cudaMemcpy(d_obstacle, buf.data(), np * sizeof(T), cudaMemcpyHostToDevice));
    }
    if ((int)opts.target_sdf.size() == np) {
        CUDA_CHECK(cudaMalloc(&d_target, np * sizeof(T)));
        std::vector<T> buf(np);
        for (int i = 0; i < np; ++i) buf[i] = T(opts.target_sdf[i]);
        CUDA_CHECK(cudaMemcpy(d_target, buf.data(), np * sizeof(T), cudaMemcpyHostToDevice));
    }
    // Obstacle/target postproc is applied inside the fused step kernels when
    // the buffers are non-null; no separate has_postproc flag is needed.

    std::vector<T> hbuf(np);
    for (int i = 0; i < np; ++i) hbuf[i] = T(initial_data[i]);
    CUDA_CHECK(cudaMemcpy(d_data, hbuf.data(), np * sizeof(T), cudaMemcpyHostToDevice));
    for (int i = 0; i < np; ++i) hbuf[i] = T(flow_x[i]);
    CUDA_CHECK(cudaMemcpy(d_flow_x, hbuf.data(), np * sizeof(T), cudaMemcpyHostToDevice));
    for (int i = 0; i < np; ++i) hbuf[i] = T(flow_y[i]);
    CUDA_CHECK(cudaMemcpy(d_flow_y, hbuf.data(), np * sizeof(T), cudaMemcpyHostToDevice));

    int block = 256;
    int grid_np = (np + block - 1) / block;

    // Max |flow| straight from the host input (the flow field is static).
    T max_abs_flow_x = T(0), max_abs_flow_y = T(0);
    for (int i = 0; i < np; ++i) {
        T fx = std::fabs(T(flow_x[i]));
        T fy = std::fabs(T(flow_y[i]));
        if (fx > max_abs_flow_x) max_abs_flow_x = fx;
        if (fy > max_abs_flow_y) max_abs_flow_y = fy;
    }

    T alpha0 = max_abs_flow_x + T(a_max) + T(b_max);
    T alpha1 = max_abs_flow_y + T(a_max) + T(b_max);
    T cfl_inv_dx = alpha0 * inv_dx0 + alpha1 * inv_dx1;
    double dt_static = (double(cfl_inv_dx) > 0.0) ? opts.factorCFL / double(cfl_inv_dx) : 1e10;

    if (opts.computeTTR) {
        std::vector<double> ttr_init(np, INFINITY);
        for (int i = 0; i < np; i++) {
            if (initial_data[i] <= 0.0) ttr_init[i] = 0.0;
        }
        CUDA_CHECK(cudaMemcpy(d_ttr, ttr_init.data(), np * sizeof(double), cudaMemcpyHostToDevice));
    }

    int restrict_flag = opts.restrictUpdate ? 1 : 0;
    double* ttr_arg = opts.computeTTR ? d_ttr : nullptr;
    T bma = T(b_max) - T(a_max);

    // `cur` tracks the buffer holding the current V. WENO5 stage 3 writes V in
    // place, so cur stays d_data there; the fused Low step ping-pongs cur<->d_y1
    // (no in-place + no V_old copy), so cur alternates.
    T* cur = d_data;
    double t = 0.0;
    int step = 0;
    while (t < opts.tMax - 1e-12 * opts.tMax) {
        double dt_d = std::min(dt_static, opts.tMax - t);
        T dt_T = T(dt_d);

        if (use_weno5) {
            // One fused launch per RK3 stage; restrict/postproc/TTR fused
            // into stage 3, which writes V in place (no V_old copy).
            lf_drifter_weno5_stage_t<T, 1><<<grid_np, block>>>(
                d_data, d_data, d_y1, d_flow_x, d_flow_y,
                nullptr, nullptr, nullptr,
                Nx, Ny, bdry0, bdry1, inv_dx0, inv_dx1,
                alpha0, alpha1, bma, dt_T, 0, t);
            lf_drifter_weno5_stage_t<T, 2><<<grid_np, block>>>(
                d_y1, d_data, d_y2, d_flow_x, d_flow_y,
                nullptr, nullptr, nullptr,
                Nx, Ny, bdry0, bdry1, inv_dx0, inv_dx1,
                alpha0, alpha1, bma, dt_T, 0, t);
            lf_drifter_weno5_stage_t<T, 3><<<grid_np, block>>>(
                d_y2, d_data, d_data, d_flow_x, d_flow_y,
                d_obstacle, d_target, ttr_arg,
                Nx, Ny, bdry0, bdry1, inv_dx0, inv_dx1,
                alpha0, alpha1, bma, dt_T, restrict_flag, t);
            cur = d_data;
        } else {
            // Fused first-order step (ping-pong cur -> pong).
            T* pong = (cur == d_data) ? d_y1 : d_data;
            lf_drifter_low_fused_t<T><<<grid_np, block>>>(
                cur, pong, d_flow_x, d_flow_y, d_obstacle, d_target, ttr_arg,
                Nx, Ny, bdry0, bdry1, inv_dx0, inv_dx1,
                alpha0, alpha1, bma, dt_T, restrict_flag, t);
            cur = pong;
        }
        if (use_labels) {
            CUDA_CHECK(cudaMemcpy(d_labels_old, d_labels,
                                  np * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToDevice));
            update_voronoi_labels_t<T><<<grid_np, block>>>(
                d_labels, d_labels_old, cur, Nx, Ny);
        }
        t += dt_d;
        step++;
    }

    SolveResult result;
    result.final_data.resize(np);
    if (opts.computeTTR) {
        result.ttr.resize(np);
        CUDA_CHECK(cudaMemcpy(result.ttr.data(), d_ttr, np * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(hbuf.data(), cur, np * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < np; ++i) result.final_data[i] = double(hbuf[i]);
    if (use_labels) {
        result.labels.resize(np);
        CUDA_CHECK(cudaMemcpy(result.labels.data(), d_labels,
                              np * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    }
    result.final_time = t;

    cudaFree(d_data);
    if (d_data_old) cudaFree(d_data_old);
    if (d_rhs)  cudaFree(d_rhs);
    if (d_pad0) cudaFree(d_pad0);
    if (d_pad1) cudaFree(d_pad1);
    cudaFree(d_flow_x); cudaFree(d_flow_y);
    if (d_y1) cudaFree(d_y1);
    if (d_y2) cudaFree(d_y2);
    if (d_ttr) cudaFree(d_ttr);
    if (d_obstacle) cudaFree(d_obstacle);
    if (d_target)   cudaFree(d_target);
    if (d_labels)     cudaFree(d_labels);
    if (d_labels_old) cudaFree(d_labels_old);
    return result;
}

SolveResult solveGPU_Drifter(const Grid& grid, const ArrayND& initial_data,
                              const SolveOptions& opts,
                              const ArrayND& flow_x, const ArrayND& flow_y,
                              double a_max, double b_max)
{
    if (opts.precision == Precision::FP32) {
        return solve_drifter_impl<float>(grid, initial_data, opts, flow_x, flow_y, a_max, b_max);
    }
    return solve_drifter_impl<double>(grid, initial_data, opts, flow_x, flow_y, a_max, b_max);
}

} // namespace gpu
} // namespace hjb
