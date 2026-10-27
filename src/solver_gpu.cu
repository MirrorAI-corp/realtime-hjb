// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#include "hjb/options.hpp"
#include "hjb/level_set/shapes.hpp"
#include "detail/value_postproc.cuh"
#include "detail/weno5.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cmath>
#include <algorithm>

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

// ============================================================
// Per-dimension 1D ghost fill for 2D grids
// Matches CPU padWithGhosts exactly
// ============================================================

// Pad along dim 0: input Nx*Ny -> output (Nx+2*gw)*Ny
__global__ void padDim0_kernel(const double* data, double* padded,
                                int Nx, int Ny, int gw, int bdry) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pNx = Nx + 2 * gw;
    if (tid >= pNx * Ny) return;
    int pi = tid / Ny;  // padded row
    int j = tid % Ny;   // column (unchanged)

    int ii = pi - gw;  // interior row index

    if (ii >= 0 && ii < Nx) {
        // Interior: direct copy
        padded[pi * Ny + j] = data[ii * Ny + j];
    } else if (ii < 0) {
        // Left ghost
        int g = -ii - 1;  // 0-based ghost index from boundary
        if (bdry == 0) { // Extrapolate
            double v0 = data[0 * Ny + j];
            double v1 = data[1 * Ny + j];
            padded[pi * Ny + j] = v0 - (g + 1) * (v1 - v0);
        } else { // Periodic
            int src = ((Nx - 1 - g) % Nx + Nx) % Nx;
            padded[pi * Ny + j] = data[src * Ny + j];
        }
    } else {
        // Right ghost
        int g = ii - Nx;  // 0-based ghost index from boundary
        if (bdry == 0) {
            double vn1 = data[(Nx - 1) * Ny + j];
            double vn2 = data[(Nx - 2) * Ny + j];
            padded[pi * Ny + j] = vn1 + (g + 1) * (vn1 - vn2);
        } else {
            int src = g % Nx;
            padded[pi * Ny + j] = data[src * Ny + j];
        }
    }
}

// Pad along dim 1: input Nx*Ny -> output Nx*(Ny+2*gw)
__global__ void padDim1_kernel(const double* data, double* padded,
                                int Nx, int Ny, int gw, int bdry) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pNy = Ny + 2 * gw;
    if (tid >= Nx * pNy) return;
    int i = tid / pNy;    // row (unchanged)
    int pj = tid % pNy;   // padded column

    int jj = pj - gw;  // interior column index

    if (jj >= 0 && jj < Ny) {
        padded[i * pNy + pj] = data[i * Ny + jj];
    } else if (jj < 0) {
        int g = -jj - 1;
        if (bdry == 0) {
            double v0 = data[i * Ny + 0];
            double v1 = data[i * Ny + 1];
            padded[i * pNy + pj] = v0 - (g + 1) * (v1 - v0);
        } else {
            int src = ((Ny - 1 - g) % Ny + Ny) % Ny;
            padded[i * pNy + pj] = data[i * Ny + src];
        }
    } else {
        int g = jj - Ny;
        if (bdry == 0) {
            double vn1 = data[i * Ny + Ny - 1];
            double vn2 = data[i * Ny + Ny - 2];
            padded[i * pNy + pj] = vn1 + (g + 1) * (vn1 - vn2);
        } else {
            int src = g % Ny;
            padded[i * pNy + pj] = data[i * Ny + src];
        }
    }
}

// ============================================================
// ENO2 device helpers
// ============================================================
__device__ double eno2_left(double Dm2, double Dm1, double D0, double inv_dx) {
    double D2a = Dm1 - Dm2;
    double D2b = D0  - Dm1;
    double base = Dm1 * inv_dx;
    return base + ((fabs(D2a) <= fabs(D2b)) ? D2a : D2b) * inv_dx * 0.5;
}

__device__ double eno2_right(double Dm1, double D0, double Dp1, double inv_dx) {
    double D2a = D0  - Dm1;
    double D2b = Dp1 - D0;
    double base = D0 * inv_dx;
    return base - ((fabs(D2a) <= fabs(D2b)) ? D2a : D2b) * inv_dx * 0.5;
}

// weno5_recon lives in detail/weno5.cuh.

// ============================================================
// Fused derivative+Hamiltonian kernels using per-dim padded buffers
// pad0: (Nx+2*gw)*Ny: padded along dim 0, stride = Ny
// pad1: Nx*(Ny+2*gw): padded along dim 1, stride = Ny+2*gw
// ============================================================

// --- Low (1st-order upwind) ---
__global__ void lf_holo_l2_low_kernel(
    const double* pad0, const double* pad1,
    double* rhs, double* alpha_buf,
    int Nx, int Ny, int gw, double inv_dx0, double inv_dx1, double uBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int i = idx / Ny, j = idx % Ny;

    // Dim 0 derivs from pad0: stride = Ny, offset = (i+gw)*Ny + j
    int p0 = (i + gw) * Ny + j;
    double dL0 = (pad0[p0] - pad0[p0 - Ny]) * inv_dx0;
    double dR0 = (pad0[p0 + Ny] - pad0[p0]) * inv_dx0;

    // Dim 1 derivs from pad1: stride = Ny+2*gw, offset = i*(Ny+2*gw) + (j+gw)
    int pNy = Ny + 2 * gw;
    int p1 = i * pNy + (j + gw);
    double dL1 = (pad1[p1] - pad1[p1 - 1]) * inv_dx1;
    double dR1 = (pad1[p1 + 1] - pad1[p1]) * inv_dx1;

    double dc0 = 0.5*(dL0+dR0), dc1 = 0.5*(dL1+dR1);
    double H = uBound * sqrt(dc0*dc0 + dc1*dc1 + 1e-30);
    alpha_buf[idx] = uBound * inv_dx0 + uBound * inv_dx1;
    double diss = 0.5*(uBound*(dR0-dL0) + uBound*(dR1-dL1));
    rhs[idx] = -H + diss;
}

// --- Medium (ENO2) ---
__global__ void lf_holo_l2_eno2_kernel(
    const double* pad0, const double* pad1,
    double* rhs, double* alpha_buf,
    int Nx, int Ny, int gw, double inv_dx0, double inv_dx1, double uBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int i = idx / Ny, j = idx % Ny;

    int s0 = Ny;  // stride for pad0
    int p0 = (i + gw) * s0 + j;
    double D_m2 = pad0[p0 - s0] - pad0[p0 - 2*s0];
    double D_m1 = pad0[p0]      - pad0[p0 - s0];
    double D_0  = pad0[p0 + s0] - pad0[p0];
    double D_p1 = pad0[p0 + 2*s0] - pad0[p0 + s0];
    double dL0 = eno2_left(D_m2, D_m1, D_0, inv_dx0);
    double dR0 = eno2_right(D_m1, D_0, D_p1, inv_dx0);

    int pNy = Ny + 2 * gw;
    int p1 = i * pNy + (j + gw);
    D_m2 = pad1[p1 - 1] - pad1[p1 - 2];
    D_m1 = pad1[p1]     - pad1[p1 - 1];
    D_0  = pad1[p1 + 1] - pad1[p1];
    D_p1 = pad1[p1 + 2] - pad1[p1 + 1];
    double dL1 = eno2_left(D_m2, D_m1, D_0, inv_dx1);
    double dR1 = eno2_right(D_m1, D_0, D_p1, inv_dx1);

    double dc0 = 0.5*(dL0+dR0), dc1 = 0.5*(dL1+dR1);
    double H = uBound * sqrt(dc0*dc0 + dc1*dc1 + 1e-30);
    alpha_buf[idx] = uBound * inv_dx0 + uBound * inv_dx1;
    double diss = 0.5*(uBound*(dR0-dL0) + uBound*(dR1-dL1));
    rhs[idx] = -H + diss;
}

// --- VeryHigh (WENO5) ---
__global__ void lf_holo_l2_weno5_kernel(
    const double* pad0, const double* pad1,
    double* rhs, double* alpha_buf,
    int Nx, int Ny, int gw, double inv_dx0, double inv_dx1, double uBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int i = idx / Ny, j = idx % Ny;

    int s0 = Ny;
    int p0 = (i + gw) * s0 + j;
    // Left-biased stencil along dim 0
    double v1 = (pad0[p0 - 2*s0] - pad0[p0 - 3*s0]) * inv_dx0;
    double v2 = (pad0[p0 - s0]   - pad0[p0 - 2*s0]) * inv_dx0;
    double v3 = (pad0[p0]        - pad0[p0 - s0])    * inv_dx0;
    double v4 = (pad0[p0 + s0]   - pad0[p0])         * inv_dx0;
    double v5 = (pad0[p0 + 2*s0] - pad0[p0 + s0])    * inv_dx0;
    double dL0 = weno5_recon(v1, v2, v3, v4, v5);
    // Right-biased
    double w1 = (pad0[p0 + 3*s0] - pad0[p0 + 2*s0]) * inv_dx0;
    double dR0 = weno5_recon(w1, v5, v4, v3, v2);

    int pNy = Ny + 2 * gw;
    int p1 = i * pNy + (j + gw);
    v1 = (pad1[p1-2] - pad1[p1-3]) * inv_dx1;
    v2 = (pad1[p1-1] - pad1[p1-2]) * inv_dx1;
    v3 = (pad1[p1]   - pad1[p1-1]) * inv_dx1;
    v4 = (pad1[p1+1] - pad1[p1])   * inv_dx1;
    v5 = (pad1[p1+2] - pad1[p1+1]) * inv_dx1;
    double dL1 = weno5_recon(v1, v2, v3, v4, v5);
    w1 = (pad1[p1+3] - pad1[p1+2]) * inv_dx1;
    double dR1 = weno5_recon(w1, v5, v4, v3, v2);

    double dc0 = 0.5*(dL0+dR0), dc1 = 0.5*(dL1+dR1);
    double H = uBound * sqrt(dc0*dc0 + dc1*dc1 + 1e-30);
    alpha_buf[idx] = uBound * inv_dx0 + uBound * inv_dx1;
    double diss = 0.5*(uBound*(dR0-dL0) + uBound*(dR1-dL1));
    rhs[idx] = -H + diss;
}

// ============================================================
// RK update kernels
// ============================================================
__global__ void euler_update_kernel(double* data, const double* data_old,
                                     const double* rhs, double dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = data_old[i] + dt * rhs[i];
}

// Device-resident-dt variant: reads dt from device memory. Pairs with
// compute_dt_2d_kernel so an entire integration step can be sequenced
// on the GPU without a host roundtrip per stage.
__global__ void euler_update_kernel_p(double* data, const double* data_old,
                                       const double* rhs, const double* dt_ptr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = data_old[i] + (*dt_ptr) * rhs[i];
}

// dt = clamp(factorCFL / max(alpha), 0, t_remaining). Single-thread kernel.
__global__ void compute_dt_2d_kernel(const double* alpha_max, double* dt_out,
                                      double factorCFL, double t_remaining) {
    double a = *alpha_max;
    double dt = (a > 0.0) ? (factorCFL / a) : 1e10;
    if (dt > t_remaining) dt = t_remaining;
    if (dt < 0.0) dt = 0.0;
    *dt_out = dt;
}

__global__ void restrict_update_kernel(double* data, const double* data_old, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = fmin(data[i], data_old[i]);
}

__global__ void rk2_combine_kernel(double* data, const double* data_old,
                                    const double* y1, const double* rhs2,
                                    double dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = 0.5 * (data_old[i] + y1[i] + dt * rhs2[i]);
}

__global__ void rk3_stage2_kernel(double* y2, const double* data_old,
                                   const double* y1, const double* rhs2,
                                   double dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y2[i] = 0.75 * data_old[i] + 0.25 * (y1[i] + dt * rhs2[i]);
}

__global__ void rk3_stage3_kernel(double* data, const double* data_old,
                                   const double* y2, const double* rhs3,
                                   double dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = (1.0/3.0) * data_old[i] + (2.0/3.0) * (y2[i] + dt * rhs3[i]);
}

__global__ void ttr_update_kernel(double* ttr, const double* data_old,
                                   const double* data_new, double t, double dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (isinf(ttr[i]) && data_new[i] <= 0.0) {
        if (data_old[i] > 0.0) {
            double frac = data_old[i] / (data_old[i] - data_new[i]);
            ttr[i] = t + frac * dt;
        } else {
            ttr[i] = t;
        }
    }
}

// rk3_stage_write (the shared SSP-RK3 stage epilogue) and the direct-fetch
// fetchB / weno5_axis stencil helpers live in detail/weno5.cuh.

// Fused per-RK3-stage Holonomic-L2 WENO5 kernel (direct fetch, no pads):
// H = uBound*||grad V||, global-LF dissipation. The CFL wavespeed
// uBound*(1/dx0 + 1/dx1) is constant, so dt is host-computed.
template <int STAGE>
__global__ void holo_l2_weno5_stage_kernel(
    const double* __restrict__ src, const double* vold, double* out,
    const double* __restrict__ obstacle, const double* __restrict__ target,
    double* ttr,
    int Nx, int Ny, int bdry0, int bdry1,
    double inv_dx0, double inv_dx1, double uBound,
    double dt, int restrict_update, double t_now)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int i = idx / Ny, j = idx % Ny;

    double dL0, dR0, dL1, dR1;
    weno5_axis<double>(src, i, Nx, j, Ny, bdry0, inv_dx0, dL0, dR0);
    weno5_axis<double>(src, j, Ny, i * Ny, 1, bdry1, inv_dx1, dL1, dR1);
    double dc0 = 0.5*(dL0+dR0), dc1 = 0.5*(dL1+dR1);

    double H = uBound * sqrt(dc0*dc0 + dc1*dc1 + 1e-30);
    double diss = 0.5*(uBound*(dR0-dL0) + uBound*(dR1-dL1));
    double r = -H + diss;

    rk3_stage_write<double, STAGE>(idx, src[idx], r, dt, vold, out,
                                   obstacle, target, ttr, restrict_update, t_now);
}

// Fused per-RK3-stage double-integrator WENO5 kernel:
// H_std = x2*p1 - inputBound*|p2|; LLF dissipation. The CFL wavespeed
// max|x2|/dx0 + inputBound/dx1 is constant (grid-corner max), host-computed.
template <int STAGE>
__global__ void lf_di2d_weno5_stage(const double* __restrict__ src,
                                    const double* vold, double* out,
                                    const double* __restrict__ obstacle,
                                    const double* __restrict__ target,
                                    double* ttr,
                                    int Nx, int Ny, double x1_lo, double dx0, double dx1,
                                    double inputBound,
                                    double dt, int restrict_update, double t_now) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int i = idx / Ny, j = idx % Ny;
    double inv_dx0 = 1.0 / dx0, inv_dx1 = 1.0 / dx1;
    double x2 = x1_lo + j * dx1;

    double dL0, dR0, dL1, dR1;
    weno5_axis<double>(src, i, Nx, j, Ny, 0, inv_dx0, dL0, dR0);   // dim0: stride Ny, base j
    weno5_axis<double>(src, j, Ny, i * Ny, 1, 0, inv_dx1, dL1, dR1); // dim1: stride 1, base i*Ny
    double dc0 = 0.5 * (dL0 + dR0), dc1 = 0.5 * (dL1 + dR1);

    double H_std = x2 * dc0 - inputBound * fabs(dc1);
    double a0 = fabs(x2), a1 = inputBound;
    double diss = 0.5 * (a0 * (dR0 - dL0) + a1 * (dR1 - dL1));
    double r = H_std + diss;   // rhs = -(-H_std) + diss

    rk3_stage_write<double, STAGE>(idx, src[idx], r, dt, vold, out,
                                   obstacle, target, ttr, restrict_update, t_now);
}

// ============================================================
// GPU Solver for Holonomic L2: fused WENO5 path.
// One kernel per RK3 stage (direct-fetch stencil + Hamiltonian + stage
// combine + restrict/postproc/TTR fused), dt host-computed from the
// constant wavespeed. 3 launches/step, no pads, no rhs/alpha round-trip,
// no V -> V_old memcpy.
// ============================================================
static SolveResult solveGPU_HolonomicL2_weno5_fused(
    const Grid& grid, const ArrayND& initial_data,
    const SolveOptions& opts, double inputBound)
{
    int Nx = grid.N[0], Ny = grid.N[1];
    int np = Nx * Ny;
    double inv_dx0 = 1.0 / grid.dx[0];
    double inv_dx1 = 1.0 / grid.dx[1];
    int bdry0 = (grid.bdry[0] == BoundaryType::Periodic) ? 1 : 0;
    int bdry1 = (grid.bdry[1] == BoundaryType::Periodic) ? 1 : 0;

    double *d_data, *d_y1, *d_y2, *d_ttr;
    CUDA_CHECK(cudaMalloc(&d_data, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y1, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y2, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ttr, np * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_data, initial_data.data(), np * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> ttr_init(np, INFINITY);
    for (int i = 0; i < np; i++)
        if (initial_data[i] <= 0.0) ttr_init[i] = 0.0;
    CUDA_CHECK(cudaMemcpy(d_ttr, ttr_init.data(), np * sizeof(double), cudaMemcpyHostToDevice));

    double *d_obstacle = nullptr, *d_target = nullptr;
    if ((int)opts.obstacle_sdf.size() == np) {
        CUDA_CHECK(cudaMalloc(&d_obstacle, np * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_obstacle, opts.obstacle_sdf.data(), np * sizeof(double), cudaMemcpyHostToDevice));
    }
    if ((int)opts.target_sdf.size() == np) {
        CUDA_CHECK(cudaMalloc(&d_target, np * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_target, opts.target_sdf.data(), np * sizeof(double), cudaMemcpyHostToDevice));
    }

    int block = 256;
    int grid_np = (np + block - 1) / block;
    int restrict_flag = opts.restrictUpdate ? 1 : 0;
    double* ttr_arg = opts.computeTTR ? d_ttr : nullptr;

    // Constant wavespeed, so dt is static.
    double alpha_const = inputBound * inv_dx0 + inputBound * inv_dx1;
    double dt_static = (alpha_const > 0.0) ? opts.factorCFL / alpha_const : 1e10;

    double t = 0.0;
    while (t < opts.tMax - 1e-12 * opts.tMax) {
        double dt = std::min(dt_static, opts.tMax - t);
        holo_l2_weno5_stage_kernel<1><<<grid_np, block>>>(
            d_data, d_data, d_y1, nullptr, nullptr, nullptr,
            Nx, Ny, bdry0, bdry1, inv_dx0, inv_dx1, inputBound, dt, 0, t);
        holo_l2_weno5_stage_kernel<2><<<grid_np, block>>>(
            d_y1, d_data, d_y2, nullptr, nullptr, nullptr,
            Nx, Ny, bdry0, bdry1, inv_dx0, inv_dx1, inputBound, dt, 0, t);
        holo_l2_weno5_stage_kernel<3><<<grid_np, block>>>(
            d_y2, d_data, d_data, d_obstacle, d_target, ttr_arg,
            Nx, Ny, bdry0, bdry1, inv_dx0, inv_dx1, inputBound, dt, restrict_flag, t);
        t += dt;
    }

    SolveResult result;
    result.ttr.resize(np);
    result.final_data.resize(np);
    CUDA_CHECK(cudaMemcpy(result.ttr.data(), d_ttr, np * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.final_data.data(), d_data, np * sizeof(double), cudaMemcpyDeviceToHost));
    result.final_time = t;

    cudaFree(d_data); cudaFree(d_y1); cudaFree(d_y2); cudaFree(d_ttr);
    if (d_obstacle) cudaFree(d_obstacle);
    if (d_target)   cudaFree(d_target);
    return result;
}

// ============================================================
// GPU Solver for Holonomic L2
// ============================================================
SolveResult solveGPU_HolonomicL2(const Grid& grid, const ArrayND& initial_data,
                                  const SolveOptions& opts, double inputBound) {
    // WENO5 (the verification/benchmark configuration) takes the fused
    // per-stage path; Low/ENO2 keep the padded-buffer path below.
    if (opts.accuracy != AccuracyLevel::Low && opts.accuracy != AccuracyLevel::Medium) {
        return solveGPU_HolonomicL2_weno5_fused(grid, initial_data, opts, inputBound);
    }
    int Nx = grid.N[0], Ny = grid.N[1];
    int np = Nx * Ny;
    int gw;
    switch (opts.accuracy) {
        case AccuracyLevel::Low:      gw = 1; break;
        case AccuracyLevel::Medium:   gw = 2; break;
        default:                      gw = 3; break;
    }
    int pad0_size = (Nx + 2*gw) * Ny;       // padded along dim 0
    int pad1_size = Nx * (Ny + 2*gw);       // padded along dim 1
    double inv_dx0 = 1.0 / grid.dx[0];
    double inv_dx1 = 1.0 / grid.dx[1];

    int bdry0 = (grid.bdry[0] == BoundaryType::Periodic) ? 1 : 0;
    int bdry1 = (grid.bdry[1] == BoundaryType::Periodic) ? 1 : 0;

    // Device memory
    double *d_data, *d_data_old, *d_rhs, *d_pad0, *d_pad1, *d_alpha, *d_ttr;
    double *d_y1, *d_y2;
    CUDA_CHECK(cudaMalloc(&d_data, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_data_old, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pad0, pad0_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pad1, pad1_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ttr, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y1, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y2, np * sizeof(double)));

    // CUB reduction
    double *d_max_alpha;
    CUDA_CHECK(cudaMalloc(&d_max_alpha, sizeof(double)));
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Max(d_temp, temp_bytes, d_alpha, d_max_alpha, np);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    // Copy initial data
    CUDA_CHECK(cudaMemcpy(d_data, initial_data.data(), np * sizeof(double), cudaMemcpyHostToDevice));

    // Init TTR
    std::vector<double> ttr_init(np, INFINITY);
    for (int i = 0; i < np; i++)
        if (initial_data[i] <= 0.0) ttr_init[i] = 0.0;
    CUDA_CHECK(cudaMemcpy(d_ttr, ttr_init.data(), np * sizeof(double), cudaMemcpyHostToDevice));

    // Value postprocessor buffers (obstacle / target SDFs).
    double *d_obstacle = nullptr, *d_target = nullptr;
    if ((int)opts.obstacle_sdf.size() == np) {
        CUDA_CHECK(cudaMalloc(&d_obstacle, np * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_obstacle, opts.obstacle_sdf.data(), np * sizeof(double), cudaMemcpyHostToDevice));
    }
    if ((int)opts.target_sdf.size() == np) {
        CUDA_CHECK(cudaMalloc(&d_target, np * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_target, opts.target_sdf.data(), np * sizeof(double), cudaMemcpyHostToDevice));
    }
    const bool has_postproc = (d_obstacle != nullptr) || (d_target != nullptr);

    int block = 256;
    int grid_np = (np + block - 1) / block;
    int grid_p0 = (pad0_size + block - 1) / block;
    int grid_p1 = (pad1_size + block - 1) / block;

    // Compute RHS lambda
    // CFL dt is constant for these time-invariant Hamiltonians: compute the
    // global-max-wavespeed reduction + host sync once, then cache and reuse
    // instead of every RK substage.
    double dt_cached = -1.0;
    auto computeRHS = [&](double* src) -> double {
        // Build per-dimension padded buffers
        padDim0_kernel<<<grid_p0, block>>>(src, d_pad0, Nx, Ny, gw, bdry0);
        padDim1_kernel<<<grid_p1, block>>>(src, d_pad1, Nx, Ny, gw, bdry1);

        // Fused derivative + Hamiltonian
        switch (opts.accuracy) {
            case AccuracyLevel::Low:
                lf_holo_l2_low_kernel<<<grid_np, block>>>(
                    d_pad0, d_pad1, d_rhs, d_alpha,
                    Nx, Ny, gw, inv_dx0, inv_dx1, inputBound);
                break;
            case AccuracyLevel::Medium:
                lf_holo_l2_eno2_kernel<<<grid_np, block>>>(
                    d_pad0, d_pad1, d_rhs, d_alpha,
                    Nx, Ny, gw, inv_dx0, inv_dx1, inputBound);
                break;
            default:
                lf_holo_l2_weno5_kernel<<<grid_np, block>>>(
                    d_pad0, d_pad1, d_rhs, d_alpha,
                    Nx, Ny, gw, inv_dx0, inv_dx1, inputBound);
                break;
        }

        // CFL dt (computed once, then cached; constant for time-invariant H)
        if (dt_cached < 0.0) {
            cub::DeviceReduce::Max(d_temp, temp_bytes, d_alpha, d_max_alpha, np);
            double h_max;
            CUDA_CHECK(cudaMemcpy(&h_max, d_max_alpha, sizeof(double), cudaMemcpyDeviceToHost));
            dt_cached = (h_max > 0.0) ? opts.factorCFL / h_max : 1e10;
        }
        return dt_cached;
    };

    double t = 0.0;
    while (t < opts.tMax - 1e-12 * opts.tMax) {
        CUDA_CHECK(cudaMemcpy(d_data_old, d_data, np * sizeof(double), cudaMemcpyDeviceToDevice));

        if (opts.accuracy == AccuracyLevel::Low) {
            double dt = computeRHS(d_data);
            dt = std::min(dt, opts.tMax - t);
            euler_update_kernel<<<grid_np, block>>>(d_data, d_data_old, d_rhs, dt, np);
            if (opts.restrictUpdate)
                restrict_update_kernel<<<grid_np, block>>>(d_data, d_data_old, np);
            if (has_postproc)
                apply_value_postproc_t<double><<<grid_np, block>>>(d_data, d_obstacle, d_target, np);
            if (opts.computeTTR)
                ttr_update_kernel<<<grid_np, block>>>(d_ttr, d_data_old, d_data, t, dt, np);
            t += dt;
        } else if (opts.accuracy == AccuracyLevel::Medium) {
            double dt = computeRHS(d_data);
            dt = std::min(dt, opts.tMax - t);
            euler_update_kernel<<<grid_np, block>>>(d_y1, d_data_old, d_rhs, dt, np);
            computeRHS(d_y1);
            rk2_combine_kernel<<<grid_np, block>>>(d_data, d_data_old, d_y1, d_rhs, dt, np);
            if (opts.restrictUpdate)
                restrict_update_kernel<<<grid_np, block>>>(d_data, d_data_old, np);
            if (has_postproc)
                apply_value_postproc_t<double><<<grid_np, block>>>(d_data, d_obstacle, d_target, np);
            if (opts.computeTTR)
                ttr_update_kernel<<<grid_np, block>>>(d_ttr, d_data_old, d_data, t, dt, np);
            t += dt;
        } else {
            double dt = computeRHS(d_data);
            dt = std::min(dt, opts.tMax - t);
            euler_update_kernel<<<grid_np, block>>>(d_y1, d_data_old, d_rhs, dt, np);
            computeRHS(d_y1);
            rk3_stage2_kernel<<<grid_np, block>>>(d_y2, d_data_old, d_y1, d_rhs, dt, np);
            computeRHS(d_y2);
            rk3_stage3_kernel<<<grid_np, block>>>(d_data, d_data_old, d_y2, d_rhs, dt, np);
            if (opts.restrictUpdate)
                restrict_update_kernel<<<grid_np, block>>>(d_data, d_data_old, np);
            if (has_postproc)
                apply_value_postproc_t<double><<<grid_np, block>>>(d_data, d_obstacle, d_target, np);
            if (opts.computeTTR)
                ttr_update_kernel<<<grid_np, block>>>(d_ttr, d_data_old, d_data, t, dt, np);
            t += dt;
        }
    }

    // Copy back
    SolveResult result;
    result.ttr.resize(np);
    result.final_data.resize(np);
    CUDA_CHECK(cudaMemcpy(result.ttr.data(), d_ttr, np * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.final_data.data(), d_data, np * sizeof(double), cudaMemcpyDeviceToHost));
    result.final_time = t;

    cudaFree(d_data); cudaFree(d_data_old); cudaFree(d_rhs);
    cudaFree(d_pad0); cudaFree(d_pad1);
    cudaFree(d_alpha); cudaFree(d_ttr);
    cudaFree(d_y1); cudaFree(d_y2);
    if (d_obstacle) cudaFree(d_obstacle);
    if (d_target)   cudaFree(d_target);
    cudaFree(d_max_alpha); cudaFree(d_temp);

    return result;
}


// ============================================================
// 2D Double Integrator: WENO5 + RK3 + LLF
//   State: (x1, x2) where x1 is position, x2 is velocity.
//   Dynamics: dx1/dt = x2, dx2/dt = u, |u| <= inputBound.
//   Mitchell Hamiltonian H_std = -(x2*p1 - inputBound*|p2|) (control_mode="min").
//   We use kernel-H = -H_std so that rhs = -H + diss = +H_std + diss matches
//   hj_reachability's backward integration (see solver_gpu_3d.cu for derivation).
// ============================================================
__global__ void lf_di2d_weno5_kernel(
    const double* pad0, const double* pad1,
    double* rhs, double* alpha_buf,
    int Nx, int Ny, int gw,
    double x0_lo, double dx0, double x1_lo, double dx1,
    double inputBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;
    int i = idx / Ny, j = idx % Ny;

    double inv_dx0 = 1.0 / dx0;
    double inv_dx1 = 1.0 / dx1;
    double x2 = x1_lo + j * dx1;  // velocity = second state coordinate

    int s0 = Ny;
    int p0 = (i + gw) * s0 + j;
    double v1 = (pad0[p0 - 2*s0] - pad0[p0 - 3*s0]) * inv_dx0;
    double v2 = (pad0[p0 - s0]   - pad0[p0 - 2*s0]) * inv_dx0;
    double v3 = (pad0[p0]        - pad0[p0 - s0])    * inv_dx0;
    double v4 = (pad0[p0 + s0]   - pad0[p0])         * inv_dx0;
    double v5 = (pad0[p0 + 2*s0] - pad0[p0 + s0])    * inv_dx0;
    double dL0 = weno5_recon(v1, v2, v3, v4, v5);
    double w1 = (pad0[p0 + 3*s0] - pad0[p0 + 2*s0]) * inv_dx0;
    double dR0 = weno5_recon(w1, v5, v4, v3, v2);

    int pNy = Ny + 2 * gw;
    int p1 = i * pNy + (j + gw);
    v1 = (pad1[p1-2] - pad1[p1-3]) * inv_dx1;
    v2 = (pad1[p1-1] - pad1[p1-2]) * inv_dx1;
    v3 = (pad1[p1]   - pad1[p1-1]) * inv_dx1;
    v4 = (pad1[p1+1] - pad1[p1])   * inv_dx1;
    v5 = (pad1[p1+2] - pad1[p1+1]) * inv_dx1;
    double dL1 = weno5_recon(v1, v2, v3, v4, v5);
    w1 = (pad1[p1+3] - pad1[p1+2]) * inv_dx1;
    double dR1 = weno5_recon(w1, v5, v4, v3, v2);

    double dc0 = 0.5*(dL0+dR0), dc1 = 0.5*(dL1+dR1);

    // Standard Mitchell Hamiltonian (control_mode="min"):
    //     H_std = min_u (∇V · f) = x2 * p1 - inputBound * |p2|
    // For our forward-integration to match hj_reachability's backward integration
    // (rhs = +H_std + diss == -H + diss), we set kernel H = -H_std.
    double H_std = x2 * dc0 - inputBound * fabs(dc1);
    double H = -H_std;

    // LLF dissipation coefficients: |dH/dp_d|
    //   |dH/dp0| = |x2|       (varies per cell)
    //   |dH/dp1| = inputBound  (constant)
    double a0 = fabs(x2);
    double a1 = inputBound;
    double diss = 0.5 * (a0 * (dR0 - dL0) + a1 * (dR1 - dL1));

    rhs[idx] = -H + diss;
    alpha_buf[idx] = a0 * inv_dx0 + a1 * inv_dx1;
}

// ------------------------------------------------------------------
// Fused WENO5: read V directly with inline ghost handling, no padded
// global buffers. fetchB / weno5_axis live in detail/weno5.cuh (shared
// with the drifter solver).
SolveResult solveGPU_DoubleIntegrator2D(
    const Grid& grid, const ArrayND& initial_data,
    const SolveOptions& opts, double inputBound)
{
    int Nx = grid.N[0], Ny = grid.N[1];
    int np = Nx * Ny;
    double dx0 = grid.dx[0], dx1 = grid.dx[1];
    double x1_lo = grid.min[1];

    double *d_data, *d_ttr = nullptr;
    double *d_y1, *d_y2;
    CUDA_CHECK(cudaMalloc(&d_data, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y1, np * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y2, np * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_data, initial_data.data(), np * sizeof(double), cudaMemcpyHostToDevice));

    if (opts.computeTTR) {
        CUDA_CHECK(cudaMalloc(&d_ttr, np * sizeof(double)));
        std::vector<double> ttr_init(np, INFINITY);
        for (int i = 0; i < np; i++)
            if (initial_data[i] <= 0.0) ttr_init[i] = 0.0;
        CUDA_CHECK(cudaMemcpy(d_ttr, ttr_init.data(), np * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Value postprocessor buffers (obstacle / target SDFs).
    double *d_obstacle = nullptr, *d_target = nullptr;
    if ((int)opts.obstacle_sdf.size() == np) {
        CUDA_CHECK(cudaMalloc(&d_obstacle, np * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_obstacle, opts.obstacle_sdf.data(), np * sizeof(double), cudaMemcpyHostToDevice));
    }
    if ((int)opts.target_sdf.size() == np) {
        CUDA_CHECK(cudaMalloc(&d_target, np * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_target, opts.target_sdf.data(), np * sizeof(double), cudaMemcpyHostToDevice));
    }

    int block = 256;
    int grid_np = (np + block - 1) / block;
    int restrict_flag = opts.restrictUpdate ? 1 : 0;
    double* ttr_arg = opts.computeTTR ? d_ttr : nullptr;

    // The global max wavespeed max|x2|/dx0 + inputBound/dx1 is attained at a
    // velocity-axis grid extreme (|x2| is maximized at j=0 or j=Ny-1), so the
    // CFL dt is computed directly on the host; no per-cell alpha buffer, no
    // reduction, no device->host sync.
    double max_abs_x2 = std::max(std::fabs(x1_lo), std::fabs(x1_lo + (Ny - 1) * dx1));
    double alpha_const = max_abs_x2 / dx0 + inputBound / dx1;
    double dt_static = (alpha_const > 0.0) ? opts.factorCFL / alpha_const : 1e10;

    double t = 0.0;
    while (t < opts.tMax - 1e-12 * opts.tMax) {
        double dt = std::min(dt_static, opts.tMax - t);
        lf_di2d_weno5_stage<1><<<grid_np, block>>>(
            d_data, d_data, d_y1, nullptr, nullptr, nullptr,
            Nx, Ny, x1_lo, dx0, dx1, inputBound, dt, 0, t);
        lf_di2d_weno5_stage<2><<<grid_np, block>>>(
            d_y1, d_data, d_y2, nullptr, nullptr, nullptr,
            Nx, Ny, x1_lo, dx0, dx1, inputBound, dt, 0, t);
        lf_di2d_weno5_stage<3><<<grid_np, block>>>(
            d_y2, d_data, d_data, d_obstacle, d_target, ttr_arg,
            Nx, Ny, x1_lo, dx0, dx1, inputBound, dt, restrict_flag, t);
        t += dt;
    }

    SolveResult result;
    if (opts.computeTTR) {
        result.ttr.resize(np);
        CUDA_CHECK(cudaMemcpy(result.ttr.data(), d_ttr, np * sizeof(double), cudaMemcpyDeviceToHost));
    }
    result.final_data.resize(np);
    CUDA_CHECK(cudaMemcpy(result.final_data.data(), d_data, np * sizeof(double), cudaMemcpyDeviceToHost));
    result.final_time = t;

    cudaFree(d_data);
    cudaFree(d_y1); cudaFree(d_y2);
    if (d_ttr) cudaFree(d_ttr);
    if (d_obstacle) cudaFree(d_obstacle);
    if (d_target)   cudaFree(d_target);
    return result;
}



} // namespace gpu
} // namespace hjb
