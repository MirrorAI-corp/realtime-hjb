// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

/// Full 4D pursuit-evasion HJBI solver with WENO5 + RK3 + LLF.
///
/// State: (x_ego, y_ego, x_obs, y_obs)
/// H_std = -v_ego * ||(p0,p1)|| + v_obs * ||(p2,p3)||
///
/// Templated on `Real` in {float, double}. FP32 is about 30 times faster on
/// consumer GPUs and agrees with FP64 on the tube to within 0.1 %.
///
#include "hjb/options.hpp"
#include "detail/weno5.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
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

// weno5_recon lives in detail/weno5.cuh.

template <typename T>
__global__ void pad4d_kernel_t(
    const T* src, T* dst,
    int N0, int N1, int N2, int N3,
    int dim_d, int gw, int pNd, int np_padded)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np_padded) return;

    int pN[4] = {N0, N1, N2, N3};
    pN[dim_d] = pNd;

    int idx[4];
    int rem = tid;
    for (int d = 3; d >= 0; d--) { idx[d] = rem % pN[d]; rem /= pN[d]; }

    int Nd_orig = (dim_d==0)?N0:(dim_d==1)?N1:(dim_d==2)?N2:N3;
    int ii = idx[dim_d] - gw;

    int si[4] = {idx[0], idx[1], idx[2], idx[3]};

    if (ii >= 0 && ii < Nd_orig) {
        si[dim_d] = ii;
        dst[tid] = src[((si[0]*N1 + si[1])*N2 + si[2])*N3 + si[3]];
    } else if (ii < 0) {
        si[dim_d] = 0;
        int si1[4] = {idx[0], idx[1], idx[2], idx[3]};
        si1[dim_d] = 1;
        T v0 = src[((si[0]*N1+si[1])*N2+si[2])*N3+si[3]];
        T v1 = src[((si1[0]*N1+si1[1])*N2+si1[2])*N3+si1[3]];
        dst[tid] = v0 + T(-ii)*(v0-v1);
    } else {
        int g = ii - Nd_orig;
        si[dim_d] = Nd_orig-1;
        int si1[4] = {idx[0], idx[1], idx[2], idx[3]};
        si1[dim_d] = Nd_orig-2;
        T vn1 = src[((si[0]*N1+si[1])*N2+si[2])*N3+si[3]];
        T vn2 = src[((si1[0]*N1+si1[1])*N2+si1[2])*N3+si1[3]];
        dst[tid] = vn1 + T(g+1)*(vn1-vn2);
    }
}

// Clamp-boundary read of a 4D global-memory cell. Used by the fused
// kernel below; matches Neumann/extrapolate behavior for ghost cells.
template <typename T>
__device__ __forceinline__ T load_cell_4d_clamp(
    const T* data, int x0, int x1, int x2, int x3,
    int N0, int N1, int N2, int N3)
{
    int i0 = max(0, min(N0 - 1, x0));
    int i1 = max(0, min(N1 - 1, x1));
    int i2 = max(0, min(N2 - 1, x2));
    int i3 = max(0, min(N3 - 1, x3));
    return data[((i0 * N1 + i1) * N2 + i2) * N3 + i3];
}

// Fused 4D PE kernel. Loads a (BX+6)(BY+6)(BZ+6)(BW+6) ghost-padded tile
// into shared memory once per block, then computes WENO5 derivatives in
// all four dimensions plus Lax-Friedrichs dissipation plus the Mitchell
// Hamiltonian in a single pass. Replaces 4 padDim4D kernels + 1 lf_pe
// kernel per RK3 stage (5 launches -> 1). Memory-traffic and launch-overhead
// reduction, especially impactful on the embedded SoC where the L2 cache
// is much smaller than the desktop GPU.
//
// Block dim is flat 1D (BX*BY*BZ*BW); a per-thread decode maps it back to
// the 4D index. Grid dim is 3D: (nb01 = nb0*nb1, nb2, nb3).
//
// Shared tile fits in Ada static shared at FP32 (40 KB) and in Thor static
// shared at both FP32 (40 KB) and FP64 (80 KB).


// ------------------------------------------------------------------
// Direct-fetch WENO5 helpers: read V along one axis straight from the
// unpadded global array, applying the same linear-extrapolation ghost
// rule as pad4d_kernel_t inline. Removes the 4 pad kernels (4 full-grid
// writes + reads) per RK stage, and lets all four axes hit the SAME
// array in L2 instead of four disjoint padded copies (4x the cache
// footprint at N=51 FP64, where one field alone is ~54 MB).
// ------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T fetch_axis4(const T* __restrict__ V, int p, int N,
                                          int base, int stride) {
    if (p >= 0 && p < N) return V[base + p * stride];
    if (p < 0) {
        T v0 = V[base], v1 = V[base + stride];
        return v0 + T(p) * (v1 - v0);                       // p < 0
    }
    T vn1 = V[base + (N - 1) * stride], vn2 = V[base + (N - 2) * stride];
    return vn1 + T(p - (N - 1)) * (vn1 - vn2);              // p >= N
}

template <typename T>
__device__ __forceinline__ void weno5_axis4(const T* __restrict__ V, int idxc,
                                             int N, int base, int stride,
                                             T inv_dx, T& dL, T& dR) {
    T f[7];
#pragma unroll
    for (int k = -3; k <= 3; ++k)
        f[k + 3] = fetch_axis4<T>(V, idxc + k, N, base, stride);
    T D0 = (f[1] - f[0]) * inv_dx, D1 = (f[2] - f[1]) * inv_dx, D2 = (f[3] - f[2]) * inv_dx;
    T D3 = (f[4] - f[3]) * inv_dx, D4 = (f[5] - f[4]) * inv_dx, D5 = (f[6] - f[5]) * inv_dx;
    dL = weno5_recon<T>(D0, D1, D2, D3, D4);
    dR = weno5_recon<T>(D5, D4, D3, D2, D1);
}

// Fused per-RK3-stage 4D PE kernel: WENO5 (direct fetch) + LLF + Hamiltonian
// + RK stage combine + wall-mask + value clamp (+ restrict/TTR in stage 3),
// all in-register. Replaces, PER STAGE: 4 pad kernels + 1 lf kernel + 1 CUB
// reduction + 1 update kernel + 1 mask kernel + 1 clamp kernel (9 launches,
// ~14 full-grid memory sweeps) with ONE launch (~3 sweeps). The CFL wavespeed
// alpha = v_ego_max*(1/dx0+1/dx1) + v_obs*(1/dx2+1/dx3) is a constant, so dt
// is host-computed and passed by value; no reduction, no per-step sync.
//   STAGE 1:  out = src + dt*rhs(src)                      (src == V)
//   STAGE 2:  out = 3/4 vold + 1/4 (src + dt*rhs(src))     (src == y1)
//   STAGE 3:  out = 1/3 vold + 2/3 (src + dt*rhs(src))     (src == y2)
// Stage 3 writes V in place (vold/out alias; vold is only read at the
// thread's own cell, the stencil reads come from y2).
template <typename T, int STAGE>
__global__ void pe4d_stage_kernel(
    const T* __restrict__ src, const T* vold, T* out,
    const T* __restrict__ ego_speed,
    const T* __restrict__ init, const T* __restrict__ mask,
    double* ttr,
    int N0, int N1, int N2, int N3,
    T inv_dx0, T inv_dx1, T inv_dx2, T inv_dx3,
    T v_ego_max, T v_obs,
    T dt, T vclamp_lo, T vclamp_hi,
    int restrict_update, double t_now, int np)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    int i3 = tid % N3; int rem = tid / N3;
    int i2 = rem % N2; rem /= N2;
    int i1 = rem % N1; int i0 = rem / N1;

    int s2 = N3, s1 = N2 * N3, s0 = N1 * s1;

    T dL0,dR0,dL1,dR1,dL2,dR2,dL3,dR3;
    weno5_axis4<T>(src, i0, N0, tid - i0 * s0, s0, inv_dx0, dL0, dR0);
    weno5_axis4<T>(src, i1, N1, tid - i1 * s1, s1, inv_dx1, dL1, dR1);
    weno5_axis4<T>(src, i2, N2, tid - i2 * s2, s2, inv_dx2, dL2, dR2);
    weno5_axis4<T>(src, i3, N3, tid - i3,      1,  inv_dx3, dL3, dR3);

    T dc0 = T(0.5)*(dL0+dR0), dc1 = T(0.5)*(dL1+dR1);
    T dc2 = T(0.5)*(dL2+dR2), dc3 = T(0.5)*(dL3+dR3);

    // Standard Mitchell Hamiltonian for ego(min)+obs(max):
    //     H_std = -v_ego * ||(p0,p1)|| + v_obs * ||(p2,p3)||
    // hj_reachability integrates BACKWARD with V_t + H_std = 0; we integrate
    // forward with rhs = -H + diss, so kernel-H = -H_std.
    T v_ego = ego_speed ? ego_speed[i0*N1 + i1] : v_ego_max;
    T ego_grad = sqrt(dc0*dc0 + dc1*dc1 + T(1e-30));
    T obs_grad = sqrt(dc2*dc2 + dc3*dc3 + T(1e-30));
    T H_std, a0, a1, a2, a3;
    {
        H_std = -v_ego * ego_grad + v_obs * obs_grad;
#ifdef HJB_PE4D_GLOBAL_LF
        // -DHJB_PE4D_GLOBAL_LF=ON: global Lax-Friedrichs bounds.
        a0 = v_ego; a1 = v_ego; a2 = v_obs; a3 = v_obs;
#else
        a0 = v_ego * fabs(dc0) / ego_grad;
        a1 = v_ego * fabs(dc1) / ego_grad;
        a2 = v_obs * fabs(dc2) / obs_grad;
        a3 = v_obs * fabs(dc3) / obs_grad;
#endif
    }
    T diss = T(0.5)*(a0*(dR0-dL0) + a1*(dR1-dL1) + a2*(dR2-dL2) + a3*(dR3-dL3));

    T r = H_std + diss;   // rhs = -(-H_std) + diss
    T center = src[tid];

    T v, vo = T(0);
    if (STAGE == 1) {
        v = center + dt * r;
    } else if (STAGE == 2) {
        v = T(0.75) * vold[tid] + T(0.25) * (center + dt * r);
    } else {
        vo = vold[tid];
        v = T(1.0/3.0) * vo + T(2.0/3.0) * (center + dt * r);
    }
    if (mask && mask[tid] > T(0.5)) v = init[tid];
    v = fmax(vclamp_lo, fmin(vclamp_hi, v));
    if (STAGE == 3) {
        if (restrict_update) v = fmin(v, vo);
        if (ttr && isinf(ttr[tid]) && v <= T(0)) {
            ttr[tid] = (vo > T(0))
                ? t_now + double(vo) / double(vo - v) * double(dt)
                : t_now;
        }
    }
    out[tid] = v;
}



template <typename T> __global__ void euler4d_t(T* d, const T* o, const T* r, T dt, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=o[i]+dt*r[i];
}
template <typename T> __global__ void rk3_s2_t(T* y2, const T* o, const T* y1, const T* r, T dt, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y2[i]=T(0.75)*o[i]+T(0.25)*(y1[i]+dt*r[i]);
}
template <typename T> __global__ void rk3_s3_t(T* d, const T* o, const T* y2, const T* r, T dt, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=(T(1)/T(3))*o[i]+(T(2)/T(3))*(y2[i]+dt*r[i]);
}
// Device-resident-dt variants. Reads dt from device memory each step,
// letting the CFL pipeline run end-to-end on the GPU without host syncs.
template <typename T> __global__ void euler4d_p(T* d, const T* o, const T* r, const T* dt_p, int n) {
    T dt=*dt_p; int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=o[i]+dt*r[i];
}
template <typename T> __global__ void rk3_s2_p(T* y2, const T* o, const T* y1, const T* r, const T* dt_p, int n) {
    T dt=*dt_p; int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y2[i]=T(0.75)*o[i]+T(0.25)*(y1[i]+dt*r[i]);
}
template <typename T> __global__ void rk3_s3_p(T* d, const T* o, const T* y2, const T* r, const T* dt_p, int n) {
    T dt=*dt_p; int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=(T(1)/T(3))*o[i]+(T(2)/T(3))*(y2[i]+dt*r[i]);
}
template <typename T> __global__ void compute_dt_4d_t(
    const T* alpha_max, T* dt_out, T factorCFL, T t_remaining)
{
    T a = *alpha_max;
    T dt = (a > T(0)) ? (factorCFL / a) : T(1e10);
    if (dt > t_remaining) dt = t_remaining;
    if (dt < T(0)) dt = T(0);
    *dt_out = dt;
}

// Graph-friendly variant: derives t_remaining from device-side counters and
// updates the accumulator in place, so the kernel signature only carries
// device pointers + per-call constants and can be captured into a graph.
template <typename T> __global__ void compute_dt_4d_g(
    const T* alpha_max, T* dt_out, T* t_accum,
    T factorCFL, T tMax)
{
    T a = *alpha_max;
    T dt = (a > T(0)) ? (factorCFL / a) : T(1e10);
    T remaining = tMax - *t_accum;
    if (dt > remaining) dt = remaining;
    if (dt < T(0)) dt = T(0);
    *dt_out = dt;
    *t_accum += dt;
}
template <typename T> __global__ void apply_wall_mask_t(T* d, const T* init, const T* mask, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && mask[i]>T(0.5)) d[i]=init[i];
}
template <typename T> __global__ void clamp_values_t(T* d, T vmin, T vmax, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) d[i]=fmax(vmin, fmin(vmax, d[i]));
}
template <typename T> __global__ void restrict4d_t(T* d, const T* o, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=fmin(d[i],o[i]);
}
template <typename T> __global__ void ttr4d_t(double* t, const T* o, const T* c, double time, double dt, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && isinf(t[i]) && c[i]<=T(0))
        t[i]=(o[i]>T(0))?time+double(o[i])/double(o[i]-c[i])*dt:time;
}

template <typename T>
static SolveResult solve_pe4d_impl(
    const Grid& grid, const ArrayND& initial_data,
    const SolveOptions& opts, double v_ego_max, double v_obs,
    const std::vector<double>& wall_mask,
    const std::vector<double>& ego_speed_map)
{
    int N0=grid.N[0],N1=grid.N[1],N2=grid.N[2],N3=grid.N[3];
    int np=grid.total_points;
    T inv_dx[4] = {T(1.0/grid.dx[0]), T(1.0/grid.dx[1]),
                    T(1.0/grid.dx[2]), T(1.0/grid.dx[3])};

    const char* dtype = (sizeof(T) == 4) ? "fp32" : "fp64";
    if (opts.verbose) printf("  4D WENO5+RK3+LLF (fused-stage %s): %dx%dx%dx%d = %d pts\n", dtype, N0,N1,N2,N3,np);

    T *d_data,*d_y1,*d_y2;
    CK(cudaMalloc(&d_data,np*sizeof(T)));
    CK(cudaMalloc(&d_y1,  np*sizeof(T)));
    CK(cudaMalloc(&d_y2,  np*sizeof(T)));

    T *d_init=nullptr, *d_mask=nullptr, *d_ego_speed=nullptr;
    bool has_mask = !wall_mask.empty() && (int)wall_mask.size()==np;
    bool has_speed_map = !ego_speed_map.empty() && (int)ego_speed_map.size()==N0*N1;

    std::vector<T> host_buf(np);
    for (int i = 0; i < np; ++i) host_buf[i] = T(initial_data[i]);
    CK(cudaMemcpy(d_data, host_buf.data(), np*sizeof(T), cudaMemcpyHostToDevice));

    if(has_mask){
        CK(cudaMalloc(&d_init, np*sizeof(T)));
        CK(cudaMalloc(&d_mask, np*sizeof(T)));
        CK(cudaMemcpy(d_init, host_buf.data(), np*sizeof(T), cudaMemcpyHostToDevice));
        std::vector<T> mask_buf(np);
        for (int i = 0; i < np; ++i) mask_buf[i] = T(wall_mask[i]);
        CK(cudaMemcpy(d_mask, mask_buf.data(), np*sizeof(T), cudaMemcpyHostToDevice));
        if (opts.verbose) printf("  Wall mask: %d states frozen\n",(int)std::count(wall_mask.begin(),wall_mask.end(),1.0));
    }
    if(has_speed_map){
        CK(cudaMalloc(&d_ego_speed, N0*N1*sizeof(T)));
        std::vector<T> sp_buf(N0*N1);
        for (int i = 0; i < N0*N1; ++i) sp_buf[i] = T(ego_speed_map[i]);
        CK(cudaMemcpy(d_ego_speed, sp_buf.data(), N0*N1*sizeof(T), cudaMemcpyHostToDevice));
        double minspd=*std::min_element(ego_speed_map.begin(),ego_speed_map.end());
        double maxspd=*std::max_element(ego_speed_map.begin(),ego_speed_map.end());
        int nzero=0; for(auto s:ego_speed_map) if(s<0.01) nzero++;
        if (opts.verbose) printf("  Ego speed map: [%.2f, %.2f], %d zero-speed cells\n",minspd,maxspd,nzero);
    }

    double* d_ttr = nullptr;
    if (opts.computeTTR) {
        CK(cudaMalloc(&d_ttr, np*sizeof(double)));
        std::vector<double> ttr_init(np, INFINITY);
        for (int i=0; i<np; i++) if (initial_data[i] <= 0.0) ttr_init[i] = 0.0;
        CK(cudaMemcpy(d_ttr, ttr_init.data(), np*sizeof(double), cudaMemcpyHostToDevice));
    }

    int block=256;
    int grid_np = (np + block - 1) / block;

    // The CFL wavespeed is the constant
    //     alpha = v_ego_max*(1/dx0 + 1/dx1) + v_obs*(1/dx2 + 1/dx3)
    // so dt is computed once on the host and the step loop runs
    // asynchronously.
    T alpha_const = T(v_ego_max)*inv_dx[0] + T(v_ego_max)*inv_dx[1]
                  + T(v_obs)*inv_dx[2]     + T(v_obs)*inv_dx[3];
    double dt_static = (double(alpha_const) > 0.0)
                     ? opts.factorCFL / double(alpha_const) : 1e10;

    T vclamp_lo = T(opts.value_clip_lo), vclamp_hi = T(opts.value_clip_hi);
    int restrict_flag = opts.restrictUpdate ? 1 : 0;

    // One fused launch per RK3 stage (see pe4d_stage_kernel). Stage 3 writes
    // d_data in place, so no V -> V_old memcpy is needed either.
    auto launch_stage = [&](int stage, const T* src, const T* vold, T* out,
                            T dt, double t_now) {
        auto args = [&](auto stage_c) {
            constexpr int Sc = decltype(stage_c)::value;
            pe4d_stage_kernel<T, Sc><<<grid_np, block>>>(
                src, vold, out, d_ego_speed, d_init, d_mask,
                (Sc == 3 && opts.computeTTR) ? d_ttr : nullptr,
                N0, N1, N2, N3,
                inv_dx[0], inv_dx[1], inv_dx[2], inv_dx[3],
                T(v_ego_max), T(v_obs),
                dt, vclamp_lo, vclamp_hi,
                restrict_flag, t_now, np);
        };
        if (stage == 1)      args(std::integral_constant<int,1>{});
        else if (stage == 2) args(std::integral_constant<int,2>{});
        else                 args(std::integral_constant<int,3>{});
    };

    double t = 0; int steps = 0;
    while (t < opts.tMax - 1e-12 * opts.tMax) {
        double dt_d = std::min(dt_static, opts.tMax - t);
        T dt = T(dt_d);
        launch_stage(1, d_data, d_data, d_y1, dt, t);
        launch_stage(2, d_y1,   d_data, d_y2, dt, t);
        launch_stage(3, d_y2,   d_data, d_data, dt, t);
        t += dt_d; steps++;
        if (opts.verbose && steps % 50 == 0) printf("  step %d, t=%.3f/%.3f\n", steps, t, opts.tMax);
    }
    CK(cudaDeviceSynchronize());
    if (opts.verbose) printf("  Done (%s): %d steps\n", dtype, steps);

    SolveResult result;
    if (opts.computeTTR) {
        result.ttr.resize(np);
        CK(cudaMemcpy(result.ttr.data(), d_ttr, np*sizeof(double), cudaMemcpyDeviceToHost));
    }
    result.final_data.resize(np);
    CK(cudaMemcpy(host_buf.data(), d_data, np*sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < np; ++i) result.final_data[i] = double(host_buf[i]);
    result.final_time = t;

    cudaFree(d_data);cudaFree(d_y1);cudaFree(d_y2);
    if(d_init)cudaFree(d_init);
    if(d_mask)cudaFree(d_mask);
    if(d_ego_speed)cudaFree(d_ego_speed);
    if(d_ttr)cudaFree(d_ttr);
    return result;
}

SolveResult solveGPU_PursuitEvasion4D(
    const Grid& grid, const ArrayND& initial_data,
    const SolveOptions& opts, double v_ego_max, double v_obs,
    const std::vector<double>& wall_mask,
    const std::vector<double>& ego_speed_map)
{
    if (opts.precision == Precision::FP32) {
        return solve_pe4d_impl<float>(grid, initial_data, opts, v_ego_max, v_obs,
                                       wall_mask, ego_speed_map);
    }
    return solve_pe4d_impl<double>(grid, initial_data, opts, v_ego_max, v_obs,
                                    wall_mask, ego_speed_map);
}

} // namespace gpu
} // namespace hjb
