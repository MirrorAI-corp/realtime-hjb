// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

/// Shared templated WENO5 reconstruction used by all GPU solvers.
///
/// One copy lives here instead of being duplicated in solver_gpu*.cu so that
/// any future tuning to the reconstruction (smoothness indicator weights,
/// epsilon, etc.) only needs to happen once.

#include <cuda_runtime.h>

namespace hjb {
namespace gpu {

/// Fifth-order WENO reconstruction. Used by every GPU stencil kernel.
/// `v1..v5` are five consecutive divided differences (in the upwind direction)
/// along one spatial axis. Returns the WENO-weighted point-value estimate of
/// the derivative at the cell interface.
///
/// Division-lean formulation. The textbook form costs 13 divisions per call
/// (9 divides-by-constant in p0..p2 plus 0.1/d0^2, 0.6/d1^2, 0.3/d2^2 and the
/// final weight normalization). FP64 division on consumer GPUs (sm_89: 1:64
/// FP64 ALU, software div ~36 instr) made that THE bottleneck of every FP64
/// stencil kernel. Equivalent algebra with one division:
///   - divides-by-constant become multiplies by compile-time reciprocals
///     (1/3, 7/6, ... are rounded once at compile time: <=1 ulp difference);
///   - unnormalized weights are cross-multiplied by the other two smoothness
///     denominators, so only the final normalization divides:
///       a_k = c_k / d_k  ==>  w_k ∝ c_k * prod_{j!=k} d_j
/// Identical to the textbook form up to round-off.
template <typename T>
__device__ inline T weno5_recon(T v1, T v2, T v3, T v4, T v5) {
    const T eps = T(1e-6);
    T p0 = T(1.0/3.0)*v1 - T(7.0/6.0)*v2 + T(11.0/6.0)*v3;
    T p1 = T(-1.0/6.0)*v2 + T(5.0/6.0)*v3 + T(1.0/3.0)*v4;
    T p2 = T(1.0/3.0)*v3 + T(5.0/6.0)*v4 - T(1.0/6.0)*v5;
    T b0 = T(13.0/12.0)*(v1-T(2)*v2+v3)*(v1-T(2)*v2+v3)
         + T(0.25)*(v1-T(4)*v2+T(3)*v3)*(v1-T(4)*v2+T(3)*v3);
    T b1 = T(13.0/12.0)*(v2-T(2)*v3+v4)*(v2-T(2)*v3+v4)
         + T(0.25)*(v2-v4)*(v2-v4);
    T b2 = T(13.0/12.0)*(v3-T(2)*v4+v5)*(v3-T(2)*v4+v5)
         + T(0.25)*(T(3)*v3-T(4)*v4+v5)*(T(3)*v3-T(4)*v4+v5);
    T d0 = eps + b0, d1 = eps + b1, d2 = eps + b2;
    d0 *= d0; d1 *= d1; d2 *= d2;
    // Unnormalized weights scaled by d0*d1*d2 (positive, cancels in the ratio).
    T w0 = T(0.1) * d1 * d2;
    T w1 = T(0.6) * d0 * d2;
    T w2 = T(0.3) * d0 * d1;
    return (w0*p0 + w1*p1 + w2*p2) / (w0 + w1 + w2);
}

/// Boundary-aware 1D fetch along one axis of a row-major 2D field.
/// Returns V at (possibly out-of-bounds) position `p`, applying the same
/// linear-extrapolation / periodic ghost rule as the padDim* kernels.
/// Index into V = p*stride + base.
template <typename T>
__device__ __forceinline__ T fetchB(const T* __restrict__ V, int p, int N,
                                     int base, int stride, int bdry) {
    if (p >= 0 && p < N) return V[p * stride + base];
    if (bdry == 1) { int s = ((p % N) + N) % N; return V[s * stride + base]; }
    if (p < 0) {
        T v0 = V[base], v1 = V[stride + base];
        return v0 + (T)p * (v1 - v0);                      // p<0
    }
    T vn1 = V[(N - 1) * stride + base], vn2 = V[(N - 2) * stride + base];
    return vn1 + (T)(p - (N - 1)) * (vn1 - vn2);           // p>=N
}

/// Per-axis WENO5 left/right derivatives from 7 directly-fetched values.
template <typename T>
__device__ __forceinline__ void weno5_axis(const T* __restrict__ V, int idxc,
                                            int N, int base, int stride, int bdry,
                                            T inv_dx, T& dL, T& dR) {
    T f[7];
#pragma unroll
    for (int k = -3; k <= 3; ++k) f[k + 3] = fetchB<T>(V, idxc + k, N, base, stride, bdry);
    T D0 = (f[1] - f[0]) * inv_dx, D1 = (f[2] - f[1]) * inv_dx, D2 = (f[3] - f[2]) * inv_dx;
    T D3 = (f[4] - f[3]) * inv_dx, D4 = (f[5] - f[4]) * inv_dx, D5 = (f[6] - f[5]) * inv_dx;
    dL = weno5_recon(D0, D1, D2, D3, D4);
    dR = weno5_recon(D5, D4, D3, D2, D1);
}

/// Generic SSP-RK3 stage epilogue shared by the fused per-stage kernels.
/// `r` is the stage RHS at this cell, `center == src[idx]`. Stage 3 also
/// applies restrict / obstacle-target postproc / TTR (all fused), and may
/// write `out == vold` in place: vold is only read at the thread's own cell.
///   STAGE 1:  out = center + dt*r                       (src == vold == V)
///   STAGE 2:  out = 3/4 vold + 1/4 (center + dt*r)      (src == y1)
///   STAGE 3:  out = 1/3 vold + 2/3 (center + dt*r)      (src == y2)
template <typename T, int STAGE>
__device__ __forceinline__ void rk3_stage_write(
    int idx, T center, T r, T dt, const T* vold, T* out,
    const T* obstacle, const T* target, double* ttr,
    int restrict_update, double t_now)
{
    T v;
    if (STAGE == 1) {
        v = center + dt * r;
    } else if (STAGE == 2) {
        v = T(0.75) * vold[idx] + T(0.25) * (center + dt * r);
    } else {
        T vo = vold[idx];
        v = T(1.0/3.0) * vo + T(2.0/3.0) * (center + dt * r);
        if (restrict_update) v = fmin(v, vo);
        if (obstacle) v = fmax(v, obstacle[idx]);
        if (target)   v = fmin(v, target[idx]);
        if (ttr && isinf(ttr[idx]) && v <= T(0)) {
            ttr[idx] = (vo > T(0))
                ? t_now + double(vo) / double(vo - v) * double(dt)
                : t_now;
        }
    }
    out[idx] = v;
}

} // namespace gpu
} // namespace hjb
