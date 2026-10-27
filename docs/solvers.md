# Solvers

All solvers are in namespace `hjb::gpu`, take `(grid, initial_data, opts, ...)`
and return a `SolveResult`. The Hamiltonians below use the sign convention of
[Concepts](concepts.md): V evolves as dV/dt = H(x, grad V) in time-to-go, with
H = min or max over the inputs of grad V . f.

Options every solver reads: `tMax`, `factorCFL`, `computeTTR`,
`restrictUpdate`. The table lists the others.

| Solver | Scheme | Precision | Other options |
|---|---|---|---|
| `solveGPU_HolonomicL2` | WENO5 + RK3, ENO2 or first order (`accuracy`) | FP64 | `obstacle_sdf`, `target_sdf` |
| `solveGPU_DoubleIntegrator2D` | WENO5 + RK3 | FP64 | `obstacle_sdf`, `target_sdf` |
| `solveGPU_Drifter` | WENO5 + RK3 or first order (`accuracy`) | FP32, FP64 | `obstacle_sdf`, `target_sdf`, `initial_labels` |
| `solveGPU_Air3D` | WENO5 + RK3 | FP32, FP64 | `obstacle_sdf`, `target_sdf`, `verbose` |
| `solveGPU_PursuitEvasion4D` | WENO5 + RK3 | FP32, FP64 | `value_clip_lo`, `value_clip_hi`, `verbose` |

## solveGPU_HolonomicL2

```cpp
SolveResult solveGPU_HolonomicL2(const Grid& grid, const ArrayND& initial_data,
                                 const SolveOptions& opts, double inputBound);
```

A point that moves in any direction at speed up to `inputBound` (2D).

- Dynamics: x' = u, |u| <= inputBound.
- Hamiltonian: H = -inputBound |p|. The target grows at speed inputBound.
- Result: `ttr` is distance to the target divided by inputBound.
- Dissipation: global Lax-Friedrichs.

## solveGPU_DoubleIntegrator2D

```cpp
SolveResult solveGPU_DoubleIntegrator2D(const Grid& grid, const ArrayND& initial_data,
                                        const SolveOptions& opts, double inputBound);
```

A mass on a line with bounded acceleration. Dimension 0 is position, 1 is
velocity.

- Dynamics: x1' = x2, x2' = u, |u| <= inputBound.
- Hamiltonian: H = x2 p1 - inputBound |p2| (the control steers into the target).
- Dissipation: Lax-Friedrichs with per-cell bounds |x2| and inputBound.

## solveGPU_Drifter

```cpp
SolveResult solveGPU_Drifter(const Grid& grid, const ArrayND& initial_data,
                             const SolveOptions& opts,
                             const ArrayND& flow_x, const ArrayND& flow_y,
                             double a_max, double b_max);
```

A vehicle in a spatially varying flow field (wind, current), with a bounded
disturbance (2D). `flow_x` and `flow_y` give the flow at every cell.

- Dynamics: x' = w(x) + a_max u + b_max d, |u| <= 1, |d| <= 1.
- Hamiltonian: H = w(x) . p - (a_max - b_max) |p|.
- Result: `ttr` is the guaranteed time for a vehicle starting at each cell to
  reach the target, with the flow and against the worst disturbance.
- Forward reachability: to get the cells a vehicle starting in the target can
  reach, and how soon, pass the negated flow (-w).
- Labels: set `initial_labels` to partition the domain between several agents
  in one solve (see [Concepts](concepts.md)).
- Dissipation: Lax-Friedrichs with bounds max|w_x| + a_max + b_max and
  max|w_y| + a_max + b_max.
- `solveGPU_DrifterLow` is the same solver with `accuracy` forced to `Low`.

## solveGPU_Air3D

```cpp
SolveResult solveGPU_Air3D(const Grid& grid, const ArrayND& initial_data,
                           const SolveOptions& opts,
                           double v_a, double v_b, double omega_max);
```

The two-aircraft collision-avoidance game of Mitchell, Bayen and Tomlin
(IEEE TAC 2005). The state (x, y, psi) is the pursuer's position and heading
relative to the evader; dimension 2 must be periodic on [0, 2 pi).

- Inputs: the evader (control) turns at |omega_e| <= omega_max to avoid; the
  pursuer (disturbance) turns at |omega_p| <= omega_max to collide. Speeds are
  v_a (evader) and v_b (pursuer).
- Dynamics: x' = -v_a + v_b cos psi + omega_e y, y' = v_b sin psi - omega_e x,
  psi' = omega_p - omega_e.
- Hamiltonian: H = (-v_a + v_b cos psi) p0 + v_b sin(psi) p1
  + omega_max |y p0 - x p1 - p2| - omega_max |p2|.
- Result: with the collision set as target, {V <= 0} is the set of relative
  states from which the pursuer can force a collision within tMax.
- Dissipation: Lax-Friedrichs with per-cell bounds.
- Implementation: one fused kernel per Runge-Kutta stage, replayed as a CUDA
  graph, with no host synchronization inside the horizon.

## solveGPU_PursuitEvasion4D

```cpp
SolveResult solveGPU_PursuitEvasion4D(const Grid& grid, const ArrayND& initial_data,
                                      const SolveOptions& opts,
                                      double v_ego, double v_obs,
                                      const std::vector<double>& wall_mask = {},
                                      const std::vector<double>& ego_speed_map = {});
```

Two agents in the plane. The state is (x_ego, y_ego, x_obs, y_obs). The ego
agent is the pursuer (control, speed up to v_ego); the other agent evades
(disturbance, speed up to v_obs).

- Hamiltonian: H = -v_ego |(p0, p1)| + v_obs |(p2, p3)|.
- Result: with the capture set (for example, the agents closer than r) as
  target, {V <= 0} is the set of joint states from which the ego agent can
  capture the other within tMax.
- `wall_mask`: one value per grid cell; 1.0 freezes the cell at its initial
  value (use it for states where either agent is inside a wall).
- `ego_speed_map`: the ego agent's speed limit as a function of its own
  position, N0 x N1 values; it must not exceed v_ego.
- `value_clip_lo`, `value_clip_hi`: V is clamped to this range after every
  stage (default -50 to 50). Widen it for large domains.
- Dissipation: Lax-Friedrichs with coefficients from the local gradient
  direction, v |p_i| / |p_pair|. hj_reachability uses global coefficients
  instead; build with `-DHJB_PE4D_GLOBAL_LF=ON` to match it.
- Memory and time: a 51^4 grid (6.8 million cells) uses about 330 MB of GPU
  memory, including the CUDA context, and takes 0.23 s in FP32 or 6.3 s in FP64
  on an RTX 4000 Ada.

## Writing a new solver

The solvers share a structure that a new system can follow: a CUDA kernel per
Runge-Kutta stage that reconstructs the gradient with WENO5
(`src/detail/weno5.cuh`), evaluates the Hamiltonian and the Lax-Friedrichs
term, and applies the stage update, the tube restriction and the
post-processors (`src/detail/value_postproc.cuh`). `src/solver_gpu_3d.cu` is
the most complete example. A new system needs its Hamiltonian and a bound on
|dH/dp_i| in each dimension for the time step and the dissipation.
