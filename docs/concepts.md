# Concepts

## Value function and target

A solve starts from a function V0 on the grid that is negative exactly on the
target set and positive outside it. The signed distance to the target is the
usual choice, because the scheme is most accurate when |grad V| is close to 1.

The solver evolves V under the Hamilton-Jacobi-Isaacs equation of the chosen
system. After a horizon T, the set {V <= 0} has grown (or, in an avoidance
game, the unsafe set has grown) to contain every state from which the target
can be reached within T. Only the sign of V and its zero level set have a
direct meaning; the magnitude is a distance-like quantity that the scheme
needs.

## Time convention

The solvers integrate in time-to-go: they start at 0 and stop at
`SolveOptions::tMax`, and V at the end describes what can happen within tMax
seconds from now. This is the same as integrating the backward HJI equation
from 0 to -tMax in the convention of Mitchell's ToolboxLS and hj_reachability;
the sign of time is the only difference.

## Tube or set: restrictUpdate

With `restrictUpdate = true` (the default) the solver applies
V <- min(V_new, V_old) after every time step, so V never increases. The result
is the backward reachable tube: states that reach the target at any time up to
tMax. With `restrictUpdate = false` the result is the backward reachable set:
states that are in the target at exactly tMax.

For safety, use the tube. A state outside the tube of the unsafe set is safe
for the whole horizon.

## Control and disturbance

Each solver's Hamiltonian fixes which input is the control and which is the
disturbance, and what each one wants:

- **Reach problems** (holonomic, double integrator, drifter, 4D pursuer): the
  control drives the state into the target and the disturbance resists. The
  tube is where the control can guarantee arrival.
- **Avoid problems** (Air3D): the target is the unsafe set, the control tries to
  stay out and the disturbance tries to push in. The tube is where the
  disturbance can force entry, so the complement of the tube is the safe set.

[Solvers](solvers.md) lists the roles for every solver.

## Time-to-reach

With `computeTTR = true`, `SolveResult::ttr` holds, for every cell, the first
time V becomes non-positive there: the minimum time to reach the target from
that state under the worst-case disturbance. Cells that are not reached within
tMax hold +infinity (`hjb::kInfinity`). Cells inside the target hold 0.

The TTR map answers "how long until..." questions directly. For example,
`examples/time_to_reach_2d.cpp` gives 0.43 time units to reach a target 0.5
units away with the wind behind the vehicle, and 1.42 against it.

## Several agents: capture-basin partition

The drifter solver can carry agent labels with the front. Start from the union
of the agents' initial sets, give each cell of agent k's set label k, and set
`SolveOptions::initial_labels`. One solve then returns, in
`SolveResult::labels`, which agent reaches each cell first. This is the
Voronoi partition of the agents under the system's dynamics (for a uniform
flow, a set of Apollonius-like curves; for a varying flow, curved boundaries
that no closed form gives). The cost is one solve, whatever the number of
agents. `examples/wildfire/` uses it to dispatch a drone fleet.

## Obstacles and reach-avoid

`obstacle_sdf` and `target_sdf` apply per-cell limits after every step:
V <- max(V, obstacle_sdf) and V <- min(V, target_sdf). Passing the negated
signed distance of an obstacle as `obstacle_sdf` keeps states inside the
obstacle out of the tube. Together they give the reach-avoid value function
min(target_sdf, max(V, obstacle_sdf)). The 4D solver uses `wall_mask` instead.

## Grid and boundaries

`Grid::create` places N points on [min, max] in each dimension. For a periodic
dimension the spacing is (max - min) / N and the point at max is omitted, so a
heading on [0, 2 pi) with N = 51 has spacing 2 pi / 51. For the other dimensions
the spacing is (max - min) / (N - 1). Non-periodic boundaries extrapolate
linearly.

Make the domain large enough that the tube does not touch a non-periodic
boundary within the horizon, or check that it does not matter for your
question: values near the boundary depend on the extrapolation.

## Accuracy and precision

`AccuracyLevel::VeryHigh` selects fifth-order WENO with third-order TVD
Runge-Kutta, the scheme used for all verification results. `Low` selects
first-order upwind with forward Euler: one stage per time step instead of
three, and a smaller stencil, at the cost of more smearing of the front. Air3D and the 4D solver always use the high-order
scheme.

`Precision::FP32` runs the whole solve in single precision. On consumer and
workstation GPUs, where FP64 throughput is 1/64 of FP32, it is 20 to 30 times
faster. On the benchmark problems the FP32 tube matches the FP64 tube exactly
in 2D and 3D, and its cell count is within 0.1 % of the FP64 count in 4D. Use FP64 when you compare
against another library or study convergence.

## Time step

The solvers choose dt from the CFL condition with
`SolveOptions::factorCFL` (default 0.75), using a bound on |dH/dp| for each
dimension. The bound depends only on the system parameters and the grid, so dt
is computed once per solve, and the time loop needs no per-step reduction or
host synchronization.
