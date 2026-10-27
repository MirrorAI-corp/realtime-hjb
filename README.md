# RealtimeHJB

RealtimeHJB is a CUDA library for Hamilton-Jacobi reachability on a grid. It
computes backward reachable tubes and minimum time-to-reach maps for 2D, 3D
and 4D systems in milliseconds on one GPU, fast enough to recompute them inside
a robot's control loop.

It uses the standard level-set scheme of Mitchell's ToolboxLS and the JAX
library [hj_reachability](https://github.com/StanfordASL/hj_reachability):
fifth-order WENO in space, third-order TVD Runge-Kutta in time and
Lax-Friedrichs dissipation. Results agree with hj_reachability to within the
discretization error (see [docs/performance.md](docs/performance.md)).

| Problem | Grid | Time on an RTX 4000 Ada (FP32) |
|---|---|---|
| Air3D collision avoidance (3D) | 51 x 51 x 51 | 8.9 ms |
| Air3D collision avoidance (3D) | 101 x 101 x 101 | 102 ms |
| Vehicle in a wind field, time-to-reach (2D) | 200 x 200 | 6.8 to 13.6 ms |
| Pursuit-evasion (4D) | 31^4 | 16.4 ms |

These are 6 to 18 times faster than hj_reachability on the same GPU.

## Solvers

| Function | System | State |
|---|---|---|
| `solveGPU_HolonomicL2` | point robot, speed bound | 2D |
| `solveGPU_DoubleIntegrator2D` | position and velocity, acceleration bound | 2D |
| `solveGPU_Drifter` | vehicle in a spatially varying flow field, with a bounded disturbance | 2D |
| `solveGPU_Air3D` | two-aircraft collision avoidance game | 3D |
| `solveGPU_PursuitEvasion4D` | two agents in the plane, optional walls and speed maps | 4D |

Each takes a grid, an initial value function whose zero sublevel set is the
target, and options (horizon, scheme order, FP32 or FP64). It returns the value
function at the horizon and, optionally, the time-to-reach of every cell. The
drifter solver can also split the reachable region between several agents (a
capture-basin Voronoi partition) in one solve.

## Quick start

Requirements: Linux, an NVIDIA GPU, CUDA 12 or 13, CMake 3.18 or later and a
C++17 compiler.

```bash
git clone https://github.com/MirrorAI-corp/realtime-hjb.git
cd realtime-hjb
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89-real   # your GPU, see below
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/examples/air3d_brt
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability without the dot
(`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` prints it): `89-real`
for RTX 40 series and RTX Ada, `86-real` for RTX 30, `90-real` for H100,
`110-real` for Jetson AGX Thor. With CMake 3.24 or later, `native` also works.

```cpp
#include <hjb/hjb.hpp>

int main() {
    using hjb::BoundaryType;
    hjb::Grid grid = hjb::Grid::create(
        3, {-20, -20, 0}, {20, 20, 2 * M_PI}, {51, 51, 51},
        {BoundaryType::Extrapolate, BoundaryType::Extrapolate, BoundaryType::Periodic});

    // Collision set: within 5 m in (x, y), for every relative heading.
    hjb::ArrayND v0 = hjb::shapeCylinder(grid, {false, false, true}, {0, 0, 0}, 5.0);

    hjb::SolveOptions opts;
    opts.accuracy = hjb::AccuracyLevel::VeryHigh;  // WENO5 + TVD-RK3
    opts.precision = hjb::Precision::FP32;
    opts.tMax = 2.0;                                // horizon, seconds

    hjb::SolveResult r = hjb::gpu::solveGPU_Air3D(grid, v0, opts, 5.0, 5.0, 1.0);
    // r.final_data <= 0 marks every state from which a collision can be forced
    // within 2 s.
}
```

## Documentation

- [Getting started](docs/getting-started.md): build, run the examples, write your first solve.
- [Concepts](docs/concepts.md): value functions, reachable tubes, time-to-reach and sign conventions.
- [Solvers](docs/solvers.md): dynamics, Hamiltonian, parameters and options for each solver.
- [Using RealtimeHJB in your project](docs/using-in-your-project.md): CMake `find_package`, `add_subdirectory` and `FetchContent`.
- [Performance](docs/performance.md): timings, FP32 versus FP64, comparison with hj_reachability.

## Repository layout

| Path | Contents |
|---|---|
| `include/hjb/`, `src/` | the library |
| `examples/` | small programs that each solve one problem, plus a closed-loop wildfire-fleet simulation |
| `tests/` | regression and analytic tests (`ctest`) |

## Citing

If you use RealtimeHJB in your research, please cite:

K. Weekly and A. M. Bayen, "Online Hamilton-Jacobi Reachability Onboard the
Robot Control Loop: A Real-Time GPU Solver for Safety-Critical Autonomy,"
2026.

## License

Copyright 2026 Mirror AI, Corp. Licensed under the Apache License, Version 2.0;
see [LICENSE](LICENSE) and [NOTICE](NOTICE).
