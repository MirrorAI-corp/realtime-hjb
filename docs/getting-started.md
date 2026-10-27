# Getting started

## Requirements

- Linux (tested on Ubuntu 22.04)
- An NVIDIA GPU and a driver that supports your CUDA toolkit
- CUDA toolkit 12 or 13 (tested with 13.2)
- CMake 3.18 or later and a C++17 compiler (tested with GCC 11)

The library has no other dependencies.

## Build

```bash
git clone https://github.com/MirrorAI-corp/realtime-hjb.git
cd realtime-hjb
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89-real
cmake --build build -j
```

`CMAKE_CUDA_ARCHITECTURES` selects the GPU the code is compiled for. Use your
GPU's compute capability without the dot, which
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` prints: `89-real` for
RTX 40 series and RTX Ada (the default), `86-real` for RTX 30, `90-real` for
H100, `110-real` for Jetson AGX Thor. With CMake 3.24 or later, `native` picks
the GPU in your machine.

Build options:

| Option | Default | Effect |
|---|---|---|
| `HJB_BUILD_EXAMPLES` | ON | build `examples/` |
| `HJB_BUILD_TESTS` | ON | build `tests/` |
| `HJB_PE4D_GLOBAL_LF` | OFF | 4D solver: global instead of gradient-local Lax-Friedrichs coefficients |

The first two default to OFF when the library is added to another project.

## Check the build

```bash
ctest --test-dir build --output-on-failure
```

The tests take about a second. They compare time-to-reach maps with closed-form
answers and check the Air3D tube against the value verified against
hj_reachability.

## Run the examples

```bash
cd build/examples
./air3d_brt                 # Air3D tube on a 51^3 grid, FP32
./air3d_brt 101 fp64        # finer grid, double precision
./time_to_reach_2d          # time-to-reach for a vehicle in wind
./pursuit_evasion_4d        # 4D pursuit-evasion on a 31^4 grid
```

`air3d_brt` prints the number of cells in the tube and the solve time, then
writes `air3d_V.npy`. `time_to_reach_2d` writes `ttr.npy`. To plot them (needs
numpy and matplotlib):

```bash
python ../../examples/plot_slice.py air3d_V.npy
python ../../examples/plot_slice.py ttr.npy
```

The examples are short and commented; they are the best starting point for your
own code.

## Write your own solve

Every solve follows the same four steps.

**1. Make a grid.** Give the number of dimensions, the lower and upper corners,
the number of points per dimension and the boundary condition of each
dimension:

```cpp
#include <hjb/hjb.hpp>
using hjb::BoundaryType;

hjb::Grid grid = hjb::Grid::create(
    2, {-1.0, -1.0}, {1.0, 1.0}, {200, 200},
    {BoundaryType::Extrapolate, BoundaryType::Extrapolate});
```

Use `Periodic` for angles. Grid values are stored row-major (the last dimension
varies fastest); `grid.flatIndex2D(i, j)` and `grid.unflatIndex(flat, idx)`
convert between flat and multi-indices.

**2. Describe the target.** The initial value function must be negative exactly
on the target set. A signed distance works best. Use the helpers in
`shapes.hpp`, or fill an `ArrayND` yourself:

```cpp
hjb::ArrayND target = hjb::shapeSphere(grid, {0.0, 0.0}, 0.1);   // disk of radius 0.1

hjb::ArrayND custom(grid);
int idx[hjb::kMaxDim];
for (int i = 0; i < grid.total_points; ++i) {
    grid.unflatIndex(i, idx);
    double x = grid.coord(0, idx[0]), y = grid.coord(1, idx[1]);
    custom[i] = std::max(std::fabs(x), std::fabs(y)) - 0.1;  // square of half-width 0.1
}
```

**3. Set the options.**

```cpp
hjb::SolveOptions opts;
opts.accuracy = hjb::AccuracyLevel::VeryHigh;  // WENO5 + TVD-RK3 (Low: first order, faster)
opts.precision = hjb::Precision::FP32;         // FP64 for verification-grade results
opts.tMax = 1.5;                               // horizon
opts.computeTTR = true;                        // also return time-to-reach
```

**4. Solve and read the result.**

```cpp
hjb::ArrayND wind_x(grid), wind_y(grid);
wind_x.fill(0.4);
wind_y.fill(0.0);
hjb::SolveResult r = hjb::gpu::solveGPU_Drifter(grid, target, opts, wind_x, wind_y,
                                               /*a_max=*/1.0, /*b_max=*/0.25);

// r.final_data[i] <= 0: cell i can reach the target within tMax.
// r.ttr[i]: the guaranteed time to reach it (+inf if not within tMax).
hjb::writeNpy("ttr.npy", r.ttr.data(), {200, 200});
```

The solvers copy the inputs to the GPU, run the whole horizon there and copy
the result back; there is no device state to manage between calls. The first
call in a process also creates the CUDA context, which takes longer than the
solve itself, so time the second call when you benchmark.

## Next steps

- [Concepts](concepts.md) explains what the value function, the tube and the
  time-to-reach mean, and the sign conventions.
- [Solvers](solvers.md) gives the dynamics and options of each solver.
- [Using RealtimeHJB in your project](using-in-your-project.md) shows how to
  link the library from another CMake project.
