# Changelog

Versions follow [Semantic Versioning](https://semver.org/). Until 1.0 the API
can change between minor versions.

## [0.1.1] - 2026-10-27

- `examples/wildfire/plot_frame.py` draws `fleet_sim` frames; the frame layout
  is documented in `fleet_sim.cpp`.
- The container image's working directory is writable by any user, so the
  examples run with `docker run --user`.
- Docs: mount and `--user` for the container examples; the FetchContent
  example pins a release tag instead of `main`.

## [0.1.0] - 2026-10-27

First public release.

- GPU solvers for five problem classes: holonomic and double-integrator reach
  problems (FP64), the Air3D collision-avoidance game, a 4D pursuit-evasion
  game, and a 2D drifter in a wind field with agent labels for capture-basin
  partitioning (FP32 or FP64).
- Time-to-reach maps, reach-avoid terms (obstacle_sdf, target_sdf) and
  backward reachable tubes or sets.
- CMake package `RealtimeHJB::hjb` for `find_package`, `add_subdirectory` and
  `FetchContent`.
- Examples: Air3D tube, 2D time-to-reach, 4D pursuit-evasion, and a closed-loop
  wildfire-response simulation with six drones.
- Prebuilt static libraries for Linux x86_64 (CUDA 12 and 13) and aarch64
  (CUDA 13, including Jetson AGX Thor).
- Container image `ghcr.io/mirrorai-corp/realtime-hjb` with the examples and
  tests, for x86_64 and arm64.
- `Grid::create` and `writeNpy` reject invalid arguments with
  `std::invalid_argument`.
