# Using RealtimeHJB in your project

Link against the CMake target `RealtimeHJB::hjb` and include `<hjb/hjb.hpp>`.
Your project does not need to enable CUDA as a language; the target brings the
include paths, the `HJB_USE_CUDA` definition and the CUDA runtime with it. You
do need the CUDA toolkit installed, because the library links against
`libcudart`.

## Installed package

Build and install once:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89-real   # your GPU
cmake --build build -j
cmake --install build --prefix /opt/realtime-hjb
```

Then, in your project:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_planner LANGUAGES CXX)

find_package(RealtimeHJB 0.1 REQUIRED)
add_executable(my_planner main.cpp)
target_link_libraries(my_planner PRIVATE RealtimeHJB::hjb)
```

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/realtime-hjb
```

## Source in your tree

Add the repository as a subdirectory (for example a git submodule). Examples,
tests and tools are off by default in this mode.

```cmake
add_subdirectory(third_party/realtime-hjb)
target_link_libraries(my_planner PRIVATE RealtimeHJB::hjb)
```

## FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(realtime_hjb
    GIT_REPOSITORY https://github.com/MirrorAI-corp/realtime-hjb.git
    GIT_TAG main)
FetchContent_MakeAvailable(realtime_hjb)
target_link_libraries(my_planner PRIVATE RealtimeHJB::hjb)
```

## GPU architecture

The library is compiled for one GPU architecture, `89-real` unless you set
`CMAKE_CUDA_ARCHITECTURES` when configuring RealtimeHJB (for example
`86-real`, a list such as `"86-real;89-real"`, or `native` with CMake 3.24 or
later). A binary built for
the wrong architecture fails at the first kernel launch with
"no kernel image is available for execution on the device".

## Memory and concurrency

Each solve allocates its device buffers, runs the horizon and frees them before
it returns, so no GPU state survives between calls. Concurrent solves from
several host threads have not been tested.
