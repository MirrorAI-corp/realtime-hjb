#!/bin/bash -eu
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

# Builds the fuzz targets from the host-side sources (no CUDA needed).
cd "$SRC/realtime-hjb"
for target in fuzz/*_fuzzer.cpp; do
    name=$(basename "$target" .cpp)
    $CXX $CXXFLAGS -std=c++17 -I include \
        src/core/grid.cpp src/core/array_nd.cpp src/level_set/shapes.cpp src/io/npy_writer.cpp \
        "$target" $LIB_FUZZING_ENGINE -o "$OUT/$name"
done
