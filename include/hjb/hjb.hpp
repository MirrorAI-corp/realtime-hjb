// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mirror AI, Corp.

#pragma once

/// Single include for the library: grid and array types, solve options,
/// initial-value shapes, .npy output, and the GPU solvers.

#include "core/types.hpp"
#include "core/grid.hpp"
#include "core/array_nd.hpp"
#include "options.hpp"
#include "level_set/shapes.hpp"
#include "io/npy_writer.hpp"
#include "solver_gpu.hpp"
#include "solver_gpu_3d.hpp"
#include "solver_gpu_4d.hpp"
#include "solver_gpu_drifter.hpp"
#include <hjb/version.hpp>
