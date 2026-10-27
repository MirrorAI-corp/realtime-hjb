# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

foreach(ex air3d_brt time_to_reach_2d pursuit_evasion_4d)
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} PRIVATE RealtimeHJB::hjb)
endforeach()

# Larger example: closed-loop wildfire response with a fleet of drones
# (fleet_sim) and capture-basin Voronoi partitioning (wildfire_voronoi).
add_subdirectory(wildfire)
