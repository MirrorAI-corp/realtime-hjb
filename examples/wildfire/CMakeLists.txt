# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

add_executable(wildfire_voronoi wildfire_voronoi.cpp)
target_link_libraries(wildfire_voronoi PRIVATE RealtimeHJB::hjb)

add_executable(fleet_sim fleet_sim.cpp)
target_link_libraries(fleet_sim PRIVATE RealtimeHJB::hjb)
