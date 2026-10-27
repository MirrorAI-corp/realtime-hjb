# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

# Builds a program against an installed RealtimeHJB (used by CI).
cmake_minimum_required(VERSION 3.18)
project(RealtimeHJBConsumer LANGUAGES CXX)
find_package(RealtimeHJB 0.1 REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE RealtimeHJB::hjb)
