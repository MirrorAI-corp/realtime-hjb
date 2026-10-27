# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

cmake_minimum_required(VERSION 3.18)
project(RealtimeHJB VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(HJB_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(HJB_TOP_LEVEL ON)
endif()

option(HJB_BUILD_EXAMPLES "Build the example programs" ${HJB_TOP_LEVEL})
option(HJB_BUILD_TESTS "Build the regression tests (ctest)" ${HJB_TOP_LEVEL})
option(HJB_PE4D_GLOBAL_LF
       "4D pursuit-evasion: global instead of gradient-local Lax-Friedrichs coefficients" OFF)

# GPU architecture. Default sm_89 (RTX 40-series / RTX Ada). Override with e.g.
# -DCMAKE_CUDA_ARCHITECTURES=86-real (RTX 30), 90-real (H100), 110-real
# (Jetson AGX Thor), or native (CMake 3.24 or later). A `-real` target embeds
# machine code only, which avoids PTX JIT failures when the toolkit is newer
# than the driver.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 89-real)
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)

configure_file(cmake/version.hpp.in ${PROJECT_BINARY_DIR}/include/hjb/version.hpp @ONLY)

add_library(hjb STATIC
    src/core/grid.cpp
    src/core/array_nd.cpp
    src/io/npy_writer.cpp
    src/level_set/shapes.cpp
    src/solver_gpu.cu
    src/solver_gpu_3d.cu
    src/solver_gpu_4d.cu
    src/solver_gpu_drifter.cu
)
add_library(RealtimeHJB::hjb ALIAS hjb)
target_include_directories(hjb
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src)
# The solver headers and ArrayND's device members are compiled in only when
# HJB_USE_CUDA is defined, so it is part of the public interface.
target_compile_definitions(hjb PUBLIC HJB_USE_CUDA)
target_compile_options(hjb PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
if(HJB_PE4D_GLOBAL_LF)
    target_compile_definitions(hjb PRIVATE HJB_PE4D_GLOBAL_LF)
endif()
target_link_libraries(hjb PUBLIC CUDA::cudart)
set_target_properties(hjb PROPERTIES
    OUTPUT_NAME realtime_hjb
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF)

if(HJB_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(HJB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- install / package --------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS hjb EXPORT RealtimeHJBTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/hjb DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/RealtimeHJB)
install(FILES ${PROJECT_BINARY_DIR}/include/hjb/version.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hjb)
install(EXPORT RealtimeHJBTargets NAMESPACE RealtimeHJB::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RealtimeHJB)
configure_package_config_file(cmake/RealtimeHJBConfig.cmake.in
        ${PROJECT_BINARY_DIR}/RealtimeHJBConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RealtimeHJB)
write_basic_package_version_file(${PROJECT_BINARY_DIR}/RealtimeHJBConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)
install(FILES ${PROJECT_BINARY_DIR}/RealtimeHJBConfig.cmake
              ${PROJECT_BINARY_DIR}/RealtimeHJBConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RealtimeHJB)
