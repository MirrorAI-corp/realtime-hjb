# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

# RealtimeHJB with its examples and tests, on the CUDA 13.2 runtime.
#   docker run --rm --gpus all ghcr.io/mirrorai-corp/realtime-hjb air3d_brt
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04@sha256:f9492f2eea77fbc3d0c14fa8738f35946b42da72917bf5959d284ca39b4f209a AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
ARG CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
          -DHJB_BUILD_EXAMPLES=ON -DHJB_BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=/opt/realtime-hjb \
 && cmake --build build -j "$(nproc)" \
 && cmake --install build \
 && install -D -t /opt/realtime-hjb/bin \
      build/examples/air3d_brt build/examples/time_to_reach_2d build/examples/pursuit_evasion_4d \
      build/examples/wildfire/wildfire_voronoi build/examples/wildfire/fleet_sim build/tests/hjb_tests

FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04@sha256:7fada70c5ed1b85cb3c15b53f51300fdacff1c31466a20e4f2b29ad952fd63a2
COPY --from=build /opt/realtime-hjb /opt/realtime-hjb
ENV PATH=/opt/realtime-hjb/bin:$PATH
# Writable by any user, so the examples also run with --user and no mount.
RUN mkdir -p /work && chmod 1777 /work
LABEL org.opencontainers.image.source="https://github.com/MirrorAI-corp/realtime-hjb" \
      org.opencontainers.image.description="RealtimeHJB: GPU Hamilton-Jacobi reachability, with examples and tests" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /work
CMD ["hjb_tests"]
