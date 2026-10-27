# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

FROM gcr.io/oss-fuzz-base/base-builder@sha256:1f9cf74bc3d7029b528494542f9d9851b256dceed38d3f420989434dc5ae0760
COPY . $SRC/realtime-hjb
COPY .clusterfuzzlite/build.sh $SRC/
WORKDIR $SRC/realtime-hjb
