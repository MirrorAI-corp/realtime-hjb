# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mirror AI, Corp.

add_executable(hjb_tests test_main.cpp)
target_link_libraries(hjb_tests PRIVATE RealtimeHJB::hjb)
add_test(NAME hjb_tests COMMAND hjb_tests)
