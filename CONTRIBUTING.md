# Contributing

RealtimeHJB is developed by Mirror AI, Corp. Changes to this repository are
made by its maintainers.

## Issues

Bug reports, questions and feature requests are welcome as
[issues](https://github.com/MirrorAI-corp/realtime-hjb/issues). For a bug,
include the RealtimeHJB version, the GPU, the CUDA version and the smallest
program that shows the problem.

## Pull requests

We do not accept unsolicited pull requests. If you want to change the code,
open an issue first and describe the change. If a maintainer asks you for a
pull request, it must:

- build with `cmake --build build` and pass `ctest --test-dir build` on a GPU;
- add a test for new behavior;
- carry the SPDX license header on new files;
- be signed off under the [Developer Certificate of Origin](https://developercertificate.org/)
  (`git commit -s`).

Contributions are licensed under the Apache License, Version 2.0.
