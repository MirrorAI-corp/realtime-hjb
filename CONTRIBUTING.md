# Contributing

RealtimeHJB is developed by Mirror AI, Corp. Changes to this repository are
made by its maintainers.

## Issues

Bug reports, questions and feature requests are welcome as
[issues](https://github.com/MirrorAI-corp/realtime-hjb/issues). For a bug,
include the RealtimeHJB version, the GPU, the CUDA version and the smallest
program that shows the problem.

## Pull requests

Only collaborators can open pull requests. If you want to change the code,
open an issue and describe the change; if it is accepted, a maintainer makes
it or adds you as a collaborator for it. A pull request must:

- build with `cmake --build build` and pass `ctest --test-dir build` on a GPU;
- add a test for new behavior;
- carry the SPDX license header on new files;
- be signed off under the [Developer Certificate of Origin](https://developercertificate.org/)
  (`git commit -s`).

Contributions are licensed under the Apache License, Version 2.0.
