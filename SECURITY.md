# Security policy

## Reporting a vulnerability

Report security problems privately through GitHub's vulnerability reporting:
<https://github.com/MirrorAI-corp/realtime-hjb/security/advisories/new>. Please
do not open a public issue for a security problem.

We acknowledge reports within 7 days and aim to publish a fix and a security
advisory within 90 days of the report. We agree the disclosure date with the
reporter and credit them in the advisory unless they ask us not to.

## Supported versions

Security fixes go into the latest release.

## Scope

RealtimeHJB is a numerical library. It reads no files and opens no network
connections; `writeNpy` writes the file the caller names, and the Air3D solver
reads one environment variable, `HJB_AIR3D_STEP_MODE`. Reports about memory
safety in the host code, the CUDA kernels or the input checks are in scope.
The value functions it computes are only as good as the model and grid you give
it: a reachable set that is wrong because the dynamics or the resolution do not
match the real system is not a security vulnerability.
