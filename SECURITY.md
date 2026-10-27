# Security policy

## Reporting a vulnerability

Report security problems privately through GitHub: on the repository's
**Security** tab, choose **Report a vulnerability**. Please do not open a public
issue for a security problem.

We will confirm the report and agree a disclosure date with you once a fix is
ready.

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
