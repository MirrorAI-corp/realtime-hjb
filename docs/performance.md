# Performance

All timings are for one solve of the full horizon, measured on the host
around the library call (including the copies to and from the GPU), after one
untimed warm-up solve in the same process. Each number is the median of
repeated runs.

Desktop: NVIDIA RTX 4000 Ada Generation (20 GB), CUDA 13.2.
Embedded: NVIDIA Jetson AGX Thor developer kit, 120 W mode.

## Single precision (real-time mode)

| Problem | Grid | RealtimeHJB | Rate | hj_reachability (JAX) | torch.compile |
|---|---|---|---|---|---|
| Air3D | 51^3 | 8.91 ms | 112 Hz | 52.3 ms | 50.0 ms |
| Air3D | 101^3 | 102 ms | 9.8 Hz | 804 ms | 178 ms |
| Drifter, constant wind | 200^2 | 6.82 ms | 147 Hz | 86.4 ms | 107 ms |
| Drifter, spatially varying wind | 200^2 | 13.6 ms | 74 Hz | 159 ms | 223 ms |
| Pursuit-evasion 4D | 31^4 | 16.4 ms | 61 Hz | 150 ms | 27.6 ms |
| Pursuit-evasion 4D | 51^4 | 222 ms | 4.5 Hz | 3885 ms | 573 ms |

hj_reachability 0.7.0 ran on the same GPU. The torch.compile column is a
PyTorch implementation of the same scheme.

## Double precision

| Problem | Grid | RealtimeHJB | hj_reachability | torch.compile |
|---|---|---|---|---|
| Holonomic | 200^2 | 67 ms | 280 ms | 96 ms |
| Double integrator | 200^2 | 134 ms | 271 ms | 185 ms |
| Drifter, constant wind | 200^2 | 147 ms | 561 ms | 206 ms |
| Drifter, spatially varying wind | 200^2 | 284 ms | 1128 ms | 400 ms |
| Air3D | 101^3 | 3.09 s | 7.04 s | 4.37 s |
| Pursuit-evasion 4D | 51^4 | 6.25 s | 28.5 s | 8.41 s |

FP64 is 20 to 33 times slower than FP32 on this GPU, which, like other
workstation and consumer GPUs, runs FP64 at 1/64 of the FP32 rate.

## Jetson AGX Thor (FP32)

| Problem | Grid | RealtimeHJB | Rate | hj_reachability |
|---|---|---|---|---|
| Air3D | 41^3 | 15.1 ms | 66 Hz | 68.5 ms |
| Air3D | 51^3 | 28.9 ms | 35 Hz | 110 ms |
| Air3D | 101^3 | 355 ms | 2.8 Hz | 1848 ms |
| Drifter, constant wind | 200^2 | 18.6 ms | 54 Hz | 167 ms |
| Drifter, spatially varying wind | 200^2 | 37.1 ms | 27 Hz | 289 ms |
| Pursuit-evasion 4D | 31^4 | 53.9 ms | 19 Hz | 392 ms |

## Accuracy against hj_reachability

In FP64 on the same grids, the 2D and 3D results match hj_reachability to
within 0.002 to 0.022 in V. On Air3D at 51^3 the two tubes differ by 4 of
about 11,200 cells. The 4D solver uses gradient-local Lax-Friedrichs
coefficients (see [Solvers](solvers.md)), so its value function differs more
from hj_reachability's; the 4D tubes agree to within 0.6 % when that choice
is aligned.

## Choosing grid and precision

- Use FP32 for anything inside a control loop. Use FP64 to check convergence
  or to compare with another library.
- Cost grows with the number of cells times the number of time steps, and the
  number of steps grows with the grid resolution: doubling N in 3D costs up to
  16 times more (measured: 11 times for Air3D from 51^3 to 101^3).
- On the desktop GPU in FP32, Air3D up to 71^3 (3.6 x 10^5 cells), the 4D
  problem up to 31^4 (9.2 x 10^5 cells) and the 2D drifter up to 200^2 solve at
  30 Hz or faster.

## Measuring on your hardware

`air3d_brt` and the other examples print the time of one solve after a
warm-up call, the same measurement used above:

```bash
./build/examples/air3d_brt 51
./build/examples/air3d_brt 101
```
