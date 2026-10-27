# Wildfire response with a drone fleet

Two programs that use the drifter solver's agent labels to split a region
between several drones in a wind field.

`wildfire_voronoi` computes, for six drones in a 20 km x 20 km region, which
drone can reach each cell first under two wind scenarios. It does this twice:
once as a single labelled solve and once as six separate solves followed by an
argmin, and reports the time of each and how many cells agree.

```bash
./build/examples/wildfire/wildfire_voronoi out_dir
```

On an RTX 4000 Ada the labelled solve takes 15 to 17 ms and the six separate
solves 69 to 77 ms; 99.5 % of the cells get the same drone.

`fleet_sim` runs a 15-minute closed-loop simulation. Six drones patrol from
home stations; three fires start at different times; the wind turns and gusts.
At every 10 Hz tick the program recomputes the partition from the drones'
current positions and sends each new fire to the drone that can reach it first
within a 10-minute horizon.

```bash
./build/examples/wildfire/fleet_sim out_dir
```

It writes one frame per simulated second (about 310 MB) and `run_summary.json`
with the mean solve time per tick (4.0 ms on an RTX 4000 Ada, 9.7 ms on a
Jetson AGX Thor) and the number of fires suppressed.
