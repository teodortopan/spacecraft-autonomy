# Spacecraft Autonomy

> **Scope:** an autonomous flight-software / GNC stack for spacecraft. This repo currently
> implements the **GNC landing core** — the foundation for a larger autonomy stack (mission
> planning, AI copilot, terrain-aware landing, telemetry, health monitoring) on the roadmap.

The current core is a simulator for autonomously landing under powered descent, demonstrating the
full **Guidance, Navigation & Control (GNC)** loop:

> **estimate state → guide → control → actuate**

validated with Monte-Carlo dispersions. Pure Python, runs on a laptop.

This project is a working demonstration of three general autonomy capabilities:

- **State Estimation & Sensor Fusion** — an Extended Kalman Filter fusing heterogeneous sensors
  (IMU + position + range) to recover full vehicle state from noisy measurements.
- **Optimal Control** — an LQR controller derived from the linearized rigid-body dynamics.
- **Trajectory Optimization** — fuel-optimal powered-descent guidance via convex programming
  (lossless convexification, after Açıkmeşe & Ploen).

Demonstrated here on a powered-descent landing; the same core transfers directly to other
vehicles (the next iteration applies it to a quadrotor).

## The GNC loop

```
                +--------------------+
   sensors ---> |  Navigation (EKF)  | --- state estimate -->+
   (noisy)      +--------------------+                       |
                                                             v
                +--------------------+            +---------------------+
   target ----> | Guidance (convex)  | --traj--> | Control (LQR / PID) | --[thrust, gimbal]-->
                +--------------------+            +---------------------+                       |
                                                                                               v
                                                                                  +-------------------+
                                                                                  | Vehicle dynamics  |
                                                                                  | (rigid body + g)  |
                                                                                  +-------------------+
                                                                                               |
                                                                                  state --------+--> sensors
```

## Status / approach

This repo is built as a **learning project**. The simulation infrastructure (dynamics, integration,
noise, visualization, Monte-Carlo, tests) is fully implemented. The core algorithms — PID, LQR,
the Kalman filter / EKF, and the convex guidance problem — are **implemented by the author by hand**
as the learning exercise, and live as documented stubs until each week is completed.

See [`LEARNING.md`](./LEARNING.md) for the study plan and progress tracker.

## Quickstart

```bash
python3 -m venv rocket-env
source rocket-env/bin/activate
pip install -r requirements.txt

python -m pytest tests/        # physics sanity checks
python scripts/run_week1_lqr.py
```

Until a given week's algorithm is implemented, its script prints a clear message telling you
which file to fill in (it will not crash with a traceback).

## How to run each milestone

| Week | Script | What it shows |
|------|--------|---------------|
| 1 | `scripts/run_week1_lqr.py` | 2D rocket lands softly under PID then LQR, on true state |
| 2 | `scripts/run_week2_ekf.py` | Noisy sensors + EKF estimate; lands "blind" on the estimate |
| 3 | `scripts/run_week3_guidance.py` | Convex fuel-optimal guidance, tracked; fuel vs LQR-only |
| 4 | `scripts/run_week4_montecarlo.py` | 3D + Monte-Carlo dispersions; landing success rate |

## Configuration

All vehicle, environment, simulation, noise, and Monte-Carlo parameters live in
[`config/vehicle.yaml`](./config/vehicle.yaml).

## Results

_(placeholder — fill in with the landing animation, estimate-vs-truth plots, fuel comparison,
and Monte-Carlo success rate as each week is completed.)_
