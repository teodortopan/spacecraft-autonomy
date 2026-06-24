"""Week 2 — Make it land blind. Noisy sensors + EKF; the controller lands on the ESTIMATE.

Wires the owner's EKF (src/estimation/ekf.py) as the simulator's observer and reuses the Week-1
LQR controller — but now the controller sees the EKF estimate instead of the true state. Plots
estimate-vs-truth. Requires Week 1 (LQR) and Week 2 (EKF) to be implemented.
"""

from __future__ import annotations

import os

import _bootstrap  # noqa: F401
from _bootstrap import run_guarded, OUTPUTS

import numpy as np

from src.config import load_config
from src.dynamics import make_dynamics
from src.simulator import Simulator, SensorModel, evaluate_landing
from src.control.lqr import linearize, compute_lqr_gain, LQRController
from src.estimation.ekf import ExtendedKalmanFilter
from src.viz import plots


def main():
    cfg = load_config()
    dynamics = make_dynamics(cfg, dim=2)
    params = dynamics.params
    dt = cfg["simulation"]["dt"]

    # --- Controller (Week 1) ---
    state_eq = np.zeros(6)
    control_eq = np.array([params.initial_mass * params.gravity, 0.0])
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 50.0, 10.0])
    R = np.diag([1e-9, 100.0])
    A, B = linearize(dynamics, state_eq, control_eq, params.initial_mass)
    K = compute_lqr_gain(A, B, Q, R)
    controller = LQRController(K, state_eq, control_eq)

    # --- Estimator (Week 2) ---
    sim = Simulator(dynamics, cfg, rng=np.random.default_rng(1))
    x_true0 = sim.initial_state()
    x_hat0 = x_true0 + np.array([20.0, 2.0, 20.0, 2.0, 0.05, 0.02])  # imperfect initial guess
    P0 = np.diag([50.0, 5.0, 50.0, 5.0, 0.1, 0.05]) ** 2
    Q_ekf = np.diag([1e-2, 1e-1, 1e-2, 1e-1, 1e-3, 1e-3])
    R_ekf = sim.sensor_model.noise_cov()

    ekf = ExtendedKalmanFilter(dynamics, Q_ekf, R_ekf, x_hat0, P0, params.initial_mass)

    def observer(t, measurement, u_prev):
        ekf.predict(u_prev, dt)
        ekf.update(measurement)
        return ekf.x

    result = sim.run(controller, observer=observer)

    report = evaluate_landing(result, cfg)
    print("Week 2 — LQR on EKF-estimated state (noisy sensors)")
    for k, v in report.items():
        print(f"  {k:18s}: {v}")

    outdir = os.path.join(OUTPUTS, "week2")
    plots.save_figure(plots.plot_trajectory(result, "Week 2 — land blind"),
                      os.path.join(outdir, "trajectory.png"))
    plots.save_figure(plots.plot_estimate_vs_truth(result),
                      os.path.join(outdir, "estimate_vs_truth.png"))
    print(f"\nSaved plots to {outdir}/")


if __name__ == "__main__":
    run_guarded(main)
