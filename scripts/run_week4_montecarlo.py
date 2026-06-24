"""Week 4 — Make it real. Monte-Carlo dispersions; report a landing success rate.

Runs N dispersed closed-loop landings (LQR + EKF) and reports the success rate + touchdown
statistics — the way aerospace proves robustness. Requires Week 1 (LQR); add the EKF to also
disperse the estimator. Runs the 2D model now; extending the dynamics to 3D is the Week-4
build (see ``make_dynamics`` / LEARNING.md) — flip ``DIM = 3`` once implemented.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _bootstrap import run_guarded

import numpy as np

from src.config import load_config
from src.dynamics import make_dynamics
from src.control.lqr import linearize, compute_lqr_gain, LQRController
from src.estimation.ekf import ExtendedKalmanFilter
from src.montecarlo import run_montecarlo

DIM = 2          # set to 3 after the 3D dynamics extension lands
USE_EKF = False  # set True to also disperse the estimator (needs Week 2)


def make_controller_factory(cfg):
    """Build a fresh LQR controller for each Monte-Carlo run."""
    def factory(dynamics, _cfg):
        p = dynamics.params
        state_eq = np.zeros(6)
        control_eq = np.array([p.initial_mass * p.gravity, 0.0])
        Q = np.diag([1.0, 1.0, 1.0, 1.0, 50.0, 10.0])
        R = np.diag([1e-9, 100.0])
        A, B = linearize(dynamics, state_eq, control_eq, p.initial_mass)
        K = compute_lqr_gain(A, B, Q, R)
        return LQRController(K, state_eq, control_eq)
    return factory


def make_observer_factory(cfg):
    dt = cfg["simulation"]["dt"]

    def factory(dynamics, _cfg):
        p = dynamics.params
        x_hat0 = np.array([0.0, 0.0, 1000.0, -80.0, 0.0, 0.0])
        P0 = np.diag([50.0, 5.0, 50.0, 5.0, 0.1, 0.05]) ** 2
        Q_ekf = np.diag([1e-2, 1e-1, 1e-2, 1e-1, 1e-3, 1e-3])
        # R is rebuilt from the run's own sensor model inside the sim; reuse nominal here.
        from src.simulator import SensorModel
        R_ekf = SensorModel(cfg["noise"]["sensor_std"], np.random.default_rng(0)).noise_cov()
        ekf = ExtendedKalmanFilter(dynamics, Q_ekf, R_ekf, x_hat0, P0, p.initial_mass)

        def observer(t, measurement, u_prev):
            ekf.predict(u_prev, dt)
            ekf.update(measurement)
            return ekf.x

        return observer
    return factory


def main():
    cfg = load_config()
    _ = make_dynamics(cfg, dim=DIM)  # validates DIM is supported before the campaign

    controller_factory = make_controller_factory(cfg)
    observer_factory = make_observer_factory(cfg) if USE_EKF else None

    mc = run_montecarlo(
        cfg, controller_factory, observer_factory=observer_factory, dim=DIM, seed=0
    )
    mc.print_report()


if __name__ == "__main__":
    run_guarded(main)
