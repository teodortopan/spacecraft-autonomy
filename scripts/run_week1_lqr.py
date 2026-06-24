"""Week 1 — Make it land. LQR controller on the TRUE state, 2D planar landing.

Wires the owner's ``linearize`` + ``compute_lqr_gain`` + ``LQRController`` (src/control/lqr.py)
into the simulator. Until those are implemented, prints a pointer and exits cleanly.

(``src/control/pid.py`` is available as a warm-up before LQR — see LEARNING.md Week 1.)
"""

from __future__ import annotations

import os

import _bootstrap  # noqa: F401  (sets sys.path)
from _bootstrap import run_guarded, OUTPUTS

import numpy as np

from src.config import load_config
from src.dynamics import make_dynamics
from src.simulator import Simulator, evaluate_landing
from src.control.lqr import linearize, compute_lqr_gain, LQRController
from src.viz import plots


def main():
    cfg = load_config()
    dynamics = make_dynamics(cfg, dim=2)
    params = dynamics.params

    # Operating point: regulate to the pad (origin), upright, at rest. Hover thrust balances g.
    state_eq = np.zeros(6)
    m_design = params.initial_mass
    control_eq = np.array([m_design * params.gravity, 0.0])

    # LQR weights — TUNE THESE (owner). Penalize position/attitude error vs control effort.
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 50.0, 10.0])
    R = np.diag([1e-9, 100.0])  # thrust is large-valued (N); scale R[0] small accordingly

    # --- Owner-implemented control design ---
    A, B = linearize(dynamics, state_eq, control_eq, m_design)
    K = compute_lqr_gain(A, B, Q, R)
    controller = LQRController(K, state_eq, control_eq)

    # --- Simulate on the true state (no estimator yet) ---
    sim = Simulator(dynamics, cfg, rng=np.random.default_rng(0))
    result = sim.run(controller, enable_process_noise=False, enable_sensor_noise=False)

    report = evaluate_landing(result, cfg)
    print("Week 1 — LQR on true state")
    for k, v in report.items():
        print(f"  {k:18s}: {v}")

    outdir = os.path.join(OUTPUTS, "week1")
    plots.save_figure(plots.plot_trajectory(result, "Week 1 — LQR landing"),
                      os.path.join(outdir, "trajectory.png"))
    plots.save_figure(plots.plot_controls(result, max_thrust=params.max_thrust),
                      os.path.join(outdir, "controls.png"))
    print(f"\nSaved plots to {outdir}/")
    print("Tip: plots.animate_landing(result, params.length, save_path='.../landing.gif')")


if __name__ == "__main__":
    run_guarded(main)
