"""Week 3 — Make it optimal. Convex powered-descent guidance, then track it; compare fuel.

Wires the owner's ``solve_powered_descent`` (src/control/guidance.py) to compute a fuel-optimal
trajectory, then tracks it with the LQR controller and reports fuel used vs an LQR-only baseline.
Requires Week 1 (LQR) and Week 3 (guidance) to be implemented.
"""

from __future__ import annotations

import os

import _bootstrap  # noqa: F401
from _bootstrap import run_guarded, OUTPUTS

import numpy as np

from src.config import load_config
from src.dynamics import make_dynamics
from src.simulator import Simulator, evaluate_landing
from src.control.lqr import linearize, compute_lqr_gain, LQRController
from src.control.guidance import solve_powered_descent
from src.viz import plots


def main():
    cfg = load_config()
    dynamics = make_dynamics(cfg, dim=2)
    params = dynamics.params

    sim = Simulator(dynamics, cfg, rng=np.random.default_rng(2))
    x0 = sim.initial_state()

    # --- Guidance (Week 3): fuel-optimal trajectory ---
    solution = solve_powered_descent(dynamics, x0, cfg, n_nodes=60)
    print(f"Guidance solved: status={solution.solver_status}, fuel={solution.fuel_used:.1f} kg")

    # --- Track the optimal trajectory with LQR (reuse Week-1 controller about hover) ---
    state_eq = np.zeros(6)
    control_eq = np.array([params.initial_mass * params.gravity, 0.0])
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 50.0, 10.0])
    R = np.diag([1e-9, 100.0])
    A, B = linearize(dynamics, state_eq, control_eq, params.initial_mass)
    K = compute_lqr_gain(A, B, Q, R)
    # NOTE(owner): a trajectory-tracking controller feeds the guidance reference into the law,
    # i.e. u = u_ref(t) - K (x - x_ref(t)). Wire solution -> reference here once implemented.
    controller = LQRController(K, state_eq, control_eq)

    result = sim.run(controller, enable_process_noise=False, enable_sensor_noise=False)
    report = evaluate_landing(result, cfg)

    print("Week 3 — guidance + tracking")
    for k, v in report.items():
        print(f"  {k:18s}: {v}")
    print(f"\nFuel — guidance(optimal): {solution.fuel_used:.1f} kg | "
          f"tracked(actual): {report['propellant_used']:.1f} kg")

    outdir = os.path.join(OUTPUTS, "week3")
    plots.save_figure(plots.plot_trajectory(result, "Week 3 — guided descent"),
                      os.path.join(outdir, "trajectory.png"))
    print(f"\nSaved plots to {outdir}/")


if __name__ == "__main__":
    run_guarded(main)
