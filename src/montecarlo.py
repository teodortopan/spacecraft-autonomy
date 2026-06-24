"""Monte-Carlo dispersion runner. INFRASTRUCTURE — fully implemented.

Runs N closed-loop sims with randomized initial conditions and reports a landing success rate
plus touchdown statistics — the way aerospace demonstrates robustness (Week 4).

The GNC pieces are supplied as *factories* so each run gets a fresh controller/observer (an
estimator carries internal state, so it must be re-created per run):

    controller_factory(dynamics, cfg) -> controller(t, nav_state) -> u
    observer_factory(dynamics, cfg)   -> observer(t, measurement, u_prev) -> est   (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .dynamics import RocketDynamics, make_dynamics, X, VX, Y, VY, THETA, OMEGA
from .simulator import Simulator, evaluate_landing

ControllerFactory = Callable[[RocketDynamics, dict], Callable]
ObserverFactory = Callable[[RocketDynamics, dict], Callable]


@dataclass
class MonteCarloResult:
    n_runs: int
    success_rate: float
    per_run: list           # list of evaluate_landing dicts
    summary: dict           # aggregate stats

    def print_report(self) -> None:
        print(f"\nMonte-Carlo: {self.n_runs} runs")
        print(f"  success rate : {self.success_rate * 100:.1f}%")
        for key in ("touchdown_speed", "touchdown_tilt", "horizontal_offset", "propellant_used"):
            st = self.summary[key]
            print(
                f"  {key:18s}: mean {st['mean']:8.3f}  p95 {st['p95']:8.3f}  max {st['max']:8.3f}"
            )
        reasons = self.summary["reasons"]
        print(f"  termination  : {reasons}")


def _sample_initial_state(cfg: dict, rng: np.random.Generator) -> np.ndarray:
    ic = cfg["initial_state"]
    disp = cfg["montecarlo"]["dispersions"]
    nominal = np.array(
        [ic["x"], ic["vx"], ic["y"], ic["vy"], ic["theta"], ic["omega"]], dtype=float
    )
    std = np.array(
        [disp["x"], disp["vx"], disp["y"], disp["vy"], disp["theta"], disp["omega"]],
        dtype=float,
    )
    x0 = nominal + std * rng.normal(size=6)
    x0[Y] = max(x0[Y], 10.0)  # keep the vehicle above ground at t=0
    return x0


def run_montecarlo(
    cfg: dict,
    controller_factory: ControllerFactory,
    observer_factory: Optional[ObserverFactory] = None,
    n_runs: Optional[int] = None,
    seed: int = 0,
    dim: int = 2,
) -> MonteCarloResult:
    """Run the dispersion campaign and aggregate results."""
    n = int(n_runs if n_runs is not None else cfg["montecarlo"]["n_runs"])
    master = np.random.default_rng(seed)
    per_run = []

    for i in range(n):
        rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        dynamics = make_dynamics(cfg, dim=dim)
        sim = Simulator(dynamics, cfg, rng=rng)
        x0 = _sample_initial_state(cfg, rng)

        controller = controller_factory(dynamics, cfg)
        observer = observer_factory(dynamics, cfg) if observer_factory else None

        result = sim.run(controller, observer=observer, x0=x0)
        per_run.append(evaluate_landing(result, cfg))

    return _aggregate(per_run)


def _aggregate(per_run: list) -> MonteCarloResult:
    n = len(per_run)
    successes = sum(r["success"] for r in per_run)

    def stats(key: str) -> dict:
        vals = np.array([r[key] for r in per_run], dtype=float)
        return {
            "mean": float(np.mean(vals)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)),
        }

    reasons: dict = {}
    for r in per_run:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    summary = {
        "touchdown_speed": stats("touchdown_speed"),
        "touchdown_tilt": stats("touchdown_tilt"),
        "horizontal_offset": stats("horizontal_offset"),
        "propellant_used": stats("propellant_used"),
        "reasons": reasons,
    }
    return MonteCarloResult(
        n_runs=n,
        success_rate=successes / n if n else 0.0,
        per_run=per_run,
        summary=summary,
    )
