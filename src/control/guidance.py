"""Convex powered-descent guidance — STUB for the owner to implement (Week 3, the SpaceX algorithm).

Goal: compute a *fuel-optimal* descent trajectory from the current state to a soft landing at the
pad, subject to thrust and geometry constraints — then track it (e.g. with the LQR controller).

Continuous problem (translational only; attitude handled by the tracker)
------------------------------------------------------------------------
State r (position), v (velocity), mass m; control T (thrust vector), with ‖T‖ ∈ [T_min, T_max].

    minimize    fuel  ≈  ∫ ‖T‖ / (Isp g0) dt        (equivalently maximize final mass)
    subject to  ṙ = v
                v̇ = T / m  +  g
                ṁ = -‖T‖ / (Isp g0)
                T_min ≤ ‖T‖ ≤ T_max                 (lower bound is NON-convex as written)
                glide-slope:  position stays within a cone above the pad
                r(0), v(0), m(0) given;  r(t_f) = pad, v(t_f) = 0   (soft landing)

Lossless convexification (Açıkmeşe & Ploen, 2007)
-------------------------------------------------
The thrust *lower* bound ‖T‖ ≥ T_min makes the feasible set non-convex. Introduce a slack
σ with ‖T‖ ≤ σ and relax the bound to  T_min ≤ σ ≤ T_max ; a change of variables
(u = T/m, with a log-mass substitution z = ln m) turns the dynamics + constraints into a
Second-Order Cone Program. Açıkmeşe & Ploen prove the relaxation is *lossless* (the optimum
satisfies ‖T‖ = σ), so the convex solution solves the original problem.

Implementation: discretize over N nodes, build the SOCP in CVXPY, solve, and return the optimal
state/thrust trajectory for the tracker to follow. Compare fuel used vs the LQR-only baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamics import RocketDynamics


@dataclass
class GuidanceSolution:
    """Optimal trajectory returned by the solver."""

    t: np.ndarray            # (N,)   time grid
    positions: np.ndarray    # (N, d) optimal position trajectory
    velocities: np.ndarray   # (N, d)
    thrusts: np.ndarray      # (N, d) optimal thrust vector
    masses: np.ndarray       # (N,)
    fuel_used: float
    solver_status: str


def solve_powered_descent(
    dynamics: RocketDynamics,
    x0: np.ndarray,
    cfg: dict,
    n_nodes: int = 60,
) -> GuidanceSolution:
    """Solve the convex powered-descent guidance problem and return the optimal trajectory.

    Build the discretized SOCP in CVXPY per the formulation above (objective = fuel; dynamics as
    equality constraints; thrust cone + glide-slope as conic constraints; initial state from x0,
    terminal soft-landing constraints), solve it, and pack the result into a GuidanceSolution.
    """
    # TODO(owner): formulate and solve the SOCP in CVXPY (lossless convexification).
    raise NotImplementedError(
        "solve_powered_descent not implemented — see src/control/guidance.py TODO(owner)"
    )
