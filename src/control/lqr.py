"""Linear-Quadratic Regulator — STUB for the owner to implement (Week 1, the real controller).

Idea
----
Linearize the dynamics about an operating point (for landing, a *hover* equilibrium: upright,
thrust balancing gravity, zero gimbal). This gives a linear model of the error dynamics:

    δẋ = A δx + B δu,     δx = x - x_eq,   δu = u - u_eq

LQR chooses δu = -K δx to minimize the infinite-horizon cost

    J = ∫₀^∞ ( δxᵀ Q δx  +  δuᵀ R δu ) dt ,   Q ⪰ 0,  R ≻ 0

The optimal gain is

    K = R⁻¹ Bᵀ P

where P = Pᵀ ⪰ 0 solves the Continuous-time Algebraic Riccati Equation (CARE):

    AᵀP + PA - P B R⁻¹ Bᵀ P + Q = 0

The closed-loop control law applied to the nonlinear plant is then

    u = u_eq - K (x - x_eq)

Three things to implement: (1) linearize to get A, B; (2) solve CARE for K; (3) the feedback law.
Tuning Q vs R trades state error against control effort. You may cross-check K against
``scipy.linalg.solve_continuous_are`` or ``control.lqr`` — but derive it yourself first.
"""

from __future__ import annotations

import numpy as np

from ..dynamics import RocketDynamics, STATE_DIM, CONTROL_DIM


def linearize(
    dynamics: RocketDynamics,
    state_eq: np.ndarray,
    control_eq: np.ndarray,
    mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, B) — the Jacobians ∂ṡ/∂s and ∂ṡ/∂u of the dynamics at the operating point.

    A is (6, 6), B is (6, 2). You can derive these analytically from the equations of motion in
    ``src/dynamics.py`` (recommended — it builds the intuition), or compute them with finite
    differences of ``dynamics.derivatives``. Evaluate at ``(state_eq, control_eq, mass)``.
    """
    # TODO(owner): build A = ∂f/∂x and B = ∂f/∂u at the equilibrium.
    raise NotImplementedError(
        "linearize not implemented — see src/control/lqr.py TODO(owner)"
    )


def compute_lqr_gain(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Solve the CARE and return the LQR gain K = R⁻¹ Bᵀ P  (shape (2, 6)).

    Steps: solve  AᵀP + PA - P B R⁻¹ Bᵀ P + Q = 0  for P ⪰ 0, then K = R⁻¹ Bᵀ P.
    """
    # TODO(owner): solve the Continuous Algebraic Riccati Equation, then form K.
    raise NotImplementedError(
        "compute_lqr_gain not implemented — see src/control/lqr.py TODO(owner)"
    )


class LQRController:
    """Closed-loop LQR controller: u = u_eq - K (x - x_eq), saturated by the simulator.

    Construct it with the equilibrium operating point and the gain K from ``compute_lqr_gain``.
    """

    def __init__(self, K: np.ndarray, state_eq: np.ndarray, control_eq: np.ndarray):
        self.K = np.asarray(K, dtype=float)
        self.state_eq = np.asarray(state_eq, dtype=float)
        self.control_eq = np.asarray(control_eq, dtype=float)

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return the control command for the current (estimated) state."""
        # TODO(owner): implement the feedback law u = u_eq - K (x - x_eq).
        raise NotImplementedError(
            "LQRController.__call__ not implemented — see src/control/lqr.py TODO(owner)"
        )
