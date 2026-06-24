"""Extended Kalman Filter — STUB for the owner to implement (Week 2, land "blind").

The EKF estimates the state of a *nonlinear* system by linearizing about the current estimate.
Here it fuses heterogeneous sensors — position, velocity, IMU attitude/gyro, and a NONLINEAR
range-to-pad measurement — to reconstruct the full 6-state from noisy data, so the controller
can land on the estimate.

Nonlinear model:
    xₖ = f(xₖ₋₁, uₖ₋₁) + wₖ ,   wₖ ~ N(0, Q)
    zₖ = h(xₖ) + vₖ ,           vₖ ~ N(0, R)

Jacobians (derive these BY HAND — that's the Week-2 exercise):
    F = ∂f/∂x |(x̂, u)         H = ∂h/∂x |(x̂)

Predict (time update):
    x̂⁻ = f(x̂, u)              (propagate through the *nonlinear* dynamics)
    P⁻  = F P Fᵀ + Q

Update (measurement update):
    y  = z - h(x̂⁻)            (innovation, using the nonlinear measurement function)
    S  = H P⁻ Hᵀ + R
    K  = P⁻ Hᵀ S⁻¹
    x̂  = x̂⁻ + K y
    P  = (I - K H) P⁻

The measurement layout and the noise-free h(x) are defined by the simulator's sensor model
(``src.simulator.SensorModel.measurement_function`` and the M_* indices); R is
``SensorModel.noise_cov()``. The range row z = sqrt(x² + y²) is what makes H state-dependent
and forces the EKF (vs. a plain KF). For propagation you can reuse ``dynamics.step`` for f, and
derive F either analytically or by linearizing the equations of motion.
"""

from __future__ import annotations

import numpy as np

from ..dynamics import RocketDynamics


class ExtendedKalmanFilter:
    """Discrete-time EKF for the 6-state rocket with heterogeneous sensor fusion."""

    def __init__(
        self,
        dynamics: RocketDynamics,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        mass0: float,
    ):
        self.dynamics = dynamics
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)
        self.mass = float(mass0)

    # --- Jacobians (owner derives these by hand) ---
    def state_jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """F = ∂f/∂x for the discrete propagation step, evaluated at (x, u)."""
        # TODO(owner): derive and return the (6, 6) state-transition Jacobian.
        raise NotImplementedError(
            "EKF.state_jacobian not implemented — see src/estimation/ekf.py TODO(owner)"
        )

    def measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """H = ∂h/∂x of the sensor model, evaluated at x (the range row is nonlinear)."""
        # TODO(owner): derive and return the (7, 6) measurement Jacobian.
        raise NotImplementedError(
            "EKF.measurement_jacobian not implemented — see src/estimation/ekf.py TODO(owner)"
        )

    def predict(self, u: np.ndarray, dt: float) -> None:
        """Time update: x̂⁻ = f(x̂, u) through the nonlinear dynamics; P⁻ = F P Fᵀ + Q."""
        # TODO(owner): implement the nonlinear predict step (reuse dynamics.step for f).
        raise NotImplementedError(
            "EKF.predict not implemented — see src/estimation/ekf.py TODO(owner)"
        )

    def update(self, z: np.ndarray) -> None:
        """Measurement update with the nonlinear h(x): innovation, gain, state & covariance."""
        # TODO(owner): implement the EKF update (use SensorModel.measurement_function for h).
        raise NotImplementedError(
            "EKF.update not implemented — see src/estimation/ekf.py TODO(owner)"
        )
