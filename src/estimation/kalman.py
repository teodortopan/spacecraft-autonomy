"""Linear Kalman Filter — STUB for the owner to implement (Week 2, foundation for the EKF).

A KF estimates the state of a *linear* system with Gaussian process and measurement noise,
maintaining a mean x̂ and covariance P.

Model:
    xₖ = F xₖ₋₁ + B uₖ₋₁ + wₖ ,   wₖ ~ N(0, Q)
    zₖ = H xₖ + vₖ ,               vₖ ~ N(0, R)

Predict (time update):
    x̂⁻ = F x̂ + B u
    P⁻  = F P Fᵀ + Q

Update (measurement update), given measurement z:
    y  = z - H x̂⁻                 (innovation)
    S  = H P⁻ Hᵀ + R              (innovation covariance)
    K  = P⁻ Hᵀ S⁻¹                (Kalman gain)
    x̂  = x̂⁻ + K y
    P  = (I - K H) P⁻             (Joseph form is more numerically stable)

Implement this first on the linear model; the EKF in ``ekf.py`` is the same recipe with F and H
replaced by Jacobians of nonlinear models. Master this, then generalize.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Discrete-time linear Kalman filter."""

    def __init__(
        self,
        F: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ):
        self.F = np.asarray(F, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)

    def predict(self, u: np.ndarray) -> None:
        """Time update: propagate the mean and covariance forward one step."""
        # x̂⁻ = F x̂ + B u ;   P⁻ = F P Fᵀ + Q
        # TODO(owner): implement the predict step.
        raise NotImplementedError(
            "KalmanFilter.predict not implemented — see src/estimation/kalman.py TODO(owner)"
        )

    def update(self, z: np.ndarray) -> None:
        """Measurement update: fuse measurement z into the estimate."""
        # y = z - H x̂⁻ ;  S = H P⁻ Hᵀ + R ;  K = P⁻ Hᵀ S⁻¹ ;
        # x̂ = x̂⁻ + K y ;  P = (I - K H) P⁻
        # TODO(owner): implement the update step.
        raise NotImplementedError(
            "KalmanFilter.update not implemented — see src/estimation/kalman.py TODO(owner)"
        )
