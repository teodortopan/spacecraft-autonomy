"""Rigid-body equations of motion for the powered-descent rocket.

This is INFRASTRUCTURE — fully implemented. The GNC algorithms (LQR/EKF/guidance) that the
owner writes will *linearize* and *call* this module, so the dynamics here must be correct,
analytic, and clean.

2D planar model
---------------
State (6):    s = [x, vx, y, vy, theta, omega]
    x, y      position (m)        — y is altitude, ground at y = 0
    vx, vy    velocity (m/s)      — vy < 0 means descending
    theta     attitude (rad)      — 0 = upright (body axis along +y), +CCW
    omega     angular rate (rad/s)

Control (2):  u = [thrust, gimbal]
    thrust    engine thrust magnitude (N), >= 0
    gimbal    thrust-vector deflection from the body axis (rad)

Mass depletes as propellant burns: ṁ = -thrust / (Isp * g0). Mass is NOT part of the 6-state
the GNC sees (so the controller/estimator stay 6-dimensional and easy to linearize); instead
it is carried alongside and passed explicitly into ``derivatives`` / ``step``. The simulator
integrates the augmented [state, mass] system (see :mod:`src.simulator`).

Force / torque derivation (engine mounted a distance ``l_cm`` below the center of mass)
---------------------------------------------------------------------------------------
Body "up" axis at attitude theta:           b = (-sin theta,  cos theta)
Thrust vector (deflected by gimbal delta):  F = T * (-sin(theta+delta), cos(theta+delta))
Engine position relative to CM:             r = -l_cm * b = (l_cm sin theta, -l_cm cos theta)
Torque (scalar, z):  tau = r_x F_y - r_y F_x = -T * l_cm * sin(delta)

Hence:
    ax     = -T sin(theta + delta) / m
    ay     =  T cos(theta + delta) / m - g
    alpha  = -T * l_cm * sin(delta) / I
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# --- State / control index constants (use these instead of magic numbers) ---
X, VX, Y, VY, THETA, OMEGA = 0, 1, 2, 3, 4, 5
STATE_DIM = 6

THRUST, GIMBAL = 0, 1
CONTROL_DIM = 2


@dataclass
class VehicleParams:
    """Physical parameters of the vehicle + environment, loaded from config/vehicle.yaml."""

    dry_mass: float           # kg, mass with no propellant
    initial_propellant: float # kg
    max_thrust: float         # N
    min_thrust_fraction: float  # fraction of max_thrust when the engine is lit (guidance constraint)
    isp: float                # s
    max_gimbal_angle: float   # rad
    inertia: float            # kg*m^2 about the CM
    length: float             # m (overall length; used for geometry + drawing)
    gravity: float            # m/s^2
    g0: float                 # m/s^2 (standard gravity for the Isp relation)

    @property
    def l_cm(self) -> float:
        """Distance from the center of mass to the engine gimbal point (m).

        Toy assumption: CM at the geometric center, engine at the tail => l_cm = length / 2.
        """
        return self.length / 2.0

    @property
    def initial_mass(self) -> float:
        return self.dry_mass + self.initial_propellant

    @classmethod
    def from_config(cls, cfg: dict) -> "VehicleParams":
        v = cfg["vehicle"]
        e = cfg["environment"]
        return cls(
            dry_mass=float(v["dry_mass"]),
            initial_propellant=float(v["initial_propellant"]),
            max_thrust=float(v["max_thrust"]),
            min_thrust_fraction=float(v["min_thrust_fraction"]),
            isp=float(v["isp"]),
            max_gimbal_angle=float(v["max_gimbal_angle"]),
            inertia=float(v["inertia"]),
            length=float(v["length"]),
            gravity=float(e["gravity"]),
            g0=float(e["g0"]),
        )


class RocketDynamics:
    """Base class. Subclass for 2D (here) and later 3D — keep the same call signatures."""

    state_dim: int = STATE_DIM
    control_dim: int = CONTROL_DIM

    def __init__(self, params: VehicleParams):
        self.params = params

    def mass_flow_rate(self, control: np.ndarray) -> float:
        """ṁ (kg/s, negative — mass decreases). Depends only on thrust magnitude."""
        thrust = float(control[THRUST])
        return -abs(thrust) / (self.params.isp * self.params.g0)

    def clip_control(self, control: np.ndarray) -> np.ndarray:
        """Saturate a commanded control to the actuator limits.

        Thrust is clamped to [0, max_thrust]; the engine may be off (0). The
        ``min_thrust_fraction`` (minimum throttle *while lit*) is a guidance-level constraint,
        not enforced here so that zero-thrust (free-fall) remains representable.
        """
        p = self.params
        u = np.asarray(control, dtype=float).copy()
        u[THRUST] = np.clip(u[THRUST], 0.0, p.max_thrust)
        u[GIMBAL] = np.clip(u[GIMBAL], -p.max_gimbal_angle, p.max_gimbal_angle)
        return u


class Rocket2DDynamics(RocketDynamics):
    """Planar (2D) rigid-body rocket dynamics."""

    def derivatives(self, state: np.ndarray, control: np.ndarray, mass: float) -> np.ndarray:
        """Continuous-time state derivative ṡ for the 6-state, given current ``mass`` (kg).

        Control is saturated to actuator limits before use. Returns a length-6 array.
        """
        p = self.params
        s = np.asarray(state, dtype=float)
        u = self.clip_control(control)

        theta = s[THETA]
        omega = s[OMEGA]
        thrust = u[THRUST]
        delta = u[GIMBAL]
        m = float(mass)

        # Translational acceleration (world frame).
        ax = -thrust * np.sin(theta + delta) / m
        ay = thrust * np.cos(theta + delta) / m - p.gravity

        # Rotational acceleration about the CM from the gimballed thrust.
        alpha = -thrust * p.l_cm * np.sin(delta) / p.inertia

        ds = np.empty(STATE_DIM, dtype=float)
        ds[X] = s[VX]
        ds[VX] = ax
        ds[Y] = s[VY]
        ds[VY] = ay
        ds[THETA] = omega
        ds[OMEGA] = alpha
        return ds

    def augmented_derivatives(self, aug: np.ndarray, control: np.ndarray) -> np.ndarray:
        """ṡ for the augmented state [x, vx, y, vy, theta, omega, mass] (length 7).

        Used by the integrator so that mass depletion is integrated consistently with the
        rigid-body states within a single RK4 step.
        """
        state = aug[:STATE_DIM]
        mass = aug[STATE_DIM]
        ds = self.derivatives(state, control, mass)
        dmass = self.mass_flow_rate(self.clip_control(control))
        return np.concatenate([ds, [dmass]])

    def step(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: float,
        mass: float,
        integrator: str = "rk4",
    ) -> tuple[np.ndarray, float]:
        """Advance the 6-state and mass by ``dt`` under a zero-order-hold control.

        Returns ``(next_state, next_mass)``. Mass is floored at the dry mass (out of propellant
        => engine produces no more depletion, but thrust is still allowed for a toy model;
        a stricter model would also zero thrust here).
        """
        aug = np.concatenate([np.asarray(state, dtype=float), [float(mass)]])

        if integrator == "euler":
            aug_next = aug + dt * self.augmented_derivatives(aug, control)
        elif integrator == "rk4":
            k1 = self.augmented_derivatives(aug, control)
            k2 = self.augmented_derivatives(aug + 0.5 * dt * k1, control)
            k3 = self.augmented_derivatives(aug + 0.5 * dt * k2, control)
            k4 = self.augmented_derivatives(aug + dt * k3, control)
            aug_next = aug + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown integrator {integrator!r} (use 'rk4' or 'euler')")

        next_state = aug_next[:STATE_DIM]
        next_mass = max(aug_next[STATE_DIM], self.params.dry_mass)
        return next_state, next_mass


def make_dynamics(cfg: dict, dim: int = 2) -> RocketDynamics:
    """Factory: build a dynamics model from a loaded config dict.

    ``dim=2`` is implemented now; ``dim=3`` is the Week-4 extension.
    """
    params = VehicleParams.from_config(cfg)
    if dim == 2:
        return Rocket2DDynamics(params)
    raise NotImplementedError("3D dynamics arrive in Week 4 — see LEARNING.md")
