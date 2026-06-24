"""Closed-loop simulation: integrate the dynamics, inject noise, log everything.

INFRASTRUCTURE — fully implemented. This is the harness the owner's GNC code plugs into.

Decoupling from the owner's algorithms
--------------------------------------
The simulator never imports the stubbed control/estimation files. Instead the weekly scripts
wrap the owner's code into two simple callables and hand them to :meth:`Simulator.run`:

    controller(t, nav_state) -> u           # nav_state is the estimate if an observer is given,
                                            # otherwise the true state
    observer(t, measurement, u_prev) -> est # optional; the Week-2 EKF wrapper

So "land on the true state" (Week 1) is just ``run(controller)``; "land blind" (Week 2) is
``run(controller, observer=ekf_wrapper)`` — and the controller transparently receives the
estimate. Estimates are logged for estimate-vs-truth plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .dynamics import RocketDynamics, STATE_DIM, CONTROL_DIM, X, VX, Y, VY, THETA, OMEGA

# --- Sensor measurement layout (what the owner's EKF must model with h(x), H) ---
# z = [x, y, vx, vy, theta, omega, range].  The last row (slant range to the pad at the
# origin, sqrt(x^2 + y^2)) is NONLINEAR — it is why an EKF (not a plain KF) is needed.
M_X, M_Y, M_VX, M_VY, M_THETA, M_OMEGA, M_RANGE = range(7)
MEAS_DIM = 7

Controller = Callable[[float, np.ndarray], np.ndarray]
Observer = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


class SensorModel:
    """Maps true state -> noisy measurements. The owner's EKF mirrors ``measurement_function``.

    Provide the owner with: the noise-free measurement function h(x), the layout above, and the
    measurement-noise covariance R. The owner writes the matching Jacobian H = dh/dx by hand.
    """

    def __init__(self, sensor_std: dict, rng: np.random.Generator):
        self.std = sensor_std
        self.rng = rng

    @staticmethod
    def measurement_function(state: np.ndarray) -> np.ndarray:
        """Noise-free h(x): the measurement you would get from a perfect sensor suite."""
        s = np.asarray(state, dtype=float)
        rng_to_pad = float(np.hypot(s[X], s[Y]))
        return np.array(
            [s[X], s[Y], s[VX], s[VY], s[THETA], s[OMEGA], rng_to_pad], dtype=float
        )

    def _sigma_vector(self) -> np.ndarray:
        d = self.std
        return np.array(
            [
                d["position"], d["position"],
                d["velocity"], d["velocity"],
                d["attitude"], d["gyro"],
                d["altitude"],   # range/altimeter noise
            ],
            dtype=float,
        )

    def noise_cov(self) -> np.ndarray:
        """Measurement-noise covariance R (diagonal), for the owner's EKF update."""
        return np.diag(self._sigma_vector() ** 2)

    def measure(self, state: np.ndarray) -> np.ndarray:
        """A noisy measurement of ``state``."""
        return self.measurement_function(state) + self.rng.normal(0.0, self._sigma_vector())


@dataclass
class SimulationResult:
    """Full history of a run. Time-major arrays of shape (N, ...)."""

    t: np.ndarray                       # (N,)
    states: np.ndarray                  # (N, 6)   true state
    controls: np.ndarray                # (N, 2)   applied (saturated) control
    masses: np.ndarray                  # (N,)     vehicle mass
    measurements: Optional[np.ndarray] = None   # (N, 7) or None
    estimates: Optional[np.ndarray] = None       # (N, 6) or None (observer output)
    terminated_reason: str = ""         # "landed", "max_time", "diverged"

    @property
    def final_state(self) -> np.ndarray:
        return self.states[-1]

    @property
    def propellant_used(self) -> float:
        return float(self.masses[0] - self.masses[-1])


@dataclass
class Simulator:
    """Fixed-step closed-loop simulator with optional process + sensor noise."""

    dynamics: RocketDynamics
    cfg: dict
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def __post_init__(self):
        sim = self.cfg["simulation"]
        self.dt = float(sim["dt"])
        self.max_time = float(sim["max_time"])
        self.integrator = str(sim.get("integrator", "rk4"))
        self.ground = float(sim["ground_altitude"])
        self._process_std = self._build_process_std()
        self.sensor_model = SensorModel(self.cfg["noise"]["sensor_std"], self.rng)

    def _build_process_std(self) -> np.ndarray:
        d = self.cfg["noise"]["process_std"]
        return np.array(
            [d["x"], d["vx"], d["y"], d["vy"], d["theta"], d["omega"]], dtype=float
        )

    def initial_state(self) -> np.ndarray:
        ic = self.cfg["initial_state"]
        return np.array(
            [ic["x"], ic["vx"], ic["y"], ic["vy"], ic["theta"], ic["omega"]], dtype=float
        )

    def run(
        self,
        controller: Controller,
        observer: Optional[Observer] = None,
        x0: Optional[np.ndarray] = None,
        mass0: Optional[float] = None,
        enable_process_noise: bool = True,
        enable_sensor_noise: bool = True,
    ) -> SimulationResult:
        """Run the closed loop until landing, divergence, or ``max_time``.

        ``controller`` sees the estimate when ``observer`` is provided, else the true state.
        Sensor measurements (and estimates) are logged whenever an observer is present, or when
        sensor noise is on, so the weekly scripts can plot estimate-vs-truth.
        """
        params = self.dynamics.params
        state = self.initial_state() if x0 is None else np.asarray(x0, dtype=float).copy()
        mass = params.initial_mass if mass0 is None else float(mass0)

        n_max = int(np.ceil(self.max_time / self.dt)) + 1
        t_hist, s_hist, u_hist, m_hist = [], [], [], []
        z_hist, e_hist = [], []
        log_meas = observer is not None or enable_sensor_noise

        u_prev = np.zeros(CONTROL_DIM)
        reason = "max_time"

        for k in range(n_max):
            t = k * self.dt

            # --- Navigation: measurement -> estimate (or pass through truth) ---
            measurement = None
            if log_meas:
                measurement = (
                    self.sensor_model.measure(state)
                    if enable_sensor_noise
                    else SensorModel.measurement_function(state)
                )

            if observer is not None:
                nav_state = np.asarray(observer(t, measurement, u_prev), dtype=float)
            else:
                nav_state = state.copy()

            # --- Guidance/Control: produce a command, saturate to actuator limits ---
            u = self.dynamics.clip_control(np.asarray(controller(t, nav_state), dtype=float))

            # --- Log this instant ---
            t_hist.append(t)
            s_hist.append(state.copy())
            u_hist.append(u.copy())
            m_hist.append(mass)
            if log_meas:
                z_hist.append(measurement)
                e_hist.append(nav_state.copy())

            # --- Terminate at ground contact (evaluated before stepping through it) ---
            if state[Y] <= self.ground and k > 0:
                reason = "landed"
                break
            if not np.all(np.isfinite(state)) or abs(state[X]) > 1e6 or state[Y] > 1e6:
                reason = "diverged"
                break

            # --- Integrate true dynamics one step, then inject process noise ---
            state, mass = self.dynamics.step(
                state, u, self.dt, mass, integrator=self.integrator
            )
            if enable_process_noise and np.any(self._process_std > 0):
                # Brownian disturbance: per-step perturbation ~ N(0, (sigma*sqrt(dt))^2).
                state = state + self._process_std * np.sqrt(self.dt) * self.rng.normal(
                    size=STATE_DIM
                )
            u_prev = u

        return SimulationResult(
            t=np.array(t_hist),
            states=np.array(s_hist),
            controls=np.array(u_hist),
            masses=np.array(m_hist),
            measurements=np.array(z_hist) if z_hist else None,
            estimates=np.array(e_hist) if e_hist else None,
            terminated_reason=reason,
        )


def evaluate_landing(result: SimulationResult, cfg: dict) -> dict:
    """Score a run against the soft-landing gates in ``cfg['landing']``.

    Returns a dict with the final touchdown metrics and a boolean ``success``.
    """
    gates = cfg["landing"]
    s = result.final_state
    speed = float(np.hypot(s[VX], s[VY]))
    tilt = abs(float(s[THETA]))
    spin = abs(float(s[OMEGA]))
    offset = abs(float(s[X]))

    success = (
        result.terminated_reason == "landed"
        and speed <= gates["max_touchdown_speed"]
        and tilt <= gates["max_touchdown_tilt"]
        and spin <= gates["max_touchdown_omega"]
        and offset <= gates["max_horizontal_offset"]
    )
    return {
        "success": bool(success),
        "reason": result.terminated_reason,
        "touchdown_speed": speed,
        "touchdown_tilt": tilt,
        "touchdown_omega": spin,
        "horizontal_offset": offset,
        "propellant_used": result.propellant_used,
    }
