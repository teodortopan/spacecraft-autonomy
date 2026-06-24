"""PID control — STUB for the owner to implement (Week 1 warm-up).

A PID controller drives a tracking error e(t) = setpoint - measurement to zero:

    u(t) = Kp * e(t)  +  Ki * ∫ e(τ) dτ  +  Kd * de(t)/dt

In discrete time with step dt:
    integral   += e * dt
    derivative  = (e - e_prev) / dt
    u           = Kp*e + Ki*integral + Kd*derivative

For the rocket you will typically compose a few PID loops (e.g. an outer loop on altitude /
vertical speed commanding thrust, and an attitude loop commanding gimbal). Watch for integral
wind-up when the actuator saturates.

This is a warm-up before LQR — get a feel for feedback, then move to ``lqr.py``.
"""

from __future__ import annotations


class PID:
    """Single-channel PID controller."""

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limits: tuple[float, float] | None = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.reset()

    def reset(self) -> None:
        """Clear the integrator and derivative memory."""
        self._integral = 0.0
        self._prev_error = None

    def update(self, error: float, dt: float) -> float:
        """Return the control output for the current ``error`` over timestep ``dt``.

        Implement the discrete PID law above. Remember to:
          - accumulate the integral (consider anti-wind-up when clamping to output_limits),
          - compute the derivative from the previous error (handle the first call),
          - clamp to output_limits if provided.
        """
        # Governing law:  u = Kp*e + Ki*∫e dt + Kd*de/dt   (see module docstring)
        # TODO(owner): implement the PID update.
        raise NotImplementedError(
            "PID.update not implemented — see src/control/pid.py TODO(owner)"
        )
