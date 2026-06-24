"""Physics sanity checks for the dynamics + simulator infrastructure.

These exercise ONLY the implemented infrastructure (no control/estimation stubs), so they must
pass on a fresh clone. If one fails, the physics is wrong — fix it before building on top.
"""

import numpy as np
import pytest

from src.config import load_config
from src.dynamics import (
    Rocket2DDynamics, VehicleParams, make_dynamics,
    X, VX, Y, VY, THETA, OMEGA, THRUST, GIMBAL,
)
from src.simulator import Simulator, SensorModel, evaluate_landing


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def dyn(cfg):
    return make_dynamics(cfg, dim=2)


def test_zero_thrust_is_free_fall(dyn):
    """No thrust => purely gravitational acceleration, no horizontal or angular accel."""
    state = np.array([10.0, 5.0, 500.0, -20.0, 0.1, 0.0])
    ds = dyn.derivatives(state, np.array([0.0, 0.0]), mass=dyn.params.initial_mass)
    assert ds[VX] == pytest.approx(0.0)
    assert ds[VY] == pytest.approx(-dyn.params.gravity)
    assert ds[OMEGA] == pytest.approx(0.0)
    # Kinematic rows just echo the velocities.
    assert ds[X] == pytest.approx(state[VX])
    assert ds[Y] == pytest.approx(state[VY])


def test_hover_thrust_balances_gravity(dyn):
    """Upright, thrust = m*g, zero gimbal => zero net acceleration."""
    m = dyn.params.initial_mass
    state = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    control = np.array([m * dyn.params.gravity, 0.0])
    ds = dyn.derivatives(state, control, mass=m)
    assert ds[VX] == pytest.approx(0.0, abs=1e-9)
    assert ds[VY] == pytest.approx(0.0, abs=1e-9)
    assert ds[OMEGA] == pytest.approx(0.0, abs=1e-9)


def test_free_fall_integration_matches_analytic(dyn):
    """RK4 of constant-gravity free fall is exact: y = y0 + vy0 t - 0.5 g t^2."""
    g = dyn.params.gravity
    state = np.array([0.0, 0.0, 1000.0, 0.0, 0.0, 0.0])
    mass = dyn.params.initial_mass
    dt, T = 0.02, 5.0
    n = int(T / dt)
    for _ in range(n):
        state, mass = dyn.step(state, np.array([0.0, 0.0]), dt, mass)
    t = n * dt
    assert state[VY] == pytest.approx(-g * t, rel=1e-6)
    assert state[Y] == pytest.approx(1000.0 - 0.5 * g * t**2, rel=1e-6)


def test_mass_depletes_with_thrust(dyn):
    """ṁ = -T / (Isp g0); mass should drop by ~that over a step, and not below dry mass."""
    p = dyn.params
    thrust = p.max_thrust
    expected_mdot = -thrust / (p.isp * p.g0)
    state = np.array([0.0, 0.0, 500.0, 0.0, 0.0, 0.0])
    m0 = p.initial_mass
    dt = 0.1
    _, m1 = dyn.step(state, np.array([thrust, 0.0]), dt, m0)
    assert (m1 - m0) == pytest.approx(expected_mdot * dt, rel=1e-3)
    assert m1 >= p.dry_mass


def test_gimbal_produces_torque(dyn):
    """A nonzero gimbal angle creates angular acceleration (tau = -T l_cm sin(delta))."""
    p = dyn.params
    state = np.zeros(6)
    thrust = 0.5 * p.max_thrust
    delta = 0.1
    ds = dyn.derivatives(state, np.array([thrust, delta]), mass=p.initial_mass)
    expected_alpha = -thrust * p.l_cm * np.sin(delta) / p.inertia
    assert ds[OMEGA] == pytest.approx(expected_alpha)
    assert ds[OMEGA] != 0.0


def test_tilt_produces_horizontal_acceleration(dyn):
    """Tilting the vehicle points thrust off-vertical => horizontal acceleration appears."""
    p = dyn.params
    m = p.initial_mass
    theta = 0.2
    thrust = m * p.gravity
    state = np.array([0.0, 0.0, 100.0, 0.0, theta, 0.0])
    ds = dyn.derivatives(state, np.array([thrust, 0.0]), mass=m)
    assert ds[VX] == pytest.approx(-thrust * np.sin(theta) / m)


def test_control_saturation(dyn):
    """clip_control clamps thrust to [0, max] and gimbal to [-max, max]."""
    p = dyn.params
    u = dyn.clip_control(np.array([10 * p.max_thrust, 10 * p.max_gimbal_angle]))
    assert u[THRUST] == pytest.approx(p.max_thrust)
    assert u[GIMBAL] == pytest.approx(p.max_gimbal_angle)
    u2 = dyn.clip_control(np.array([-5.0, -10 * p.max_gimbal_angle]))
    assert u2[THRUST] == pytest.approx(0.0)
    assert u2[GIMBAL] == pytest.approx(-p.max_gimbal_angle)


def test_sensor_model_mean_and_shape(cfg):
    """Noise-free measurement function returns the right layout incl. nonlinear range row."""
    state = np.array([30.0, 1.0, 40.0, -2.0, 0.05, 0.01])
    z = SensorModel.measurement_function(state)
    assert z.shape == (7,)
    assert z[6] == pytest.approx(np.hypot(30.0, 40.0))  # range to pad = 50
    R = SensorModel(cfg["noise"]["sensor_std"], np.random.default_rng(0)).noise_cov()
    assert R.shape == (7, 7)


def test_simulator_runs_closed_loop_smoke(cfg, dyn):
    """A trivial hover controller drives a full episode without touching any stub."""
    sim = Simulator(dyn, cfg, rng=np.random.default_rng(0))
    m = dyn.params.initial_mass

    def hover_controller(t, state):
        return np.array([m * dyn.params.gravity * 1.05, 0.0])  # slightly > hover => slows descent

    result = sim.run(hover_controller, enable_process_noise=False, enable_sensor_noise=False)
    assert result.states.shape[1] == 6
    assert result.controls.shape[1] == 2
    assert len(result.t) == len(result.states)
    assert result.terminated_reason in {"landed", "max_time", "diverged"}
    report = evaluate_landing(result, cfg)
    assert set(report) >= {"success", "touchdown_speed", "propellant_used"}
