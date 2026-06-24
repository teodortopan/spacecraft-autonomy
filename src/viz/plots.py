"""Plots + landing animation. INFRASTRUCTURE — fully implemented.

All functions return a Matplotlib figure (or animation) and never call ``plt.show()`` themselves,
so they work headless. Use :func:`save_figure` / pass ``save_path`` to write files, or call
``plt.show()`` from a script.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg") if not os.environ.get("DISPLAY") and os.name != "nt" else None
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..dynamics import X, VX, Y, VY, THETA, OMEGA
from ..simulator import (
    SimulationResult,
    M_X, M_Y, M_VX, M_VY, M_THETA, M_OMEGA, M_RANGE,
)

_STATE_LABELS = ["x (m)", "vx (m/s)", "y (m)", "vy (m/s)", "theta (rad)", "omega (rad/s)"]


def save_figure(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def plot_trajectory(result: SimulationResult, title: str = "Trajectory"):
    """Ground-track (x vs y) plus altitude and speed time histories."""
    s, t = result.states, result.t
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(s[:, X], s[:, Y], color="tab:blue")
    axes[0].scatter([s[0, X]], [s[0, Y]], c="green", label="start", zorder=5)
    axes[0].scatter([s[-1, X]], [s[-1, Y]], c="red", label="touchdown", zorder=5)
    axes[0].axhline(0.0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("altitude y (m)")
    axes[0].set_title("Ground track"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(t, s[:, Y], color="tab:blue")
    axes[1].axhline(0.0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("altitude y (m)")
    axes[1].set_title("Altitude"); axes[1].grid(alpha=0.3)

    speed = np.hypot(s[:, VX], s[:, VY])
    axes[2].plot(t, speed, color="tab:purple")
    axes[2].set_xlabel("t (s)"); axes[2].set_ylabel("speed (m/s)")
    axes[2].set_title("Speed"); axes[2].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_controls(result: SimulationResult, max_thrust: Optional[float] = None):
    """Thrust and gimbal command histories."""
    t, u = result.t, result.controls
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(t, u[:, 0], color="tab:red")
    if max_thrust:
        axes[0].axhline(max_thrust, color="k", ls="--", lw=0.8, label="max")
        axes[0].legend()
    axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("thrust (N)")
    axes[0].set_title("Thrust"); axes[0].grid(alpha=0.3)

    axes[1].plot(t, np.degrees(u[:, 1]), color="tab:orange")
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("gimbal (deg)")
    axes[1].set_title("Gimbal"); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_estimate_vs_truth(result: SimulationResult):
    """Overlay EKF estimate on the true state for each of the 6 states (Week 2)."""
    if result.estimates is None:
        raise ValueError("result has no estimates — run with an observer to compare")
    t, s, e = result.t, result.states, result.estimates
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for i, ax in enumerate(axes.ravel()):
        ax.plot(t, s[:, i], color="k", lw=1.5, label="truth")
        ax.plot(t, e[:, i], color="tab:red", lw=1.2, ls="--", label="estimate")
        ax.set_xlabel("t (s)"); ax.set_ylabel(_STATE_LABELS[i]); ax.grid(alpha=0.3)
        if i == 0:
            ax.legend()
    fig.suptitle("EKF estimate vs. truth")
    fig.tight_layout()
    return fig


def animate_landing(
    result: SimulationResult,
    vehicle_length: float = 30.0,
    save_path: Optional[str] = None,
    fps: int = 30,
):
    """Matplotlib animation of the descent. Draws the rocket as an oriented body with a flame
    sized by thrust. If ``save_path`` ends in .mp4/.gif it is written (needs ffmpeg/pillow)."""
    s, u = result.states, result.controls
    L = vehicle_length

    xs, ys = s[:, X], s[:, Y]
    pad = max(50.0, 0.15 * (np.max(ys) - np.min(ys) + 1.0))
    xmin, xmax = np.min(xs) - pad, np.max(xs) + pad
    ymin, ymax = min(-0.05 * np.max(ys), -pad * 0.2), np.max(ys) + pad

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.axhline(0.0, color="saddlebrown", lw=2)
    ax.scatter([0], [0], marker="x", c="k", s=80, zorder=3, label="pad")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("Powered descent")

    (path_line,) = ax.plot([], [], color="tab:blue", lw=1, alpha=0.6)
    (body_line,) = ax.plot([], [], color="k", lw=4, solid_capstyle="round")
    (flame_line,) = ax.plot([], [], color="tab:orange", lw=3, solid_capstyle="round")
    max_thrust = max(np.max(u[:, 0]), 1e-9)

    def body_endpoints(k):
        theta = s[k, THETA]
        up = np.array([-np.sin(theta), np.cos(theta)])  # nose direction
        center = np.array([xs[k], ys[k]])
        nose = center + 0.5 * L * up
        tail = center - 0.5 * L * up
        return nose, tail, up

    def init():
        for ln in (path_line, body_line, flame_line):
            ln.set_data([], [])
        return path_line, body_line, flame_line

    def update(k):
        nose, tail, up = body_endpoints(k)
        body_line.set_data([tail[0], nose[0]], [tail[1], nose[1]])
        path_line.set_data(xs[: k + 1], ys[: k + 1])
        # Flame: from the tail, opposite the gimballed thrust, length ~ throttle.
        theta, delta = s[k, THETA], u[k, 1]
        thrust_dir = np.array([-np.sin(theta + delta), np.cos(theta + delta)])
        flame_len = 0.6 * L * (u[k, 0] / max_thrust)
        flame_tip = tail - flame_len * thrust_dir
        flame_line.set_data([tail[0], flame_tip[0]], [tail[1], flame_tip[1]])
        return path_line, body_line, flame_line

    anim = FuncAnimation(
        fig, update, frames=len(s), init_func=init, blit=True, interval=1000 / fps
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, fps=fps)
    return anim
