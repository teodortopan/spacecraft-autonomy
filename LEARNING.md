# LEARNING.md — study plan & progress tracker

The project IS the curriculum. Watch/derive one concept, then immediately implement it.
Golden ratio: ~1 part watch/derive : ~3 parts build. Never more than 1–2 days of pure theory
before building again.

**The division of labor:** the simulation infrastructure is already built. *You* implement the
five core algorithm files by hand — that's the whole point. Each is stubbed with its governing
equations in comments and a `# TODO(owner)` marking where your code goes.

## Dependency chain (study in this order)

Each topic unlocks the next:

1. **Linear algebra refresh** (light, ongoing) — matrix mult, transpose/inverse, least squares,
   eigenvalues, positive-definiteness. → powers everything.
2. **Dynamics & state-space** (ẋ = Ax + Bu) — representing a physical system; stability via
   eigenvalues. → Week 1. *Files: `src/dynamics.py` (already built — read it).*
3. **Classical control (PID)** — first working controller; feedback intuition. → Week 1.
   *File: `src/control/pid.py`.*
4. **State-space control (LQR)** — optimal linear control; the real controller. → Week 1.
   *File: `src/control/lqr.py`.*
5. **Probability / Gaussians refresh** — mean, covariance, conditioning; the language of
   estimation. → Week 2.
6. **Kalman Filter → EKF** — state estimation / sensor fusion; needs Jacobians. → Week 2.
   *Files: `src/estimation/kalman.py`, `src/estimation/ekf.py`.*
7. **Convex optimization (+ CVXPY)** — the tool for fuel-optimal guidance. → Week 3.
8. **Powered-descent guidance (convexification)** — the SpaceX landing algorithm class. → Week 3.
   *File: `src/control/guidance.py`.*
9. **Monte-Carlo validation** — how aerospace proves robustness. → Week 4.
   *File: `src/montecarlo.py` (already built — read + use it).*

**Put your new learning energy on #5 (probability) and #6 (Kalman/EKF)** — those are the likely
gaps. Calculus + linear algebra are just dusting-off.

## Resources

**Primary (use every week):**
- Steve Brunton — *Control Bootcamp* playlist (dynamics, state-space, PID, LQR) → Weeks 1, 3.
- Steve Brunton — *Kalman Filter* playlist (estimation, KF, EKF) → Week 2.
- 3Blue1Brown — *Essence of Linear Algebra* (intuition refresh, as needed).
- CVXPY docs & tutorial — cvxpy.org → Week 3.

**Reference (dip in when stuck):**
- Russ Tedrake — *Underactuated Robotics* (underactuated.csail.mit.edu) — rigorous version.
- Brian Douglas — *Control System Lectures* (YouTube) — best control intuition.
- Gilbert Strang — MIT 18.06 Linear Algebra (OCW) — rigorous linear algebra.
- Boyd & Vandenberghe — *Convex Optimization* (free PDF) — Week 3 reference.
- Thrun, Burgard, Fox — *Probabilistic Robotics*, Ch. 3 — Kalman/EKF depth.

**The key paper (Week 3):**
- Açıkmeşe & Ploen (2007), "Convex Programming Approach to Powered Descent Guidance for Mars
  Landing." The lossless-convexification result — skim for problem formulation + constraints.

## Weekly tracker

### Week 1 — Make it land (control)
- [ ] Study: Brunton Control Bootcamp #1–4
- [ ] Re-derive ẋ = Ax + Bu and the eigenvalue stability check on paper
- [ ] Implement `src/control/pid.py`
- [ ] Implement `src/control/lqr.py` (`compute_lqr_gain(A, B, Q, R)`)
- [ ] `python scripts/run_week1_lqr.py` → 2D rocket lands softly under LQR on true state
- [ ] Repo public + pinned

### Week 2 — Make it land blind (estimation)  ← mid-month on-track marker
- [ ] Study: Brunton Kalman Filter series
- [ ] Derive the EKF Jacobians (F, H) by hand once
- [ ] Implement `src/estimation/kalman.py`
- [ ] Implement `src/estimation/ekf.py` (predict + update) — fuse IMU + position + range
- [ ] `python scripts/run_week2_ekf.py` → lands on EKF-estimated state; estimate-vs-truth plots

### Week 3 — Make it optimal (the SpaceX algorithm)
- [ ] Study: CVXPY tutorial + skim Açıkmeşe & Ploen
- [ ] Implement `src/control/guidance.py` (convex powered descent in CVXPY)
- [ ] `python scripts/run_week3_guidance.py` → fuel-optimal trajectory tracked; fuel vs LQR-only

### Week 4 — Make it real (3D + robustness + ship)
- [ ] Extend dynamics to 3D
- [ ] `python scripts/run_week4_montecarlo.py` → dispersion success rate + clean visuals
- [ ] Write + publish the blog post; link from resume / portfolio / LinkedIn

## After Week 4 → the quadrotor

The same GNC core transfers to a quadrotor in a new repo, where **C++/ROS2** and an **ML layer**
(learned dynamics or an RL/imitation policy benchmarked vs the classical controller) enter. That
project is what reaches both autonomy companies (C++ flight software) and physical-AI labs.
