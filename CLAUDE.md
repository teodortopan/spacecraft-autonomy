# CLAUDE.md

Guidance for Claude Code (and any AI agent) working in this repository.

## What this project is

This repo (`spacecraft-autonomy`) is the **GNC core** of a planned larger autonomous-spacecraft
flight-software stack. The current code is a 2D→3D **powered-descent landing simulator** that
demonstrates the full GNC loop — **estimate → guide → control → actuate** — validated with
Monte-Carlo dispersions. It is a **learning project** for the repo owner (Teodor). The point is for the owner to implement the
core estimation/control/guidance math *by hand* so he can defend every line in interviews.

## The contract (read before writing any code)

- **Agent builds infrastructure.** Dynamics, simulator, visualization, Monte-Carlo, tests,
  config, scripts, docs — implement these fully.
- **Owner implements the algorithms.** The bodies of these five files are the owner's learning
  exercise. **Leave them STUBBED:** signature + docstring + the governing equations as comments
  + `# TODO(owner)`. Do **not** fill them in, even if asked to "just make it work," unless the
  owner explicitly overrides that file by name.
  - `src/control/pid.py`
  - `src/control/lqr.py`
  - `src/control/guidance.py`
  - `src/estimation/kalman.py`
  - `src/estimation/ekf.py`
- When a stub is unfilled, the wiring (scripts) must **fail gracefully** with a clear message
  pointing at the `TODO(owner)`, not raise a raw traceback.

## Git / commit rules

- **Do NOT co-author commits.** Do not add a `Co-Authored-By:` trailer, "Generated with Claude
  Code", or any similar attribution to commit messages or PR bodies. Commits should read as the
  owner's own work.
- Keep commit messages plain and conventional (e.g. `feat: add 2D dynamics`, `test: hover
  equilibrium check`). No AI/tool attribution lines anywhere.

## Commands

```bash
python3 -m venv rocket-env && source rocket-env/bin/activate
pip install -r requirements.txt

python -m pytest tests/                 # physics sanity checks (must pass)
python scripts/run_week1_lqr.py         # each week's entry point
```

## Layout

- `src/dynamics.py` — equations of motion (INFRA, implemented).
- `src/simulator.py` — RK4 integration + noise injection + history logging (INFRA).
- `src/viz/plots.py` — trajectory plots + landing animation (INFRA).
- `src/montecarlo.py` — dispersion runner (INFRA).
- `src/control/`, `src/estimation/` — STUBS (owner implements).
- `scripts/run_weekN_*.py` — weekly wiring that imports the stubs.
- `config/vehicle.yaml` — vehicle + sim + noise + Monte-Carlo parameters.
- `LEARNING.md` — the study/dependency chain and per-week tracker.
