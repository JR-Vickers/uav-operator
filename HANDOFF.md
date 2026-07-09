# HANDOFF.md

## 2026-07-09

Changed:
- Kept the single-file top-level `uav_operator.py` module and aligned the repo
  instructions with that simpler layout.
- Kept `load_environment(**kwargs) -> vf.Environment` as the entry point,
  currently raising `NotImplementedError` until the Day 1 loop exists.
- Updated `pyproject.toml` metadata, tags, dependencies, and hatch include list
  to ship only `uav_operator.py` plus `pyproject.toml`.
- Replaced the placeholder README with the current environment pitch, task
  description, planned config fields, and planned rubric metrics.

Verified:
- `uv run python -c "import uav_operator; ..."` imports the top-level module
  from `uav_operator.py`.
- `uv` rebuilds package metadata and reports the updated project summary.

Broken / not done:
- The environment itself is still not implemented; `load_environment()` raises
  intentionally.
- `vf-eval` / `prime eval run` cannot be expected to pass until the Day 1
  `vf.MultiTurnEnv` skeleton is built.

Next action:
- Implement PLAN.md Day 1: `SimState`, a minimal `vf.MultiTurnEnv`, three
  tools (`get_telemetry`, `file_flight_plan`, `command_rtl`), five T0
  scenarios, straight-line autopilot, mission-value reward, and rollout logging.
