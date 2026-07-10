# HANDOFF.md

## 2026-07-10 docs quality bar

Changed:
- Added a concise professional-environment quality bar to `SPEC.md`.
- Refreshed `README.md` so it reflects the implemented Day 1 harness and the
  current model-eval smoke command.
- Clarified in README that the core sim is event-driven and analytic, not a 3D
  drone simulator.

Verified:
- Documentation-only change; checked the edited sections with `sed`.

Broken / not done:
- No code changes in this pass.

Next action:
- Keep Day 2 implementation aligned with the new SPEC quality bar.

## 2026-07-10

Changed:
- Implemented the PLAN.md Day 1 skeleton in the top-level `uav_operator.py`.
- Added `UAVOperatorEnv`, a `vf.MultiTurnEnv` subclass with
  `load_environment(**kwargs) -> vf.Environment` as the entry point.
- Added serializable dataclasses for `Site`, `Waypoint`, `Mission`,
  `Aircraft`, and `SimState`.
- Added five hardcoded T0 Bay Area scenarios with generated briefings.
- Added the Day 1 tool set: `get_telemetry`, `file_flight_plan`, and
  `command_rtl`.
- Added analytic straight-line autopilot segment execution with flat wind,
  simulated tool latency, battery burn, landing reserve burn, and invalid
  action time penalties.
- Added full simulator snapshots to `state["sim_log"]` at briefing and after
  each environment/tool event.
- Added mission-value-only reward computed from simulator state, not model
  text.
- Added terminal handling for landed completed/failed missions, lost aircraft,
  and sim-time cap.
- Added focused Day 1 tests in `tests/test_day1_env.py` for environment
  loading, successful mission flow, invalid-action penalties, JSON-serializable
  sim logs, and deterministic replay.
- Added `docs/HACKS.md` with the first reward-hacking note for
  `mission_value` success-claiming.

Verified:
- `uv run python -c "import uav_operator; ..."` loads the environment, builds
  the 5-example eval dataset, and exposes the 3 tool definitions.
- `uv run ruff check .` passes.
- `uv run --with pytest pytest -q` passes: 4 tests.
- A synthetic Verifiers smoke path initializes state, executes tool calls,
  scores the rubric, and produces reward `1.0` with a populated `sim_log`.
- `uv run vf-eval --help` works, confirming the eval CLI is installed.

Broken / not done:
- I did not run a live `vf-eval uav-operator -m <model> -n 2 -r 1` because it
  requires configured model/provider credentials.
- Day 2 geography/energy scope is not implemented yet: static world JSON,
  airspace polygons, obstacle raster, LOW_BATT_RTL, and GEOFENCE_HOLD remain
  next.
- Rollout JSON export/rendering scripts are still not present.

Next action:
- Run a real `vf-eval` against an available model/provider.
- Then implement PLAN.md Day 2: world data builder, richer geometry/energy
  tests, and the first two autopilot failsafes.

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
