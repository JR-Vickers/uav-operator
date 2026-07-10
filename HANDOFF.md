# HANDOFF.md

## 2026-07-10 Day 3 wind + full console

Changed:
- Added seeded per-episode wind fields with base flow, 2-4 translating
  perturbation blobs, optional moving gust fronts, forecast ETA/error, and
  altitude shear.
- Switched execution noise to a serialized NumPy bit-generator state so wind
  generation and segment multipliers stay on one deterministic RNG stream.
- Updated route validation, segment timing, and energy burn to use the same
  `wind_at(...)` helper exposed for tests/renderer use.
- Expanded the operator console to the full SPEC §3.2 tool set:
  weather/airspace/mission/sites reads plus route amendment, altitude/speed
  changes, hold/resume, land now, release payload, abort, failsafe override,
  and alert acknowledgement.
- Added structured sim-time costs for read, routine action, high-impact action,
  and invalid tool calls.
- Added override and acknowledgement state to simulator snapshots without
  adding any text-parsing reward path.
- Added `assets/rollouts/day3_scripted_rollout.json` as a deterministic local
  rollout artifact generated from the environment itself.
- Updated README status and taskset config notes for Day 3.

Verified:
- `uv run ruff check .` passes.
- `uv run --with pytest pytest -q` passes: 10 tests.

Broken / not done:
- I did not run live `vf-eval uav-operator -m <model> -n 2 -r 1`; it still
  depends on configured model/provider credentials.
- Reward is still Day 1 mission-value-only by design; full SPEC §5 rubric,
  baselines, and T0-T1 generator remain Day 4.

Next action:
- Implement PLAN.md Day 4: full reward v0, rulebook/reckless baselines,
  T0-T1 scenario generator, determinism check in CI shape, and first model
  contact through Prime Inference if credentials are available.

Troubleshooting update:
- Fixed `vf-eval --state-columns sim_state,sim_log` response serialization
  failure by storing NumPy PCG `rng_state.state.state` and `rng_state.state.inc`
  as decimal strings in persisted sim state/logs, then converting them back to
  ints only when assigning to NumPy.
- Added a msgpack regression test for saved `sim_state` / `sim_log` columns.
- User reran live eval successfully:
  `uv run vf-eval uav-operator -m poolside/laguna-m.1 -n 5 -r 1 --save-results --state-columns sim_state,sim_log`.
  Output path `outputs/evals/uav-operator--poolside--laguna-m.1/6fee7f55`
  had avg reward 0.8, avg turns 6.8, and avg error 0.0.
- The single zero-reward rollout was a valid sim/training failure: low
  commanded airspeed into strong headwind caused near-zero groundspeed,
  battery depletion, and `LOW_BATT_RTL`; not a pipeline failure.

## 2026-07-10 Day 2 geography + energy

Changed:
- Added embedded static Bay Area world data in `uav_operator.py`: expanded
  recovery sites, target points, simplified airspace polygons, and coarse
  obstacle cells.
- Added `scripts/build_world.py` and generated `data/world.json` for dev and
  renderer inspection while keeping the runtime wheel lean.
- Added bearing, polygon/route intersection, min-safe-altitude, wind-component,
  groundspeed, and calibrated energy helpers.
- Replaced Day 1 segment execution with deterministic analytic segment metrics
  including groundspeed, energy, and seeded execution multipliers.
- Added route validation for bounds/envelope errors, min-safe-alt warnings,
  airspace advisories, and unauthorized-airspace prediction.
- Implemented simulator-owned `GEOFENCE_HOLD` and `LOW_BATT_RTL` failsafes
  with structured alerts logged into `sim_state` / `sim_log`.
- Updated tests for Day 2 geometry, energy calibration, geofence hold,
  low-battery RTL, and deterministic replay.
- Updated `README.md` status and static-world note.

Verified:
- `uv run ruff check .` passes.
- `uv run --with pytest pytest -q` passes: 7 tests.

Broken / not done:
- Plain `uv run pytest -q` does not work because `pytest` is not installed in
  the base uv environment; use `uv run --with pytest pytest -q`.
- I did not run live `vf-eval uav-operator -m <model> -n 2 -r 1`; it still
  depends on configured model/provider credentials.
- Day 3 scope remains open: seeded wind field, full console tool set,
  sim-time tool costs across new tools, and determinism CI expansion.

Next action:
- Implement PLAN.md Day 3: seeded wind perturbations/gust front, altitude
  shear, the full SPEC §3.2 operator console, validation warnings, and the
  formal determinism CI test.

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
