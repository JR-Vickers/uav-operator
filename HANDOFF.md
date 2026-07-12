# HANDOFF.md

## 2026-07-12 Day 5 complete: frontier calibration curve

Changed:
- Built the primary Day 5 curve from four saved, zero-provider-error
  GPT-4.1-nano dev runs and committed the PNG plus a machine-readable summary.
- Added per-tier 95% confidence intervals, rollout-count annotations, mission
  completion counts, run IDs, and saved-results provenance.
- Recorded the curve and interpretation in README; marked Day 5 complete in
  PLAN. Laguna is intentionally excluded because its T0/T1 coverage is
  incomplete and its T2/T3 runs include provider errors.
- Reworked `scripts/plot_day5_scores.py` to derive statistics directly from
  saved `vf-eval` metadata/results rather than accepting hand-entered means.

Calibration:
- T0 `f32c2f03`: 0.899 mean reward, 95% CI [0.833, 0.965], 10/10 complete.
- T1 `f2cef9f9`: 0.134 mean reward, 95% CI [-0.020, 0.288], 7/30 complete.
- T2 `ea07497e`: -0.137 mean reward, 95% CI [-0.154, -0.119], 0/30 complete.
- T3 `ef609452`: -0.199 mean reward, 95% CI [-0.231, -0.167], 0/30 complete.

Verified:
- `uv run ruff check .` passes.
- `uv run pytest -q` passes: 21 tests.
- `git diff --check` passes.

Next action:
- Begin Day 6 red-team round 1 and freeze the rollout-JSON schema before
  renderer work.

## 2026-07-12 ground-state loop pricing

Changed:
- Added a serialized ground no-progress streak covering telemetry, weather,
  airspace, mission/site reads, and holds while a mission remains pending.
- After three consecutive no-progress calls, the console returns a structured
  warning and charges deterministic latency escalating from 30 to 120 seconds.
  Mission/alert-resolving actions reset the streak; configuration churn does
  not, and the simulator never chooses an action for the model.
- Added exact warning, latency, hold, evasion, and reset regression coverage;
  updated SPEC and `docs/HACKS.md`.

Verified:
- `uv run ruff check .` passes.
- `uv run pytest -q` passes: 21 tests.
- `git diff --check` passes.

Next action:
- Re-score saved T2 run `8a5e4f78` under the canonical hard-safety fix.

## 2026-07-12 TFR feasibility admission fix

Changed:
- Fixed the analytic feasibility solver to orient rectangular TFR detours for
  the direction of travel and to validate detours on both the outbound and
  recovery legs.
- Added a regression proving that the 15-scenario T2 dev calibration set
  retains all 9 seeded TFR compositions and that every scenario remains
  solver-feasible.

Verified:
- `uv run ruff check .` passes.
- `uv run pytest -q` passes: 20 tests.
- T2 dev retains 9 TFR scenarios; T3 dev retains 11; both have zero infeasible
  scenarios across their 15 calibration seeds.

Next action:
- Add the ground-state telemetry/hold loop regression, then re-score saved T2
  run `8a5e4f78` under the canonical hard-safety fix.

## 2026-07-12 T2 calibration reward audit

Changed:
- Canonicalized overlapping battery-depletion terminal evidence so one
  physical aircraft loss receives one `-5.0` hard-safety penalty rather than
  separate penalties for the detailed violation, lost status, and critical
  battery condition.
- Added a regression reproducing the overlap observed in saved Laguna T2 run
  `8a5e4f78`, and documented the failure mode in `docs/HACKS.md`.
- Made fresh-clone setup reproducible from `pyproject.toml` and the committed
  `uv.lock`: the wheel keeps only `numpy` and `verifiers` at runtime, while
  pytest, Ruff, msgpack, and matplotlib live in the development group.

Verified:
- `uv run ruff check .` passes.
- `uv run pytest -q` passes: 19 tests.
- `git diff --check` passes.
- `uv sync --locked --all-groups` succeeds, and the built wheel declares only
  `numpy` and `verifiers` as runtime dependencies.

Next action:
- Investigate the T2 dev split's missing `TFR_POPUP` events and decide how to
  make repeated ground-state read/hold loops terminate more informatively.

## 2026-07-10 Day 5 T2/T3 generation + calibration

Changed:
- Replaced the alternating Day 4 dataset with generated T0-T3 scenarios and
  composed T2/T3 event sets drawn only from the active Day 4 taxonomy.
- Added a deterministic analytic feasibility check for route/recovery candidates
  and a conservative TFR-to-wind fallback that preserves composed-event count.
- Added non-overlapping, stratified 300 train / 60 dev / 60 final-eval splits;
  `load_environment()` now uses train rows for training and held eval rows for
  evaluation.
- Added repeat pricing for known-geofence filings and invalid/inactive failsafe
  overrides, with regression tests.
- Added `scripts/day5_calibration.py` and `scripts/plot_day5_scores.py`; baseline
  CLI now supports `--tier all`.

Verified:
- `uv run ruff check .` passes.
- `uv run --with pytest pytest -q` passes: 18 tests.
- `uv run python scripts/day5_calibration.py --episodes 2 --output /tmp/day5_calibration.json` passes.

Broken / not done:
- The Day 4 live-frontier postmortem and Day 5 final live calibration with
  `poolside/laguna-m.1` and `gpt-5-nano` have not been run in this session.
- No final frontier plot is committed until those saved evaluation results exist.

Next action:
- Run the two-model, two-rollout-per-scenario final evaluation across T0-T3,
  inspect the T1 postmortem, and calibrate only against the 60-row dev split.

## 2026-07-10 Day 4 reward v0 + first contact

Changed:
- Added seeded Day 4 scenario generation with default 20-example mixed T0/T1
  eval rows and `tier` taskset selection.
- Added serializable event state to `SimState`: event queue, active and
  acknowledged events, closed recovery sites, procedure violations, hard-safety
  violations, and generated scenario par.
- Implemented T1 interrupts for `WIND_SHIFT`, `TFR_POPUP`, `BATT_DEGRADE`, and
  `SITE_CLOSED`; mutating tools pause on active event alerts until the operator
  acknowledges them.
- Added dynamic TFRs as active airspace zones, closed-site avoidance for nearest
  recovery selection, and event effects on wind/battery/site availability.
- Replaced the mission-only rubric with Day 4 reward components:
  `mission_value`, `hard_safety`, `margin_policy`, `procedure`, and
  `efficiency`, all computed from saved `sim_log` snapshots.
- Added `scripts/baselines.py` with rulebook and reckless policies that drive
  the environment through the same tool-call path as model rollouts.
- Moved the local `vf-eval` summary tweak into the repo as
  `patches/verifiers_eval_utils_run_results.patch` plus
  `scripts/apply_vf_eval_summary_patch.py` so the run-id / results-path footer
  can be re-applied after a fresh install.
- Added Day 4 tests for deterministic T0/T1 dataset generation, event
  interrupt logging, prose-inert reward behavior, and rulebook > reckless.
- Updated README, SPEC reward notes, and `docs/HACKS.md`.

Verified:
- `uv run ruff check .` passes.
- `uv run --with pytest pytest -q` passes: 15 tests.
- `uv run python scripts/baselines.py --policy both --episodes 20 --tier mixed_day4`
  passes the local gate: rulebook avg reward ≈ 0.7485, reckless avg reward
  ≈ 0.6682.
- Started live first-contact smoke:
  `uv run vf-eval uav-operator -m poolside/laguna-m.1 -n 2 -r 1 --state-columns sim_state,sim_log`.
  The first rollout completed with reward 1.0, but the second rollout produced
  no further output after ~5 minutes and was interrupted. The command also
  reported no local `./configs/endpoints.toml` registry, so use the workspace
  Prime endpoint config or explicit `-b/-k` flags for the next live run.

Broken / not done:
- The Day 4 live frontier-model gate is not fully satisfied yet: no 20-episode
  Prime Inference run has been completed, and the interrupted 2-rollout smoke
  did not produce a saved output path.
- `TRAFFIC_ADVISORY` and `PAYLOAD_ISSUE` remain cut per the Day 4 fallback.
- Dynamic TFRs are enforced as geofence holds/procedure penalties; explicit
  post-incursion hard-safety logging is still future work once route-through
  authorization semantics are richer.

Next action:
- Re-run `vf-eval` with explicit endpoint config and a shorter generation cap
  if needed, then inspect whether a T1 rollout contains a non-rulebook action.
- If frontier rollouts remain boring, increase T1 decision density before
  adding T2/T3 world scope.

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
