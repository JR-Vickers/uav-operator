# HANDOFF.md

## 2026-07-15 Day 8 smoke stopped at step 0

Changed:
- Replaced the obsolete planned orchestrator/trainer/inference TOMLs and manual
  pod selection with one Hosted Training config at
  `configs/day8_laguna_t1_smoke.toml`.
- Configured a 50-step T1-only RL/GRPO smoke on
  `poolside/Laguna-XS-2.1`: batch 128, 8 rollouts/example, 96 maximum in flight,
  learning rate `3e-5`, LoRA alpha 32, 1,024 tokens, temperature 0.7, and
  thinking disabled. Hosted Training selects infrastructure.
- Configured a pre-training baseline and evaluation every 10 steps across all
  15 T1 dev rows, with two rollouts/example, temperature 0, 10-step
  checkpoints (retain two), and 10-step adapter uploads (retain three).
- Pinned `max_examples` in the Hosted arguments so public environment `0.1.1`
  exposes exactly 75 T1 train and 15 T1 dev rows; no environment republish is
  needed and final-eval rows remain excluded.
- Added `scripts/capture_day8_training.py` to archive metadata, metrics, reward
  distributions, sampled rollouts, token usage/cost, truncation, provider
  errors, checkpoints, logs, pricing, wallet, and Hub status in one JSON
  artifact.
- Documented that Laguna is a zero-credit pipeline-validation choice while its
  effective prices remain zero. At 33.4B total parameters it is not yet the
  project's final "small model." Preparation made Day 10 provisionally 300
  Laguna steps, but the stopped smoke now suspends that path.

Launch protocol satisfied:
- Preparation invoked no training command. The later launch followed an
  explicit user confirmation, renewed availability/pricing/wallet/Hub checks,
  and the CLI's interactive prompt without `--yes`.
- Monitoring enforced the documented stop rules; repeated `ModelError`
  failures stopped the run before step 1.

Verified:
- Prime CLI `0.6.16` accepts the TOML with its current Hosted Training schema;
  the parsed config preserves the pinned Hub version and all requested values.
- Ruff passes and all 40 tests pass, including exact TOML assertions, local
  instantiation of the 75-row T1 train and 15-row T1 dev variants, disjoint
  train/dev/final-eval seeds, and reporting-summary coverage.
- `uv build` succeeds for both sdist and wheel; `git diff --check` passes.
- Live preflight on 2026-07-15: Laguna is available and not at capacity;
  effective training, inference-input, and inference-output prices are each
  `$0.00/M` tokens. Personal wallet balance is `$57.9186`. Public Hub version
  `0.1.1` has quality action `SUCCESS` (job `sf8qph79xpnv6ctn9q34tlsn`).
- No Hosted Training launch command was invoked during the preparation commit.

Next action:
- Obtain the underlying platform/provider detail for the generic `ModelError`,
  then review and gate `configs/day8_laguna_t1_diagnostic.toml`: one step,
  batch 16, two rollouts/example, four in flight, eight train rows, and two dev
  rows. It preserves the 40-turn/1,024-token limits to isolate concurrency.
  Any retry requires a renewed live gate and explicit approval; the 150- and
  300-step runs are suspended.

Smoke outcome:
- User explicitly approved launch. The renewed gate remained healthy: Laguna
  available/not at capacity and free, wallet `$57.9186`, Hub action `SUCCESS`.
- Launched run `ed7ap9lbtm3lpy6pqeav7lrt` without `--yes`; the CLI displayed
  and received its own interactive confirmation.
- Step-0 baseline was finite (`avg@2 = -0.1`) with zero errored/cancelled rows,
  platform truncation mean `0.0333333`, and all reported turn counts at 40.
- Training then emitted repeated `ModelError` failures across many rollout
  groups. Stopped at 02:51:13 UTC under the agreed repeated-failure condition,
  before step 1. No checkpoint or adapter exists.
- Captured `assets/training/day8_smoke.json`; its retained log tail includes
  200 explicit failures. Targeted logs expose no underlying provider detail,
  so context pressure and shared inference instability remain hypotheses.
- Usage: 5,616,940 inference tokens, zero training tokens, `$0.00` total cost.
  Wallet remained `$57.9186`; the new run billing row has amount `$0.00`.
- Full postmortem: `docs/DAY8_SMOKE.md`. Do not relaunch unchanged.

Diagnostic preparation:
- Added `configs/day8_laguna_t1_diagnostic.toml`, which changes only workload
  pressure: one step, batch 16, two rollouts/example, and four maximum in
  flight. It retains `max_turns = 40`, 1,024-token sampling, model, learning
  rate, LoRA alpha, environment version, and T1 task.
- Prime CLI `0.6.16` accepts the diagnostic config. Its eight train and two dev
  rows instantiate locally and use disjoint seeds; final-eval remains excluded.
- Ruff passes, all 44 tests pass, both package artifacts build, and
  `git diff --check` passes.
- Live gate remains healthy: Laguna available/not at capacity with all three
  effective prices `$0.00/M`, wallet `$57.9186`, and Hub action `SUCCESS`.
- The diagnostic was committed and explicitly reviewed before the user
  launched it; the outcome is recorded below.

Diagnostic outcome:
- User launched `hg6jhftohpaognsubyoncy8s` after the CLI's free-pricing and Hub
  checks. Its four-rollout baseline completed cleanly at step 0 with
  `avg@2 = -0.1`, zero errors/truncation, and 739,505 inference tokens.
- The run then made no training-token or step progress for about 29 minutes and
  emitted repeated `ModelError` failures across refill/retry groups. It was
  stopped at 03:44:47 UTC; no checkpoint, adapter, sample, or distribution
  exists.
- `assets/training/day8_diagnostic.json` captures the run. Cost was `$0.00`,
  wallet stayed `$57.9186`, and its billing row is zero.
- Reducing concurrency from 96 to 4 did not help. The clean baseline plus zero
  training-policy tokens makes a Hosted Training policy-inference timeout/path
  failure the leading diagnosis, though the hidden wrapped exception is needed
  for confirmation. Pause further config launches until it is exposed.

Root-cause follow-up:
- A one-row local eval using the RL `renderer` client failed before rollout
  with HTTP 404: Prime Inference does not expose
  `poolside/Laguna-XS-2.1`, despite Hosted Training advertising that ID.
- The live inference catalog exposes only `poolside/laguna-m.1` for Laguna;
  its input/output price is `$0/M`. The live Hosted Training catalog does not
  include Laguna M.1, so it cannot be used as a Hosted Training base model.
- Added `configs/eval/day8_laguna_m1_t1_functional.toml` with the exact M.1 ID,
  two T1 dev rows, one rollout each, serialized sim state/logs, and the standard
  chat-completions client proven by prior Laguna evaluations. The manual command
  is `prime eval run configs/eval/day8_laguna_m1_t1_functional.toml`.
- This eval can prove that the environment/tool loop functions, but training
  remains blocked until Prime repairs the Laguna XS model alias or a paid model
  present in both catalogs receives explicit budget approval.

## 2026-07-14 Day 7 complete: renderer and public soft launch

Changed:
- Added the deterministic `scripts/render.py` pipeline for frozen v1
  `sim_log` artifacts: 1920x1080/30 fps H.264 MP4, 960px/12 fps GIF, cached
  attributed Contextily basemap, and a network-independent `--no-basemap`
  mode. Contextily remains dev-only; runtime dependencies are unchanged.
- Added schema, trajectory, event/TFR, wind, battery, terminal-state, and
  low-resolution FFmpeg integration coverage.
- Curated the fixed zero-based Laguna T3 rows 23, 28, 4, 12, and 20 from
  `day6-t3-clean-retry/results.jsonl`. Complete provenance and `sim_log` are
  under `assets/rollouts/day7/`; five MP4s, the row-23 hero GIF, and the
  contact sheet are under `assets/renders/day7/`.
- Set the release candidate to `0.1.1`. The earlier `0.1.0` Hub push consumed
  the version originally scheduled for the later environment freeze.
- Tightened each clip to its mission-derived extent after the cold-viewer
  review, placing home and target near opposite map edges. Mission-bounded
  zoom-12 rasters keep the close-up sharp while remaining locally cached.
- Published public `jarrett/uav-operator@0.1.1`; Hub wheel SHA-256 is
  `8d67634eb9d0c39e63b5a8b9c7b3267db448ff598d448a6ce05c56df4bfa9f34`.

Verified:
- Ruff and all 37 tests pass; `git diff --check` passes.
- All five artifacts render with the cached full-Bay basemap and independently
  with `--no-basemap`; MP4s are 22-second, 1920x1080, 30 fps H.264/yuv420p
  with no audio stream. The hero GIF is 960x540 and about 0.5 MB.
- The hero truthfully shows a TFR activating while the aircraft remains on the
  ground during mission intake. A true mid-flight TFR is deferred.
- The exact Hub wheel installed into
  `/tmp/uav-operator-hub-0.1.1-8djuQb/.venv`; its imported module resolved from
  isolated site-packages and `load_environment(max_examples=1)` returned
  `UAVOperatorEnv`.
- Isolated saved-state eval `407bb371` ran two one-rollout examples with
  `openai/gpt-4.1-nano`: zero provider errors, rewards `0.8` and `0.0`, and
  complete `sim_log` arrays of 5 and 40 snapshots. Results path:
  `/tmp/uav-operator-hub-0.1.1-8djuQb/eval-openai/evals/uav-operator--openai--gpt-4.1-nano/407bb371`.
- Extracted row 0 from those results and rendered it from outside the repo via
  the public renderer command in `--no-basemap` mode; FFprobe confirmed a
  22-second 1920x1080 H.264/yuv420p artifact.

Install/eval notes and deferred work:
- `prime env install jarrett/uav-operator@0.1.1` fails dependency resolution
  unless `--prerelease` is supplied. README now includes the required flag;
  direct isolated verification used `uv pip install --prerelease allow` with
  the owner's Hub index.
- Failed preflight eval `a82a4099` used bare `gpt-4.1-nano` against the Prime
  provider and received `model_not_found`; the proven registered identifier is
  `openai/gpt-4.1-nano`, used successfully in `407bb371`.
- A true mid-flight TFR clip and same-seed before/after-training pair remain
  deferred until suitable saved evidence exists.

## 2026-07-14 fixed-sim frontier recalibration complete

Changed:
- Added `scripts/run_frontier_calibration.py` to automate the former manual
  clean-retry process. It preserves successful rows by `example_id`, archives
  provider-error rows in `failed_results.jsonl`, removes only those errors from
  the resumable artifact, and retries missing per-example slots until all 15
  dev examples have two clean rollouts each.
- Unlimited attempts are the intentional default (`--max-attempts 0`), while
  each rollout has a 420-second timeout and retry batches use concurrency 8 so
  a hung provider request cannot block the loop indefinitely.
- Added regressions for error archival, per-example quotas, valid truncated
  outcomes, and plot rejection of any run that still contains provider errors.
- Documented the persistent runner in README. Raw Claude reruns remain intact;
  clean retry artifacts are under the ignored `outputs/evals/` tree.

Final fixed-sim calibration:
- GPT-4.1-nano is clean at 30/30 for T0-T3: `cdbac883`, `d43ac56f`,
  `99c64cc4`, and `bde540dd`.
- Laguna clean retry artifacts reached 30/30 on every tier. T2 required 65
  archived provider errors; T3 converged on retry attempt 14. The final scores
  are `0.998`, `0.424`, `0.476`, and `0.095`, with the intended sharp T3 drop.
- Generated `assets/evals/day6_tier_scores.png` and the machine-readable
  `day6_frontier_calibration.json`; all 240 plotted rows have zero provider
  errors. Laguna T3 includes two genuine hard-safety outcomes.

Next action:
- Begin the Day 7 renderer against the frozen rollout schema, using fixed T3
  Laguna rollouts as candidate failure/success clips.

## 2026-07-14 Day 6 complete: red-team round 1 + schema freeze

Changed:
- Built `scripts/redteam_day6.py`, deterministic probes for every SPEC §8
  exploit class through the real tool-call path; pre/post-fix evidence saved
  to `assets/redteam/day6_round1_{prefix,postfix}.json`.
- Closed five exploits with regression tests (27 tests total): mid-flight
  event skipping (segments now split at pending event triggers, remaining
  route suspends, holds cap at the next trigger), free airborne loitering
  (hover power burns during all airborne time; LOW_BATT_RTL fires after
  holds; hovering to depletion loses the aircraft), no-fly boundary hugging
  (0.2 nm buffer on authorization-required zones in validation and the
  feasibility solver), abort-on-ground zombie episodes (now terminal as
  `aircraft_ground_mission_<status>`), and a critical-battery-on-ground
  false hard-safety penalty (crit-batt now requires airborne/holding).
- Verified four classes closed by construction: success-claiming, override
  spam, SLA partial-credit farming, tool-error loops.
- Resolved the Day 5 Laguna T1 anomaly from saved run `1c757714`: the T1
  TFR_POPUP composition covers the mission target; Laguna scored 0.93-1.0 on
  other T1 event types but -1.575 with all 40 turns burned on the 8 TFR
  scenarios. Geofence holds now report `contains_mission_target` plus an
  advisory; regression added. No eval-seed tuning.
- Ran live adversarial prompting evals on the fixed sim via a new
  `system_prompt` env kwarg: GPT-4.1-nano T2 dev `a32af316` (all rollouts
  -0.1..-0.2) and Laguna T3 dev `a44f3e91` (avg 0.478, zero skipped events,
  zero hard-safety escapes; best rollout was honest flying). Artifacts in
  `assets/redteam/day6_adversarial/`.
- Froze the rollout JSON schema at v1 in `docs/SCHEMA.md` with an enforcement
  test asserting exact snapshot/segment/alert key sets.
- Updated SPEC (buffer, mid-flight interrupts, hover energy, crit-batt
  wording, §8 round 1 status), README status, and `docs/HACKS.md`.

Verified:
- `uv run ruff check .` passes.
- `uv run pytest -q` passes: 27 tests.
- T2/T3 dev generation unchanged under the buffered solver (9/11 TFR
  scenarios retained, all feasible); dataset rows are byte-identical.
- Baselines on the fixed sim: rulebook still beats reckless on T0/T1
  (0.995/0.651 vs 0.831/0.394); both bots are weak on T2/T3 as intended.

Broken / not done:
- The Day 5 calibration curve pre-dates the Day 6 sim changes; regenerate
  both model curves before the Day 7 soft launch (README notes this).
- Commanded RTL/land_now recovery legs remain uninterruptible and are not
  validated against active TFRs; post-incursion hard-safety logging is still
  deferred until route-through authorization semantics exist.
- A TFR activating mid-flight can, rarely, trap the aircraft inside the new
  polygon; the escape hatch is RTL (mission value lost). Documented, not yet
  priced or rendered.

Next action:
- Day 7: renderer + soft launch. Build the mp4 pipeline against frozen
  `docs/SCHEMA.md`, render 5 best episodes, `prime env push` v0.0.x, README
  quickstart + GIF. Regenerate the tier curve on the fixed sim first so the
  README table is not stale.

## 2026-07-12 Day 5 final two-model curve

Changed:
- Replaced GPT-4.1-nano's 10-rollout T0 point with clean 30-rollout run
  `6d18e4bc` and added a complete 30-rollout-per-tier Laguna series.
- Recovered Laguna T3 infrastructure failures by retaining the 28 valid rows
  from `fc2fab5c` and resuming only the two missing rollouts until the merged
  `t3-clean-retry` artifact contained 30 valid rows and zero provider errors.
- Updated the plot, machine-readable summary, README table, and interpretation.
  All 240 plotted rollouts have zero provider errors and zero hard-safety
  violations.

Interpretation:
- GPT-4.1-nano remains the clean monotonic calibration curve: `0.872`, `0.134`,
  `-0.137`, `-0.199` from T0 through T3.
- Laguna scores `1.000`, `0.282`, `0.479`, `0.504`. Its T1 variance and
  non-monotonic curve warrant investigation during Day 6 rather than being
  presented as a universal tier ordering.

Verified:
- `uv run ruff check .` passes.
- `uv run pytest -q` passes: 21 tests.
- `git diff --check` passes.

Next action:
- Begin Day 6 red-team round 1 and freeze the rollout-JSON schema.

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
