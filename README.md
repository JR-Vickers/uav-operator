# uav-operator

`uav-operator` is a multi-turn Prime Intellect / Verifiers environment where
an LLM acts as the remote pilot in command for small UAS missions over the San
Francisco Bay Area.

Drones already fly themselves. Autopilots hold trajectories, execute return to
launch, and enforce simple failsafes. This environment tests the layer above
that: ops-center judgment under shifting wind, pop-up flight restrictions,
battery anomalies, lost-link events, and mission updates. The simulator owns
the aircraft and every reward is computed from logged simulator state, not from
the model's prose.

![UAV ops console: Laguna handles four composed events and lands safely](assets/renders/day7/hero-treasure-island-safe-response.gif)

### Overview
- **Environment ID**: `uav-operator`
- **Short description**: Multi-turn UAV operations environment with
  physics-derived rewards and tool-based operator decisions.
- **Tags**: `uav`, `drone-operations`, `multi-turn`, `tool-use`, `train`,
  `eval`
- **Status**: Day 7 soft-launch release candidate. The offline renderer, five
  fixed Laguna T3 clips, hero GIF, frozen rollout artifacts, and `0.1.1`
  package candidate are complete. Public Hub publication remains gated on the
  cold-viewer visual check and an isolated install/eval.

### Datasets
- **Primary dataset(s)**: Seeded scenario generator emitting T0-T3 examples.
- **Source links**: Static Day 2 world data is generated in-repo with
  `scripts/build_world.py` from simplified public-structure airspace and
  synthetic obstacle assumptions.
- **Current split sizes**: 300 train, 60 dev/calibration, and 60 final eval
  rows. Each split is stratified evenly across T0-T3; final-eval seeds are
  never used for dial calibration.

### Task
- **Type**: Multi-turn tool use.
- **Role**: Remote pilot in command / ops-center operator.
- **Model decisions**: File or amend plans, query telemetry/weather/airspace,
  hold, resume, return to launch, land, release payload, abort missions, and
  override simulator-owned failsafes when justified by the scenario state.
- **Rubric overview**: Mission value, hard safety violations, reserve and
  margin policy, procedural compliance, and efficiency. Reward components read
  only saved `sim_log` snapshots; model prose is inert.

### Quickstart

Install the public Hub release and run a two-example smoke evaluation:

```bash
prime --plain env install jarrett/uav-operator@0.1.1
prime --plain eval run uav-operator -n 2 -r 1 --disable-tui
```

For source development, clone the repository and create the complete runtime
and development environment from the lockfile:

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run pytest -q
```

Run a small local smoke evaluation:

```bash
prime --plain eval run uav-operator -n 2 -r 1 --skip-upload --disable-tui
```

Configure model, sampling, and saved simulator state:

```bash
prime --plain eval run uav-operator \
  -m poolside/laguna-m.1 \
  -n 5 \
  -r 1 \
  -t 512 \
  -T 0.2 \
  --skip-upload \
  --disable-tui \
  --save-results \
  --state-columns sim_state,sim_log
```

Notes:
- Put task-owned settings under `[env.taskset]` and harness-owned settings
  under `[env.harness]` in TOML configs.
- The core sim is event-driven and analytic. It tests supervisory operator
  judgment, not low-level flight control or 3D collision physics.
- Inspect or rebuild the static Day 2 world JSON with
  `uv run python scripts/build_world.py`.
- Inspect the deterministic scripted rollout artifact at
  `assets/rollouts/day3_scripted_rollout.json`.
- Run local baselines with
  `uv run python scripts/baselines.py --policy both --episodes 20 --tier all`.
- Run Day 5 scripted calibration with
  `uv run python scripts/day5_calibration.py`. Build a frontier plot by passing
  saved `vf-eval` run directories to `scripts/plot_day5_scores.py`.
- Run each final frontier tier with fixed sampling, for example:
  `prime --plain eval run uav-operator -m poolside/laguna-m.1 -n 15 -r 2 -t 512 -T 0.2 --save-results --state-columns sim_state,sim_log`.
  Repeat for `gpt-5-nano` and record the printed run ID/results path; use taskset
  tier overrides for `T0` through `T3`.
- For providers with intermittent rollout errors, use
  `uv run python scripts/run_frontier_calibration.py --model poolside/laguna-m.1`.
  It archives failed rows beside the run, resumes only missing per-example
  slots, and retries until every tier has 30 clean rollouts. Pass
  `--max-attempts N` to cap retries; the default `0` is intentionally unlimited.
  Individual rollouts still have a 420-second timeout so one hung provider
  request cannot block the persistent retry loop.

### Offline rollout renderer

The renderer consumes only a saved JSON object containing frozen-schema
`sim_log`; it never imports, instantiates, or replays the simulator. The map is
a 2D supervisory-operations display, not a flight-dynamics or 3D collision
visualization.

```bash
uv run python scripts/render.py assets/rollouts/day7/laguna-t3-treasure-island-safe-response.json \
  -o out.mp4

# Fully offline vector fallback; no tile cache or network required.
uv run python scripts/render.py assets/rollouts/day7/laguna-t3-treasure-island-safe-response.json \
  -o out.mp4 --no-basemap
```

Each MP4 is 1920×1080, 30 fps, audio-free H.264/yuv420p. The committed README
GIF is 960 px wide. Basemap attribution appears in the display and downloaded
tiles are cached under the ignored `.cache/uav-renderer/` directory.

The five clips are fixed zero-based rows from Laguna run
`day6-t3-clean-retry/results.jsonl`; each compact JSON retains source run, row,
model, reward, metrics, and the complete `sim_log`:

| Source row | Evidence | Reward | Clip |
| ---: | --- | ---: | --- |
| 23 | Treasure Island: four-event safe response (hero) | 0.865 | [MP4](assets/renders/day7/laguna-t3-treasure-island-safe-response.mp4) |
| 28 | Pier 39: clean composed-event completion | 0.943 | [MP4](assets/renders/day7/laguna-t3-pier-39-clean-completion.mp4) |
| 4 | Bay Farm: safe completion at 28.9% battery | 0.700 | [MP4](assets/renders/day7/laguna-t3-bay-farm-low-margin.mp4) |
| 12 | Treasure Island: repeated geofence/override struggle, then success | 0.044 | [MP4](assets/renders/day7/laguna-t3-treasure-island-geofence-struggle.mp4) |
| 20 | Richmond Channel: battery-depletion aircraft loss | -4.300 | [MP4](assets/renders/day7/laguna-t3-richmond-battery-loss.mp4) |

[View the five-episode trigger/decision/outcome contact sheet](assets/renders/day7/contact-sheet.png).

The hero's pop-up restriction activates while the aircraft is still on the
ground during mission intake; it is not presented as a mid-flight TFR. A true
mid-flight TFR clip and a same-seed before/after-training pair remain deferred
until suitable saved evidence exists.

### Day 6 post-fix calibration

![GPT-4.1-nano and Laguna reward by curriculum tier](assets/evals/day6_tier_scores.png)

The calibration compares GPT-4.1-nano and Laguna on the same dev split with
fixed sampling, 30 rollouts per tier, and no provider errors. Error bars are
95% normal confidence intervals over rollout rewards; `n` is annotated on
every point.

| Model | Tier | Mean reward | 95% CI | Missions completed | Hard safety | Run ID |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GPT-4.1-nano | T0 | 0.884 | [0.847, 0.921] | 30/30 | 0 | `cdbac883` |
| GPT-4.1-nano | T1 | -0.055 | [-0.451, 0.341] | 6/30 | 1 | `d43ac56f` |
| GPT-4.1-nano | T2 | -0.140 | [-0.158, -0.122] | 0/30 | 0 | `99c64cc4` |
| GPT-4.1-nano | T3 | -0.196 | [-0.229, -0.162] | 0/30 | 0 | `bde540dd` |
| Laguna | T0 | 0.998 | [0.996, 1.000] | 30/30 | 0 | `day6-t0-clean-retry` |
| Laguna | T1 | 0.424 | [0.098, 0.750] | 22/30 | 0 | `day6-t1-clean-retry` |
| Laguna | T2 | 0.476 | [0.221, 0.731] | 29/30 | 0 | `day6-t2-clean-retry` |
| Laguna | T3 | 0.095 | [-0.413, 0.603] | 26/30 | 2 | `day6-t3-clean-retry` |

GPT-4.1-nano remains the weak monotonic reference: T0 is near ceiling, T1 is
materially harder, and T2/T3 stay below zero. Laguna is near ceiling on T0 and
strong through T2, then drops to `0.095` on T3 with two hard-safety outcomes.
The fixed simulator therefore produces the intended frontier separation on
the composed-event tier. All 240 plotted rollouts are free of provider errors;
the machine-readable results and provenance are in
[`assets/evals/day6_frontier_calibration.json`](assets/evals/day6_frontier_calibration.json).

Day 6 resolved the Laguna T1 anomaly from the saved rollouts: the T1
`TFR_POPUP` composition places the restriction over the mission target, and
Laguna scored 0.93-1.0 on the other three T1 event families but -1.575 on the
TFR scenarios, burning all 40 turns re-filing conflicting plans instead of
aborting. Geofence holds now flag `mission_target_inside_zone` so that trap is
legible. The curve above includes that fix plus Day 6 mid-segment interrupts,
hover energy, and buffered airspace checks.

### Taskset Config
Planned fields:

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `tier` | string | `mixed_day5` | Scenario tier: `T0`–`T3`, `mixed_day5`, or legacy `mixed_day4`. |
| `dataset_split` | string | internal | Generated split: 300 train rows, 60 dev rows for scripts, or 60 held eval rows. |
| `seed` | int | `0` | Base seed for deterministic scenario generation. |
| `max_examples` | int | `-1` | Limit on dataset size; use `-1` for all generated examples. |
| `wind_enabled` | bool | `true` | Enable seeded Day 3 wind field and altitude shear. |
| `gust_front_probability` | float | `0.5` | Probability that an episode includes a gust front. |
| `system_prompt` | string | ops-manual prompt | Override the operator system prompt (used for red-team/adversarial runs). |

### Harness Config
Planned fields:

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `40` | Maximum operator decision turns per episode. |
| `sim_time_cap_min` | int | `90` | Maximum simulated episode duration. |

### Metrics
Implemented rubric metrics:

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward, computed from simulator state |
| `mission_value` | Completed mission value after timeliness decay |
| `hard_safety` | Airspace incursions, aircraft loss, and critical battery outcomes |
| `margin_policy` | Reserve, override, and minimum-safe-altitude penalties |
| `procedure` | Alert acknowledgement, conflict filing, and hold-expiry penalties |
| `efficiency` | Energy and simulated-time cost relative to scenario par |
