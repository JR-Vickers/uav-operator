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

### Overview
- **Environment ID**: `uav-operator`
- **Short description**: Multi-turn UAV operations environment with
  physics-derived rewards and tool-based operator decisions.
- **Tags**: `uav`, `drone-operations`, `multi-turn`, `tool-use`, `train`,
  `eval`
- **Status**: Day 4 implemented: seeded T0-T1 generator, event interrupts for
  wind shift / pop-up TFR / battery degrade / site closure, full SPEC §5 reward
  v0, rulebook and reckless baselines, deterministic sim logging, and the full
  SPEC §3.2 console. T2/T3 generation and renderer remain next.

### Datasets
- **Primary dataset(s)**: Seeded scenario generator emitting T0-T1 examples.
- **Source links**: Static Day 2 world data is generated in-repo with
  `scripts/build_world.py` from simplified public-structure airspace and
  synthetic obstacle assumptions.
- **Current split sizes**: default Day 4 eval set is 20 mixed T0-T1 examples.
  v0.1 target remains at least 300 train and 60 eval scenarios, stratified
  across T0-T3 difficulty tiers.

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
  `uv run python scripts/baselines.py --policy both --episodes 20`.

### Taskset Config
Planned fields:

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `tier` | string | `mixed_day4` | Scenario tier selection: `T0`, `T1`, or mixed Day 4 T0/T1. |
| `seed` | int | `0` | Base seed for deterministic scenario generation. |
| `max_examples` | int | `-1` | Limit on dataset size; use `-1` for all generated examples. |
| `wind_enabled` | bool | `true` | Enable seeded Day 3 wind field and altitude shear. |
| `gust_front_probability` | float | `0.5` | Probability that an episode includes a gust front. |

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
