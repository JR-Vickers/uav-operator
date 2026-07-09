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
- **Status**: Build skeleton. The package metadata and import path are in
  place; the Day 1 end-to-end environment loop is the next implementation
  target.

### Datasets
- **Primary dataset(s)**: Seeded scenario generator, planned.
- **Source links**: Generated in-repo from public-structure airspace and
  synthetic weather assumptions.
- **Split sizes**: v0.1 target is at least 300 train and 60 eval scenarios,
  stratified across T0-T3 difficulty tiers.

### Task
- **Type**: Multi-turn tool use.
- **Role**: Remote pilot in command / ops-center operator.
- **Model decisions**: File or amend plans, query telemetry/weather/airspace,
  hold, resume, return to launch, land, release payload, abort missions, and
  override simulator-owned failsafes when justified by the scenario state.
- **Rubric overview**: Mission value, hard safety violations, reserve and
  margin policy, procedural compliance, and efficiency. Reward components are
  derived only from simulator state and `sim_log`.

### Quickstart
Once the Day 1 environment skeleton is implemented, run an evaluation with
default settings:

```bash
prime eval run uav-operator
```

Configure model and sampling:

```bash
prime eval run uav-operator -m openai/gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- Put task-owned settings under `[env.taskset]` and harness-owned settings
  under `[env.harness]` in TOML configs.
- Until the Day 1 loop is implemented, `load_environment()` intentionally
  raises `NotImplementedError`.

### Taskset Config
Planned fields:

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `tier` | string | `mixed` | Scenario tier selection: `T0`, `T1`, `T2`, `T3`, or `mixed`. |
| `seed` | int | `0` | Base seed for deterministic scenario generation. |
| `max_examples` | int | `-1` | Limit on dataset size; use `-1` for all generated examples. |

### Harness Config
Planned fields:

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `40` | Maximum operator decision turns per episode. |
| `sim_time_cap_min` | int | `90` | Maximum simulated episode duration. |

### Metrics
Planned rubric metrics:

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward, computed from simulator state |
| `mission_value` | Completed mission value after timeliness decay |
| `hard_safety` | Airspace incursions, aircraft loss, and critical battery outcomes |
| `margin_policy` | Reserve, override, and minimum-safe-altitude penalties |
| `procedure` | Alert acknowledgement, conflict filing, and hold-expiry penalties |
| `efficiency` | Energy and simulated-time cost relative to scenario par |
