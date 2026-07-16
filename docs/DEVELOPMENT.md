# Development

This guide covers source setup and the repository-only tools that are not
shipped in the Hub wheel. For the public environment arguments, see
[Configuration](CONFIGURATION.md).

## Source setup

Python 3.10 or newer and `uv` are required. Install the locked runtime and
development dependency groups from the repository root:

```bash
uv sync --locked --all-groups
```

The runtime package depends only on Verifiers, NumPy, and the standard library.
Matplotlib and Contextily are development dependencies used by the renderer.
MP4 output also requires `ffmpeg` to be available on `PATH`.

## Checks

Run the standard local checkpoint:

```bash
uv run ruff check .
uv run pytest -q
uv build
git diff --check
```

The wheel is intentionally lean. `[tool.hatch.build]` in `pyproject.toml`
includes only `uav_operator.py` and `pyproject.toml`.

## Local evaluations

Run a small source evaluation without uploading results:

```bash
prime --plain eval run uav-operator \
  -n 2 \
  -r 1 \
  --skip-upload \
  --disable-tui
```

Choose a model and sampling settings with the harness flags:

```bash
prime --plain eval run uav-operator \
  -m poolside/laguna-m.1 \
  -n 5 \
  -r 1 \
  -t 512 \
  -T 0.2 \
  --skip-upload \
  --disable-tui
```

Environment arguments such as `tier`, `dataset_split`, and `max_turns` are most
reproducibly stored in a TOML config. See
[`configs/eval/day8_laguna_m1_t1_functional.toml`](../configs/eval/day8_laguna_m1_t1_functional.toml)
for a working saved-state example.

## Saving simulator state

The renderer and evidence tooling require the full simulator snapshots:

```bash
prime --plain eval run uav-operator \
  -m poolside/laguna-m.1 \
  -n 2 \
  -r 1 \
  --save-results \
  --state-columns sim_state,sim_log \
  --skip-upload \
  --disable-tui
```

`sim_state` stores the final serializable simulator state. `sim_log` stores a
full snapshot at every decision point and is the source of truth for reward
recomputation and offline rendering.

## Scripted baselines

Compare the rulebook and reckless policies across all tiers:

```bash
uv run python scripts/baselines.py \
  --policy both \
  --episodes 20 \
  --tier all
```

The baseline script also accepts `--seed`, a single tier, or one policy.

Generate the scripted four-tier calibration artifact:

```bash
uv run python scripts/day5_calibration.py
```

Its defaults are seed `0`, 15 episodes per tier, and output
`outputs/day5/scripted_calibration.json`.

## Frontier calibration

For a direct evaluation, run each tier against the development split with
fixed sampling and saved logs. A TOML config is preferred for exact
reproducibility.

For providers with intermittent rollout failures, use the persistent runner:

```bash
uv run python scripts/run_frontier_calibration.py \
  --model poolside/laguna-m.1
```

By default it runs T0–T3 on the development split with 15 examples, two
rollouts per example, 512 output tokens, temperature `0.2`, concurrency `8`,
and a 420-second per-rollout timeout. It archives failed rows and retries only
missing example/rollout slots until each tier has 30 clean rollouts.

The default `--max-attempts 0` retries indefinitely. Use a positive cap for
bounded automation, and use repeated `--resume-tier TIER=PATH` arguments to
continue from existing run directories.

Build a plot and optional machine-readable summary from completed Verifiers run
directories:

```bash
uv run python scripts/plot_day5_scores.py \
  outputs/evals/model-t0/run \
  outputs/evals/model-t1/run \
  outputs/evals/model-t2/run \
  outputs/evals/model-t3/run \
  -o outputs/tier-scores.png \
  --summary-output outputs/frontier-calibration.json
```

## World data

The simulator's static world structures are defined in `uav_operator.py`.
Rebuild the inspection/renderer JSON with:

```bash
uv run python scripts/build_world.py
```

The default output is `data/world.json`; pass `-o PATH` to write elsewhere.
Rebuilding this file does not change the source-of-truth simulator constants.

## Offline rendering

Render a frozen rollout to MP4:

```bash
uv run python scripts/render.py \
  assets/rollouts/day7/laguna-t3-treasure-island-safe-response.json \
  -o out.mp4
```

The default basemap path uses Contextily. Downloaded mission-scale tiles are
cached under `.cache/uav-renderer/`. If tile access is unavailable, use the
fully offline vector fallback:

```bash
uv run python scripts/render.py \
  assets/rollouts/day7/laguna-t3-treasure-island-safe-response.json \
  -o out.mp4 \
  --no-basemap
```

The output extension may be `.mp4` or `.gif`. Input must be one JSON object
containing the frozen v1 `sim_log` described in [Rollout schema](SCHEMA.md).
Renderer failures return a structured JSON error on stderr.

## Historical training status

Hosted-training incident reports and retained evidence are tracked separately
from the public landing page:

- [Free-training eligibility report](DAY8_FREE_TRAINING.md)
- [Hosted-training failure report](DAY8_SMOKE.md)
- [Checkpoint evidence workflow](TRAINING_EVIDENCE.md)

These documents are operational history, not evidence of a successful
checkpoint or learned improvement.
