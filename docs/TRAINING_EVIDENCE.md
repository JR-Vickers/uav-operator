# Checkpoint training-evidence pipeline

`scripts/training_evidence.py` prepares and validates the post-training
evaluation path without launching training, deploying a checkpoint, or running
inference. Generated configs, manifests, summaries, and plots belong under
ignored `outputs/training-evidence/`. Only captured, validated results may later
be promoted into a committed result or learning claim.

## Safety boundary

`prepare` calls only these read-only Prime surfaces: training-run metadata,
Hosted model pricing, wallet snapshot, checkpoint listing, and adapter listing.
It never calls `prime train`, `prime deployments create`, or `prime eval run`.
Instead, it records the exact manual next command in its manifest.

An adapter must match the requested training run, base model, and step and have
both `status=READY` and `deployment_status=DEPLOYED`. Missing or ambiguous
metadata, pricing, wallet state, provenance, or readiness fails closed. The
composite inference ID is `{base_model}:{adapter_id}`.

A checkpoint must match the requested run and step and have `status=READY`.
Preparation then stops with `status=awaiting_adapter_deployment` and records:

```bash
prime deployments create --checkpoint-id <checkpoint-id>
```

That command is intentionally interactive and must be run by a person. After
deployment reaches `DEPLOYED`, rerun `prepare` with the resulting adapter ID.

## Prepare a milestone

Use a separate directory for every step, including the step-0 baseline:

```bash
uv run python scripts/training_evidence.py prepare \
  --run-id <training-run-id> \
  --base-model <hosted-base-model> \
  --step 10 \
  --adapter-id <deployed-adapter-id> \
  --output-dir outputs/training-evidence/<training-run-id>/step-10
```

The generated `eval.toml` freezes the T1 dev workload at 15 rows, two rollouts
per row, temperature 0, 1,024 tokens per model turn, 20 environment turns,
concurrency four, and no retries. It saves `sim_state` and `sim_log`. The
manifest records the config SHA-256, all 15 dev seeds, all excluded final-eval
seeds, provenance IDs, effective prices, wallet snapshot, a 614,400-token
configured output ceiling, and the exact manual eval command.

The output token ceiling is `15 × 2 × 20 × 1,024`. Input tokens are explicitly
marked unbounded because context size is provider/model dependent. If effective
inference prices are nonzero, calculate a current cost estimate and obtain
explicit approval before manually running the eval command. Preparation itself
has no inference workload.

The manual command uses the milestone directory as vf-eval's output base.
vf-eval writes `metadata.json` and `results.jsonl` in its normal generated
`evals/<environment--model>/<eval-run-id>/` child. Keep the manifest and config
at the milestone root, do not edit any captured file, and use the printed leaf
run directory in `summarize`.

## Summarize milestones

Pass every expected step explicitly. Step 0 is mandatory:

```bash
uv run python scripts/training_evidence.py summarize \
  0=outputs/training-evidence/<run-id>/step-0/evals/<environment--model>/<eval-id> \
  10=outputs/training-evidence/<run-id>/step-10/evals/<environment--model>/<eval-id> \
  20=outputs/training-evidence/<run-id>/step-20/evals/<environment--model>/<eval-id> \
  --expected-steps 0,10,20 \
  --output-json outputs/training-evidence/<run-id>/summary.json \
  --output-plot outputs/training-evidence/<run-id>/reward-by-step.png \
  --mode captured
```

Validation requires exactly two rollouts for each of seeds 10000–10014, no
seed 20000–20014, the T1 dev split, exact sampling/workload settings, a config
whose hash matches its manifest, finite rewards/metrics/token counts, zero row
errors, and nonempty schema-complete simulator logs. A max-turn outcome is
valid evidence and is reported as truncation rather than a provider error.

The summary contains per-step mean reward and a normal 95% confidence interval,
mission completion rate, hard-safety count, max-turn truncation rate, mean
turns, aggregate token use, all model/training/checkpoint/adapter/eval IDs, and
absolute source paths. Same-seed comparison averages both rollouts per seed,
ranks step-0-to-final reward deltas, and retains both source row indices for
future before/after rendering.

## Synthetic fixture mode

Tests and plumbing demos may provide Prime-shaped JSON snapshots with
`--fixture-metadata-dir`. That option forces `provenance=synthetic_fixture` into
the manifest. Summarize such runs only with `--mode synthetic_fixture`.
Fixture summaries retain that provenance and fixture plots are watermarked
`SYNTHETIC FIXTURE — NOT RESULTS`. Captured mode rejects fixture provenance;
renaming or copying a fixture directory cannot make it captured evidence.

The fixture metadata directory uses `run.json`, `models.json`, `wallet.json`,
and either `adapters.json` or `checkpoints.json`, each matching the corresponding
Prime CLI JSON envelope. This mode exists only to exercise plumbing. Never put
its metrics or chart under `assets/`, the README leaderboard, a writeup result,
or a learning claim.

## Summary schema

The top-level JSON fields are:

- `schema_version`, `kind`, `provenance`, and nullable `watermark`;
- `expected_steps` and `interval`;
- `milestones[]`, with aggregate metrics, provenance IDs, and source paths;
- `same_seed_baseline_to_final`, with baseline/final step IDs and ranked seed
  records containing aggregate rewards, deltas, source paths, and row indices.

Treat schema changes as evidence-interface changes: update this runbook and the
strict tests together.
