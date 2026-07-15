# Day 8 Hosted Training smoke postmortem

- Run: `ed7ap9lbtm3lpy6pqeav7lrt`
- Model: `poolside/Laguna-XS-2.1`
- Config: `configs/day8_laguna_t1_smoke.toml`
- Evidence: `assets/training/day8_smoke.json`

## Outcome

The run launched after the explicit approval and repeated live pricing gate,
completed its pre-training evaluation at step 0, then hit the mandatory stop
condition for repeated model/provider failures before the first training step.
It was manually stopped at 02:51:13 UTC. Do not restart or relaunch this config
unchanged.

The smoke therefore did its pipeline-validation job but did not pass Day 8:
there is no training metric, reward distribution, sampled-rollout artifact,
checkpoint, or adapter. The 300-step Day 10 run is not authorized by this
result.

## Timeline

- 02:39:16 UTC: Hosted Training created the run.
- 02:39:37 UTC: run entered `RUNNING`; train and eval env-servers were healthy.
- 02:49:06 UTC: pre-training evaluation published finite step-0 metrics.
- 02:49:41–02:51:12 UTC: the orchestrator logged repeated `ModelError` rollout
  failures across many distinct training groups.
- 02:51:13 UTC: run stopped under the documented repeated-failure rule.
- 02:53:17 UTC: reproducible evidence bundle captured and wallet reconciled.

## Evidence

Pre-training evaluation over 30 rollouts reported:

- `avg@2 = -0.1`
- `errored_count = 0`, `cancelled_count = 0`, `no_response/mean = 0`
- `is_truncated/mean = 0.0333333`
- completion length mean/min/max = `1238.8 / 691 / 11029`
- reported turn count mean/min/max = `40 / 40 / 40`

Immediately afterward, the retained 5,000-line log tail contained 200 explicit
`Rollout failed in group ... Error: ModelError` records. Filtered ERROR-level,
env-server, context, rate-limit, timeout, and capacity searches exposed no more
specific provider message. The exact upstream cause is therefore unresolved;
calling it a context-length failure would be an inference, not established
evidence. The baseline's all-40-turn metric and 11,029-token maximum completion
make turn/context pressure a Day 9 hypothesis, while the burst across distinct
groups also leaves shared inference instability as a live hypothesis.

Usage and billing reconciled exactly:

- inference: 5,616,940 tokens (5,579,776 input; 37,164 output), `$0.00`
- training: 0 tokens, `$0.00`
- total reported cost: `$0.00`
- wallet before and after: `$57.9186`
- Prime billing row `w6wr0rud100tbk7vockow4zf` is tied to this run with
  `amount_usd = 0.0`

No final-eval seed was configured or used. Reward metrics came from the pinned
simulator-state-only environment `jarrett/uav-operator@0.1.1`.

## Day 9 decision

Stop and re-plan before any main run. First obtain the underlying `ModelError`
detail from Prime support/platform telemetry. The first diagnostic is
`configs/day8_laguna_t1_diagnostic.toml`: one optimizer step, batch 16, two
rollouts/example, four maximum in flight, eight T1 train rows, and two T1 dev
rows. It deliberately preserves `max_turns = 40` and 1,024-token sampling so a
success isolates the original 96-way training burst as the material change. If
it reproduces `ModelError`, the next diagnostic should reduce turn/context
limits rather than merely retrying. Any launch needs a fresh config review,
pricing/wallet gate, and explicit approval. The former provisional 300-step
Laguna run is suspended; even the 150-step fallback is inappropriate until a
diagnostic reaches at least one healthy training step.

## One-step concurrency diagnostic

Run `hg6jhftohpaognsubyoncy8s` tested the first hypothesis with batch 16 and
four maximum in-flight rollouts while retaining the original 40-turn and
1,024-token limits. It also failed at step 0.

- Started at 03:12:53 UTC and published a clean four-rollout baseline at
  03:15:15 UTC: `avg@2 = -0.1`, zero errors/cancellations/truncation, and
  completion lengths 812–886.
- Made no training-step or training-token progress for approximately 29
  minutes. It was stopped at 03:44:47 UTC under both the repeated-failure and
  15-minute no-progress rules.
- The retained log tail contains 200 `ModelError` rollout failures spanning
  far more groups than the configured 16-rollout batch, consistent with
  repeated refill/retry attempts.
- Usage contains only the baseline: 739,505 inference tokens, zero training
  tokens, and `$0.00`. No sample, distribution, checkpoint, or adapter exists.
- Wallet remained `$57.9186`; billing row `cwi9xmm4qy5t8x1w561tete8` records
  `$0.00` for this run.

Evidence is in `assets/training/day8_diagnostic.json`. Lowering concurrency
from 96 to 4 did not fix the failure, so concurrency alone is rejected. Because
the clean baseline used the same environment and context limits while training
produced no token record before its apparent timeout, the leading diagnosis is
a Hosted Training policy-inference path failure. This remains an inference:
the platform still exposes only the wrapper `ModelError`. Do not launch another
config experiment until the underlying exception is available or Prime
confirms the service condition.

## Model-catalog diagnosis and functional check

The policy-inference failure is now bounded to a model-catalog mismatch rather
than the UAV environment:

- `prime train models --output json` advertises and accepts
  `poolside/Laguna-XS-2.1` for Hosted Training.
- Prime Inference returns HTTP 404 for that exact ID: `Model
  'poolside/Laguna-XS-2.1' not found or unavailable`.
- `prime inference models --output json --search Laguna` lists only
  `poolside/laguna-m.1` at zero input and output cost.
- Prime's token-preserving training renderer registry keys Laguna XS.2 by the
  different exact ID `poolside/Laguna-XS.2`. Hosted Training exposes neither
  that canonical ID nor a user-facing renderer override.

This explains why changing rollout concurrency did not help: the failing
training policy path cannot resolve the advertised model consistently. It also
explains why the base-model evaluation is not sufficient proof of a healthy
training path; eval and RL training use different inference clients.

`configs/eval/day8_laguna_m1_t1_functional.toml` uses the exact live inference
ID `poolside/laguna-m.1` and the standard chat-completions client already proven
by the Day 6/7 Laguna rollouts. It is a two-row T1 dev check with saved
`sim_state` and `sim_log`, so success means the model actually called tools and
the resulting rollout remains renderable. Run it manually with:

```bash
prime eval run configs/eval/day8_laguna_m1_t1_functional.toml
```

This command is an inference evaluation, not training. `poolside/laguna-m.1`
is absent from the live Hosted Training catalog and therefore cannot replace
the base model in a Hosted Training TOML. A functioning Hosted Training run
requires Prime to repair the Laguna XS alias or a separately approved model
that appears in both catalogs.
