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
