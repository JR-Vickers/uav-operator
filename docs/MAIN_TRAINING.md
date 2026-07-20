# Main training protocol

This recovery protocol prepares **40 additional warm-started updates** from selected
smoke checkpoint `g1akido7qfo58e3my36wnqrz`. It does not describe one
uninterrupted 60-step optimizer trajectory: `checkpoint_id` proves weight
warm-start provenance, but optimizer-state restoration must not be claimed
unless later platform metadata explicitly proves it.

No command in this document authorizes paid work. Each phase is prepared,
manually approved, launched, captured, and validated before the next phase may
be prepared. The tool never launches training itself.

## Frozen curriculum

| Phase | Updates | Training rows | Input | Eval | Retention |
|---|---:|---|---|---|---|
| B | 20 (global 20→40) | 40% T1 / 60% T2 | smoke step 20 | T0–T3 at 20, 30, 40 | steps 30, 40 |
| C | 20 (global 40→60) | 25% T1 / 45% T2 / 30% T3 | passing Phase-B step 40 | T0–T3 at 40, 50, 60 | steps 50, 60 |

Every training tier is its own `[[env]]` using
`jarrett/uav-operator@0.1.1`, the deterministic `train` split, 75 rows, and a
static ratio. Training remains Qwen3.5-2B RL/GRPO with batch 16, two
rollouts/example, four in flight, learning rate `3e-5`, LoRA alpha 32,
temperature 0.7, 1,024 tokens, thinking disabled, 20 turns, and zero retries.

Each hosted evaluation milestone uses one deterministic `mixed_day5` dev
environment containing six examples from each T0–T3 tier (24 total), one
rollout/example, temperature 0, 1,024 tokens, 20 turns, zero retries, and the
initial step. No T0 or final-eval row enters training.
Simulator, reward, prompt, generator, and split semantics remain frozen.
Training entries use unique names such as `train_t1`; the evaluation entry is
`dev_mixed`. The single mixed entry preserves tier coverage while avoiding the
six-container startup pattern that failed twice before training began.

Prime interprets Hosted Training `max_steps` as an absolute global target when
warm-starting. Therefore the recovery TOMLs use `max_steps = 40` and `60`;
these represent 20 and 20 additional updates respectively. An
initial Phase-A launch attempt with `max_steps = 10` was rejected by the API
before run creation or billing, which supplied this platform evidence.

## Budget ledger

| Item | Expected | Hard ceiling |
|---|---:|---:|
| Existing diagnostic, smoke, and checkpoint comparison | $3.0469 | $3.0469 |
| Failed Phase A including clean validation (sunk) | $1.4739461 | $1.4739461 |
| Phase B | $1.90 | $2.90 |
| Phase C | $1.95 | $2.40 |
| Post-training dev comparison | reserved | $1.75 |
| T2/T3 red-team | reserved | $0.75 |
| Frozen final evaluation | reserved | $1.25 |

Expected total is `$12.1208461`; hard commitments total `$13.5708461`,
leaving `$1.4291539` below the aggregate `$15.00` cap. A pricing change recomputes
the phase projection from the paid smoke's measured tokens. It cannot silently
consume the unallocated amount. Missing pricing, capacity, wallet, Hub status,
or billing evidence fails closed.

On 2026-07-20 the first live Phase-A preparation stopped at this gate: the
effective inference-input price was `$0.05/M`, producing a conservative
`$1.4413` projection against the original `$1.25` ceiling. The user then
approved a narrow increase to `$1.50`, leaving the `$15.00` aggregate cap and
all downstream reserves unchanged.

On 2026-07-20 live pricing projected recovery Phase B at `$2.8825466`, above
its original `$2.35` ceiling. The user explicitly approved raising only the
Phase B ceiling to `$2.90`; the aggregate `$15.00` cap and downstream reserves
remain unchanged.

## Manual commands and evidence

Generated bundles are ignored under `outputs/main-training/`. `prepare`
validates the source run/model/step and exact READY checkpoint, refreshes model
pricing/capacity, wallet, and Hub status, then writes `train.toml`, a SHA-256
manifest, usage projection, stop limits, and the exact manual launch command.

```bash
uv run python scripts/main_training.py prepare B
```

After reviewing the bundle and receiving separate approval, the operator runs
the manifest's `manual_launch_command`. The command is never executed by
`main_training.py`. Capture the completed phase with:

```bash
uv run python scripts/main_training.py capture B RUN_ID \
  --manifest outputs/main-training/phase-b/manifest.json \
  --exact-evaluation outputs/main-training/phase-b/exact-step-40.json
```

Only a passing capture is promoted to
`assets/training/main_phase_b.json` and its curve to
`assets/training/main_phase_b_curve.png`. Phase C adds the predecessor gate:

```bash
uv run python scripts/main_training.py prepare C \
  --predecessor-capture assets/training/main_phase_b.json
uv run python scripts/main_training.py budget assets/training/main_phase_*.json
```

Capture retrieves the run, exact TOML, metrics, distributions, usage, logs,
rollouts, checkpoints, and the deployment registry. It requires completed
expected steps, finite rewards, zero provider errors/cancelled rows, exact
READY retained adapters, matching provenance, and a reconciled phase cost
below its ceiling. `passed` records that scientific result independently from
`continuation_ready`, which additionally requires both retained checkpoint
milestones and exactly one READY final checkpoint. Model prose is stored only if the platform
rollout response necessarily includes it; it is never read or judged by a
gate. Reward and safety evidence remain simulator-derived.

## Stop and approval gates

Stop the active run, or withhold the next approval, for any of:

- non-finite reward or metrics;
- provider errors or cancelled rows;
- 15 minutes without optimizer progress;
- missing expected optimizer/evaluation steps;
- no exact final READY adapter (scientific failure), or no exact final READY
  checkpoint (continuation blocker);
- any negative `hard_safety` reward component (a positive penalty magnitude);
- more than 50% max-turn truncation at an evaluation milestone;
- the phase or cumulative cost ceiling being reached or breached.

After a passing capture, report actual phase and cumulative cost, reward by
tier, safety events, truncation by milestone, exact final checkpoint identity,
and the next `prepare` command before requesting approval.

Exploratory training rollouts remain diagnostic evidence, including any
simulator-derived safety penalty. The launch safety gate is evaluated on the
deterministic exact-checkpoint dev workload. Hosted milestone evaluations that
mix policy versions are not accepted as exact-checkpoint evidence.

## Phase B exact step-40 recovery (2026-07-20)

Completed run `aygdxtalsw85xbznsj0k288m` reports its original `$1.9938` cost.
The deployment registry exposes READY step-40 adapter
`j21ahkcyttbbu3ponk9on8m5`; checkpoint `ezalshw3415w0z9kb8z56vji` has not
completed upload. Platform-created null/step-39 aliases remain evidence but do
not satisfy or invalidate the exact step-40 lookup.

```bash
uv run python scripts/main_training.py prepare-exact-b
uv run python scripts/main_training.py summarize-exact-b VF_EVAL_RUN_DIR \
  --manifest outputs/main-training/phase-b/exact-manifest.json
```

The emitted manual workload is 24 `mixed_day5` dev rows—seeds 10000–10023,
six per T0–T3—at temperature 0, one rollout per example, 20 turns, zero
retries, with saved `sim_state`/`sim_log`. A failure scientifically fails Phase
B. After a pass, refresh the checkpoint. If it is still unavailable, Phase C
stays blocked. A fallback from READY step-30 checkpoint
`hq9o5apo977l7hpkk5n0tpaf` would be a new stochastic branch; add its projected
cost to the `$15` ledger and obtain separate approval before launch.

Two initial Phase-B launches (`b2oubcurqhhfcsoh3443wru5` and
`ju9j3zjlligjfgt3pwy8fxyy`) each started exactly five of six environment
containers before a different eval slot failed. Both ended at zero steps,
zero tokens, and `$0.00`. The recovery config consolidates the four hosted dev
slots into `dev_mixed`, reducing the total to three containers without changing
the dev rows per tier. Evidence is in
`assets/training/main_phase_b_startup_failures.json`.

## Phase A result (2026-07-20)

Run `r6n3cud398dkrhnsoqwxpsjk` completed global steps 20 through 30 and produced
exact step-30 adapter `npn2nu01kpb3u402x1igr86v`. Training cost `$1.1958`; the
clean exact-adapter T0-T3 validation cost `$0.2781461`; total Phase A cost
`$1.4739461`, below the `$1.50` ceiling.

The capture failed the preregistered truncation gate. T2 and T3 each reached
the 20-turn limit in 4/6 episodes (66.7%). T0 and T1 had 0/6 and 2/6 max-turn
stops. All 24 evaluations completed with saved simulator state/logs, zero
provider errors, and zero hard-safety penalties. Forensics classify the eight
T2/T3 truncations as seven ground/pending stalls and one airborne holding
stall, with repeated planning, polling, holding, route amendment, and
no-tool-call behavior. Phase A and its step-23 training safety event remain
diagnostic evidence, but step 30 is abandoned as a parent. No simulator,
reward, prompt, dataset, or turn-limit change was made. See
`assets/training/main_phase_a_exact_step30.json` and
`assets/training/main_phase_a_recovery_decision.json`.

## Post-Phase-C selection

Deploy smoke step 20, failed Phase A step 30, Phase B step 40, and Phase C step
60. Evaluate all four on
identical T1–T3 dev workloads: six seeds per tier, two rollouts per seed,
temperature 0, 20 turns, no retries, and saved `sim_state`/`sim_log`. Report
paired-seed results. Smoke and failed Phase A are references only; the winner
is selected only among passing Phase B and C by fewer hard-safety violations, higher
mean reward, more completions, then fewer max-turn truncations.

Run the existing T2/T3 adversarial protocol on the selected winner. Only after
selection and red-team completion may the winner be evaluated once on all 60
untouched final-eval episodes (15 each from T0–T3).
