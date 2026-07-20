# Main training protocol

This protocol prepares **50 additional warm-started updates** from selected
smoke checkpoint `g1akido7qfo58e3my36wnqrz`. It does not describe one
uninterrupted 70-step optimizer trajectory: `checkpoint_id` proves weight
warm-start provenance, but optimizer-state restoration must not be claimed
unless later platform metadata explicitly proves it.

No command in this document authorizes paid work. Each phase is prepared,
manually approved, launched, captured, and validated before the next phase may
be prepared. The tool never launches training itself.

## Frozen curriculum

| Phase | Updates | Training rows | Input | Eval | Retention |
|---|---:|---|---|---|---|
| A | 10 (global 20→30) | 100% T1 | smoke step 20 | T0–T3 at 20, 30 | steps 25, 30 |
| B | 20 (global 30→50) | 40% T1 / 60% T2 | Phase-A final READY checkpoint | T0–T3 at 30, 40, 50 | steps 40, 50 |
| C | 20 (global 50→70) | 25% T1 / 45% T2 / 30% T3 | Phase-B final READY checkpoint | T0–T3 at 50, 60, 70 | steps 60, 70 |

Every training tier is its own `[[env]]` using
`jarrett/uav-operator@0.1.1`, the deterministic `train` split, 75 rows, and a
static ratio. Training remains Qwen3.5-2B RL/GRPO with batch 16, two
rollouts/example, four in flight, learning rate `3e-5`, LoRA alpha 32,
temperature 0.7, 1,024 tokens, thinking disabled, 20 turns, and zero retries.

Each evaluation milestone uses four separate dev environments (T0–T3), six
examples per tier, one rollout/example, temperature 0, 1,024 tokens, 20 turns,
zero retries, and the initial step. No T0 or final-eval row enters training.
Simulator, reward, prompt, generator, and split semantics remain frozen.
Training entries use unique names such as `train_t1`; evaluation entries use
`dev_t0` through `dev_t3`. Prime rejects repeated instances of the same Hub ID
unless each entry has a unique name.

Prime interprets Hosted Training `max_steps` as an absolute global target when
warm-starting. Therefore the three TOMLs use `max_steps = 30`, `50`, and `70`;
these still represent 10, 20, and 20 additional updates respectively. An
initial Phase-A launch attempt with `max_steps = 10` was rejected by the API
before run creation or billing, which supplied this platform evidence.

## Budget ledger

| Item | Expected | Hard ceiling |
|---|---:|---:|
| Existing diagnostic, smoke, and checkpoint comparison | $3.0469 | $3.0469 |
| Phase A | $1.45 | $1.50 |
| Phase B | $1.90 | $2.35 |
| Phase C | $1.95 | $2.40 |
| Post-training dev comparison | reserved | $1.75 |
| T2/T3 red-team | reserved | $0.75 |
| Frozen final evaluation | reserved | $1.25 |

Expected total is `$12.0969`; the sum of all hard ceilings is `$13.0469`,
leaving `$1.9531` below the aggregate `$15.00` cap. A pricing change recomputes
the phase projection from the paid smoke's measured tokens. It cannot silently
consume the unallocated amount. Missing pricing, capacity, wallet, Hub status,
or billing evidence fails closed.

On 2026-07-20 the first live Phase-A preparation stopped at this gate: the
effective inference-input price was `$0.05/M`, producing a conservative
`$1.4413` projection against the original `$1.25` ceiling. The user then
approved a narrow increase to `$1.50`, leaving the `$15.00` aggregate cap and
all downstream reserves unchanged.

## Manual commands and evidence

Generated bundles are ignored under `outputs/main-training/`. `prepare`
validates the source run/model/step and exact READY checkpoint, refreshes model
pricing/capacity, wallet, and Hub status, then writes `train.toml`, a SHA-256
manifest, usage projection, stop limits, and the exact manual launch command.

```bash
uv run python scripts/main_training.py prepare A
```

After reviewing the bundle and receiving separate approval, the operator runs
the manifest's `manual_launch_command`. The command is never executed by
`main_training.py`. Capture the completed phase with:

```bash
uv run python scripts/main_training.py capture A RUN_ID \
  --manifest outputs/main-training/phase-a/manifest.json
```

Only a passing capture is promoted to
`assets/training/main_phase_a.json` and its curve to
`assets/training/main_phase_a_curve.png`. Phase B and C add the predecessor
gate:

```bash
uv run python scripts/main_training.py prepare B \
  --predecessor-capture assets/training/main_phase_a.json
uv run python scripts/main_training.py prepare C \
  --predecessor-capture assets/training/main_phase_b.json
uv run python scripts/main_training.py budget assets/training/main_phase_*.json
```

Capture retrieves the run, exact TOML, metrics, distributions, usage, logs,
rollouts, checkpoints, and adapters. It requires completed expected steps,
finite rewards, zero provider errors/cancelled rows, exact retained artifacts,
a final READY checkpoint, matching checkpoint provenance, and a reconciled
phase cost below its ceiling. Model prose is stored only if the platform
rollout response necessarily includes it; it is never read or judged by a
gate. Reward and safety evidence remain simulator-derived.

## Stop and approval gates

Stop the active run, or withhold the next approval, for any of:

- non-finite reward or metrics;
- provider errors or cancelled rows;
- 15 minutes without optimizer progress;
- missing expected optimizer/evaluation steps;
- no exact final READY checkpoint and adapter;
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

## Phase A result (2026-07-20)

Run `r6n3cud398dkrhnsoqwxpsjk` completed global steps 20 through 30 and produced
exact step-30 adapter `npn2nu01kpb3u402x1igr86v`. Training cost `$1.1958`; the
clean exact-adapter T0-T3 validation cost `$0.2781461`; total Phase A cost
`$1.4739461`, below the `$1.50` ceiling.

The capture failed the preregistered truncation gate. T2 and T3 each reached
the 20-turn limit in 4/6 episodes (66.7%). T0 and T1 had 0/6 and 2/6 max-turn
stops. All 24 evaluations completed with saved simulator state/logs, zero
provider errors, and zero hard-safety penalties. Phase B is withheld. See
`assets/training/main_phase_a_exact_step30.json`.

## Post-Phase-C selection

Deploy the smoke-step-20 and Phase-A/B/C checkpoints. Evaluate all four on
identical T1–T3 dev workloads: six seeds per tier, two rollouts per seed,
temperature 0, 20 turns, no retries, and saved `sim_state`/`sim_log`. Report
paired-seed results. The smoke checkpoint is a before-training reference only;
the winner is selected among A/B/C by fewer hard-safety violations, higher
mean reward, more completions, then fewer max-turn truncations.

Run the existing T2/T3 adversarial protocol on the selected winner. Only after
selection and red-team completion may the winner be evaluated once on all 60
untouched final-eval episodes (15 each from T0–T3).
