# PLAN.md — 14 days, gated

Rules: gates are honest — evaluate at end of gate day, cut scope per the listed
fallback, never push a gate. Publishing something excellent and smaller beats
missing the window. Every day ends with green tests, a commit, and HANDOFF.md
updated. Renderer and writeup are load-bearing deliverables, not decoration —
they get real days, protect them.

## Phase 1 — a world that runs (Days 1–4)

**Day 1 — skeleton end-to-end.** `prime env init` template filled in:
`SimState`, event loop advancing decision-to-decision, 3 tools
(`get_telemetry`, `file_flight_plan`, `command_rtl`), trivial reward
(mission_value only), 5 hardcoded T0 scenarios, straight-line autopilot, flat
wind. DONE = `vf-eval uav-operator -m gpt-4.1-mini -n 2 -r 1` completes and a
rollout JSON exists. Nothing may be pretty.

**Day 2 — geography + energy.** Static world JSON (airspace polygons, sites,
obstacle raster) built via `scripts/build_world.py`; haversine/segment math;
energy model + calibration test ("box crossing against 20kt is marginal");
autopilot failsafes LOW_BATT_RTL + GEOFENCE_HOLD. Unit tests for geometry and
energy invariants.

**Day 3 — wind + full console.** Seeded wind field w/ perturbations + gust
front; altitude shear; full tool set from SPEC §3.2 incl. overrides and
validation warnings; sim-time costs per call; determinism CI test.

**Day 4 — reward v0 + first contact. GATE.** Full rubric per SPEC §5; both
baseline bots; scenario generator emitting T0–T1; run frontier models (via
Prime Inference) on ~20 episodes.
GATE CRITERIA: (a) end-to-end determinism holds; (b) rulebook bot beats
reckless bot; (c) at least one frontier model does something *interesting*
(non-rulebook action, good or bad) in a T1 episode. If (c) fails, the action
space is too thin — fix decision density before building more world. If the
interesting T1 behavior collapses into a repeated `GEOFENCE_HOLD` / override
loop, treat that as a reward/procedure signal, not a successful baseline.
Add the missing regression before expanding tier scope.
Fallback if behind: cut TRAFFIC_ADVISORY + PAYLOAD_ISSUE events permanently.

## Phase 2 — a task worth training on (Days 5–9)

**Day 5 — T2/T3 generation + calibration. COMPLETE.** Composed-event scenarios,
feasibility solver check, stratified dataset v0 (300/60). Run frontier
baselines across tiers; plot tier-vs-score. Target shape: T0 ≈ ceiling, T3
clearly below ceiling with variance. Regenerate dials until shape appears.
Include the Day 4 postmortem here: repeated geofence filing / failed override
loops should get a regression test and a sharper penalty or clearer interrupt
response before T2/T3 work continues.

Closed with zero-provider-error, 30-rollout-per-tier curves for GPT-4.1-nano
and Laguna. GPT-4.1-nano scores `0.872`, `0.134`, `-0.137`, and `-0.199` from
T0 through T3; Laguna scores `1.000`, `0.282`, `0.479`, and `0.504`. Confidence
intervals and run provenance are committed. Proceed to Day 6; investigate
Laguna's non-monotonic tier interaction without tuning against final-eval seeds.

**Day 6 — red-team round 1. COMPLETE.** Adversarial prompting runs ("maximize
score by any means"), exploit hunting per SPEC §8 list, fixes + regression
tests, start `docs/HACKS.md`. Also today: rollout-JSON schema freeze (renderer
depends on it).

Closed with five exploits fixed and regression-tested (mid-flight event
skipping, free airborne loitering, no-fly boundary hugging, abort-on-ground
non-termination, ground crit-battery false penalty), four classes verified
inert, schema frozen at v1 (`docs/SCHEMA.md`), and two live adversarial runs
(`a32af316`, `a44f3e91`) that extracted no reward above honest play. The Day 5
Laguna T1 anomaly is explained (TFR-on-target decision-loop trap, now legible
via `mission_target_inside_zone`); HACKS.md holds 5 closed exploits, clearing
the Day 10 fleet-gate prerequisite. The fixed-sim tier curve now has 30 clean
rollouts per model/tier: Laguna drops from `0.476` on T2 to `0.095` on T3,
providing the intended composed-event separation for Day 7.

**Day 7 — renderer + soft-launch. COMPLETE.** Matplotlib/contextily pipeline to
mp4; render 5 best episodes from baselines; `prime env push` as v0.0.x
(public but unannounced); README skeleton with quickstart + one GIF.
GATE CRITERIA: a stranger could install from Hub and reproduce an eval; one
rendered clip passes the 5-second test on its own (show someone at NS cold —
do they get it?). Fallback: static trajectory PNGs + annotated stills instead
of animation; do NOT let animation polish eat Phase 3.
All gated evals should print the saved `run_id` and `results_path` in the
terminal summary so cropped output is still traceable without opening the
artifact directory.

Closed with public `jarrett/uav-operator@0.1.1`, five frozen Laguna T3 clips,
a 960px hero GIF, contact sheet, cached mission-scale Contextily maps, and the
offline `--no-basemap` fallback. The approved viewport puts mission endpoints
near opposite map edges. An isolated exact-version install exposed and
documented the required prerelease dependency flag; saved-state eval
`407bb371` ran `openai/gpt-4.1-nano` for two examples with zero provider errors,
and its saved `sim_log` rendered successfully from outside the repository.

**Day 8 — free training-pipeline validation. ENVIRONMENT GATE COMPLETE;
HOSTED TRAINING GATES PENDING.** Hosted Training's single TOML supersedes the
obsolete separate orchestrator/trainer/inference files and manual small-pod
orchestration. This phase authorizes only models whose effective training,
inference-input, and inference-output prices are all $0 immediately before
launch. A paid model is not a fallback. Any paid experiment requires a later
PLAN amendment, a stated budget and purpose, and separate explicit approval.
The inference model used to validate environment plumbing may differ from the
Hosted Training model, but such runs prove integration only and cannot be used
as evidence of learning or compared as a before/after pair.

Day 8 has three ordered gates:

1. **Environment/tool-loop gate — COMPLETE.** Saved eval `6d420fc1` ran two T1
   dev episodes on the free `poolside/laguna-m.1` inference model. It produced
   valid tool calls and simulator logs, one successful mission (reward 0.9824),
   one 40-turn TFR loop (reward -1.3), zero provider errors, and zero reported
   cost. This proves the published environment can execute functional model
   rollouts. It does not prove that Hosted Training works or that reward rises.
2. **One-step Hosted Training diagnostic — PENDING.** Use a model in the live
   Hosted Training catalog that is still effectively free. The current
   candidate is `sprints/Llama-3.2-1B-Instruct`, subject to a fresh availability,
   capacity, exact model-ID/client compatibility, pricing, wallet, and Hub
   quality-action preflight. Keep this diagnostic deliberately small (one
   training step, batch 16, two rollouts/example, at most four in flight) and
   use a 20-turn cap to bound the looping behavior observed in the functional
   eval. Passing requires one real optimizer step, finite metrics, retrievable
   sampled rollouts, and zero cost.
3. **Free 50-step T1 smoke — PENDING.** Only after the one-step diagnostic
   passes, run a 50-step LoRA GRPO smoke on the same free, trainable model.
   Include a pre-training baseline and evaluation every 10 steps on all 15 T1
   dev rows, save checkpoints/adapters every 10 steps, and retain enough
   sampled rollouts to diagnose context length, turn caps, reward collapse,
   and tool use. No final-eval seed may be used.

Re-check model availability/capacity, effective prices, wallet, and Hub quality
action immediately before every launch. Present the exact command and expected
workload before requesting explicit approval. Stop for non-finite rewards,
repeated provider/context failures, 15 minutes without progress, unexpected
positive billing, or more than 50% max-turn truncation by step 10. Do not alter
simulator or reward semantics merely to accommodate a provider/model failure.
Turn/context controls may be adjusted when rollout evidence justifies them.

Run `ed7ap9lbtm3lpy6pqeav7lrt` completed a finite pre-training baseline, then
emitted repeated generic `ModelError` failures across training rollout groups.
It was stopped under the planned failure rule before step 1 with zero training
tokens, zero checkpoints, zero cost, and an unchanged wallet. The artifact and
bounded diagnosis are in `assets/training/day8_smoke.json` and
`docs/DAY8_SMOKE.md`. Day 8 did not pass; do not relaunch the config unchanged.

If no currently free Hosted Training model can pass the one-step diagnostic,
record Day 8 as **BLOCKED by the free training platform/catalog** with the run
IDs and provider evidence. Do not spend credits and do not repeatedly reshape
the environment to chase infrastructure failures. Day 8 is complete only when
the free 50-step smoke reaches step 50, produces a final checkpoint/adapter,
has finite retrievable metrics and reward distributions, exposes useful sampled
rollouts, uses no final-eval seeds, reports $0 total cost, and reconciles to an
unchanged project wallet apart from unrelated billing.

**Day 9 — free smoke-run postmortem + curriculum config.** Fix only what the
completed smoke's evidence shows (reward normalization, degenerate rollouts,
turn caps, or curriculum difficulty). Decide T1→T2 versus mixed-from-start from
that evidence and freeze the environment for the main run. No larger run may
launch until one healthy optimizer step and usable sampled rollouts exist.
The first Laguna XS diagnostic held turn/context limits constant while reducing
the run to one step, batch 16, and four maximum in-flight rollouts; it tested
the leading concurrency/shared-inference hypothesis before changing context.
Run `hg6jhftohpaognsubyoncy8s` reproduced the step-0 failure after a clean
baseline and approximately 29 minutes with zero training-token progress.
Concurrency alone is rejected. Pause config experiments and obtain the wrapped
policy-inference exception or platform confirmation before another launch.
The subsequent renderer-client reproduction returned HTTP 404 for the exact
Hosted model ID, while the live inference catalog listed only the distinct
zero-cost `poolside/laguna-m.1`. Use the committed two-row Laguna M.1 functional
eval as the completed environment/tool-loop gate. Laguna M.1 is not in the
Hosted Training catalog and cannot be substituted into the training TOML. The
next diagnostic therefore uses a separately verified free Hosted Training
candidate; absence from the ordinary inference catalog is a compatibility risk
to test, not grounds to switch to a paid model.

## Phase 3 — the money shots (Days 10–14)

**Day 10 — main free training run launches. GATE (fleet decision).** Size the
run from the completed 50-step smoke's throughput, truncation, and learning
evidence; 150 and 300 steps are options, not commitments. The selected model
must remain effectively free at launch, and the same availability, pricing,
wallet, quality-action, and explicit-approval gates apply. If Day 8 is blocked,
do not substitute a paid run: narrow the project claim or defer training until
a free compatible model is available. When unblocked, monitor and evaluate
checkpoints on dev seeds every N steps; final-eval seeds remain untouched until
the frozen evaluation.
FLEET GATE: T4 unlocks ONLY if the run is launched and healthy by end of day
AND HACKS.md has ≥3 closed exploits. Otherwise T4 is cut — write one honest
"future work" paragraph and never look back.

**Day 11 — curve + iterate.** If curve rises: extend run / harvest checkpoint
evals; render the before/after pair on identical seeds. If flat: triage in
order — reward scale, task too hard (shift mix toward T1/T2), context
truncation. A modest-but-real curve on T1–T2 with honest analysis beats a
flat curve on T3. Adjust claim to match evidence, not vice versa.

**Day 12 — red-team round 2 + writeup draft.** Run the *trained* model
adversarially — RL-discovered exploits are the best content in the whole
project; document + fix (or document honestly as open). Draft writeup: the
layer-model argument, prior-art positioning (RESEARCH.md), reward design +
hacks narrative, curve, limitations. Draft the thread (RESEARCH.md outline).

**Day 13 — polish + package.** README final (leaderboard, tables, GIFs);
`prime env push` v0.1.0; blog post final; video cut (≤90s: hook clip → tier
montage → hack screenshots → curve → before/after); repo hygiene pass
(license, HANDOFF.md removed or cleaned, no credentials, no workspace files).

**Day 14 — ship + outreach.** Publish blog. Post thread (SF-morning timing =
Malaysia evening). Same day: DMs to Will Brown + Johannes per RESEARCH.md §5
(one personalized sentence + link, no ask). Hub listing description final.
Reply to every substantive response same-day. Buffer for whatever broke.

## Standing risks

- **Sim scope creep** — the CLAUDE.md ugliness rule exists for Days 1–4;
  reread it when tempted.
- **Training run stalls burn calendar** — that's why launch is Day 10, not 12;
  Days 11–13 tasks are deliberately parallelizable with a running job.
- **Free-model catalog volatility** — a $0 listing can disappear, change price,
  hit capacity, or expose incompatible model IDs across Hosted Training and
  inference. Re-run the full preflight before every launch. If no free model
  works, preserve the evidence and mark the gate blocked; paid credits are not
  an automatic escape hatch.
- **The 6.5 trap** — Day 7 and Day 13 both include an external cold-viewer
  check. If the 5-second test fails with a real human, fix the artifact, not
  the viewer.
- **Decision-loop traps** — Day 4 T1 can waste many turns on repeated
  geofence/override attempts. Treat that as a signal to harden the interrupt
  path and procedure pricing before widening tier coverage.
