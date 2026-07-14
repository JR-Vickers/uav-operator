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

**Day 7 — renderer + soft-launch. GATE.** Matplotlib/contextily pipeline to
mp4; render 5 best episodes from baselines; `prime env push` as v0.0.x
(public but unannounced); README skeleton with quickstart + one GIF.
GATE CRITERIA: a stranger could install from Hub and reproduce an eval; one
rendered clip passes the 5-second test on its own (show someone at NS cold —
do they get it?). Fallback: static trajectory PNGs + annotated stills instead
of animation; do NOT let animation polish eat Phase 3.
All gated evals should print the saved `run_id` and `results_path` in the
terminal summary so cropped output is still traceable without opening the
artifact directory.

**Day 8 — training prep.** prime-rl configs (orchestrator/trainer/inference
TOMLs) for Qwen3-4B-Instruct LoRA GRPO on a small PI pod; smoke run (50
steps, T1-only) to shake out reward scaling, context length, turn truncation.
Budget check against PI credits; pick main-run size accordingly.

**Day 9 — smoke-run postmortem + curriculum config.** Fix what the smoke run
exposed (it will expose things: reward normalization, degenerate rollouts,
turn caps). Decide curriculum schedule (T1→T2 mix vs mixed-from-start) from
smoke evidence. Freeze env v0.1.0 for the main run.

## Phase 3 — the money shots (Days 10–14)

**Day 10 — main training run launches. GATE (fleet decision).** Main GRPO run
on PI compute; monitor; eval checkpoints on withheld seeds every N steps.
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
- **PI credit exhaustion** — smoke run sizes the main run; if credits are
  tight, shrink model context/turns before shrinking the run count evidence.
- **The 6.5 trap** — Day 7 and Day 13 both include an external cold-viewer
  check. If the 5-second test fails with a real human, fix the artifact, not
  the viewer.
- **Decision-loop traps** — Day 4 T1 can waste many turns on repeated
  geofence/override attempts. Treat that as a signal to harden the interrupt
  path and procedure pricing before widening tier coverage.
