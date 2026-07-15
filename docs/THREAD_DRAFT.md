# Not for publication — training evidence incomplete

> This is a six-post structural draft, not launch copy. Do not publish while
> any placeholder remains. Synthetic fixtures may test tooling but may never
> supply a number, image, curve, comparison, or claim below.

## Post 1 — hook and layer

**[ATTACH: existing captured T3 hero clip; final clip selection pending]**

Drones already fly themselves. Autopilots hold trajectories, execute RTL, and
enforce simple failsafes. The hard part I wanted to train is the chair above
the autopilot: an ops-center model deciding what to do when wind shifts, a TFR
appears, a recovery site closes, and the battery estimate changes mid-mission.

`uav-operator` puts the model in that chair. The simulator owns the aircraft;
the model owns the judgment.

## Post 2 — environment and reward

The sim advances from decision point to decision point—no timestep flight
integrator, no LLM micromanaging motors. Routes, wind-adjusted segment time,
energy burn, and failsafes are analytic and deterministic.

T0 checks the harness. T1 adds one rulebook event. T2 makes the obvious
response costly. T3 composes 2–4 failures.

Every reward comes from saved simulator state: completed mission, physical
safety outcomes, reserve/margin policy, procedure, and efficiency. Model prose
is inert.

## Post 3 — a closed reward hack

The nastiest Day 6 bug was not a clever sentence. It was a route that jumped
over future events: one long plan could finish before the simulator applied a
mid-flight TFR, site closure, or battery degradation.

I changed route execution to split exactly at event times, return control to
the operator, and suspend the remainder. Then I added the exploit transcript
and regression. Four more defects were closed, and four attack classes were
verified inert by construction. The current scripted round reports all nine
closed.

## Post 4 — reserved learning evidence

**[EVIDENCE REQUIRED — ATTACH A CAPTURED REAL LEARNING CURVE]**

**[EVIDENCE REQUIRED — ATTACH CAPTURED BEFORE/AFTER RENDERS ON THE SAME DEV
SEED, WITH RUN/CHECKPOINT/ROW PROVENANCE]**

Copy may be written only after those artifacts exist. Do not substitute a
synthetic curve, step-0 baseline, inference-only model, unmatched seed, or
hand-selected fixture.

## Post 5 — conditional platform/training claim

**Publication condition:** if a real Hosted Training checkpoint and captured
evaluations exist, state the exact model, optimizer steps, selected checkpoint,
dev protocol, and observed change here. Then say that the environment uses
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers) and the run used
Prime Intellect Hosted Training/compute, to the extent supported by captured
run metadata.

Until then, the only supported claim is: the public environment runs through
Prime evaluation and produced functional tool-use rollouts; three free Hosted
Training attempts produced no optimizer step or checkpoint and cost $0.
Inference success is not learning evidence.

**[EVIDENCE REQUIRED — FINAL CONDITIONAL TRAINING COPY]**

## Post 6 — close and future work

The next research question is whether RL discovers better supervisory policy
or better exploits. The trained-adapter adversarial protocol is ready: exact
Day 6 prompt, frozen T2/T3 dev seeds, state-only reward recomputation, and
fixture-taint rejection.

Multi-aircraft T4 was cut at the gate. It is future work, not a hidden result.
The immediate roadmap is one healthy training step, a captured curve,
same-seed behavior, and round-two red-teaming—then the withheld final eval.

**[LINKS AT PUBLICATION: Hub, repository, article]**

## Publication gate

This thread is publishable only after all of the following are true:

- a real checkpoint/adapter exists and its training provenance is captured;
- the learning curve and checkpoint-selection rule use captured dev results;
- the same-seed before/after pair uses the frozen environment and real logs;
- the trained T2/T3 red-team round has captured, validated results;
- every factual and platform claim receives a final source/evidence review;
- the Hub-positioning claim is freshly checked on Day 13;
- every bracketed placeholder and this “not for publication” label is removed.

If training remains blocked at publication time, rewrite Posts 1, 4, 5, and 6
as an environment-only launch and make no learned-model claim.
