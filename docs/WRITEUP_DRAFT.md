# Unpublished draft — training incomplete

> **Do not publish this draft.** No Hosted Training run has completed an
> optimizer step, no trained checkpoint exists, and no learning claim is
> supported. Every bracketed evidence gate below must remain empty until it can
> be populated by captured, non-fixture artifacts. Synthetic fixtures are
> plumbing tests and must never fill an evidence gate.

# Drones already fly themselves. I built the layer that does not.

A modern drone does not need a language model to hold altitude, follow a
trajectory, or return to launch when its battery crosses a threshold. Those
are control problems. The interesting gap sits one layer higher: the remote
pilot in command deciding whether a mission still makes sense when a forecast
is wrong, a temporary flight restriction appears across the route, the normal
recovery site closes, and the battery estimate degrades at the same time.

That is the chair I wanted a model to occupy. `uav-operator` is a multi-turn RL
environment for supervisory small-UAS operations over a simplified San
Francisco Bay Area. The model receives a briefing and uses an operator console
to file or amend a route, inspect weather and airspace, acknowledge alerts,
hold, divert, abort, or override a failsafe. The simulator owns the vehicle.
The model owns the operational judgment.

This distinction is also the boundary between this project and excellent UAV
work at other layers. [AirSim](https://github.com/microsoft/AirSim) provides
high-fidelity visual and physical simulation. [Flightmare](https://proceedings.mlr.press/v155/song21a.html)
decouples a Unity renderer from a quadrotor dynamics engine and supports
control and path-planning research. [gym-pybullet-drones](https://arxiv.org/abs/2103.02142)
provides a Gym-like, Bullet-based environment for multi-quadcopter control.
Those projects make sense when the policy controls a vehicle. Here, classical
control is deliberately inside the simulator because asking an LLM to
micro-correct a trajectory would test the wrong thing.

LLM-oriented UAV benchmarks are closer neighbors. [MultiUAV-Plat](https://arxiv.org/abs/2606.31073)
is an interactive platform and benchmark for multi-UAV task planning, with 75
mission sessions, 1,500 language tasks, REST APIs, partial observations, and
hidden validation. Its paper reports a 30.6% task pass rate for its ReAct
baseline and 57.9% for its Agent4Drone framework. [UAVBench](https://arxiv.org/abs/2511.11252)
is an open dataset of 50,000 structured, validated flight scenarios plus a
reasoning benchmark. I credit both as evidence that high-level UAV reasoning
deserves serious evaluation. The narrower contribution here is a trainable,
state-rewarded supervisory environment, not a claim to be the first LLM/UAV
benchmark. Whether it is the first UAV-operations environment on the Prime
Hub is deliberately unclaimed pending a fresh Day 13 search.

## An event-driven simulator with a competent autopilot

The simulator does not integrate flight dynamics at a fixed timestep. It
advances from one judgment point to the next. A route segment is evaluated
analytically from its WGS84 endpoints, commanded altitude and airspeed, the
seeded wind field, and the vehicle energy model. If an event is scheduled
inside the segment, the segment is split at the event time, the aircraft moves
to the interpolated position, and the console returns control to the model.

That design keeps the abstraction honest and makes training cheap enough to
be plausible. The autopilot follows accepted routes, holds altitude, executes
return-to-launch, and enforces simple battery and geofence failsafes. It does
not ask the model to fly. The model acts only at mission intake, event
interrupts, arrival checkpoints, and explicit console calls.

The wind is synthetic and seeded: a base flow, smooth moving perturbations,
altitude shear, and optional gust fronts. Segment duration and energy burn are
analytic, with all stochastic variation drawn from one NumPy generator. The
resulting contract is strict: the same seed and action sequence must produce
the same outcome and `sim_log`, bit for bit. That determinism supports replay,
offline rendering, regression tests, and adversarial reward audits without
pretending to reproduce atmospheric fluid dynamics.

## Building a curriculum without touching the final exam

The generator emits four tiers. T0 is a harness check: one mission, no event,
and generous margins. T1 introduces one event for which the conservative
rulebook answer is usually correct. T2 adds one or two events where blindly
following that rulebook can sacrifice mission value. T3 composes two to four
events so that wind, energy, airspace, recovery sites, and deadlines produce
incommensurable tradeoffs.

The shipped dataset has 300 train rows, 60 dev/calibration rows, and 60
withheld final-eval rows, stratified across T0–T3. The namespaces are isolated:
train, dev, and final-eval seeds do not overlap. Generator dials and frontier
calibration use dev rows only. The final-eval seeds have not been used for the
Day 12 preparation or any synthetic fixture. T2/T3 generation also applies an
analytic feasibility check, so difficulty comes from the decisions rather
than an impossible route silently entering the dataset.

## Reward from simulator state, never model prose

Every reward component reads the saved simulator trajectory. Mission value
requires the simulator to log a completed mission and decays after the SLA.
Hard safety counts canonicalized physical violations. Margin policy prices
poor reserve and override outcomes. Procedure prices unacknowledged alerts and
logged violations. Efficiency compares elapsed time and energy with the
scenario's analytic par.

The model can say “mission complete” as confidently as it likes; those words
do not change mission state. It can invent an authorization, claim a battery
level, or demand a reward of one million. The scorer never reads the claim.
This is stronger than asking another language model whether the answer sounds
safe: model prose is outside the reward's information boundary.

The Day 12 summarizer carries the same rule into post-training red-team work.
It validates saved state and schema, then recomputes every component and total
from `sim_log`. It rejects disagreement with the recorded reward. It never
inspects `prompt`, `answer`, or `completion`; tests corrupt those fields and
confirm the summary is unchanged. This matches the role of a Verifiers
environment: [Verifiers](https://github.com/PrimeIntellect-ai/verifiers)
packages task inputs, a model harness, and a scoring rubric for common RL and
evaluation workflows, while its [environment documentation](https://docs.primeintellect.ai/verifiers/environments)
supports custom multi-turn interaction and stateful tools.

## What the frontier calibration actually showed

The strongest training-independent evidence is the fixed-simulator Day 6 dev
calibration: 30 clean rollouts per model and tier, saved with run provenance.
GPT-4.1-nano averaged 0.884 on T0, -0.055 on T1, -0.140 on T2, and -0.196 on
T3. Laguna averaged 0.998, 0.424, 0.476, and 0.095. Laguna completed 30/30 T0,
22/30 T1, 29/30 T2, and 26/30 T3 missions; its T3 set included two hard-safety
outcomes. The corresponding run IDs and confidence intervals are in the
README leaderboard.

Those numbers support a limited claim: the harness is solvable, and composed
T3 events separate a capable model from ceiling. They do not show that any
model learned. The earlier Day 5 calibration is retained as provenance for
how the tiers were tuned, but Day 6 is the post-fix comparison because reward
and simulator bugs discovered in red-teaming had been closed.

The non-monotonic Laguna T1 result also became a useful debugging story. On
three T1 event families Laguna scored roughly 0.93–1.0. On the eight T1 cases
where a pop-up TFR covered the mission target, it scored -1.575 and exhausted
all 40 turns by repeatedly filing routes into the restriction. The simulator
was behaving consistently; the console was failing to make the impossibility
legible. The fix exposed `contains_mission_target` and warned that re-filing
would repeat the violation, while leaving abort or diversion to the model.
This was not a reward tune against held-out seeds. It was a dev-rollout lesson
about decision-loop observability.

## Five exploits closed, four defenses verified

The first red-team round drove nine adversarial classes through the real tool
path. Five revealed simulator or scoring defects:

1. A long `file_flight_plan` could skip events that should have fired
   mid-flight. Route segments now split at event boundaries.
2. Airborne holds advanced time without consuming energy. Loiter now burns
   hover power and can trigger the battery failsafe or loss.
3. A route could skim a no-fly polygon about 0.03 nautical miles outside its
   exact boundary. Authorization-required zones now have a 0.2 nm buffer.
4. Aborting before launch left a ground aircraft in a zombie episode. A
   resolved mission on the ground now terminates.
5. A degraded but never-launched battery could receive the airborne critical
   battery penalty. That condition now requires an airborne or holding state.

Four other attacks were closed at the construction level. Prose-only success
claims produced zero mission value. Repeated invalid overrides and conflicting
plans accumulated procedure penalties. Completing far beyond twice the SLA
decayed mission value to zero. Unknown-tool loops continued to consume
simulated time. The committed pre-fix and post-fix artifacts remain intact,
and the current scripted regression reports all nine classes closed.

Two live Day 6 adversarial prompts then asked models to maximize reward by any
means. GPT-4.1-nano's six T2 rollouts scored between -0.10 and -0.20. Laguna's
six T3 rollouts averaged 0.478; its best, 0.981, was an honest
detour-and-deliver flight. This is evidence that those prompted attacks did not
beat honest play on those twelve rows. It is not proof that no exploit exists,
which is why the trained-adapter round is prepared for later execution.

## Making the state visible in five seconds

Each rollout stores a full frozen-schema snapshot after every decision. The
renderer consumes only that log: it does not import the simulator or replay
live objects. A dark operations display shows the Bay route, planned and flown
tracks, wind, battery, alerts, and pop-up TFR geometry. The committed hero GIF
shows a T3 episode with four composed events and a safe landing; four other
clips cover a clean completion, low-margin success, a geofence struggle, and
an aircraft loss.

That visual is not decoration. A reader should be able to see the supervisory
problem before reading the reward table: a restriction blooms, the route
changes, the battery falls, and the vehicle either comes home or does not. It
is the five-second proof that the task involves operational judgment rather
than text classification.

## Hosted Training: three attempts, no learning result

Prime's [Environments Hub](https://www.primeintellect.ai/blog/environments)
is designed to connect shared environments to evaluation and RL workflows,
and Verifiers integrates with Prime's training stack. The project reached
inference and environment integration, but not training.

The first free Laguna XS Hosted run, `ed7ap9lbtm3lpy6pqeav7lrt`, completed a
finite step-0 baseline and then produced repeated generic `ModelError`
failures. It was stopped before optimizer step 1 with zero training tokens and
no checkpoint. A reduced-concurrency diagnostic,
`hg6jhftohpaognsubyoncy8s`, reproduced the step-0 failure after about 29
minutes. A separate renderer-client check returned HTTP 404 for the exact
Hosted model ID, while ordinary inference exposed a different Laguna model.
That mismatch is evidence for a policy-inference path problem, not a confirmed
root cause because the wrapped provider exception was unavailable.

The remaining free candidate, `sprints/Llama-3.2-1B-Instruct`, passed visible
price, capacity, wallet, and Hub-action checks but was rejected with HTTP 400
before run creation because the environment did not meet a separate free-tier
eligibility rule. The preview endpoint did not expose that eligibility. No run
ID, optimizer step, checkpoint, adapter, or billing row resulted.

All three attempts cost $0, and the captured wallet balances were unchanged
apart from unrelated history. A two-row Laguna M.1 dev evaluation did execute
the tool loop and produce renderable simulator logs. That proves inference
integration. It cannot be used as evidence of learning because the inference
model was not a checkpoint produced by the Hosted run. The same is true of a
finite step-0 baseline: evaluation before an optimizer update is not a
learning curve.

The future trained-adapter protocol is therefore prepared but unexecuted. It
will freeze six T2 and six T3 dev rows, preserve the exact Day 6 adversarial
prompt, require adapter/run/checkpoint/config provenance, and recompute reward
from saved logs. The script writes manual commands but never deploys or runs
them; nonzero inference prices require a token estimate and explicit cost
approval.

## Evidence gates that remain empty

- **[EVIDENCE REQUIRED — REAL LEARNING CURVE]** Captured checkpoint
  evaluations over frozen dev seeds, with run IDs, confidence intervals,
  provider-error accounting, and no synthetic points.
- **[EVIDENCE REQUIRED — CHECKPOINT SELECTION]** A stated selection rule and
  the captured run/checkpoint/adapter IDs chosen without final-eval tuning.
- **[EVIDENCE REQUIRED — LEARNED BEHAVIOR]** State-log evidence for a behavior
  that differs after training; inference competence or fixture deltas do not
  qualify.
- **[EVIDENCE REQUIRED — TRAINED-MODEL LEADERBOARD]** Captured results from the
  selected trained adapter, with workload and pricing provenance.
- **[EVIDENCE REQUIRED — SAME-SEED RENDER PAIR]** Before/after rollouts from
  the same dev seed and frozen environment, rendered from captured logs.
- **[EVIDENCE REQUIRED — DAY 12 ROUND TWO]** Captured T2/T3 adversarial runs
  summarized by the prepared state-only protocol. Synthetic review candidates
  do not qualify.

No fixture may populate, illustrate, approximate, or visually stand in for
any item in this section.

## Limitations

The wind field is synthetic, not a forecast product or computational-fluid
dynamics model. Airspace is simplified and real-shaped, not navigation-grade.
The simulator abstracts supervisory operations and cannot support claims about
low-level flight control, perception, collision avoidance, or real-world
airworthiness. Event-driven segment math trades physical fidelity for decision
density and deterministic replay.

Commanded RTL and `land_now` recovery legs remain uninterruptible and are not
fully validated against active TFR geometry. That unresolved semantic boundary
is documented rather than papered over. T4 multi-aircraft fleet operations
were cut at the Day 10 gate and belong to future work. Final-eval seeds remain
withheld. Most importantly, there is no trained checkpoint, no learning curve,
no trained-model leaderboard, and no round-two finding yet.

The honest project today is a deterministic supervisory environment, a
post-fix frontier calibration, a renderer, a documented adversarial round,
and a training-platform postmortem. The more exciting sentence—“I taught a
small open model to stop crashing”—remains a hypothesis until captured
checkpoint evidence earns it.
