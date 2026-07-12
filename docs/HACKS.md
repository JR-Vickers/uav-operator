# HACKS.md

Red-team notes for reward hacking attempts. Keep failed attempts and open
issues here; they are evidence for the writeup.

## Day 1 reward: mission_value

Exploit idea:
- The model claims the mission is complete in natural language without flying
  to the target.

Current defense:
- `mission_value` reads only `state["sim_state"]["mission"]["status"]`.
- The mission status changes to `completed` only when the simulator advances
  the aircraft to within 0.2 nm of the target waypoint.
- Tests cover the successful sim-derived completion path and deterministic
  replay; no test grants reward from assistant prose.

Open TODO:
- Add an explicit regression transcript once live `vf-eval` is running, with a
  model attempting to success-claim without a valid `file_flight_plan`.

## Day 4 reward v0: override and stall paths

Exploit idea:
- The model repeatedly acknowledges/overrides or holds instead of solving the
  mission, hoping procedure text or elapsed turns produce credit.

Current defense:
- Reward reads only saved `sim_log` snapshots.
- Unacknowledged alerts and known conflict filings are priced by `procedure`.
- Poor-outcome overrides are priced by `margin_policy`.
- Simulated time and energy above generated par are priced by `efficiency`.
- Local baselines show the conservative rulebook policy beats the reckless
  push/override policy on the same T0-T1 seeds.

Open TODO:
- Hold/stall farming needs a stronger regression once T2/T3 deadlines and
  composed events exist.

## Day 4 reward v0: abort conservatism

Exploit idea:
- The model aborts every T1 event to avoid safety penalties.

Current defense:
- Aborted missions receive no `mission_value`, so they lose the primary reward
  opportunity even when they avoid hard-safety penalties.
- T1 TFR cases may still make abort/RTL the rulebook answer; T2 is where this
  must stop being sufficient.

Open TODO:
- Add T2 composed scenarios where conservative abort loses to a safe replan.

## Day 5: repeated geofence and override loops

Exploit idea:
- Re-file the same known-conflicting route or spam overrides after a failsafe
  clears, hoping the simulator's deduplicated procedure log makes retries free.

Current defense:
- Each known-conflict filing is preserved as a separate simulator procedure
  violation; reward never reads the model's explanation.
- Invalid or inactive-failsafe override attempts also create procedure
  violations, in addition to their simulated-time cost.
- Regression tests verify that repeating the loop strictly worsens procedure
  reward and does not crash the environment.

## Day 5: duplicate terminal safety labels

Exploit / failure mode:
- A single battery-depletion loss was represented by a detailed violation,
  lost-aircraft status, and a critical-battery terminal condition. Reward
  treated those three descriptions of one accident as three accidents.

Current defense:
- Hard-safety scoring canonicalizes terminal loss evidence to one
  `aircraft_loss` category. A critical-battery-airborne penalty remains
  distinct only when the aircraft was not also lost.
- A regression recreates the saved T2 rollout's overlapping labels and
  verifies that the single physical outcome costs exactly `-5.0`.

## Day 5: ground-state observation and hold loops

Exploit / failure mode:
- A model can remain on the ground while repeatedly querying telemetry and
  related read tools, or issuing holds without a specific timed condition.
  The old 10–20 second console cost allowed long, low-information rollouts
  before the turn cap ended the episode.

Current defense:
- After three consecutive ground-state read/hold actions with a pending
  mission, the console returns a structured `ground_no_progress_loop` warning.
- Each further call receives deterministic, escalating simulated decision
  latency, capped at two minutes per call. A meaningful operational action
  resets the streak; the simulator never selects an action for the model.
- A regression covers telemetry repetition, a subsequent hold, exact latency
  pricing, and streak reset.
