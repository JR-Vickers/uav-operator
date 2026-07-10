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
