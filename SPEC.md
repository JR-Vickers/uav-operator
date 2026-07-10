# SPEC.md — uav-operator environment design

Version: draft for build. Authority order: CLAUDE.md constraints > this spec >
convenience. Change this file when the design changes; it is the single source
of truth for what we are building.

## 1. Premise and layer model

Small-UAS operations over the SF Bay Area. The **simulator owns everything
below the operator**: flight dynamics (abstracted analytically), trajectory
tracking, failsafe rulebook. The **model owns the chair**: mission acceptance,
planning, replanning under events, failsafe overrides, abort/continue
tradeoffs. The model is invoked only at decision points:

1. **Mission intake** — briefing arrives, model must file a plan (or decline).
2. **Event interrupts** — sim pauses when an event fires or a failsafe is
   about to trigger; model gets telemetry + alert, must act or acknowledge.
3. **Arrival/completion checkpoints** — waypoint reached, payload action
   available, model confirms next leg.

An episode = one operational window (sim duration 20–90 min) containing 1–3
missions and 0–5 events, ending when all missions are resolved and the
aircraft is on the ground (or lost).

### 1.1 Professional environment bar

This is only a serious RL environment if it clears these checks by v0.1:
- T0–T3 scenarios exist, with T0 as harness sanity and T2/T3 as the real task.
- Scenarios come from a seeded generator, not only memorized hand examples.
- Conservative rulebook, reckless, and simple scripted baselines are reported.
- Difficulty separates by tier: T0 near ceiling, T3 visibly below ceiling.
- `docs/HACKS.md` contains closed reward-hacking attempts with regressions.
- `(seed, action_sequence)` replays to the same `sim_log`.
- Every rollout can be rendered offline from saved `sim_log` alone.
- The docs state the abstraction honestly: supervisory ops judgment, not
  low-level flight control or a 3D drone simulator.

## 2. World model

### 2.1 Static geography (built once, bundled as JSON ≤200KB)
- Bounding box: (37.55, -122.60)–(37.95, -122.15). Great demo terrain: water
  crossings, dense urban, hills.
- **Airspace**: simplified but real-shaped polygons — SFO Class B core +
  shelves (hard floor/ceiling per shelf), OAK Class C, SQL/HWD Class D rings,
  plus 3–4 fixed no-fly polygons (stadiums flagged event-conditional). Encode
  as `{polygon, floor_ft, ceiling_ft, class, authorization_required}`.
  Authoritative-enough shapes traced manually from public charts; we are
  simulating the *structure* of airspace, and README says so.
- **Sites**: ~12 launch/land/charging sites (vertiports) + ~30 delivery or
  inspection target points, each `{latlon, name, type}`. Names real-ish
  ("Mission Bay Pad", "Alameda Depot") for demo legibility.
- **Obstacle proxy**: max-terrain+structure elevation raster at ~500m grid;
  min safe altitude per cell = elev + 100ft. No 3D obstacles.

### 2.2 Wind field (the star of the show)
- Synthetic, seeded, HRRR-styled. Base flow (direction, speed sampled per
  scenario, e.g. 240°@12kt) + 2–4 smooth Gaussian-blob perturbations that
  translate/evolve over sim time + optional **gust front**: a moving line
  across the box that adds +15–25kt and 40–90° shift behind it, with a
  forecast arrival time given in the briefing (± error).
- API: `wind_at(latlon, alt_ft, t) -> (dir_deg, speed_kt)`. Pure function of
  seed — cheap, deterministic, renderable as a quiver field.
- Altitude shear: speed scales ~+15%/1000ft, direction veers slightly. Creates
  a real decision (altitude choice trades wind for airspace ceilings/energy).

### 2.3 Aircraft and energy model (analytic — NO integration)
- One airframe class: quad, cruise 35kt airspeed (commandable 20–45kt),
  battery 400 Wh nominal, payload 0–2.5kg.
- Segment time: `t = dist / groundspeed`, groundspeed = airspeed + wind
  component along track (vector math; if headwind ≥ airspeed, segment
  infeasible — autopilot refuses, tells the operator).
- Energy: `P = P_hover(mass) + k_drag * airspeed³_scaled` per segment, plus
  climb energy `m·g·Δh` equivalent, plus 8% reserve burn on landing sequence.
  Calibrate constants so a full-battery crossing of the box against 20kt wind
  is *marginal* — margins are the gameplay.
- Execution noise: per-segment time/energy multiplier ~N(1.0, 0.03), seeded.
  Enough that the model can't plan to 0.1% margins; small enough to be fair.

### 2.4 Autopilot + failsafe rulebook (sim-owned)
- Flies filed waypoint routes at commanded alt/speed; auto-holds at waypoints
  pending operator confirm where the plan marks `confirm: true`.
- Failsafes (fire automatically UNLESS a valid operator override is active):
  - `LOW_BATT_RTL`: battery ≤ 30% → return to nearest recovery site.
  - `CRIT_BATT_LAND`: ≤ 12% → land immediately at current position (uncontrolled
    site = equipment risk + possible mission fail). NOT overridable.
  - `GEOFENCE_HOLD`: predicted incursion into unauthorized airspace → hold.
  - `LINK_LOSS`: comms event → autonomous RTL after 60s unless pre-briefed
    lost-link plan says continue.
- Overrides: `override_failsafe(id, justification_code)` where justification
  is an enum (e.g. `MISSION_CRITICAL_MARGIN_OK`, `SAFE_LANDING_ASSURED`) — an
  enum so reward logic can price it without parsing prose. Overriding
  `LOW_BATT_RTL` with margin that physics later validates = good judgment;
  override followed by `CRIT_BATT_LAND` in the water = the money screenshot.

## 3. Interface to the model

### 3.1 Environment class
`vf.MultiTurnEnv` subclass (tool-calling). `env_response` executes tool calls
against `SimState`, advances sim to next decision point, returns formatted
telemetry/alert message. Termination via `@vf.stop`: all missions resolved &
aircraft down, or aircraft lost, or `max_turns` (default 40), or sim-time cap.

### 3.2 Tools (the operator console — keep to this set)
Read: `get_telemetry()`, `get_weather(latlon?)` (current + coarse forecast incl.
gust-front ETA w/ error), `get_airspace(route?)` (zones + active TFRs along
route), `get_mission_status()`, `get_sites()`.
Act: `file_flight_plan(waypoints, alt_ft, airspeed_kt, lost_link_plan)`,
`amend_route(waypoints_from_current)`, `set_altitude(ft)`, `set_speed(kt)`,
`hold(minutes)`, `resume()`, `command_rtl(site?)`, `land_now(site_id)`,
`release_payload()` (only at delivery point), `abort_mission(mission_id)`,
`override_failsafe(id, justification_code)`, `acknowledge(alert_id)`.
- Plans are validated on filing: airspace conflicts and infeasible segments
  are returned as structured warnings (operator may file anyway — knowingly
  filing through a TFR is on them, and on the reward).
- Every tool call costs 10–30s sim time (decision latency is real).

### 3.3 Messages to the model
- System prompt: role (remote PIC), ops manual summary (failsafe rulebook,
  reward-relevant policy: reserve requirements, TFR policy, payload SLAs),
  tool docs. The ops manual IS the rubric made legible — models should lose
  for violating briefed policy, not unbriefed trivia.
- Briefings and some events carry **semantic payloads**: NOTAM-style TFR text
  ("...3nm radius of STANFORD STADIUM sfc-2000ft AGL eff 2145Z..."), a
  customer note ("gate code 4471, do not land on helipad H2 if flag is up"),
  a fragmentary ops-channel message. Parsing unstructured text into action is
  a core tested skill — inputs are text, but *verification never is* (the TFR
  polygon exists in sim ground truth; the text merely announces it).

## 4. Event taxonomy (scenario generator draws from these)

| Event | Params | The judgment it forces |
|---|---|---|
| WIND_SHIFT | Δdir, Δspeed, onset | Replan route/alt vs continue; energy math changes |
| GUST_FRONT_EARLY | ETA error realized | Race it, wait it out, or divert |
| TFR_POPUP | polygon, floor/ceil, eff. time, NOTAM text | Reroute (+time/energy) vs hold vs abort |
| BATT_DEGRADE | capacity −10..25% | All margins recompute; RTL threshold now wrong-ish |
| LINK_FLICKER | duration, recurrence | Trust lost-link plan vs preemptive RTL |
| PAYLOAD_ISSUE | jam / shift | Deliver anyway? land & check? value vs risk |
| TRAFFIC_ADVISORY | helicopter transit corridor + times, text | Hold/alt-change window compliance |
| MISSION_UPDATE | new priority / cancellation / addendum text | Re-sequence multi-mission plan |
| SITE_CLOSED | recovery site unavailable | Recompute reserves against farther site |

Tier 2+ scenarios compose events so the rulebook answer is wrong or ambiguous
(e.g., BATT_DEGRADE + TFR_POPUP: the failsafe RTL route now crosses the TFR).
The generator verifies each scenario has ≥1 feasible resolution path
(solver-checked at generation time) — no unwinnable episodes in train set.

## 5. Reward (verifiers Rubric — weighted components, all sim-derived)

| Component | Weight | Definition |
|---|---|---|
| mission_value | +1.0 scale | Per-mission value × completion × timeliness decay (briefed SLA curve); partial credit only where briefed |
| hard_safety | −5.0 each, may terminate | Logged aircraft loss and critical-battery airborne outcomes; TFR/airspace incursions join here as dynamic incursion logging matures |
| margin_policy | −0.0..−1.0 shaped | Landing reserve below briefed 20%, poor-outcome overrides, min-safe-alt violations |
| procedure | −0.1 each | Unacknowledged alerts, filing through known conflicts, expired holds |
| efficiency | −0.3 scale | Energy + sim-time cost normalized by generated scenario par |

Design invariants: (a) components computed ONLY from `sim_log`; (b) the
scripted **rulebook baseline policy** (never overrides, always obeys failsafes,
naive replan) is implemented in `scripts/baselines.py` and must score
*mediocre* on Tier 2+ — if the rulebook bot matches frontier LLMs, the tier
isn't testing judgment, regenerate it; (c) a **reckless baseline** (always
override, always push) must score *badly* — if it doesn't, hard_safety is
underpriced. These two bots are the reward's unit tests and run in CI.

Day 4 v0 implementation note: the active T1 event set is `WIND_SHIFT`,
`TFR_POPUP`, `BATT_DEGRADE`, and `SITE_CLOSED`. `TRAFFIC_ADVISORY` and
`PAYLOAD_ISSUE` are cut under the Day 4 fallback rule until later scope is
explicitly reopened.

## 6. Curriculum tiers (difficulty dials: event count/severity, margin
tightness, deadline pressure, brief ambiguity)

- **T0 sanity**: 1 mission, no events, fat margins. Any competent model ≈
  ceiling. Exists to prove the harness, floor the leaderboard.
- **T1 rulebook**: 1 event whose rulebook answer is correct. Tests parsing +
  procedure. Frontier ≈ high, small models mixed.
- **T2 judgment**: 1–2 events where naive rulebook response is suboptimal but
  safe path exists (the FDE tier — this is the product).
- **T3 bad day**: 2–4 composed events, incommensurable tradeoffs, semantic
  red herrings, tight margins. Target: frontier models visibly below ceiling.
- **T4 fleet (STRETCH — gate per PLAN.md)**: 3 aircraft, shared airspace,
  dispatcher-level tools. Do not touch before Day 10 gate.

Dataset: generator emits `{seed, tier, scenario_config, briefing_text}` rows;
v0.1 ships ≥300 train / ≥60 eval, stratified by tier. Eval seeds withheld from
any tuning decisions.

## 7. Rendering (dev-only pipeline, `renderer/`)

Input: rollout JSON (sim_log). Output: mp4/GIF. Layers: basemap tiles
(contextily, cached), airspace polygons (TFRs animate in with NOTAM ticker),
wind quiver (animated), trajectory with breadcrumb + planned-vs-flown, HUD
strip (battery bar, clock, alerts, last operator action). Style: dark ops-
console aesthetic. One command per CLAUDE.md. Money shots to engineer for:
TFR blooming mid-flight + visible reroute; battery bar going red over the Bay
after a bad override; before/after-training pair on the same seed.

## 8. Red-team protocol (`docs/HACKS.md`)

Rounds at Day 6–7 (frontier models + prompted-adversarial runs) and Day 12
(the trained model — RL finds what prompting doesn't). For each exploit:
transcript → root cause → fix → regression test. Anticipated classes to probe
first: success-claiming (should be inert by construction — verify), hold/stall
farming (time decay must dominate), boundary-hugging TFR polygons (buffer
check), override-spam (justification enum + outcome pricing), abort-everything
conservatism (mission_value opportunity cost), partial-credit farming,
tool-error time-wasting loops.
