# Rollout JSON schema — v1 (frozen 2026-07-14, Day 6)

The renderer and red-team tooling consume ONLY saved rollout artifacts, never
live sim objects. This document freezes the shape of those artifacts. Any key
addition, removal, or semantic change requires bumping this version, updating
`tests/test_day1_env.py::test_day6_rollout_snapshot_schema_is_frozen`, and
updating `renderer/`.

## Where the data lives

`vf-eval ... --save-results --state-columns sim_state,sim_log` writes
`results.jsonl`, one rollout per line. Each row carries the standard Verifiers
columns (`prompt`, `completion`, `reward`, per-component metrics, ...) plus:

- `sim_state`: the final `SimState` dict (same shape as a snapshot minus the
  three derived fields below).
- `sim_log`: list of snapshots, one per decision point — `briefing` first,
  then one per executed tool call or environment event. **The renderer reads
  only `sim_log`.**

## Snapshot (one `sim_log` entry)

Every snapshot has exactly these top-level keys:

| Key | Type | Semantics |
|---|---|---|
| `event` | str | What produced this snapshot: `briefing`, a tool name, `no_tool_call`, or `sim_time_cap_reached` |
| `battery_pct` | float | Derived: `100 * aircraft.battery_wh / 400` |
| `mission_distance_to_target_nm` | float | Derived great-circle distance |
| `seed` | int | Episode seed |
| `scenario_id` | str | e.g. `T2-010-treasure-island` |
| `tier` | str | `T0`..`T3` |
| `sim_time_s` | float | Simulated seconds since briefing |
| `aircraft` | dict | See below |
| `mission` | dict | See below |
| `home_site_id` | str | Launch/home site id |
| `wind` | dict | `base_dir_deg_from`, `base_speed_kt`, `blobs` (list), `gust_front` (dict or null) |
| `rng_state` | dict | NumPy PCG64 state; `state.state` / `state.inc` are decimal strings |
| `scenario_par` | dict | `time_s`, `energy_wh` generation-time par |
| `events` | list | Scenario events; see below |
| `active_events` | list[str] | Active, unacknowledged event ids |
| `acknowledged_events` | list[str] | Acknowledged event ids |
| `closed_sites` | list[str] | Closed recovery site ids |
| `hard_safety_violations` | list[str] | e.g. `aircraft_loss:battery_depleted` |
| `procedure_violations` | list[str] | e.g. `filed_known_conflict:<zone_id>` |
| `current_plan` | list | Filed waypoints `{lat, lon, name}` |
| `lost_link_plan` | str | Briefed lost-link behavior |
| `rng_draws` | int | Execution-noise draws so far |
| `active_failsafe` | str or null | `GEOFENCE_HOLD`, `LOW_BATT_RTL`, or null |
| `alerts` | list | `{alert_id, sim_time_s, message, metadata}` |
| `acknowledged_alerts` | list[str] | Acknowledged alert ids |
| `overrides` | list | `{id, justification_code, sim_time_s}` |
| `payload_released` | bool | |
| `hold_until_s` | float or null | End of last commanded hold |
| `current_altitude_target_ft` | float | |
| `current_airspeed_kt` | float | |
| `ground_no_progress_streak` | int | Ground stall-pricing streak |
| `last_action` | str | Last tool name (or `briefing`) |
| `last_result` | dict | Last tool result; free-form except `segments` (below) |
| `is_terminal` | bool | |
| `terminal_reason` | str or null | e.g. `aircraft_landed_mission_completed`, `aircraft_ground_mission_failed`, `aircraft_lost_battery_depleted`, `sim_time_cap_reached` |

### `aircraft`

`{lat: float, lon: float, alt_ft: float, status: str, battery_wh: float,
current_site_id: str|null}` — `status` ∈ `ground | airborne | holding |
landed | lost`.

### `mission`

`{mission_id, description, launch_site_id, target: {lat, lon, name}, value,
sla_min, status, completed_time_s, failure_reason}` — `status` ∈
`pending | completed | failed`.

### `events[]`

`{event_id, type, trigger_time_s, status, message, params}` plus
`activated_time_s` once active. `type` ∈ `WIND_SHIFT | TFR_POPUP |
BATT_DEGRADE | SITE_CLOSED`; `status` ∈ `pending | active`. TFR params carry
`zone_id`, `name`, `polygon`, `floor_ft`, `ceiling_ft`.

### Flight segments (`last_result.segments[]` after `file_flight_plan` /
`amend_route`, and `last_result.segment` for RTL/land results)

A flown (possibly partial) segment has exactly:

`{from: {lat, lon, alt_ft}, to: {lat, lon, alt_ft}, distance_nm, track_deg,
groundspeed_kt, wind: {dir_deg_from, speed_kt}, execution_multiplier,
duration_s, energy_wh, interrupted_by_event: bool, progress_fraction: float}`

- `interrupted_by_event: true` means the segment was cut at an event trigger;
  `to` is the interpolated position and `distance_nm`/`duration_s`/`energy_wh`
  are the flown fraction (`progress_fraction`).
- An infeasible segment replaces the trailing execution keys with
  `error: "segment_infeasible_groundspeed"`.
- Autonomous failsafe recovery appears as a `{"failsafe": {...}}` entry in the
  `segments` list.

## Rendering guarantees

- `(seed, action_sequence)` replays to an identical `sim_log`.
- Trajectory can be reconstructed from `aircraft.lat/lon` across snapshots or,
  at higher fidelity, from segment `from`/`to` pairs.
- Every alert, failsafe, override, event activation, and violation is present
  in the snapshot where it occurred; nothing must be recomputed from prose.
