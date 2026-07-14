from __future__ import annotations

import asyncio
import json
from typing import Any

import msgpack
import verifiers as vf

from scripts import baselines
import uav_operator


def _assistant_tool(name: str, args: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "name": name,
                "arguments": json.dumps(args),
            }
        ],
    }


def _setup_state(
    seed: int = 0,
    scenario_index: int = 0,
    tier: str = "T0",
) -> tuple[uav_operator.UAVOperatorEnv, vf.State]:
    env = uav_operator.load_environment(max_examples=1, tier=tier)
    assert isinstance(env, uav_operator.UAVOperatorEnv)
    state = vf.State(info={"seed": seed, "scenario_index": scenario_index, "tier": tier})
    asyncio.run(env.setup_state(state))
    return env, state


def _run_tool(env: uav_operator.UAVOperatorEnv, state: vf.State, name: str, args: dict[str, Any]) -> vf.Messages:
    return asyncio.run(env.env_response([_assistant_tool(name, args)], state))


def test_load_environment_exposes_day1_dataset_and_tools() -> None:
    env = uav_operator.load_environment()

    assert env.env_id == uav_operator.ENV_ID
    assert len(env.dataset) == uav_operator.DAY5_TRAIN_EXAMPLES
    assert len(env.get_eval_dataset()) == uav_operator.DAY5_EVAL_EXAMPLES
    assert [tool.name for tool in env.tool_defs or []] == [
        "get_telemetry",
        "get_weather",
        "get_airspace",
        "get_mission_status",
        "get_sites",
        "file_flight_plan",
        "amend_route",
        "set_altitude",
        "set_speed",
        "hold",
        "resume",
        "command_rtl",
        "land_now",
        "release_payload",
        "abort_mission",
        "override_failsafe",
        "acknowledge",
    ]


def test_file_plan_completes_mission_then_rtl_terminates() -> None:
    env, state = _setup_state(seed=7, scenario_index=0)
    target = state["sim_state"]["mission"]["target"]

    plan_response = _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [target],
            "alt_ft": 250,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
    )

    assert len(plan_response) == 1
    assert state["sim_state"]["mission"]["status"] == "completed"
    assert state["sim_state"]["aircraft"]["status"] == "holding"
    assert uav_operator.mission_value(state) == 1.0

    rtl_response = _run_tool(
        env,
        state,
        "command_rtl",
        {"site_id": state["sim_state"]["home_site_id"]},
    )

    assert len(rtl_response) == 2
    assert state["sim_state"]["is_terminal"] is True
    assert state["sim_state"]["terminal_reason"] == "aircraft_landed_mission_completed"
    assert state["sim_state"]["aircraft"]["status"] == "landed"
    assert json.loads(rtl_response[-1].content)["reward_preview"] == 1.0
    json.dumps(state["sim_log"])


def test_invalid_action_costs_time_without_crashing() -> None:
    env, state = _setup_state()
    start_time = state["sim_state"]["sim_time_s"]

    response = _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [{"lat": 37.0, "lon": -123.0}],
            "alt_ft": 250,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
    )

    result = json.loads(response[0].content)
    assert result["ok"] is False
    assert "route_leaves_san_francisco_bay_area_box" in result["errors"]
    assert state["sim_state"]["sim_time_s"] == start_time + uav_operator.TOOL_LATENCY_S + uav_operator.INVALID_ACTION_LATENCY_S
    assert state["sim_state"]["is_terminal"] is False


def test_day2_geometry_and_energy_invariants() -> None:
    box_crossing_nm = uav_operator.haversine_nm(
        37.75,
        uav_operator.MIN_LON,
        37.75,
        uav_operator.MAX_LON,
    )

    assert 20.0 < box_crossing_nm < 22.0
    assert 80.0 < uav_operator.bearing_deg(37.75, -122.60, 37.75, -122.15) < 100.0
    assert uav_operator._groundspeed_kt(
        airspeed_kt=35.0,
        track_deg=90.0,
        wind_dir_from_deg=90.0,
        wind_speed_kt=20.0,
    ) == 15.0

    level_energy = uav_operator._segment_energy_wh(
        distance_nm=5.0,
        airspeed_kt=35.0,
        start_alt_ft=250.0,
        end_alt_ft=250.0,
    )
    climb_energy = uav_operator._segment_energy_wh(
        distance_nm=5.0,
        airspeed_kt=35.0,
        start_alt_ft=0.0,
        end_alt_ft=400.0,
    )
    fast_energy = uav_operator._segment_energy_wh(
        distance_nm=5.0,
        airspeed_kt=45.0,
        start_alt_ft=250.0,
        end_alt_ft=250.0,
    )
    marginal_energy = uav_operator._segment_energy_wh(
        distance_nm=box_crossing_nm,
        airspeed_kt=35.0,
        start_alt_ft=0.0,
        end_alt_ft=250.0,
        track_deg=90.0,
        wind_dir_from_deg=90.0,
        wind_speed_kt=20.0,
    )

    assert climb_energy > level_energy
    assert fast_energy > level_energy
    assert 0.75 * uav_operator.BATTERY_WH < marginal_energy < uav_operator.BATTERY_WH


def test_day2_geofence_hold_blocks_predicted_unauthorized_airspace() -> None:
    env, state = _setup_state(seed=9, scenario_index=3)
    start_position = dict(state["sim_state"]["aircraft"])

    response = _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [{"lat": 37.62, "lon": -122.39, "name": "SFO core test"}],
            "alt_ft": 250,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
    )

    result = json.loads(response[0].content)
    assert result["ok"] is False
    assert result["error"] == "failsafe:GEOFENCE_HOLD"
    assert state["sim_state"]["active_failsafe"] == "GEOFENCE_HOLD"
    assert state["sim_state"]["alerts"][-1]["alert_id"] == "GEOFENCE_HOLD"
    assert state["sim_state"]["aircraft"]["lat"] == start_position["lat"]
    assert state["sim_state"]["aircraft"]["lon"] == start_position["lon"]


def test_day2_low_battery_triggers_autonomous_rtl() -> None:
    env, state = _setup_state(seed=11, scenario_index=0)
    state["sim_state"]["aircraft"]["battery_wh"] = uav_operator.LOW_BATT_RTL_THRESHOLD_WH + 1.0
    target = state["sim_state"]["mission"]["target"]

    _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [target],
            "alt_ft": 250,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
    )

    assert state["sim_state"]["active_failsafe"] == "LOW_BATT_RTL"
    assert state["sim_state"]["alerts"][-1]["alert_id"] == "LOW_BATT_RTL"
    assert state["sim_state"]["aircraft"]["status"] == "landed"
    assert state["sim_state"]["is_terminal"] is True
    assert state["sim_state"]["terminal_reason"] == "aircraft_landed_mission_completed"


def test_day3_wind_is_seeded_and_has_altitude_shear() -> None:
    sim_a = uav_operator._build_sim(seed=21, scenario_index=0, gust_front_probability=1.0)
    sim_b = uav_operator._build_sim(seed=21, scenario_index=0, gust_front_probability=1.0)
    sim_c = uav_operator._build_sim(seed=22, scenario_index=0, gust_front_probability=1.0)

    low_a = uav_operator.wind_at(37.75, -122.35, 100.0, 900.0, sim_a.wind)
    low_b = uav_operator.wind_at(37.75, -122.35, 100.0, 900.0, sim_b.wind)
    low_c = uav_operator.wind_at(37.75, -122.35, 100.0, 900.0, sim_c.wind)
    high_a = uav_operator.wind_at(37.75, -122.35, 1100.0, 900.0, sim_a.wind)
    front_late = uav_operator.wind_at(37.75, -122.35, 300.0, 7200.0, sim_a.wind)

    assert sim_a.wind == sim_b.wind
    assert sim_a.wind != sim_c.wind
    assert low_a == low_b
    assert low_a != low_c
    assert high_a[1] > low_a[1]
    assert high_a[0] != low_a[0]
    assert front_late != low_a


def test_day3_console_tools_charge_time_and_return_structured_state() -> None:
    env, state = _setup_state(seed=13, scenario_index=1)

    weather = _run_tool(env, state, "get_weather", {"alt_ft": 300})
    weather_payload = json.loads(weather[0].content)
    assert weather_payload["ok"] is True
    assert "current" in weather_payload
    assert state["sim_state"]["sim_time_s"] == uav_operator.READ_TOOL_LATENCY_S

    sites = _run_tool(env, state, "get_sites", {})
    assert json.loads(sites[0].content)["nearest_site_id"] == state["sim_state"]["home_site_id"]

    altitude = _run_tool(env, state, "set_altitude", {"ft": 300})
    assert json.loads(altitude[0].content)["alt_ft"] == 300

    speed = _run_tool(env, state, "set_speed", {"kt": 32})
    assert json.loads(speed[0].content)["airspeed_kt"] == 32

    before_hold = state["sim_state"]["sim_time_s"]
    hold = _run_tool(env, state, "hold", {"minutes": 1})
    assert json.loads(hold[0].content)["held_minutes"] == 1
    assert state["sim_state"]["sim_time_s"] == before_hold + uav_operator.TOOL_LATENCY_S + 60.0


def test_ground_state_read_and_hold_loops_warn_and_escalate_latency() -> None:
    env, state = _setup_state(seed=18, scenario_index=0, tier="T0")

    for _ in range(uav_operator.GROUND_STALL_WARNING_THRESHOLD):
        response = _run_tool(env, state, "get_telemetry", {})
        assert "operator_warning" not in json.loads(response[0].content)

    before_warning = state["sim_state"]["sim_time_s"]
    response = _run_tool(env, state, "get_telemetry", {})
    warning = json.loads(response[0].content)["operator_warning"]
    assert warning == {
        "additional_latency_s": uav_operator.GROUND_STALL_LATENCY_STEP_S,
        "code": "ground_no_progress_loop",
        "message": (
            "The aircraft remains on the ground and the mission has not advanced. "
            "Choose an operational action, continue waiting only for a specific timed condition, "
            "or abort the mission."
        ),
        "streak": 4,
    }
    assert state["sim_state"]["sim_time_s"] == (
        before_warning
        + uav_operator.GROUND_STALL_LATENCY_STEP_S
        + uav_operator.READ_TOOL_LATENCY_S
    )

    before_hold = state["sim_state"]["sim_time_s"]
    hold = _run_tool(env, state, "hold", {"minutes": 1})
    hold_warning = json.loads(hold[0].content)["operator_warning"]
    assert hold_warning["streak"] == 5
    assert hold_warning["additional_latency_s"] == 2 * uav_operator.GROUND_STALL_LATENCY_STEP_S
    assert state["sim_state"]["sim_time_s"] == (
        before_hold
        + 2 * uav_operator.GROUND_STALL_LATENCY_STEP_S
        + uav_operator.TOOL_LATENCY_S
        + 60.0
    )

    _run_tool(env, state, "set_speed", {"kt": 32})
    assert state["sim_state"]["ground_no_progress_streak"] == 5

    target = state["sim_state"]["mission"]["target"]
    _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [target],
            "alt_ft": 300,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
    )
    assert state["sim_state"]["ground_no_progress_streak"] == 0


def test_day3_override_and_acknowledge_clear_active_geofence_failsafe() -> None:
    env, state = _setup_state(seed=14, scenario_index=3)

    _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [{"lat": 37.62, "lon": -122.39, "name": "SFO core test"}],
            "alt_ft": 250,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
    )
    assert state["sim_state"]["active_failsafe"] == "GEOFENCE_HOLD"

    ack = _run_tool(env, state, "acknowledge", {"alert_id": "GEOFENCE_HOLD"})
    assert json.loads(ack[0].content)["acknowledged_alerts"] == ["GEOFENCE_HOLD"]

    override = _run_tool(
        env,
        state,
        "override_failsafe",
        {
            "id": "GEOFENCE_HOLD",
            "justification_code": "AIRSPACE_AUTH_CONFIRMED",
        },
    )
    assert json.loads(override[0].content)["ok"] is True
    assert state["sim_state"]["active_failsafe"] is None
    assert state["sim_state"]["overrides"][-1]["id"] == "GEOFENCE_HOLD"


def test_day3_saved_state_columns_are_msgpack_serializable() -> None:
    env, state = _setup_state(seed=15, scenario_index=0)
    msgpack.packb(state["sim_state"])
    msgpack.packb(state["sim_log"])

    target = state["sim_state"]["mission"]["target"]
    _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [target],
            "alt_ft": 300,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
    )

    msgpack.packb(state["sim_state"])
    msgpack.packb(state["sim_log"])


def test_same_seed_and_action_sequence_are_deterministic() -> None:
    def run_sequence() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        env, state = _setup_state(seed=12, scenario_index=2)
        target = state["sim_state"]["mission"]["target"]
        _run_tool(env, state, "get_weather", {"alt_ft": 300})
        _run_tool(env, state, "get_airspace", {"route": [target], "alt_ft": 300})
        _run_tool(env, state, "set_altitude", {"ft": 300})
        _run_tool(env, state, "set_speed", {"kt": 35})
        _run_tool(
            env,
            state,
            "file_flight_plan",
            {
                "waypoints": [target],
                "alt_ft": 300,
                "airspeed_kt": 35,
                "lost_link_plan": "return_home",
            },
        )
        _run_tool(
            env,
            state,
            "command_rtl",
            {"site_id": state["sim_state"]["home_site_id"]},
        )
        return state["sim_state"], state["sim_log"]

    first_state, first_log = run_sequence()
    second_state, second_log = run_sequence()

    assert first_state == second_state
    assert first_log == second_log


def test_day4_dataset_emits_deterministic_t0_t1_mix() -> None:
    env = uav_operator.load_environment(seed=100, max_examples=6, tier="mixed_day4")
    rows = list(env.get_eval_dataset())

    assert [row["info"]["tier"] for row in rows] == ["T0", "T1", "T0", "T1", "T0", "T1"]
    assert rows == list(uav_operator.load_environment(seed=100, max_examples=6, tier="mixed_day4").get_eval_dataset())


def test_day5_splits_are_stratified_disjoint_and_deterministic() -> None:
    train = list(uav_operator._dataset(7, -1, tier="mixed_day5", split="train"))
    dev = list(uav_operator._dataset(7, -1, tier="mixed_day5", split="dev"))
    evaluation = list(uav_operator._dataset(7, -1, tier="mixed_day5", split="eval"))

    assert len(train) == 300
    assert len(dev) == len(evaluation) == 60
    assert {row["info"]["seed"] for row in train}.isdisjoint(row["info"]["seed"] for row in dev)
    assert {row["info"]["seed"] for row in dev}.isdisjoint(row["info"]["seed"] for row in evaluation)
    for rows, expected in ((train, 75), (dev, 15), (evaluation, 15)):
        assert {tier: sum(row["info"]["tier"] == tier for row in rows) for tier in uav_operator.DAY5_TIERS} == {
            tier: expected for tier in uav_operator.DAY5_TIERS
        }
    assert evaluation == list(uav_operator._dataset(7, -1, tier="mixed_day5", split="eval"))
    assert len(uav_operator.load_environment(dataset_split="dev").get_eval_dataset()) == 60


def test_day5_composed_scenarios_are_solver_checked() -> None:
    for tier in ("T2", "T3"):
        for index in range(12):
            scenario = uav_operator._scenario_for_index(index, seed=200 + index, tier=tier)
            assert 2 <= len(scenario["events"]) <= (2 if tier == "T2" else 4)
            assert uav_operator._scenario_has_feasible_resolution(scenario, 200 + index)


def test_day5_t2_dev_split_retains_feasible_tfr_scenarios() -> None:
    scenarios = [uav_operator._scenario_for_index(index, seed=10_000 + index, tier="T2") for index in range(15)]
    tfr_scenarios = [scenario for scenario in scenarios if any(event["type"] == "TFR_POPUP" for event in scenario["events"])]

    assert len(tfr_scenarios) == 9
    assert all(uav_operator._scenario_has_feasible_resolution(scenario, 10_000 + index) for index, scenario in enumerate(scenarios))


def test_repeated_known_geofence_filing_and_invalid_override_cost_procedure() -> None:
    env, state = _setup_state(seed=9, scenario_index=3)
    conflicting_plan = {
        "waypoints": [{"lat": 37.62, "lon": -122.39, "name": "SFO core test"}],
        "alt_ft": 250,
        "airspeed_kt": 35,
        "lost_link_plan": "return_home",
    }
    _run_tool(env, state, "file_flight_plan", conflicting_plan)
    first_penalty = uav_operator.procedure(state)
    _run_tool(env, state, "file_flight_plan", conflicting_plan)
    _run_tool(
        env,
        state,
        "override_failsafe",
        {"id": "LOW_BATT_RTL", "justification_code": "MISSION_CRITICAL_MARGIN_OK"},
    )

    assert uav_operator.procedure(state) < first_penalty
    assert any(item.startswith("override_inactive_failsafe") for item in state["sim_state"]["procedure_violations"])


def test_day4_t1_event_interrupt_is_logged_and_acknowledged() -> None:
    env, state = _setup_state(seed=31, scenario_index=1, tier="T1")
    target = state["sim_state"]["mission"]["target"]

    response = _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [target],
            "alt_ft": 300,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
    )
    result = json.loads(response[0].content)

    assert result["ok"] is False
    assert result["error"] == "event_interrupt"
    assert state["sim_state"]["active_events"]
    alert_id = state["sim_state"]["active_events"][0]

    _run_tool(env, state, "acknowledge", {"alert_id": alert_id})

    assert alert_id in state["sim_state"]["acknowledged_events"]
    assert alert_id not in state["sim_state"]["active_events"]
    assert state["sim_log"][-1]["events"][0]["status"] == "active"


def test_day4_reward_components_use_logged_sim_state_not_prose() -> None:
    env, state = _setup_state(seed=40, scenario_index=0, tier="T0")
    state["messages"] = [{"role": "assistant", "content": "I completed the mission safely."}]

    assert uav_operator.reward_breakdown(state)["total"] == 0.0

    state["sim_log"][-1]["mission"]["status"] = "completed"
    state["sim_log"][-1]["mission"]["completed_time_s"] = 60.0
    state["sim_log"][-1]["aircraft"]["status"] = "lost"
    state["sim_log"][-1]["hard_safety_violations"] = ["aircraft_loss"]

    breakdown = uav_operator.reward_breakdown(state)
    assert breakdown["mission_value"] == 1.0
    assert breakdown["hard_safety"] == -5.0
    assert breakdown["total"] < 0.0


def test_hard_safety_deduplicates_terminal_battery_loss_labels() -> None:
    _, state = _setup_state(seed=41, scenario_index=0, tier="T0")
    snapshot = state["sim_log"][-1]
    snapshot["aircraft"]["status"] = "lost"
    snapshot["battery_pct"] = 0.0
    snapshot["terminal_reason"] = "aircraft_lost_battery_depleted"
    snapshot["hard_safety_violations"] = ["aircraft_loss:battery_depleted"]

    assert uav_operator.hard_safety(state) == -5.0


def test_day4_rulebook_baseline_beats_reckless_baseline() -> None:
    async def run() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rulebook = await baselines.run_many("rulebook", seed=70, episodes=8, tier="mixed_day4")
        reckless = await baselines.run_many("reckless", seed=70, episodes=8, tier="mixed_day4")
        return rulebook, reckless

    rulebook_rows, reckless_rows = asyncio.run(run())
    rulebook_avg = sum(row["reward"] for row in rulebook_rows) / len(rulebook_rows)
    reckless_avg = sum(row["reward"] for row in reckless_rows) / len(reckless_rows)

    assert rulebook_avg > reckless_avg


def test_day6_mid_flight_event_interrupt_pauses_route() -> None:
    env, state = _setup_state(seed=500, scenario_index=0, tier="T3")
    target = state["sim_state"]["mission"]["target"]
    plan = {
        "waypoints": [target],
        "alt_ft": 300,
        "airspeed_kt": 35,
        "lost_link_plan": "return_home",
    }

    first = json.loads(_run_tool(env, state, "file_flight_plan", plan)[0].content)
    assert first["error"] == "event_interrupt"
    for alert in list(state["sim_state"]["alerts"]):
        _run_tool(env, state, "acknowledge", {"alert_id": alert["alert_id"]})

    second = json.loads(_run_tool(env, state, "file_flight_plan", plan)[0].content)

    assert second["error"] == "event_interrupt"
    assert second["remaining_route_suspended"] is True
    segment = second["segments"][0]
    assert segment["interrupted_by_event"] is True
    assert 0.0 < segment["progress_fraction"] < 1.0
    assert state["sim_state"]["aircraft"]["status"] == "holding"
    assert state["sim_state"]["mission"]["status"] == "pending"
    triggered = [
        event
        for event in state["sim_state"]["events"]
        if event["status"] == "active" and event["trigger_time_s"] > 60.0
    ]
    assert triggered
    assert state["sim_state"]["sim_time_s"] == triggered[0]["trigger_time_s"]


def test_day6_airborne_hold_burns_hover_energy_and_triggers_low_batt_rtl() -> None:
    env, state = _setup_state(seed=102, scenario_index=0, tier="T0")
    launch = state["sim_state"]["aircraft"]
    target = state["sim_state"]["mission"]["target"]
    midpoint = {
        "lat": (launch["lat"] + target["lat"]) / 2.0,
        "lon": (launch["lon"] + target["lon"]) / 2.0,
        "name": "loiter point",
    }
    _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [midpoint],
            "alt_ft": 300,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
    )
    assert state["sim_state"]["aircraft"]["status"] == "holding"

    battery_before = state["sim_state"]["aircraft"]["battery_wh"]
    hold = json.loads(_run_tool(env, state, "hold", {"minutes": 30})[0].content)
    drain_wh = battery_before - state["sim_state"]["aircraft"]["battery_wh"]
    expected_wh = uav_operator.HOVER_POWER_W * (uav_operator.TOOL_LATENCY_S + 1800.0) / 3600.0
    assert abs(drain_wh - expected_wh) < 1e-6
    assert hold["interrupted_by_event"] is False

    for _ in range(4):
        if state["sim_state"]["is_terminal"]:
            break
        hold = json.loads(_run_tool(env, state, "hold", {"minutes": 30})[0].content)

    assert state["sim_state"]["active_failsafe"] == "LOW_BATT_RTL"
    assert "failsafe" in hold
    assert state["sim_state"]["aircraft"]["status"] == "landed"
    assert state["sim_state"]["is_terminal"] is True
    assert state["sim_state"]["mission"]["status"] == "failed"


def test_day6_no_fly_buffer_blocks_boundary_hugging() -> None:
    env, state = _setup_state(seed=103, scenario_index=2, tier="T0")
    hug_lat = 37.7570 + 0.0005

    response = _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [
                {"lat": hug_lat, "lon": -122.2075, "name": "hug west"},
                {"lat": hug_lat, "lon": -122.1930, "name": "hug east"},
            ],
            "alt_ft": 250,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
    )

    result = json.loads(response[0].content)
    assert result["error"] == "failsafe:GEOFENCE_HOLD"
    assert any(
        zone["zone_id"] == "oakland_coliseum_no_fly" for zone in result["conflicts"]
    )
    assert "filed_known_conflict:oakland_coliseum_no_fly" in state["sim_state"]["procedure_violations"]
    clear_offset = uav_operator.AIRSPACE_BUFFER_NM / 60.0 + 0.001
    clear = json.loads(
        _run_tool(
            env,
            state,
            "get_airspace",
            {
                "route": [
                    {"lat": 37.7570 + clear_offset, "lon": -122.2075, "name": "clear west"},
                    {"lat": 37.7570 + clear_offset, "lon": -122.1930, "name": "clear east"},
                ],
                "alt_ft": 250,
            },
        )[0].content
    )
    assert all(
        zone["zone_id"] != "oakland_coliseum_no_fly" for zone in clear["route_conflicts"]
    )


def test_day6_abort_on_ground_terminates_without_false_crit_batt() -> None:
    env, state = _setup_state(seed=105, scenario_index=0, tier="T0")
    state["sim_state"]["aircraft"]["battery_wh"] = uav_operator.BATTERY_WH * 0.10

    _run_tool(
        env,
        state,
        "abort_mission",
        {"mission_id": state["sim_state"]["mission"]["mission_id"]},
    )

    assert state["sim_state"]["is_terminal"] is True
    assert state["sim_state"]["terminal_reason"] == "aircraft_ground_mission_failed"
    assert uav_operator.hard_safety(state) == 0.0


def test_day6_geofence_hold_flags_target_inside_active_tfr() -> None:
    env, state = _setup_state(seed=31, scenario_index=1, tier="T1")
    target = state["sim_state"]["mission"]["target"]
    plan = {
        "waypoints": [target],
        "alt_ft": 300,
        "airspeed_kt": 35,
        "lost_link_plan": "return_home",
    }

    first = json.loads(_run_tool(env, state, "file_flight_plan", plan)[0].content)
    assert first["error"] == "event_interrupt"
    for alert in list(state["sim_state"]["alerts"]):
        _run_tool(env, state, "acknowledge", {"alert_id": alert["alert_id"]})

    second = json.loads(_run_tool(env, state, "file_flight_plan", plan)[0].content)

    assert second["error"] == "failsafe:GEOFENCE_HOLD"
    assert second["mission_target_inside_zone"] is True
    assert "advisory" in second
    assert any(zone["contains_mission_target"] for zone in second["conflicts"])


SCHEMA_V1_SNAPSHOT_KEYS = {
    "event",
    "battery_pct",
    "mission_distance_to_target_nm",
    "seed",
    "scenario_id",
    "tier",
    "sim_time_s",
    "aircraft",
    "mission",
    "home_site_id",
    "wind",
    "rng_state",
    "scenario_par",
    "events",
    "active_events",
    "acknowledged_events",
    "closed_sites",
    "hard_safety_violations",
    "procedure_violations",
    "current_plan",
    "lost_link_plan",
    "rng_draws",
    "active_failsafe",
    "alerts",
    "acknowledged_alerts",
    "overrides",
    "payload_released",
    "hold_until_s",
    "current_altitude_target_ft",
    "current_airspeed_kt",
    "ground_no_progress_streak",
    "last_action",
    "last_result",
    "is_terminal",
    "terminal_reason",
}
SCHEMA_V1_AIRCRAFT_KEYS = {"lat", "lon", "alt_ft", "status", "battery_wh", "current_site_id"}
SCHEMA_V1_MISSION_KEYS = {
    "mission_id",
    "description",
    "launch_site_id",
    "target",
    "value",
    "sla_min",
    "status",
    "completed_time_s",
    "failure_reason",
}
SCHEMA_V1_SEGMENT_KEYS = {
    "from",
    "to",
    "distance_nm",
    "track_deg",
    "groundspeed_kt",
    "wind",
    "execution_multiplier",
    "duration_s",
    "energy_wh",
    "interrupted_by_event",
    "progress_fraction",
}
SCHEMA_V1_ALERT_KEYS = {"alert_id", "sim_time_s", "message", "metadata"}


def test_day6_rollout_snapshot_schema_is_frozen() -> None:
    env, state = _setup_state(seed=500, scenario_index=0, tier="T3")
    target = state["sim_state"]["mission"]["target"]
    plan = {
        "waypoints": [target],
        "alt_ft": 300,
        "airspeed_kt": 35,
        "lost_link_plan": "return_home",
    }
    _run_tool(env, state, "file_flight_plan", plan)
    for alert in list(state["sim_state"]["alerts"]):
        _run_tool(env, state, "acknowledge", {"alert_id": alert["alert_id"]})
    _run_tool(env, state, "file_flight_plan", plan)
    _run_tool(env, state, "hold", {"minutes": 1})
    if not state["sim_state"]["is_terminal"]:
        _run_tool(env, state, "command_rtl", {})

    assert len(state["sim_log"]) >= 5
    segments_seen = 0
    for snapshot in state["sim_log"]:
        assert set(snapshot.keys()) == SCHEMA_V1_SNAPSHOT_KEYS
        assert set(snapshot["aircraft"].keys()) == SCHEMA_V1_AIRCRAFT_KEYS
        assert set(snapshot["mission"].keys()) == SCHEMA_V1_MISSION_KEYS
        assert set(snapshot["mission"]["target"].keys()) == {"lat", "lon", "name"}
        for alert in snapshot["alerts"]:
            assert set(alert.keys()) == SCHEMA_V1_ALERT_KEYS
        for event in snapshot["events"]:
            assert {
                "event_id",
                "type",
                "trigger_time_s",
                "status",
                "message",
                "params",
            } <= set(event.keys())
        for segment in snapshot["last_result"].get("segments", []):
            if "failsafe" in segment or "error" in segment:
                continue
            assert set(segment.keys()) == SCHEMA_V1_SEGMENT_KEYS
            segments_seen += 1
    assert segments_seen > 0
    json.dumps(state["sim_log"])
