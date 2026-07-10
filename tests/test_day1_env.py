from __future__ import annotations

import asyncio
import json
from typing import Any

import verifiers as vf

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


def _setup_state(seed: int = 0, scenario_index: int = 0) -> tuple[uav_operator.UAVOperatorEnv, vf.State]:
    env = uav_operator.load_environment(max_examples=1)
    assert isinstance(env, uav_operator.UAVOperatorEnv)
    state = vf.State(info={"seed": seed, "scenario_index": scenario_index})
    asyncio.run(env.setup_state(state))
    return env, state


def _run_tool(env: uav_operator.UAVOperatorEnv, state: vf.State, name: str, args: dict[str, Any]) -> vf.Messages:
    return asyncio.run(env.env_response([_assistant_tool(name, args)], state))


def test_load_environment_exposes_day1_dataset_and_tools() -> None:
    env = uav_operator.load_environment()

    assert env.env_id == uav_operator.ENV_ID
    assert len(env.get_eval_dataset()) == 5
    assert [tool.name for tool in env.tool_defs or []] == [
        "get_telemetry",
        "file_flight_plan",
        "command_rtl",
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


def test_same_seed_and_action_sequence_are_deterministic() -> None:
    def run_sequence() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        env, state = _setup_state(seed=12, scenario_index=2)
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
