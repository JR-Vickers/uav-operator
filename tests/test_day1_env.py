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
    assert "route_leaves_san_francisco_bay_area_box" in result["warnings"]
    assert state["sim_state"]["sim_time_s"] == start_time + uav_operator.TOOL_LATENCY_S + uav_operator.INVALID_ACTION_LATENCY_S
    assert state["sim_state"]["is_terminal"] is False


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
