"""Scripted baselines for uav-operator curriculum calibration."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Literal

import verifiers as vf

import uav_operator

Policy = Literal["rulebook", "reckless"]


def _assistant_tool(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
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


async def _run_tool(
    env: uav_operator.UAVOperatorEnv,
    state: vf.State,
    name: str,
    args: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    response = await env.env_response([_assistant_tool(name, args, f"call_{step}")], state)
    if not response:
        return {}
    content = getattr(response[0], "content", "{}")
    parsed = json.loads(str(content))
    return parsed if isinstance(parsed, dict) else {}


def _target(state: vf.State) -> dict[str, Any]:
    target = state["sim_state"]["mission"]["target"]
    return {
        "lat": float(target["lat"]),
        "lon": float(target["lon"]),
        "name": str(target.get("name", "target")),
    }


def _home_site(state: vf.State) -> str:
    return str(state["sim_state"]["home_site_id"])


def _active_alert_ids(state: vf.State) -> list[str]:
    return [
        str(alert["alert_id"])
        for alert in state["sim_state"].get("alerts", [])
        if str(alert["alert_id"]) not in state["sim_state"].get("acknowledged_alerts", [])
    ]


async def _ack_all(env: uav_operator.UAVOperatorEnv, state: vf.State, step: int) -> int:
    for alert_id in _active_alert_ids(state):
        await _run_tool(env, state, "acknowledge", {"alert_id": alert_id}, step)
        step += 1
    return step


async def _rulebook(env: uav_operator.UAVOperatorEnv, state: vf.State) -> None:
    step = 0
    target = _target(state)
    await _run_tool(env, state, "get_weather", {"alt_ft": 300}, step)
    step += 1
    await _run_tool(env, state, "get_airspace", {"route": [target], "alt_ft": 300}, step)
    step += 1
    step = await _ack_all(env, state, step)

    airspace = await _run_tool(env, state, "get_airspace", {"route": [target], "alt_ft": 300}, step)
    step += 1
    if airspace.get("route_conflicts"):
        await _run_tool(env, state, "abort_mission", {"mission_id": state["sim_state"]["mission"]["mission_id"]}, step)
        step += 1
        await _run_tool(env, state, "command_rtl", {"site_id": _home_site(state)}, step)
        return

    result = await _run_tool(
        env,
        state,
        "file_flight_plan",
        {
            "waypoints": [target],
            "alt_ft": 300,
            "airspeed_kt": 35,
            "lost_link_plan": "return_home",
        },
        step,
    )
    step += 1
    if result.get("error") == "event_interrupt":
        step = await _ack_all(env, state, step)
        await _run_tool(
            env,
            state,
            "file_flight_plan",
            {
                "waypoints": [target],
                "alt_ft": 300,
                "airspeed_kt": 35,
                "lost_link_plan": "return_home",
            },
            step,
        )
        step += 1
    if state["sim_state"]["active_failsafe"] == "GEOFENCE_HOLD":
        step = await _ack_all(env, state, step)
        await _run_tool(env, state, "command_rtl", {"site_id": _home_site(state)}, step)
        return
    if not state["sim_state"]["is_terminal"]:
        await _run_tool(env, state, "command_rtl", {"site_id": _home_site(state)}, step)


async def _reckless(env: uav_operator.UAVOperatorEnv, state: vf.State) -> None:
    step = 0
    target = _target(state)
    for _ in range(3):
        result = await _run_tool(
            env,
            state,
            "file_flight_plan",
            {
                "waypoints": [target],
                "alt_ft": 250,
                "airspeed_kt": 20,
                "lost_link_plan": "continue",
            },
            step,
        )
        step += 1
        if state["sim_state"]["is_terminal"]:
            return
        if result.get("error") == "event_interrupt":
            step = await _ack_all(env, state, step)
            continue
        active_failsafe = state["sim_state"].get("active_failsafe")
        if active_failsafe:
            await _run_tool(env, state, "acknowledge", {"alert_id": str(active_failsafe)}, step)
            step += 1
            await _run_tool(
                env,
                state,
                "override_failsafe",
                {
                    "id": str(active_failsafe),
                    "justification_code": "MISSION_CRITICAL_MARGIN_OK",
                },
                step,
            )
            step += 1
    if not state["sim_state"]["is_terminal"]:
        await _run_tool(env, state, "command_rtl", {"site_id": _home_site(state)}, step)


async def run_policy(policy: Policy, *, seed: int, scenario_index: int, tier: str) -> dict[str, Any]:
    env = uav_operator.load_environment(max_examples=1, tier=tier)
    if not isinstance(env, uav_operator.UAVOperatorEnv):
        raise TypeError("load_environment did not return UAVOperatorEnv")
    state = vf.State(info={"seed": seed, "scenario_index": scenario_index, "tier": tier})
    await env.setup_state(state)
    if policy == "rulebook":
        await _rulebook(env, state)
    else:
        await _reckless(env, state)
    breakdown = uav_operator.reward_breakdown(state)
    return {
        "policy": policy,
        "seed": seed,
        "scenario_index": scenario_index,
        "scenario_id": state["sim_state"]["scenario_id"],
        "tier": state["sim_state"]["tier"],
        "terminal_reason": state["sim_state"].get("terminal_reason"),
        "reward": breakdown["total"],
        "reward_breakdown": breakdown,
    }


async def run_many(policy: Policy, *, seed: int, episodes: int, tier: str) -> list[dict[str, Any]]:
    return [
        await run_policy(policy, seed=seed + index, scenario_index=index, tier=tier)
        for index in range(episodes)
    ]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(float(row["reward"]) for row in rows)
    components: dict[str, float] = {}
    for row in rows:
        for name, value in row["reward_breakdown"].items():
            components[name] = components.get(name, 0.0) + float(value)
    return {
        "episodes": len(rows),
        "avg_reward": total / len(rows) if rows else 0.0,
        "avg_components": {
            name: value / len(rows)
            for name, value in sorted(components.items())
        }
        if rows
        else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["rulebook", "reckless", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--tier", default="mixed_day5")
    args = parser.parse_args()

    policies: list[Policy] = ["rulebook", "reckless"] if args.policy == "both" else [args.policy]
    output: dict[str, Any] = {}
    tiers = ["T0", "T1", "T2", "T3"] if args.tier == "all" else [args.tier]
    for policy in policies:
        output[policy] = {}
        for tier in tiers:
            rows = asyncio.run(run_many(policy, seed=args.seed, episodes=args.episodes, tier=tier))
            output[policy][tier] = {
                "summary": _summarize(rows),
                "episodes": rows,
            }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
