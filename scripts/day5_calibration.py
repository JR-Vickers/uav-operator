"""Run reproducible local Day 5 scripted-baseline calibration."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from scripts import baselines


def _tier_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    rewards = [float(row["reward"]) for row in rows]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards) if rewards else 0.0
    return {"mean_reward": mean, "std_reward": variance**0.5, "episodes": float(len(rewards))}


async def _run(seed: int, episodes: int) -> dict[str, Any]:
    result: dict[str, Any] = {"split": "dev", "seed": seed, "episodes_per_tier": episodes, "policies": {}}
    for policy in ("rulebook", "reckless"):
        per_tier: dict[str, Any] = {}
        for tier in ("T0", "T1", "T2", "T3"):
            rows = await baselines.run_many(policy, seed=seed, episodes=episodes, tier=tier)
            per_tier[tier] = {"summary": _tier_summary(rows), "episodes": rows}
        result["policies"][policy] = per_tier
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("outputs/day5/scripted_calibration.json"))
    args = parser.parse_args()

    result = asyncio.run(_run(args.seed, args.episodes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
