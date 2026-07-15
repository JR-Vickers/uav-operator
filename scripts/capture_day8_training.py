#!/usr/bin/env python3
"""Capture a reproducible Hosted Training evidence bundle for Day 8."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "day8_laguna_t1_smoke.toml"
DEFAULT_STEPS = (0, 10, 25, 50)
MODEL = "poolside/Laguna-XS-2.1"


def _prime_json(args: Sequence[str]) -> dict[str, Any]:
    command = ["prime", "--plain", *args]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"prime command did not return JSON: {' '.join(command)}\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"prime command returned a non-object: {' '.join(command)}")
    return value


def _prime_text(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["prime", "--plain", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def _find_value(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and child is not None:
                return child
        for child in value.values():
            found = _find_value(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_value(child, keys)
            if found is not None:
                return found
    return None


def summarize_samples(samples_by_step: dict[str, Any]) -> dict[str, Any]:
    """Summarize stop conditions and provider errors without reading model prose."""

    total = 0
    truncated = 0
    provider_errors = 0
    by_step: dict[str, dict[str, int | float]] = {}
    for step, payload in samples_by_step.items():
        samples = payload.get("samples", []) if isinstance(payload, dict) else []
        step_truncated = 0
        step_provider_errors = 0
        for sample in samples:
            stop_condition = _find_value(sample, {"stop_condition", "stopCondition"})
            if stop_condition == "max_turns_reached":
                step_truncated += 1
            provider_error = _find_value(sample, {"provider_error", "providerError"})
            if provider_error:
                step_provider_errors += 1
        count = len(samples)
        total += count
        truncated += step_truncated
        provider_errors += step_provider_errors
        by_step[step] = {
            "sample_count": count,
            "max_turn_truncations": step_truncated,
            "truncation_rate": step_truncated / count if count else 0.0,
            "provider_errors": step_provider_errors,
        }
    return {
        "sample_count": total,
        "max_turn_truncations": truncated,
        "truncation_rate": truncated / total if total else 0.0,
        "provider_errors": provider_errors,
        "by_step": by_step,
    }


def _parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    if not steps or any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("steps must be comma-separated non-negative integers")
    return steps


def capture(run_id: str, config_path: Path, steps: Sequence[int]) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    progress = _prime_json(["train", "progress", run_id])
    sample_steps = set(progress.get("steps_with_samples", []))
    distribution_steps = set(progress.get("steps_with_distributions", []))
    requested_steps = set(steps)

    samples = {
        str(step): _prime_json(
            ["train", "rollouts", run_id, "--step", str(step), "--num", "100"]
        )
        for step in steps
        if step in sample_steps
    }
    distributions = {
        str(step): _prime_json(
            ["train", "distributions", run_id, "--type", "rewards", "--step", str(step)]
        )
        for step in steps
        if step in distribution_steps
    }
    model_catalog = _prime_json(["train", "models", "--output", "json"])
    selected_model = next(
        (item for item in model_catalog.get("models", []) if item.get("name") == MODEL),
        None,
    )

    artifact = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config": {
            "path": str(config_path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "text": config_bytes.decode("utf-8"),
        },
        "requested_milestones": list(steps),
        "missing_sample_milestones": sorted(requested_steps - sample_steps),
        "missing_distribution_milestones": sorted(requested_steps - distribution_steps),
        "run": _prime_json(["train", "get", run_id, "--output", "json"]),
        "progress": progress,
        "metrics": _prime_json(["train", "metrics", run_id]),
        "reward_distributions": distributions,
        "sampled_rollouts": samples,
        "sample_summary": summarize_samples(samples),
        "usage": _prime_json(["train", "usage", run_id, "--output", "json"]),
        "checkpoints": _prime_json(
            ["train", "checkpoints", run_id, "--output", "json"]
        ),
        "model_pricing": selected_model,
        "wallet": _prime_json(["wallet", "--limit", "100", "--output", "json"]),
        "hub_status": _prime_json(
            ["env", "status", "jarrett/uav-operator", "--output", "json"]
        ),
        "logs": _prime_text(["train", "logs", run_id, "--tail", "5000", "--raw"]),
    }
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Hosted Training run ID")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"launch config (default: {DEFAULT_CONFIG.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--steps",
        type=_parse_steps,
        default=DEFAULT_STEPS,
        help="comma-separated milestones (default: 0,10,25,50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "assets" / "training" / "day8_smoke.json",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    artifact = capture(args.run_id, config_path, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
