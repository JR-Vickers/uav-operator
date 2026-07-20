#!/usr/bin/env python3
"""Capture and validate compact evidence for the paid Day 8 smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.5-2B"
EXPECTED_EVAL_STEPS = (0, 5, 10, 15, 20, 25)
EXPECTED_CHECKPOINT_STEPS = (15, 20)
MAX_RUN_COST_USD = 3.0
ENV_KEY = "jarrett/uav-operator@0.1.1"


def _prime_json(args: Sequence[str]) -> dict[str, Any]:
    command = ["prime", "--plain", *args]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"Prime command failed: {' '.join(command)}: {detail}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prime command returned invalid JSON: {' '.join(command)}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Prime command returned a non-object: {' '.join(command)}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def validate_capture(artifact: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    run = artifact.get("run", {}).get("run", {})
    progress = artifact.get("progress", {})
    usage = artifact.get("usage", {})
    config = artifact.get("config", {}).get("parsed", {})
    metrics = artifact.get("metrics", {}).get("metrics", [])
    checkpoints = artifact.get("checkpoints", {}).get("checkpoints", [])

    if run.get("status") != "COMPLETED":
        failures.append("run status is not COMPLETED")
    if run.get("base_model") != MODEL or config.get("model") != MODEL:
        failures.append("base model mismatch")
    if run.get("max_steps") != 25 or config.get("max_steps") != 25:
        failures.append("run/config is not the bounded 25-step smoke")
    if progress.get("latest_step") != 25:
        failures.append("latest step is not 25")
    if set(progress.get("steps_with_samples", [])) != set(range(25)):
        failures.append("sample milestones are incomplete")
    if set(progress.get("steps_with_distributions", [])) != set(range(25)):
        failures.append("distribution milestones are incomplete")

    try:
        cost = _finite(usage.get("total_cost_usd"), "usage.total_cost_usd")
        training_tokens = _finite(
            usage.get("training", {}).get("tokens"), "usage.training.tokens"
        )
    except ValueError as exc:
        failures.append(str(exc))
        cost = math.nan
        training_tokens = math.nan
    if math.isfinite(cost) and cost > MAX_RUN_COST_USD:
        failures.append(f"run cost exceeds ${MAX_RUN_COST_USD:.2f}")
    if math.isfinite(training_tokens) and training_tokens <= 0:
        failures.append("training token count is not positive")

    eval_by_step = {
        item.get("step"): item.get(f"eval/{ENV_KEY}/avg@2")
        for item in metrics
        if isinstance(item, dict) and item.get(f"eval/{ENV_KEY}/avg@2") is not None
    }
    if tuple(sorted(eval_by_step)) != EXPECTED_EVAL_STEPS:
        failures.append("held-out evaluation milestones are incomplete")
    for step, value in eval_by_step.items():
        try:
            _finite(value, f"eval step {step}")
        except ValueError as exc:
            failures.append(str(exc))

    ready_steps = sorted(
        item.get("step")
        for item in checkpoints
        if isinstance(item, dict) and item.get("status") == "READY"
    )
    if tuple(ready_steps) != EXPECTED_CHECKPOINT_STEPS:
        failures.append("retained READY checkpoints are not exactly steps 15 and 20")

    return {
        "passed": not failures,
        "failures": failures,
        "run_cost_usd": cost,
        "training_tokens": training_tokens,
        "eval_by_step": {str(key): value for key, value in sorted(eval_by_step.items())},
        "ready_checkpoint_steps": ready_steps,
        "absolute_eval_gain": (
            eval_by_step.get(25) - eval_by_step.get(0)
            if isinstance(eval_by_step.get(25), (int, float))
            and isinstance(eval_by_step.get(0), (int, float))
            else None
        ),
    }


def capture(run_id: str, config_path: Path) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    parsed_config = tomllib.loads(config_bytes.decode())
    run = _prime_json(["train", "get", run_id, "--output", "json"])
    progress = _prime_json(["train", "progress", run_id])
    metrics = _prime_json(["train", "metrics", run_id])
    usage = _prime_json(["train", "usage", run_id, "--output", "json"])
    checkpoints = _prime_json(["train", "checkpoints", run_id, "--output", "json"])
    models = _prime_json(["train", "models", "--output", "json"])
    model_rows = [row for row in models.get("models", []) if row.get("name") == MODEL]
    if len(model_rows) != 1:
        raise ValueError(f"expected exactly one live Hosted entry for {MODEL}")

    artifact = {
        "schema_version": 1,
        "kind": "uav_operator_paid_smoke_capture",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config": {
            "path": str(config_path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "text": config_bytes.decode(),
            "parsed": parsed_config,
        },
        "run": run,
        "progress": progress,
        "metrics": metrics,
        "usage": usage,
        "checkpoints": checkpoints,
        "captured_model_pricing": model_rows[0],
        "content_policy": (
            "No model prompt, response, completion, or natural-language judgment is stored; "
            "reward evidence comes only from platform metrics computed by the environment."
        ),
    }
    artifact["summary"] = validate_capture(artifact)
    return artifact


def plot_curve(artifact: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = artifact["metrics"]["metrics"]
    train = [row for row in rows if row.get("reward/all/mean") is not None]
    held_out = [row for row in rows if row.get(f"eval/{ENV_KEY}/avg@2") is not None]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [row["step"] for row in train],
        [row["reward/all/mean"] for row in train],
        color="#8793a5",
        alpha=0.55,
        label="training batch mean",
    )
    axis.plot(
        [row["step"] for row in held_out],
        [row[f"eval/{ENV_KEY}/avg@2"] for row in held_out],
        marker="o",
        linewidth=2.5,
        color="#146c94",
        label="held-out T1 dev (30 rollouts)",
    )
    axis.axhline(held_out[0][f"eval/{ENV_KEY}/avg@2"], color="#b24c3d", linestyle="--", label="base eval")
    axis.set(title="Qwen3.5-2B — 25-step UAV operator smoke", xlabel="optimizer step", ylabel="state-derived reward")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    args = parser.parse_args()
    artifact = capture(args.run_id, args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n")
    plot_curve(artifact, args.plot)
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))
    if not artifact["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
