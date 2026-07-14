"""Build and plot Day 5 tier calibration from saved vf-eval runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


TIERS = ("T0", "T1", "T2", "T3")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _summarize_run(run_dir: Path) -> tuple[str, str, dict[str, Any]]:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    rows = _read_jsonl(run_dir / "results.jsonl")
    tier = str(metadata["env_args"]["tier"])
    if tier not in TIERS:
        raise ValueError(f"{run_dir}: unsupported tier {tier!r}")
    rewards = [float(row["reward"]) for row in rows]
    if not rewards:
        raise ValueError(f"{run_dir}: no rollout rewards")
    standard_error = (
        statistics.stdev(rewards) / math.sqrt(len(rewards)) if len(rewards) > 1 else 0.0
    )
    completed = sum(
        row.get("sim_state", {}).get("mission", {}).get("status") == "completed"
        if row.get("sim_state", {}).get("mission", {}).get("status") is not None
        else float(row.get("mission_value", 0.0)) > 0.0
        for row in rows
    )
    errors = sum(row.get("error") is not None for row in rows)
    if errors:
        raise ValueError(f"{run_dir}: {errors} provider-error rollout(s)")
    summary = {
        "run_id": str(metadata.get("run_id", run_dir.name)),
        "results_path": str(metadata.get("results_path", run_dir)),
        "n": len(rewards),
        "mean_reward": statistics.fmean(rewards),
        "ci95_low": statistics.fmean(rewards) - 1.96 * standard_error,
        "ci95_high": statistics.fmean(rewards) + 1.96 * standard_error,
        "missions_completed": completed,
        "provider_errors": errors,
        "mean_turns": statistics.fmean(float(row["num_turns"]) for row in rows),
        "max_turn_outcomes": sum(
            row.get("stop_condition") == "max_turns_reached" for row in rows
        ),
        "hard_safety_violations": sum(
            float(row.get("hard_safety", 0.0)) < 0.0 for row in rows
        ),
    }
    return str(metadata["model"]), tier, summary


def _build_payload(run_dirs: list[Path], title: str) -> dict[str, Any]:
    series: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        model, tier, summary = _summarize_run(run_dir)
        if tier in series.setdefault(model, {}):
            raise ValueError(f"duplicate {model} {tier} run")
        series[model][tier] = summary
    return {
        "title": title,
        "interval": "95% normal confidence interval of rollout rewards",
        "series": series,
    }


def _plot(payload: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8))
    for model, by_tier in payload["series"].items():
        present = [tier for tier in TIERS if tier in by_tier]
        means = [float(by_tier[tier]["mean_reward"]) for tier in present]
        lower = [
            mean - float(by_tier[tier]["ci95_low"])
            for mean, tier in zip(means, present)
        ]
        upper = [
            float(by_tier[tier]["ci95_high"]) - mean
            for mean, tier in zip(means, present)
        ]
        axis.errorbar(
            present, means, yerr=[lower, upper], marker="o", capsize=5, label=model
        )
        for tier, mean in zip(present, means):
            axis.annotate(
                f"n={by_tier[tier]['n']}",
                (tier, mean),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title(str(payload["title"]))
    axis.set_xlabel("Curriculum tier")
    axis.set_ylabel("Mean reward (95% CI)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path, help="vf-eval run directories")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("assets/evals/day5_tier_scores.png")
    )
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--title", default="uav-operator Day 5 frontier calibration")
    args = parser.parse_args()

    payload = _build_payload(args.runs, args.title)
    _plot(payload, args.output)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(args.output)


if __name__ == "__main__":
    main()
