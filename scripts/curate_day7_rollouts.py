#!/usr/bin/env python3
"""Extract the fixed Day 7 Laguna evidence set from clean saved results."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from render import render_contact_sheet

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs"
    / "evals"
    / "uav-operator--poolside--laguna-m.1"
    / "day6-t3-clean-retry"
    / "results.jsonl"
)
DEFAULT_OUTPUT = ROOT / "assets" / "rollouts" / "day7"


@dataclass(frozen=True)
class Selection:
    row: int
    slug: str
    scenario_id: str
    reward: float
    evidence_role: str


SELECTIONS = (
    Selection(
        23,
        "laguna-t3-treasure-island-safe-response",
        "T3-010-treasure-island",
        0.8653668784356408,
        "hero: all-event response with safe completion",
    ),
    Selection(
        28,
        "laguna-t3-pier-39-clean-completion",
        "T3-006-pier-39",
        0.9429322465677678,
        "clean composed-event completion",
    ),
    Selection(
        4,
        "laguna-t3-bay-farm-low-margin",
        "T3-002-bay-farm",
        0.7,
        "safe completion at 28.9 percent battery",
    ),
    Selection(
        12,
        "laguna-t3-treasure-island-geofence-struggle",
        "T3-010-treasure-island",
        0.04419388897303542,
        "repeated geofence and override struggle before success",
    ),
    Selection(
        20,
        "laguna-t3-richmond-battery-loss",
        "T3-014-richmond-channel",
        -4.3,
        "battery-depletion aircraft loss",
    ),
)


def curate(source: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Extract and verify the five immutable source rows."""

    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for selection in SELECTIONS:
        row = rows[selection.row]
        scenario_id = row["sim_log"][0]["scenario_id"]
        if scenario_id != selection.scenario_id:
            raise ValueError(
                f"row {selection.row}: expected {selection.scenario_id}, got {scenario_id}"
            )
        if abs(float(row["reward"]) - selection.reward) > 1e-12:
            raise ValueError(
                f"row {selection.row}: expected reward {selection.reward}, got {row['reward']}"
            )
        artifact = {
            "schema_version": 1,
            "provenance": {
                "source_run": "day6-t3-clean-retry",
                "source_results": "outputs/evals/uav-operator--poolside--laguna-m.1/day6-t3-clean-retry/results.jsonl",
                "source_row_zero_based": selection.row,
                "model": "poolside/laguna-m.1",
                "example_id": row.get("example_id"),
                "evidence_role": selection.evidence_role,
            },
            "reward": row["reward"],
            "metrics": row.get("metrics", {}),
            "sim_log": row["sim_log"],
        }
        destination = output_dir / f"{selection.slug}.json"
        destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        artifacts.append(artifact)
        print(f"row {selection.row:02d} -> {destination.relative_to(ROOT)}")
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=ROOT / "assets" / "renders" / "day7" / "contact-sheet.png",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = curate(args.source, args.output_dir)
    render_contact_sheet(artifacts, args.contact_sheet)
    print(f"contact sheet -> {args.contact_sheet.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
