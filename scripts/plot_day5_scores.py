"""Plot Day 5 tier calibration summaries; matplotlib is a dev-only dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("assets/evals/day5_tier_scores.png"))
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    payload = json.loads(args.input.read_text())
    tiers = ["T0", "T1", "T2", "T3"]
    figure, axis = plt.subplots(figsize=(7, 4))
    for policy, by_tier in payload["policies"].items():
        means = [float(by_tier[tier]["summary"]["mean_reward"]) for tier in tiers]
        stds = [float(by_tier[tier]["summary"]["std_reward"]) for tier in tiers]
        axis.errorbar(tiers, means, yerr=stds, marker="o", capsize=4, label=policy)
    axis.set_title("uav-operator Day 5 calibration")
    axis.set_xlabel("Curriculum tier")
    axis.set_ylabel("Reward")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
