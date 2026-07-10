"""Build the static Day 2 world JSON used for inspection and render tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uav_operator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/world.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(uav_operator.WORLD_DATA, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
