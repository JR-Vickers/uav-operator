"""Run frontier evals until every requested rollout completes without an error.

The verifiers resume path counts errored rows as completed rollouts. This wrapper
archives those rows, removes them from the resumable results file, and resumes
only the missing per-example slots. A max-turn or truncated rollout is a valid
model outcome; only rows with a non-null ``error`` are retried.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


TIERS = ("T0", "T1", "T2", "T3")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _replace_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )
    temporary.replace(path)


def _compact_results(run_dir: Path, rollouts_per_example: int) -> dict[str, int]:
    """Archive failed/overflow rows and retain resumable clean per-example rows."""

    results_path = run_dir / "results.jsonl"
    rows = _read_jsonl(results_path)
    clean: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    counts: dict[object, int] = {}

    for row in rows:
        if row.get("error") is not None:
            failed.append(row)
            continue
        example_id = row.get("example_id")
        count = counts.get(example_id, 0)
        if count >= rollouts_per_example:
            overflow.append(row)
            continue
        counts[example_id] = count + 1
        clean.append(row)

    _append_jsonl(run_dir / "failed_results.jsonl", failed)
    _append_jsonl(run_dir / "overflow_results.jsonl", overflow)
    _replace_jsonl(results_path, clean)
    return {
        "clean": len(clean),
        "failed_this_attempt": len(failed),
        "overflow_this_attempt": len(overflow),
    }


def _parse_resume_tiers(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        tier, separator, raw_path = value.partition("=")
        if not separator or tier not in TIERS or not raw_path:
            raise ValueError(f"--resume-tier must be TIER=PATH, got {value!r}")
        parsed[tier] = Path(raw_path)
    return parsed


def _find_run_dir(tier_root: Path) -> Path:
    candidates = sorted(
        path.parent
        for path in tier_root.glob("evals/*/*/metadata.json")
        if (path.parent / "results.jsonl").exists()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one run beneath {tier_root}, found {len(candidates)}; "
            "pass --resume-tier TIER=PATH to select one"
        )
    return candidates[0]


def _run_command(
    *,
    env_id: str,
    model: str,
    tier: str,
    dataset_split: str,
    num_examples: int,
    rollouts_per_example: int,
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
    timeout: int,
    tier_root: Path,
    run_dir: Path | None,
) -> int:
    command = [
        "uv",
        "run",
        "prime",
        "--plain",
        "eval",
        "run",
        env_id,
        "-m",
        model,
        "-n",
        str(num_examples),
        "-r",
        str(rollouts_per_example),
        "-t",
        str(max_tokens),
        "-T",
        str(temperature),
        "--max-concurrent",
        str(max_concurrent),
        "--timeout",
        str(timeout),
        "--disable-tui",
        "--save-results",
        "--state-columns",
        "sim_state,sim_log",
        "-a",
        json.dumps({"tier": tier, "dataset_split": dataset_split}),
    ]
    if run_dir is None:
        command.extend(("--output-dir", str(tier_root)))
    else:
        command.extend(("--resume", str(run_dir)))
    print("running:", " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def _write_summary(
    run_dir: Path,
    *,
    tier: str,
    attempt: int,
    target: int,
    counts: dict[str, int],
    return_code: int,
) -> None:
    archived_failures = len(_read_jsonl(run_dir / "failed_results.jsonl"))
    payload = {
        "tier": tier,
        "attempts": attempt,
        "target_clean_rollouts": target,
        "clean_rollouts": counts["clean"],
        "archived_provider_errors": archived_failures,
        "last_attempt_provider_errors": counts["failed_this_attempt"],
        "last_command_return_code": return_code,
        "complete": counts["clean"] == target,
        "results_path": str(run_dir),
    }
    (run_dir / "clean_retry_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _run_tier(
    args: argparse.Namespace, tier: str, initial_run_dir: Path | None
) -> Path:
    target = args.num_examples * args.rollouts_per_example
    tier_root = args.work_dir / args.model.replace("/", "--") / tier.lower()
    tier_root.mkdir(parents=True, exist_ok=True)
    run_dir = initial_run_dir
    attempt = 0

    while True:
        if run_dir is not None and not (run_dir / "metadata.json").exists():
            raise FileNotFoundError(f"invalid resume directory: {run_dir}")
        if run_dir is not None:
            counts = _compact_results(run_dir, args.rollouts_per_example)
            if counts["clean"] == target:
                _write_summary(
                    run_dir,
                    tier=tier,
                    attempt=attempt,
                    target=target,
                    counts=counts,
                    return_code=0,
                )
                print(f"{tier}: {target}/{target} clean at {run_dir}", flush=True)
                return run_dir

        if args.max_attempts and attempt >= args.max_attempts:
            raise RuntimeError(f"{tier}: stopped after {attempt} attempts")
        attempt += 1
        return_code = _run_command(
            env_id=args.env_id,
            model=args.model,
            tier=tier,
            dataset_split=args.dataset_split,
            num_examples=args.num_examples,
            rollouts_per_example=args.rollouts_per_example,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            tier_root=tier_root,
            run_dir=run_dir,
        )
        if run_dir is None:
            run_dir = _find_run_dir(tier_root)
        counts = _compact_results(run_dir, args.rollouts_per_example)
        _write_summary(
            run_dir,
            tier=tier,
            attempt=attempt,
            target=target,
            counts=counts,
            return_code=return_code,
        )
        print(
            f"{tier}: attempt={attempt} clean={counts['clean']}/{target} "
            f"new_errors={counts['failed_this_attempt']} rc={return_code}",
            flush=True,
        )
        if counts["clean"] == target:
            return run_dir
        time.sleep(args.retry_delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--env-id", default="uav-operator")
    parser.add_argument("--tiers", nargs="+", choices=TIERS, default=list(TIERS))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--num-examples", type=int, default=15)
    parser.add_argument("--rollouts-per-example", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument(
        "--timeout",
        type=int,
        default=420,
        help="Per-rollout wall-clock timeout in seconds; overall retries remain unlimited.",
    )
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Attempts per tier; 0 retries indefinitely (default).",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("outputs/frontier-calibration")
    )
    parser.add_argument(
        "--resume-tier",
        action="append",
        default=[],
        metavar="TIER=PATH",
        help="Seed a tier from an existing vf-eval run; may be repeated.",
    )
    args = parser.parse_args()
    resume_tiers = _parse_resume_tiers(args.resume_tier)

    completed: dict[str, str] = {}
    try:
        for tier in args.tiers:
            completed[tier] = str(_run_tier(args, tier, resume_tiers.get(tier)))
    except KeyboardInterrupt:
        print("interrupted; rerun with the same --resume-tier paths to continue")
        raise SystemExit(130) from None
    print(json.dumps({"model": args.model, "runs": completed}, indent=2))


if __name__ == "__main__":
    main()
