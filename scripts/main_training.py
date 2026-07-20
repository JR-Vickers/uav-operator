#!/usr/bin/env python3
"""Prepare, capture, and budget the manual-only staged main training cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "main-training"
ASSET_ROOT = REPO_ROOT / "assets" / "training"
MODEL = "Qwen/Qwen3.5-2B"
ENVIRONMENT = "jarrett/uav-operator@0.1.1"
SMOKE_RUN_ID = "vhuh1or0bar3cht6ql0jzazs"
SMOKE_STEP = 20
SMOKE_CHECKPOINT_ID = "g1akido7qfo58e3my36wnqrz"
EXISTING_EVIDENCE_COST_USD = 3.0469
AGGREGATE_CEILING_USD = 15.0
RESERVED_COSTS_USD = {
    "post_training_dev_comparison": 1.75,
    "t2_t3_red_team": 0.75,
    "frozen_final_evaluation": 1.25,
}
PRICE_FIELDS = (
    "effective_training_price_per_mtok",
    "effective_inference_input_price_per_mtok",
    "effective_inference_output_price_per_mtok",
)


@dataclass(frozen=True)
class Phase:
    name: str
    steps: int
    mixture: tuple[tuple[str, float], ...]
    artifact_interval: int
    expected_cost_usd: float
    hard_ceiling_usd: float
    predecessor: str | None


PHASES = {
    "A": Phase("A", 10, (("T1", 1.0),), 5, 0.90, 1.25, None),
    "B": Phase("B", 20, (("T1", 0.4), ("T2", 0.6)), 10, 1.90, 2.35, "A"),
    "C": Phase(
        "C", 20, (("T1", 0.25), ("T2", 0.45), ("T3", 0.30)), 10, 1.95, 2.40, "B"
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is unavailable")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _prime_json(args: Sequence[str]) -> dict[str, Any]:
    command = ["prime", "--plain", *args]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"Prime command failed: {' '.join(command)}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Prime command returned invalid JSON: {' '.join(command)}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Prime command returned a non-object")
    return payload


def _prime_text(args: Sequence[str]) -> str:
    command = ["prime", "--plain", *args]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"Prime command failed: {' '.join(command)}: {detail}")
    return completed.stdout


def _fixture_or_live(
    fixture_dir: Path | None, name: str, args: Sequence[str]
) -> dict[str, Any]:
    if fixture_dir is not None:
        path = fixture_dir / f"{name}.json"
        if not path.is_file():
            raise ValueError(f"fixture is unavailable: {path}")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"fixture is not an object: {path}")
        return payload
    return _prime_json(args)


def _rows(payload: Mapping[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _run_object(payload: Mapping[str, Any]) -> dict[str, Any]:
    run = payload.get("run", payload)
    if not isinstance(run, dict):
        raise ValueError("run metadata is unavailable")
    return run


def _checkpoint(payload: Mapping[str, Any], checkpoint_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in _rows(payload, "checkpoints")
        if row.get("id", row.get("checkpoint_id")) == checkpoint_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one checkpoint {checkpoint_id}")
    return matches[0]


def _model_pricing(payload: Mapping[str, Any]) -> dict[str, float]:
    matches = [
        row
        for row in _rows(payload, "models", "data")
        if row.get("name", row.get("id")) == MODEL
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one Hosted Training model {MODEL}")
    row = matches[0]
    if row.get("at_capacity") is not False:
        raise ValueError("Hosted Training capacity is unavailable")
    pricing = {
        field: _finite(row.get(field), f"pricing.{field}") for field in PRICE_FIELDS
    }
    if any(value < 0 for value in pricing.values()):
        raise ValueError("pricing must not be negative")
    return pricing


def _wallet(payload: Mapping[str, Any]) -> dict[str, float | str]:
    return {
        "balance_usd": _finite(payload.get("balance_usd"), "wallet.balance_usd"),
        "total_billings": _finite(
            payload.get("total_billings"), "wallet.total_billings"
        ),
        "currency": str(payload.get("currency", "USD")),
    }


def _hub_passes(payload: Mapping[str, Any]) -> bool:
    latest = payload.get("latest_version", {})
    action = payload.get("action", {})
    return (
        isinstance(latest, dict)
        and latest.get("semantic_version") == "0.1.1"
        and isinstance(action, dict)
        and action.get("status") == "SUCCESS"
    )


def phase_toml(phase: Phase, checkpoint_id: str) -> str:
    lines = [
        f'name = "uav-operator-main-phase-{phase.name.lower()}"',
        f'model = "{MODEL}"',
        'loss = "rl"',
        f'checkpoint_id = "{checkpoint_id}"',
        f"max_steps = {phase.steps}",
        "",
        "batch_size = 16",
        "rollouts_per_example = 2",
        "max_inflight_rollouts = 4",
        "learning_rate = 3e-5",
        "lora_alpha = 32",
        "",
        "[sampling]",
        "max_tokens = 1024",
        "temperature = 0.7",
        "enable_thinking = false",
    ]
    for tier, ratio in phase.mixture:
        lines.extend(
            [
                "",
                "[[env]]",
                f'id = "{ENVIRONMENT}"',
                f"ratio = {ratio}",
                "max_retries = 0",
                f'args = {{ tier = "{tier}", dataset_split = "train", max_examples = 75, max_turns = 20 }}',
            ]
        )
    lines.extend(
        [
            "",
            "[eval]",
            "interval = 10",
            "num_examples = 24",
            "rollouts_per_example = 1",
            "skip_first_step = false",
        ]
    )
    for tier in ("T0", "T1", "T2", "T3"):
        lines.extend(
            [
                "",
                "[[eval.env]]",
                f'id = "{ENVIRONMENT}"',
                "num_examples = 6",
                "rollouts_per_example = 1",
                "max_retries = 0",
                f'args = {{ tier = "{tier}", dataset_split = "dev", max_examples = 6, max_turns = 20 }}',
            ]
        )
    lines.extend(
        [
            "",
            "[eval.sampling]",
            "max_tokens = 1024",
            "temperature = 0.0",
            "enable_thinking = false",
            "",
            "[checkpoints]",
            f"interval = {phase.artifact_interval}",
            "keep_cloud = 2",
            "",
            "[adapters]",
            f"interval = {phase.artifact_interval}",
            "keep_last = 2",
            "",
        ]
    )
    return "\n".join(lines)


def validate_phase_config(
    config: Mapping[str, Any], phase: Phase, checkpoint_id: str
) -> None:
    expected = tomllib.loads(phase_toml(phase, checkpoint_id))
    if config != expected:
        raise ValueError("generated phase config does not match the frozen protocol")
    for entry in [*config["env"], *config["eval"]["env"]]:
        if entry["args"]["dataset_split"] == "eval":
            raise ValueError("final-eval rows must not enter training preparation")


def _load_capture(path: Path, phase_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"passing Phase-{phase_name} capture is required: {path}")
    artifact = json.loads(path.read_text())
    if (
        artifact.get("phase") != phase_name
        or artifact.get("summary", {}).get("passed") is not True
    ):
        raise ValueError(f"Phase-{phase_name} capture has not passed")
    return artifact


def resolve_input(
    phase: Phase, predecessor_capture: Path | None
) -> tuple[str, str, int]:
    if phase.predecessor is None:
        return SMOKE_CHECKPOINT_ID, SMOKE_RUN_ID, SMOKE_STEP
    if predecessor_capture is None:
        raise ValueError(f"Phase-{phase.predecessor} capture is required")
    capture = _load_capture(predecessor_capture, phase.predecessor)
    final = capture["summary"].get("final_checkpoint", {})
    checkpoint_id = final.get("id", final.get("checkpoint_id"))
    run_id = capture.get("run_id")
    step = final.get("step")
    if (
        not isinstance(checkpoint_id, str)
        or not isinstance(run_id, str)
        or step != PHASES[phase.predecessor].steps
    ):
        raise ValueError("predecessor final checkpoint provenance is incomplete")
    return checkpoint_id, run_id, int(step)


def _project_usage(phase: Phase, pricing: Mapping[str, float]) -> dict[str, Any]:
    # Measured paid smoke totals divided by 25 updates; evaluation workload is
    # conservatively included in those per-update rates and the hard ceiling remains authoritative.
    tokens = {
        "training": 4_361_287 / 25 * phase.steps,
        "inference_input": 0.0,
        "inference_output": 0.0,
    }
    smoke = json.loads((ASSET_ROOT / "day8_qwen35_2b_t1_smoke_25.json").read_text())
    usage = smoke.get("usage", {}).get("usage", smoke.get("usage", {}))
    inference = usage.get("inference", {})
    if isinstance(inference, dict):
        input_tokens = inference.get("input_tokens")
        output_tokens = inference.get("output_tokens")
        if isinstance(input_tokens, (int, float)) and input_tokens >= 0:
            tokens["inference_input"] = float(input_tokens) / 25 * phase.steps
        if isinstance(output_tokens, (int, float)) and output_tokens >= 0:
            tokens["inference_output"] = float(output_tokens) / 25 * phase.steps
    cost = (
        tokens["training"] * pricing[PRICE_FIELDS[0]]
        + tokens["inference_input"] * pricing[PRICE_FIELDS[1]]
        + tokens["inference_output"] * pricing[PRICE_FIELDS[2]]
    ) / 1_000_000
    return {
        "basis": "25-step paid smoke measured tokens scaled by phase updates",
        "tokens": tokens,
        "cost_usd": cost,
    }


def budget(
    captures: Sequence[Path], projected: Mapping[str, float] | None = None
) -> dict[str, Any]:
    completed: dict[str, float] = {}
    for path in captures:
        artifact = json.loads(path.read_text())
        phase_name = artifact.get("phase")
        if (
            phase_name not in PHASES
            or artifact.get("summary", {}).get("passed") is not True
        ):
            raise ValueError(f"capture is not a passing phase artifact: {path}")
        completed[phase_name] = _finite(
            artifact["summary"].get("run_cost_usd"), f"Phase-{phase_name} cost"
        )
    for name, cost in completed.items():
        if cost > PHASES[name].hard_ceiling_usd:
            raise ValueError(f"Phase-{name} cost ceiling breached")
    failures: list[str] = []
    planned = dict(projected or {})
    remaining_phases = {
        name: completed.get(
            name,
            _finite(
                planned.get(name, phase.expected_cost_usd), f"Phase-{name} projection"
            ),
        )
        for name, phase in PHASES.items()
    }
    for name, cost in remaining_phases.items():
        if cost > PHASES[name].hard_ceiling_usd:
            failures.append(f"Phase-{name} projection exceeds its hard ceiling")
    total = (
        EXISTING_EVIDENCE_COST_USD
        + sum(remaining_phases.values())
        + sum(RESERVED_COSTS_USD.values())
    )
    hard_ceiling_total = (
        EXISTING_EVIDENCE_COST_USD
        + sum(p.hard_ceiling_usd for p in PHASES.values())
        + sum(RESERVED_COSTS_USD.values())
    )
    if total > AGGREGATE_CEILING_USD:
        failures.append("projected aggregate cost exceeds $15.00")
    if hard_ceiling_total > AGGREGATE_CEILING_USD:
        failures.append("hard-ceiling ledger exceeds $15.00")
    return {
        "passed": not failures,
        "failures": failures,
        "existing_evidence_usd": EXISTING_EVIDENCE_COST_USD,
        "phase_costs_usd": remaining_phases,
        "reserved_costs_usd": RESERVED_COSTS_USD,
        "projected_total_usd": total,
        "hard_ceiling_total_usd": hard_ceiling_total,
        "aggregate_ceiling_usd": AGGREGATE_CEILING_USD,
        "unallocated_after_hard_ceilings_usd": AGGREGATE_CEILING_USD
        - hard_ceiling_total,
    }


def prepare(
    phase_name: str,
    output_dir: Path,
    predecessor_capture: Path | None = None,
    fixture_dir: Path | None = None,
) -> dict[str, Any]:
    phase = PHASES[phase_name]
    checkpoint_id, source_run_id, source_step = resolve_input(
        phase, predecessor_capture
    )
    run_payload = _fixture_or_live(
        fixture_dir, "source_run", ["train", "get", source_run_id, "--output", "json"]
    )
    checkpoints = _fixture_or_live(
        fixture_dir,
        "source_checkpoints",
        ["train", "checkpoints", source_run_id, "--output", "json"],
    )
    models = _fixture_or_live(
        fixture_dir, "models", ["train", "models", "--output", "json"]
    )
    wallet_payload = _fixture_or_live(
        fixture_dir, "wallet", ["wallet", "--limit", "20", "--output", "json"]
    )
    hub = _fixture_or_live(
        fixture_dir,
        "hub",
        ["env", "status", "jarrett/uav-operator", "--output", "json"],
    )
    run = _run_object(run_payload)
    checkpoint = _checkpoint(checkpoints, checkpoint_id)
    if run.get("id") != source_run_id or run.get("base_model") != MODEL:
        raise ValueError("input run/model provenance mismatch")
    if checkpoint.get("step") != source_step:
        raise ValueError("input checkpoint step mismatch")
    if checkpoint.get("status") != "READY":
        raise ValueError("input checkpoint status is not READY")
    if (
        checkpoint.get("run_id", checkpoint.get("rft_run_id", source_run_id))
        != source_run_id
    ):
        raise ValueError("input checkpoint run provenance mismatch")
    if checkpoint.get("base_model", MODEL) != MODEL:
        raise ValueError("input checkpoint model provenance mismatch")
    pricing = _model_pricing(models)
    wallet = _wallet(wallet_payload)
    if not _hub_passes(hub):
        raise ValueError("Hub version/action gate failed")
    projection = _project_usage(phase, pricing)
    if projection["cost_usd"] > phase.hard_ceiling_usd:
        raise ValueError(
            f"Phase-{phase.name} projected cost ${projection['cost_usd']:.4f} "
            f"exceeds ${phase.hard_ceiling_usd:.2f} hard ceiling"
        )
    ledger = budget([], {phase.name: projection["cost_usd"]})
    if not ledger["passed"]:
        raise ValueError(
            "aggregate budget gate failed: " + "; ".join(ledger["failures"])
        )
    config_text = phase_toml(phase, checkpoint_id)
    parsed = tomllib.loads(config_text)
    validate_phase_config(parsed, phase, checkpoint_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "train.toml"
    config_path.write_text(config_text)
    manifest = {
        "schema_version": 1,
        "kind": "uav_operator_main_training_prepare",
        "prepared_at": _now(),
        "phase": phase.name,
        "status": "awaiting_explicit_manual_launch_approval",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_text.encode()),
        "input_provenance": {
            "run_id": source_run_id,
            "step": source_step,
            "checkpoint_id": checkpoint_id,
            "status": checkpoint["status"],
            "model": MODEL,
        },
        "pricing_usd_per_mtok": pricing,
        "wallet": wallet,
        "hub_status": hub,
        "projected_usage": projection,
        "stop_limits": {
            "phase_cost_usd": phase.hard_ceiling_usd,
            "aggregate_cost_usd": AGGREGATE_CEILING_USD,
            "optimizer_stall_minutes": 15,
            "max_eval_truncation_rate": 0.5,
            "positive_hard_safety_penalty_allowed": False,
        },
        "budget": ledger,
        "manual_launch_command": f"prime --plain train {config_path} --output json",
        "description": f"Phase {phase.name} is {phase.steps} additional warm-started updates; optimizer-state restoration is not claimed.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return manifest


def _metric_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _rows(payload, "metrics", "data")


def validate_capture(artifact: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    phase = PHASES[str(artifact.get("phase"))]
    run = _run_object(artifact.get("run", {}))
    config = artifact.get("config", {}).get("parsed", {})
    progress = artifact.get("progress", {})
    metrics = _metric_rows(artifact.get("metrics", {}))
    samples = artifact.get("samples_by_step", {})
    checkpoints = _rows(artifact.get("checkpoints", {}), "checkpoints")
    adapters = [
        row
        for row in _rows(artifact.get("adapters", {}), "models", "adapters")
        if row.get("rft_run_id", row.get("run_id")) == artifact.get("run_id")
    ]
    if run.get("status") != "COMPLETED":
        failures.append("run status is not COMPLETED")
    if run.get("base_model") != MODEL or config.get("model") != MODEL:
        failures.append("base model mismatch")
    if run.get("max_steps") != phase.steps or config.get("max_steps") != phase.steps:
        failures.append("expected phase steps mismatch")
    expected_train_steps = set(range(phase.steps))
    if set(progress.get("steps_with_samples", [])) != expected_train_steps:
        failures.append("sample steps are incomplete")
    if set(progress.get("steps_with_distributions", [])) != expected_train_steps:
        failures.append("distribution steps are incomplete")
    if progress.get("latest_step") != phase.steps:
        failures.append("latest optimizer step is incomplete")
    non_finite = []

    def walk(value: Any, path: str = "$") -> None:
        if isinstance(value, float) and not math.isfinite(value):
            non_finite.append(path)
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(metrics)
    if non_finite:
        failures.append("non-finite metrics: " + ", ".join(non_finite))
    expected_eval_steps = set(range(0, phase.steps + 1, 10))
    eval_steps = {
        row.get("step")
        for row in metrics
        if any(str(key).startswith("eval/") for key in row)
    }
    if eval_steps != expected_eval_steps:
        failures.append("evaluation milestones are incomplete")
    provider_errors = 0
    cancelled = 0
    truncation_by_step: dict[str, dict[str, int | float]] = {}
    safety_penalties = 0
    for step, payload in samples.items() if isinstance(samples, dict) else []:
        rows = _rows(payload, "samples", "rollouts")
        truncated = 0
        for row in rows:
            if row.get("provider_error") or row.get("error"):
                provider_errors += 1
            if str(row.get("status", "")).upper() == "CANCELLED":
                cancelled += 1
            if (
                row.get("stop_condition") == "max_turns_reached"
                or row.get("is_truncated") is True
            ):
                truncated += 1
            hard = (
                row.get("metrics", {}).get("hard_safety", 0)
                if isinstance(row.get("metrics"), dict)
                else 0
            )
            if isinstance(hard, (int, float)) and hard < 0:
                safety_penalties += 1
        truncation_by_step[str(step)] = {
            "count": truncated,
            "n": len(rows),
            "rate": truncated / len(rows) if rows else 0.0,
        }
    if provider_errors:
        failures.append("provider errors are nonzero")
    if cancelled:
        failures.append("cancelled rows are nonzero")
    if safety_penalties:
        failures.append("hard-safety penalty is positive in magnitude")
    for step in expected_eval_steps:
        summary = truncation_by_step.get(str(step))
        if summary and summary["rate"] > 0.5:
            failures.append(f"evaluation truncation exceeds 50% at step {step}")
    retained_steps = sorted(
        row.get("step") for row in checkpoints if row.get("status") == "READY"
    )
    expected_retained = [phase.artifact_interval, phase.steps]
    if retained_steps != expected_retained:
        failures.append("retained READY checkpoint steps are incomplete")
    final_matches = [
        row
        for row in checkpoints
        if row.get("step") == phase.steps and row.get("status") == "READY"
    ]
    if len(final_matches) != 1:
        failures.append("exact final READY checkpoint is unavailable")
    adapter_steps = sorted(
        row.get("step") for row in adapters if row.get("status") == "READY"
    )
    if adapter_steps != expected_retained:
        failures.append("retained READY adapter steps are incomplete")
    try:
        cost = _finite(
            artifact.get("usage", {}).get("total_cost_usd"), "usage.total_cost_usd"
        )
    except ValueError as exc:
        failures.append(str(exc))
        cost = math.nan
    if math.isfinite(cost) and cost > phase.hard_ceiling_usd:
        failures.append("phase cost ceiling breached")
    input_provenance = artifact.get("manifest", {}).get("input_provenance")
    if config.get("checkpoint_id") != (input_provenance or {}).get("checkpoint_id"):
        failures.append("warm-start provenance mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "run_cost_usd": cost,
        "provider_errors": provider_errors,
        "cancelled_rows": cancelled,
        "hard_safety_penalties": safety_penalties,
        "truncation_by_step": truncation_by_step,
        "ready_checkpoint_steps": retained_steps,
        "ready_adapter_steps": adapter_steps,
        "final_checkpoint": final_matches[0] if len(final_matches) == 1 else None,
    }


def capture(
    phase_name: str,
    run_id: str,
    manifest_path: Path,
    output: Path,
    plot: Path,
    fixture_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("phase") != phase_name:
        raise ValueError("manifest phase mismatch")
    config_path = Path(manifest["config_path"])
    config_bytes = config_path.read_bytes()
    if _sha256(config_bytes) != manifest.get("config_sha256"):
        raise ValueError("prepared config hash mismatch")

    def fetch(name: str, args: Sequence[str]) -> dict[str, Any]:
        return _fixture_or_live(fixture_dir, name, args)

    progress = fetch("progress", ["train", "progress", run_id])
    steps = progress.get("steps_with_samples", [])
    samples = {
        str(step): fetch(
            f"samples_{step}",
            ["train", "rollouts", run_id, "--step", str(step), "--output", "json"],
        )
        for step in steps
    }
    artifact = {
        "schema_version": 1,
        "kind": "uav_operator_main_training_capture",
        "captured_at": _now(),
        "phase": phase_name,
        "run_id": run_id,
        "manifest": manifest,
        "config": {
            "sha256": _sha256(config_bytes),
            "text": config_bytes.decode(),
            "parsed": tomllib.loads(config_bytes.decode()),
        },
        "run": fetch("run", ["train", "get", run_id, "--output", "json"]),
        "progress": progress,
        "metrics": fetch("metrics", ["train", "metrics", run_id]),
        "distributions": fetch("distributions", ["train", "distributions", run_id]),
        "usage": fetch("usage", ["train", "usage", run_id, "--output", "json"]),
        "logs": (
            _prime_text(["train", "logs", run_id])
            if fixture_dir is None
            else (fixture_dir / "logs.txt").read_text()
        ),
        "checkpoints": fetch(
            "checkpoints", ["train", "checkpoints", run_id, "--output", "json"]
        ),
        "adapters": fetch("adapters", ["train", "models", "--output", "json"]),
        "samples_by_step": samples,
        "content_policy": "No model prose is judged; all gates use platform status and simulator-derived metrics.",
    }
    artifact["summary"] = validate_capture(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    plot_curve(artifact, plot)
    return artifact


def plot_curve(artifact: Mapping[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _metric_rows(artifact.get("metrics", {}))
    train = [
        (row["step"], row["reward/all/mean"])
        for row in rows
        if isinstance(row.get("reward/all/mean"), (int, float))
    ]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    if train:
        axis.plot(
            [x for x, _ in train],
            [y for _, y in train],
            color="#66788a",
            label="training reward",
        )
    for tier, color in zip(
        ("T0", "T1", "T2", "T3"),
        ("#2a9d8f", "#457b9d", "#e9c46a", "#e76f51"),
        strict=True,
    ):
        points = []
        for row in rows:
            values = [
                value
                for key, value in row.items()
                if str(key).startswith("eval/")
                and tier in str(key)
                and isinstance(value, (int, float))
            ]
            if values:
                points.append((row["step"], sum(values) / len(values)))
        if points:
            axis.plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                label=f"{tier} dev",
                color=color,
            )
    axis.set(
        title=f"Main training Phase {artifact['phase']}",
        xlabel="phase optimizer step",
        ylabel="state-derived reward",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def select_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in candidates if row.get("phase") in PHASES]
    if {row.get("phase") for row in eligible} != set(PHASES):
        raise ValueError("Phase A/B/C candidate coverage is incomplete")
    seeds = [set(row.get("paired_seeds", [])) for row in candidates]
    if not seeds or any(seed_set != seeds[0] for seed_set in seeds[1:]):
        raise ValueError("candidate paired-seed coverage differs")

    def key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
        return (
            _finite(row.get("hard_safety_violations"), "hard safety"),
            -_finite(row.get("mean_reward"), "mean reward"),
            -_finite(row.get("completions"), "completions"),
            _finite(row.get("max_turn_truncations"), "truncations"),
        )

    winner = min(eligible, key=key)
    return {
        "winner": dict(winner),
        "eligible_phases": sorted(row["phase"] for row in eligible),
        "smoke_is_reference_only": any(
            row.get("phase") == "smoke" for row in candidates
        ),
        "selection_rule": "fewer hard-safety violations; higher mean reward; more completions; fewer max-turn truncations",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("phase", choices=PHASES)
    prepare_parser.add_argument("--output-dir", type=Path)
    prepare_parser.add_argument("--predecessor-capture", type=Path)
    prepare_parser.add_argument("--fixture-dir", type=Path)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("phase", choices=PHASES)
    capture_parser.add_argument("run_id")
    capture_parser.add_argument("--manifest", required=True, type=Path)
    capture_parser.add_argument("--output", type=Path)
    capture_parser.add_argument("--plot", type=Path)
    capture_parser.add_argument("--fixture-dir", type=Path)
    budget_parser = sub.add_parser("budget")
    budget_parser.add_argument("captures", nargs="*", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        result = prepare(
            args.phase,
            args.output_dir or OUTPUT_ROOT / f"phase-{args.phase.lower()}",
            args.predecessor_capture,
            args.fixture_dir,
        )
    elif args.operation == "capture":
        result = capture(
            args.phase,
            args.run_id,
            args.manifest,
            args.output or ASSET_ROOT / f"main_phase_{args.phase.lower()}.json",
            args.plot or ASSET_ROOT / f"main_phase_{args.phase.lower()}_curve.png",
            args.fixture_dir,
        )
    else:
        result = budget(args.captures)
    summary = result.get("summary", result)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if summary.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
