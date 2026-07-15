#!/usr/bin/env python3
"""Preflight and capture reproducible Hosted Training evidence for Day 8."""

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
DEFAULT_CONFIG = REPO_ROOT / "configs" / "day8_llama_1b_t1_diagnostic.toml"
DEFAULT_PREFLIGHT = REPO_ROOT / "assets" / "training" / "day8_llama_1b_preflight.json"
DEFAULT_CAPTURE = REPO_ROOT / "assets" / "training" / "day8_llama_1b_diagnostic.json"
DEFAULT_STEPS = (0, 1)
ENVIRONMENT = "jarrett/uav-operator"
ENVIRONMENT_VERSION = "0.1.1"
ZERO_PRICE_FIELDS = (
    "effective_training_price_per_mtok",
    "effective_inference_input_price_per_mtok",
    "effective_inference_output_price_per_mtok",
)
KNOWN_FREE_TIER_DENIALS = {
    ("sprints/Llama-3.2-1B-Instruct", ENVIRONMENT): (
        "HTTP 400: Free-tier model 'sprints/Llama-3.2-1B-Instruct': "
        "'jarrett/uav-operator' does not meet the free-tier environment requirements."
    )
}


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


def _config_bundle(config_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    try:
        display_path = str(config_path.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(config_path)
    return config, {
        "path": display_path,
        "sha256": hashlib.sha256(config_bytes).hexdigest(),
        "text": config_bytes.decode("utf-8"),
    }


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


def find_non_finite(value: Any, path: str = "$") -> list[str]:
    """Return JSON-style paths to every non-finite floating-point value."""

    failures: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        failures.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            failures.extend(find_non_finite(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(find_non_finite(child, f"{path}[{index}]"))
    return failures


def _catalog_items(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = catalog.get("models", catalog.get("data", []))
    return [item for item in raw_items if isinstance(item, dict)]


def _model_name(item: dict[str, Any]) -> Any:
    return item.get("name", item.get("id"))


def validate_preflight(artifact: dict[str, Any]) -> dict[str, Any]:
    """Validate only launch blockers; inference-catalog absence is informational."""

    failures: list[str] = []
    model = artifact.get("hosted_training_model")
    if not isinstance(model, dict):
        failures.append("exact model is absent from Hosted Training")
    else:
        if model.get("at_capacity") is not False:
            failures.append("Hosted Training model is at capacity or capacity is unknown")
        for field in ZERO_PRICE_FIELDS:
            if model.get(field) != 0:
                failures.append(f"{field} is not exactly zero")

    wallet = artifact.get("wallet", {})
    if not isinstance(wallet.get("balance_usd"), (int, float)):
        failures.append("wallet balance is missing")

    hub = artifact.get("hub_status", {})
    if _find_value(hub, {"semantic_version"}) != ENVIRONMENT_VERSION:
        failures.append(f"Hub version is not {ENVIRONMENT_VERSION}")
    action = hub.get("action", {}) if isinstance(hub, dict) else {}
    if not isinstance(action, dict) or action.get("status") != "SUCCESS":
        failures.append("Hub quality action is not SUCCESS")

    eligibility = artifact.get("free_tier_environment_eligibility", {})
    if eligibility.get("status") != "eligible":
        failures.append(
            "free-tier model/environment eligibility is not affirmatively confirmed"
        )

    failures.extend(f"non-finite number at {path}" for path in find_non_finite(artifact))
    return {
        "passed": not failures,
        "failures": failures,
        "inference_catalog_exact_id_present": artifact.get(
            "inference_catalog_exact_id_present", False
        ),
        "note": (
            "Ordinary inference-catalog absence is recorded but is not a preflight "
            "failure; Hosted Training is the compatibility test."
        ),
    }


def preflight(config_path: Path) -> dict[str, Any]:
    config, config_evidence = _config_bundle(config_path)
    model_name = config.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("config must contain a non-empty model string")

    hosted_catalog = _prime_json(["train", "models", "--output", "json"])
    hosted_model = next(
        (item for item in _catalog_items(hosted_catalog) if _model_name(item) == model_name),
        None,
    )
    inference_catalog = _prime_json(
        ["inference", "models", "--output", "json", "--search", model_name]
    )
    inference_matches = [
        item
        for item in _catalog_items(inference_catalog)
        if _model_name(item) == model_name
    ]
    wallet = _prime_json(["wallet", "--limit", "20", "--output", "json"])
    hub_status = _prime_json(["env", "status", ENVIRONMENT, "--output", "json"])
    eligibility_flag = _find_value(
        hub_status, {"free_tier_eligible", "freeTierEligible"}
    )
    known_denial = KNOWN_FREE_TIER_DENIALS.get((model_name, ENVIRONMENT))
    if known_denial:
        eligibility = {
            "status": "denied",
            "source": "recorded_run_creation_response",
            "detail": known_denial,
        }
    elif eligibility_flag is True:
        eligibility = {
            "status": "eligible",
            "source": "hub_status",
            "detail": None,
        }
    else:
        eligibility = {
            "status": "unverifiable",
            "source": None,
            "detail": (
                "Prime model and environment-status responses expose no free-tier "
                "eligibility field, and /rft/runs/preview returns HTTP 405."
            ),
        }
    artifact = {
        "schema_version": 2,
        "kind": "day8_hosted_training_preflight",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "config": config_evidence,
        "model": model_name,
        "hosted_training_model": hosted_model,
        "inference_catalog_exact_id_present": bool(inference_matches),
        "inference_catalog_exact_matches": inference_matches,
        "free_tier_environment_eligibility": eligibility,
        "wallet": {
            "balance_usd": wallet.get("balance_usd"),
            "currency": wallet.get("currency"),
            "total_billings": wallet.get("total_billings"),
        },
        "hub_status": hub_status,
    }
    artifact["summary"] = validate_preflight(artifact)
    return artifact


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


def summarize_logs(logs: str) -> dict[str, Any]:
    """Report explicit platform failures and checkpoint/adapter upload evidence."""

    lines = logs.splitlines()
    return {
        "model_errors": logs.count("Error: ModelError"),
        "rollout_failures": logs.count("Rollout failed in group"),
        "provider_or_context_errors": sum(
            1
            for line in lines
            if any(marker in line.lower() for marker in ("provider error", "context length"))
        ),
        "checkpoint_evidence": [
            line for line in lines if "checkpoint" in line.lower() and "upload" in line.lower()
        ],
        "adapter_upload_evidence": [
            line for line in lines if "adapter" in line.lower() and "upload" in line.lower()
        ],
    }


def wallet_evidence(wallet: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Retain run reconciliation fields without unrelated account history."""

    return {
        "balance_usd": wallet.get("balance_usd"),
        "currency": wallet.get("currency"),
        "total_billings": wallet.get("total_billings"),
        "run_billings": [
            row
            for row in wallet.get("recent_billings", [])
            if row.get("resource_id") == run_id
        ],
    }


def billing_reconciliation(
    preflight_wallet: dict[str, Any], captured_wallet: dict[str, Any], usage: dict[str, Any]
) -> dict[str, Any]:
    before = preflight_wallet.get("balance_usd")
    after = captured_wallet.get("balance_usd")
    balance_delta = before - after if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
    run_billed = sum(
        row.get("amount_usd", 0.0)
        for row in captured_wallet.get("run_billings", [])
        if isinstance(row.get("amount_usd", 0.0), (int, float))
    )
    usage_cost = usage.get("total_cost_usd")
    unrelated_or_unreconciled = balance_delta - run_billed if balance_delta is not None else None
    return {
        "preflight_balance_usd": before,
        "captured_balance_usd": after,
        "balance_delta_usd": balance_delta,
        "run_billing_rows_total_usd": run_billed,
        "usage_total_cost_usd": usage_cost,
        "unrelated_or_unreconciled_wallet_delta_usd": unrelated_or_unreconciled,
        "run_reports_zero_cost": usage_cost == 0 and run_billed == 0,
    }


def validate_capture(artifact: dict[str, Any]) -> dict[str, Any]:
    failures = [
        f"non-finite number at {path}" for path in find_non_finite(artifact)
    ]
    preflight_summary = artifact.get("preflight", {}).get("summary", {})
    if not preflight_summary.get("passed"):
        failures.append("referenced preflight did not pass")
    if artifact.get("config", {}).get("sha256") != artifact.get("preflight", {}).get(
        "config", {}
    ).get("sha256"):
        failures.append("capture config does not match preflight config")
    if not artifact.get("billing_reconciliation", {}).get("run_reports_zero_cost"):
        failures.append("run reports positive or missing cost")
    try:
        config = tomllib.loads(artifact.get("config", {}).get("text", ""))
    except tomllib.TOMLDecodeError:
        config = {}
        failures.append("captured config text is invalid TOML")
    target_step = config.get("max_steps")
    latest_step = artifact.get("progress", {}).get("latest_step")
    training_tokens = _find_value(
        artifact.get("usage", {}).get("training", {}), {"tokens"}
    )
    if isinstance(target_step, int):
        if not isinstance(latest_step, int) or latest_step < target_step:
            failures.append(f"optimizer has not reached target step {target_step}")
        if not isinstance(training_tokens, (int, float)) or training_tokens <= 0:
            failures.append("training-token usage is not positive")
        if artifact.get("sample_summary", {}).get("sample_count", 0) <= 0:
            failures.append("no sampled rollout is retrievable")
        distributions = artifact.get("reward_distributions", {})
        required_distributions = 1 if target_step == 1 else 2
        if len(distributions) < required_distributions:
            failures.append("insufficient final/intermediate reward distributions")
        checkpoints = artifact.get("checkpoints", {}).get("checkpoints", [])
        if not any(
            checkpoint.get("status") == "READY"
            and isinstance(checkpoint.get("step"), int)
            and checkpoint["step"] >= target_step
            for checkpoint in checkpoints
        ):
            failures.append("final READY checkpoint is missing")
        if not artifact.get("log_summary", {}).get("adapter_upload_evidence"):
            failures.append("adapter-upload evidence is missing")
        if target_step > 1 and len(artifact.get("sampled_rollouts", {})) < 2:
            failures.append("final and intermediate sampled rollouts are missing")
    provider_errors = artifact.get("sample_summary", {}).get("provider_errors", 0)
    provider_errors += artifact.get("log_summary", {}).get(
        "provider_or_context_errors", 0
    )
    return {
        "passed": not failures,
        "failures": failures,
        "target_step": target_step,
        "latest_step": latest_step,
        "training_tokens": training_tokens,
        "platform_model_errors": artifact.get("log_summary", {}).get(
            "model_errors", 0
        ),
        "provider_or_context_errors": provider_errors,
        "missing_sample_milestones": artifact.get("missing_sample_milestones", []),
        "missing_distribution_milestones": artifact.get(
            "missing_distribution_milestones", []
        ),
        "run_reports_zero_cost": artifact.get("billing_reconciliation", {}).get(
            "run_reports_zero_cost", False
        ),
    }


def _parse_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must contain integers") from exc
    if not steps or any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("steps must be comma-separated non-negative integers")
    return steps


def capture(
    run_id: str, config_path: Path, preflight_path: Path, steps: Sequence[int]
) -> dict[str, Any]:
    _, config_evidence = _config_bundle(config_path)
    preflight_artifact = json.loads(preflight_path.read_text())
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
    wallet = wallet_evidence(
        _prime_json(["wallet", "--limit", "100", "--output", "json"]), run_id
    )
    usage = _prime_json(["train", "usage", run_id, "--output", "json"])
    logs = _prime_text(["train", "logs", run_id, "--tail", "5000", "--raw"])
    artifact = {
        "schema_version": 2,
        "kind": "day8_hosted_training_capture",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "config": config_evidence,
        "preflight": preflight_artifact,
        "requested_milestones": list(steps),
        "missing_sample_milestones": sorted(requested_steps - sample_steps),
        "missing_distribution_milestones": sorted(requested_steps - distribution_steps),
        "run": _prime_json(["train", "get", run_id, "--output", "json"]),
        "progress": progress,
        "metrics": _prime_json(["train", "metrics", run_id]),
        "reward_distributions": distributions,
        "sampled_rollouts": samples,
        "sample_summary": summarize_samples(samples),
        "usage": usage,
        "checkpoints": _prime_json(
            ["train", "checkpoints", run_id, "--output", "json"]
        ),
        "model_pricing": preflight_artifact.get("hosted_training_model"),
        "wallet": wallet,
        "hub_status": preflight_artifact.get("hub_status"),
        "log_summary": summarize_logs(logs),
        "logs": logs,
    }
    artifact["billing_reconciliation"] = billing_reconciliation(
        preflight_artifact.get("wallet", {}), wallet, usage
    )
    artifact["summary"] = validate_capture(artifact)
    return artifact


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preflight_parser.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("run_id", help="Hosted Training run ID")
    capture_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    capture_parser.add_argument("--preflight", type=Path, required=True)
    capture_parser.add_argument("--steps", type=_parse_steps, default=DEFAULT_STEPS)
    capture_parser.add_argument("--output", type=Path, default=DEFAULT_CAPTURE)
    args = parser.parse_args()

    if args.command == "preflight":
        artifact = preflight(args.config.resolve())
        _write_artifact(args.output, artifact)
        if not artifact["summary"]["passed"]:
            raise SystemExit(1)
    else:
        artifact = capture(
            args.run_id, args.config.resolve(), args.preflight.resolve(), args.steps
        )
        _write_artifact(args.output, artifact)
        if not artifact["summary"]["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
