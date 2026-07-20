"""Prepare and validate frozen-dev checkpoint evaluation evidence.

This module never starts training, deploys a model, or runs inference.  The
``prepare`` command performs read-only metadata checks and writes the exact
manual eval command.  ``summarize`` accepts only complete saved vf-eval runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DEV_SEEDS = tuple(range(10_000, 10_015))
FINAL_EVAL_SEEDS = tuple(range(20_000, 20_015))
NUM_EXAMPLES = 15
ROLLOUTS_PER_EXAMPLE = 2
MAX_TURNS = 20
MAX_TOKENS = 1_024
MAX_CONCURRENT = 4
PROVENANCE_VALUES = ("captured", "synthetic_fixture")
FIXTURE_WATERMARK = "SYNTHETIC FIXTURE — NOT RESULTS"
SIM_LOG_KEYS = {
    "event",
    "battery_pct",
    "mission_distance_to_target_nm",
    "seed",
    "scenario_id",
    "tier",
    "sim_time_s",
    "aircraft",
    "mission",
    "home_site_id",
    "wind",
    "rng_state",
    "scenario_par",
    "events",
    "active_events",
    "acknowledged_events",
    "closed_sites",
    "hard_safety_violations",
    "procedure_violations",
    "current_plan",
    "lost_link_plan",
    "rng_draws",
    "active_failsafe",
    "alerts",
    "acknowledged_alerts",
    "overrides",
    "payload_released",
    "hold_until_s",
    "current_altitude_target_ft",
    "current_airspeed_kt",
    "ground_no_progress_streak",
    "last_action",
    "last_result",
    "is_terminal",
    "terminal_reason",
}
AIRCRAFT_KEYS = {"lat", "lon", "alt_ft", "status", "battery_wh", "current_site_id"}
MISSION_KEYS = {
    "mission_id",
    "description",
    "launch_site_id",
    "target",
    "value",
    "sla_min",
    "status",
    "completed_time_s",
    "failure_reason",
}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every JSONL row must be an object")
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite number")
    return result


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a non-empty string")
    return value.strip()


def _canonical_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


class MetadataReader:
    """Read Prime metadata either from the CLI or explicit fixture snapshots."""

    def __init__(
        self,
        fixture_dir: Path | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.fixture_dir = fixture_dir
        self.runner = runner

    @property
    def provenance(self) -> str:
        return "synthetic_fixture" if self.fixture_dir else "captured"

    def get(self, name: str, command: Sequence[str]) -> dict[str, Any]:
        if self.fixture_dir:
            return _json(self.fixture_dir / f"{name}.json")
        completed = self.runner(
            list(command), capture_output=True, text=True, check=False
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(f"read-only metadata command failed: {' '.join(command)}: {detail}")
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"read-only metadata command returned invalid JSON: {' '.join(command)}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"metadata command returned a non-object: {' '.join(command)}")
        return value


def _find_exact(items: Any, key: str, expected: str, label: str) -> dict[str, Any]:
    if not isinstance(items, list):
        raise ValueError(f"{label}: expected a list")
    matches = [item for item in items if isinstance(item, dict) and item.get(key) == expected]
    if len(matches) != 1:
        raise ValueError(f"{label}: expected exactly one {key}={expected!r}, found {len(matches)}")
    return matches[0]


def _pricing(models_payload: dict[str, Any], base_model: str) -> dict[str, float]:
    model = _find_exact(models_payload.get("models"), "name", base_model, "training models")
    keys = (
        "effective_training_price_per_mtok",
        "effective_inference_input_price_per_mtok",
        "effective_inference_output_price_per_mtok",
    )
    pricing: dict[str, float] = {}
    for key in keys:
        raw = model.get(key)
        if raw is None:
            fallback = key.removeprefix("effective_")
            raw = model.get(fallback)
        pricing[key] = _finite_number(raw, f"pricing.{key}")
        if pricing[key] < 0:
            raise ValueError(f"pricing.{key}: must not be negative")
    return pricing


def _wallet_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "balance_usd": _finite_number(payload.get("balance_usd"), "wallet.balance_usd"),
        "currency": _require_string(payload.get("currency"), "wallet.currency"),
        "total_billings": int(_finite_number(payload.get("total_billings"), "wallet.total_billings")),
    }


def _eval_toml(model_id: str) -> str:
    return f'''model = {json.dumps(model_id)}
provider = "prime"
api_client_type = "openai_chat_completions"

env_args = {{ tier = "T1", dataset_split = "dev", max_examples = 15, max_turns = 20 }}
num_examples = 15
rollouts_per_example = 2
max_concurrent = 4
max_retries = 0
max_tokens = 1024
temperature = 0
state_columns = ["sim_state", "sim_log"]
save_results = true
disable_tui = true
verbose = true

[[eval]]
id = "uav-operator"
'''


def prepare(
    *,
    run_id: str,
    base_model: str,
    step: int,
    output_dir: Path,
    adapter_id: str | None = None,
    checkpoint_id: str | None = None,
    fixture_metadata_dir: Path | None = None,
    reader: MetadataReader | None = None,
) -> dict[str, Any]:
    """Validate provenance and write a manual evaluation bundle."""
    run_id = _require_string(run_id, "run_id")
    base_model = _require_string(base_model, "base_model")
    if step < 0:
        raise ValueError("step must be non-negative")
    if (adapter_id is None) == (checkpoint_id is None):
        raise ValueError("provide exactly one of adapter_id or checkpoint_id")
    metadata = reader or MetadataReader(fixture_metadata_dir)

    run_payload = metadata.get(
        "run", ("prime", "--plain", "train", "get", run_id, "--output", "json")
    )
    run = run_payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("run metadata is missing .run")
    if run.get("id") != run_id:
        raise ValueError("run metadata does not match requested run_id")
    if run.get("base_model") != base_model:
        raise ValueError("run metadata does not match requested base_model")

    pricing_payload = metadata.get(
        "models", ("prime", "--plain", "train", "models", "--output", "json")
    )
    pricing = _pricing(pricing_payload, base_model)
    wallet = _wallet_snapshot(
        metadata.get(
            "wallet", ("prime", "--plain", "wallet", "--limit", "20", "--output", "json")
        )
    )

    common: dict[str, Any] = {
        "schema_version": 1,
        "kind": "uav_operator_training_evidence_manifest",
        "provenance": metadata.provenance,
        "status": "ready_for_manual_eval" if adapter_id else "awaiting_adapter_deployment",
        "training_run_id": run_id,
        "base_model": base_model,
        "step": step,
        "workload": {
            "tier": "T1",
            "dataset_split": "dev",
            "num_examples": NUM_EXAMPLES,
            "rollouts_per_example": ROLLOUTS_PER_EXAMPLE,
            "expected_rows": NUM_EXAMPLES * ROLLOUTS_PER_EXAMPLE,
            "expected_dev_seeds": list(DEV_SEEDS),
            "excluded_final_eval_seeds": list(FINAL_EVAL_SEEDS),
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "max_turns": MAX_TURNS,
            "max_concurrent": MAX_CONCURRENT,
            "max_retries": 0,
            "state_columns": ["sim_state", "sim_log"],
        },
        "pricing_usd_per_mtok": pricing,
        "wallet_snapshot": wallet,
        "token_ceiling": {
            "kind": "configured_max_output_tokens_across_model_turns",
            "tokens": NUM_EXAMPLES * ROLLOUTS_PER_EXAMPLE * MAX_TURNS * MAX_TOKENS,
            "input_tokens_bounded": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if checkpoint_id:
        checkpoint_id = _require_string(checkpoint_id, "checkpoint_id")
        checkpoints_payload = metadata.get(
            "checkpoints",
            ("prime", "--plain", "train", "checkpoints", run_id, "--output", "json"),
        )
        checkpoint = _find_exact(
            checkpoints_payload.get("checkpoints"), "id", checkpoint_id, "checkpoints"
        )
        if checkpoint.get("rft_run_id") != run_id or checkpoint.get("step") != step:
            raise ValueError("checkpoint provenance does not match requested run and step")
        if checkpoint.get("status") != "READY":
            raise ValueError("checkpoint must have status READY")
        common["checkpoint_id"] = checkpoint_id
        common["adapter_id"] = None
        common["model_id"] = None
        common["config_path"] = None
        common["config_sha256"] = None
        common["manual_deployment_command"] = (
            f"prime deployments create --checkpoint-id {checkpoint_id}"
        )
        common["manual_eval_command"] = None
        _write_json(manifest_path, common)
        return common

    adapter_id = _require_string(adapter_id, "adapter_id")
    adapters_payload = metadata.get(
        "adapters",
        ("prime", "--plain", "deployments", "list", "--num", "100", "--output", "json"),
    )
    adapter = _find_exact(adapters_payload.get("models"), "id", adapter_id, "adapters")
    adapter_run_id = adapter.get("rft_run_id", adapter.get("rftRunId"))
    adapter_base = adapter.get("base_model", adapter.get("baseModel"))
    if adapter_run_id != run_id or adapter_base != base_model or adapter.get("step") != step:
        raise ValueError("adapter provenance does not match requested run, base model, and step")
    if adapter.get("status") != "READY":
        raise ValueError("adapter must have status READY")
    deployment_status = adapter.get("deployment_status", adapter.get("deploymentStatus"))
    if deployment_status != "DEPLOYED":
        raise ValueError("adapter must have deployment status DEPLOYED")

    model_id = f"{base_model}:{adapter_id}"
    config = _eval_toml(model_id)
    config_path = output_dir / "eval.toml"
    config_path.write_text(config)
    common["checkpoint_id"] = adapter.get("checkpoint_id", adapter.get("checkpointId"))
    common["adapter_id"] = adapter_id
    common["model_id"] = model_id
    common["config_path"] = str(config_path.resolve())
    common["config_sha256"] = _canonical_hash(config)
    common["manual_deployment_command"] = None
    common["manual_eval_command"] = (
        f"prime --plain eval run {config_path.resolve()} --output-dir {output_dir.resolve()} --skip-upload"
    )
    _write_json(manifest_path, common)
    return common


def _validate_manifest(manifest: dict[str, Any], step: int, mode: str) -> None:
    if manifest.get("kind") != "uav_operator_training_evidence_manifest":
        raise ValueError(f"step {step}: invalid or missing training-evidence manifest")
    if manifest.get("step") != step:
        raise ValueError(f"step {step}: manifest step mismatch")
    if manifest.get("status") != "ready_for_manual_eval":
        raise ValueError(f"step {step}: manifest is not ready for evaluation")
    provenance = manifest.get("provenance")
    if provenance not in PROVENANCE_VALUES:
        raise ValueError(f"step {step}: invalid manifest provenance")
    if mode == "captured" and provenance != "captured":
        raise ValueError("captured mode rejects fixture provenance")
    workload = manifest.get("workload")
    expected = {
        "tier": "T1",
        "dataset_split": "dev",
        "num_examples": NUM_EXAMPLES,
        "rollouts_per_example": ROLLOUTS_PER_EXAMPLE,
        "expected_rows": NUM_EXAMPLES * ROLLOUTS_PER_EXAMPLE,
        "expected_dev_seeds": list(DEV_SEEDS),
        "excluded_final_eval_seeds": list(FINAL_EVAL_SEEDS),
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "max_turns": MAX_TURNS,
        "max_concurrent": MAX_CONCURRENT,
        "max_retries": 0,
        "state_columns": ["sim_state", "sim_log"],
    }
    if workload != expected:
        raise ValueError(f"step {step}: manifest workload is not the frozen T1 dev workload")
    if manifest.get("model_id") != f"{manifest.get('base_model')}:{manifest.get('adapter_id')}":
        raise ValueError(f"step {step}: manifest composite model ID mismatch")


def _validate_metadata(metadata: dict[str, Any], manifest: dict[str, Any], run_dir: Path) -> None:
    expected = manifest["workload"]
    args = metadata.get("env_args")
    if not isinstance(args, dict):
        raise ValueError(f"{run_dir}: metadata.env_args is missing")
    fields = {
        "env_id": "uav-operator",
        "model": manifest["model_id"],
        "num_examples": expected["num_examples"],
        "rollouts_per_example": expected["rollouts_per_example"],
    }
    for key, value in fields.items():
        if metadata.get(key) != value:
            raise ValueError(f"{run_dir}: metadata {key} mismatch")
    for key in ("tier", "dataset_split", "max_examples", "max_turns"):
        expected_value = expected["num_examples"] if key == "max_examples" else expected[key]
        if args.get(key) != expected_value:
            raise ValueError(f"{run_dir}: metadata env_args.{key} mismatch")
    sampling = metadata.get("sampling_args")
    if not isinstance(sampling, dict):
        raise ValueError(f"{run_dir}: metadata.sampling_args is missing")
    if sampling.get("temperature") != 0 or sampling.get("max_tokens") != MAX_TOKENS:
        raise ValueError(f"{run_dir}: sampling settings mismatch")
    if metadata.get("state_columns") != ["sim_state", "sim_log"]:
        raise ValueError(f"{run_dir}: saved state columns mismatch")


def _row_tokens(row: dict[str, Any], label: str) -> tuple[float, float]:
    usage = row.get("token_usage")
    if not isinstance(usage, dict):
        raise ValueError(f"{label}: token_usage is missing")
    result = (
        _finite_number(usage.get("input_tokens"), f"{label}.input_tokens"),
        _finite_number(usage.get("output_tokens"), f"{label}.output_tokens"),
    )
    if any(value < 0 for value in result):
        raise ValueError(f"{label}: token counts must not be negative")
    return result


def _validate_log(row: dict[str, Any], seed: int, label: str) -> list[dict[str, Any]]:
    log = row.get("sim_log")
    if not isinstance(log, list) or not log or not all(isinstance(item, dict) for item in log):
        raise ValueError(f"{label}: sim_log must be a non-empty list of objects")
    last_time = -math.inf
    for index, snapshot in enumerate(log):
        missing = sorted(SIM_LOG_KEYS - snapshot.keys())
        if missing:
            raise ValueError(f"{label}: malformed sim_log[{index}], missing {missing}")
        if snapshot["seed"] != seed or snapshot["tier"] != "T1":
            raise ValueError(f"{label}: sim_log seed/tier mismatch")
        aircraft = snapshot["aircraft"]
        mission = snapshot["mission"]
        target = mission.get("target") if isinstance(mission, dict) else None
        if not isinstance(aircraft, dict) or not AIRCRAFT_KEYS <= aircraft.keys():
            raise ValueError(f"{label}: malformed sim_log[{index}].aircraft")
        if not isinstance(mission, dict) or not MISSION_KEYS <= mission.keys():
            raise ValueError(f"{label}: malformed sim_log[{index}].mission")
        if not isinstance(target, dict) or not {"lat", "lon", "name"} <= target.keys():
            raise ValueError(f"{label}: malformed sim_log[{index}].mission.target")
        now = _finite_number(snapshot["sim_time_s"], f"{label}.sim_log[{index}].sim_time_s")
        _finite_number(aircraft["lat"], f"{label}.sim_log[{index}].aircraft.lat")
        _finite_number(aircraft["lon"], f"{label}.sim_log[{index}].aircraft.lon")
        _finite_number(snapshot["battery_pct"], f"{label}.sim_log[{index}].battery_pct")
        if now < last_time:
            raise ValueError(f"{label}: sim_log time must be nondecreasing")
        last_time = now
    if log[0]["event"] != "briefing":
        raise ValueError(f"{label}: sim_log must start with a briefing snapshot")
    return log


def _preparation_dir(run_dir: Path) -> Path:
    """Find the prepare bundle above a vf-eval's generated nested run path."""
    candidates = []
    current = run_dir
    for _ in range(5):
        if (current / "manifest.json").is_file() and (current / "eval.toml").is_file():
            candidates.append(current)
        if current == current.parent:
            break
        current = current.parent
    if len(candidates) != 1:
        raise ValueError(
            f"{run_dir}: expected exactly one preparation manifest/config at or above run directory"
        )
    return candidates[0]


def _summarize_one(step: int, run_dir: Path, mode: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    preparation_dir = _preparation_dir(run_dir)
    manifest = _json(preparation_dir / "manifest.json")
    _validate_manifest(manifest, step, mode)
    config_path = preparation_dir / "eval.toml"
    try:
        config_text = config_path.read_text()
    except OSError as exc:
        raise ValueError(f"{run_dir}: matching eval.toml is missing") from exc
    if manifest.get("config_sha256") != _canonical_hash(config_text):
        raise ValueError(f"{run_dir}: eval config hash does not match manifest")
    metadata = _json(run_dir / "metadata.json")
    _validate_metadata(metadata, manifest, run_dir)
    rows = _jsonl(run_dir / "results.jsonl")
    if len(rows) != NUM_EXAMPLES * ROLLOUTS_PER_EXAMPLE:
        raise ValueError(f"{run_dir}: expected 30 rollout rows, found {len(rows)}")

    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    rewards: list[float] = []
    turns: list[float] = []
    completed = hard_safety = truncated = 0
    input_tokens = output_tokens = 0.0
    for index, row in enumerate(rows):
        label = f"{run_dir}/results.jsonl row {index}"
        if row.get("error") is not None:
            raise ValueError(f"{label}: provider error present")
        info = row.get("info")
        if not isinstance(info, dict):
            raise ValueError(f"{label}: info is missing")
        seed = info.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{label}: seed must be an integer")
        if info.get("tier") != "T1":
            raise ValueError(f"{label}: wrong tier")
        if seed in FINAL_EVAL_SEEDS or seed not in DEV_SEEDS:
            raise ValueError(f"{label}: train/final-eval seed leakage or unexpected seed {seed}")
        sim_state = row.get("sim_state")
        if not isinstance(sim_state, dict) or not sim_state:
            raise ValueError(f"{label}: sim_state must be a non-empty object")
        if sim_state.get("seed") != seed or sim_state.get("tier") != "T1":
            raise ValueError(f"{label}: sim_state seed/tier mismatch")
        log = _validate_log(row, seed, label)
        reward = _finite_number(row.get("reward"), f"{label}.reward")
        num_turns = _finite_number(row.get("num_turns"), f"{label}.num_turns")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{label}: metrics must be an object")
        for key, value in metrics.items():
            _finite_number(value, f"{label}.metrics.{key}")
        in_tokens, out_tokens = _row_tokens(row, label)
        input_tokens += in_tokens
        output_tokens += out_tokens
        rewards.append(reward)
        turns.append(num_turns)
        terminal = log[-1]
        completed += terminal.get("mission", {}).get("status") == "completed"
        violations = terminal.get("hard_safety_violations")
        if violations is None:
            violations = row.get("sim_state", {}).get("hard_safety_violations")
        if not isinstance(violations, list):
            raise ValueError(f"{label}: hard_safety_violations is missing")
        hard_safety += len(violations)
        is_truncated = row.get("stop_condition") == "max_turns_reached" or row.get("is_truncated") is True
        if is_truncated and row.get("stop_condition") != "max_turns_reached":
            raise ValueError(f"{label}: unsupported truncation outcome")
        if is_truncated and num_turns != MAX_TURNS:
            raise ValueError(f"{label}: max-turn truncation must occur at {MAX_TURNS} turns")
        truncated += is_truncated
        grouped[seed].append((index, row))

    counts = Counter({seed: len(values) for seed, values in grouped.items()})
    if set(counts) != set(DEV_SEEDS) or any(count != ROLLOUTS_PER_EXAMPLE for count in counts.values()):
        raise ValueError(f"{run_dir}: missing or duplicate per-seed rollouts")
    standard_error = statistics.stdev(rewards) / math.sqrt(len(rewards))
    mean = statistics.fmean(rewards)
    seed_results = {
        seed: {
            "mean_reward": statistics.fmean(float(row["reward"]) for _, row in values),
            "source_row_indices": [index for index, _ in values],
        }
        for seed, values in sorted(grouped.items())
    }
    summary = {
        "step": step,
        "n": len(rows),
        "reward_mean": mean,
        "reward_ci95_low": mean - 1.96 * standard_error,
        "reward_ci95_high": mean + 1.96 * standard_error,
        "mission_completion_count": completed,
        "completion_rate": completed / len(rows),
        "hard_safety_count": hard_safety,
        "max_turn_truncation_count": truncated,
        "truncation_rate": truncated / len(rows),
        "mean_turns": statistics.fmean(turns),
        "token_use": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "model_id": manifest["model_id"],
        "training_run_id": manifest["training_run_id"],
        "checkpoint_id": manifest.get("checkpoint_id"),
        "adapter_id": manifest["adapter_id"],
        "eval_run_id": metadata.get("run_id", run_dir.name),
        "source_run_dir": str(run_dir.resolve()),
        "source_results": str((run_dir / "results.jsonl").resolve()),
    }
    return summary, seed_results


def summarize(
    *,
    milestones: Iterable[tuple[int, Path]],
    expected_steps: Sequence[int],
    output_json: Path,
    output_plot: Path,
    mode: str,
) -> dict[str, Any]:
    """Validate milestone runs and emit curve and same-seed evidence."""
    if mode not in PROVENANCE_VALUES:
        raise ValueError(f"unsupported mode {mode!r}")
    expected = list(expected_steps)
    if not expected or len(expected) != len(set(expected)) or any(step < 0 for step in expected):
        raise ValueError("expected steps must be a non-empty unique list of non-negative integers")
    if 0 not in expected:
        raise ValueError("expected steps must include baseline step 0")
    supplied: dict[int, Path] = {}
    for step, path in milestones:
        if step in supplied:
            raise ValueError(f"duplicate milestone step {step}")
        supplied[step] = path
    if set(supplied) != set(expected):
        missing = sorted(set(expected) - set(supplied))
        extra = sorted(set(supplied) - set(expected))
        raise ValueError(f"milestone steps mismatch; missing={missing}, extra={extra}")

    summaries: list[dict[str, Any]] = []
    seed_by_step: dict[int, dict[int, dict[str, Any]]] = {}
    provenances: set[str] = set()
    run_ids: set[str] = set()
    base_models: set[str] = set()
    for step in sorted(expected):
        manifest = _json(_preparation_dir(supplied[step]) / "manifest.json")
        provenances.add(str(manifest.get("provenance")))
        run_ids.add(str(manifest.get("training_run_id")))
        base_models.add(str(manifest.get("base_model")))
        summary, seed_results = _summarize_one(step, supplied[step], mode)
        summaries.append(summary)
        seed_by_step[step] = seed_results
    if len(provenances) != 1 or len(run_ids) != 1 or len(base_models) != 1:
        raise ValueError("milestone manifests do not share provenance, training run, and base model")
    provenance = provenances.pop()
    if mode == "synthetic_fixture" and provenance != "synthetic_fixture":
        raise ValueError("synthetic fixture mode requires fixture provenance")

    final_step = max(expected)
    comparisons = []
    for seed in DEV_SEEDS:
        before = seed_by_step[0][seed]
        after = seed_by_step[final_step][seed]
        comparisons.append(
            {
                "seed": seed,
                "baseline_mean_reward": before["mean_reward"],
                "final_mean_reward": after["mean_reward"],
                "reward_delta": after["mean_reward"] - before["mean_reward"],
                "baseline_source_row_indices": before["source_row_indices"],
                "final_source_row_indices": after["source_row_indices"],
                "baseline_source_results": summaries[0]["source_results"],
                "final_source_results": summaries[-1]["source_results"],
            }
        )
    comparisons.sort(key=lambda item: (-item["reward_delta"], item["seed"]))
    payload = {
        "schema_version": 1,
        "kind": "uav_operator_training_evidence_summary",
        "provenance": provenance,
        "watermark": FIXTURE_WATERMARK if provenance == "synthetic_fixture" else None,
        "interval": "95% normal confidence interval of rollout rewards",
        "expected_steps": sorted(expected),
        "milestones": summaries,
        "same_seed_baseline_to_final": {
            "baseline_step": 0,
            "final_step": final_step,
            "ranking": comparisons,
        },
    }
    _write_json(output_json, payload)
    _plot(payload, output_plot)
    return payload


def select_checkpoints(
    *,
    checkpoints: Iterable[tuple[int, Path]],
    output_json: Path,
    output_plot: Path,
    mode: str = "captured",
) -> dict[str, Any]:
    """Validate two checkpoint-only evals and select one by frozen tie-breaks.

    This intentionally does not accept step 0: checkpoint selection must use
    the captured dev workload, not a training or baseline aggregate.
    """
    if mode not in PROVENANCE_VALUES:
        raise ValueError(f"unsupported mode {mode!r}")
    supplied = list(checkpoints)
    if len(supplied) != 2 or len({step for step, _ in supplied}) != 2:
        raise ValueError("checkpoint selection requires exactly two distinct steps")
    summaries: list[dict[str, Any]] = []
    seed_results: dict[int, dict[int, dict[str, Any]]] = {}
    identities: set[tuple[Any, ...]] = set()
    provenances: set[str] = set()
    for step, path in supplied:
        summary, per_seed = _summarize_one(step, path, mode)
        manifest = _json(_preparation_dir(path) / "manifest.json")
        if not manifest.get("checkpoint_id") or not manifest.get("adapter_id"):
            raise ValueError(f"step {step}: checkpoint and adapter identity are required")
        provenances.add(str(manifest.get("provenance")))
        identities.add((manifest.get("training_run_id"), manifest.get("base_model")))
        summaries.append(summary)
        seed_results[step] = per_seed
    if len(identities) != 1 or len(provenances) != 1:
        raise ValueError("checkpoint manifests do not share training run and base model")
    if mode == "captured" and provenances != {"captured"}:
        raise ValueError("captured mode rejects fixture provenance")
    if mode == "synthetic_fixture" and provenances != {"synthetic_fixture"}:
        raise ValueError("synthetic fixture mode requires fixture provenance")
    summaries.sort(key=lambda item: item["step"])
    ranked = sorted(
        summaries,
        key=lambda item: (
            item["hard_safety_count"],
            -item["reward_mean"],
            -item["mission_completion_count"],
            item["max_turn_truncation_count"],
        ),
    )
    winner = ranked[0]
    loser = ranked[1]
    winner_key = (
        winner["hard_safety_count"], -winner["reward_mean"],
        -winner["mission_completion_count"], winner["max_turn_truncation_count"],
    )
    loser_key = (
        loser["hard_safety_count"], -loser["reward_mean"],
        -loser["mission_completion_count"], loser["max_turn_truncation_count"],
    )
    paired = []
    for seed in DEV_SEEDS:
        left, right = (seed_results[s][seed] for s in (summaries[0]["step"], summaries[1]["step"]))
        paired.append({"seed": seed, "step_a": summaries[0]["step"], "step_b": summaries[1]["step"],
                       "step_a_mean_reward": left["mean_reward"], "step_b_mean_reward": right["mean_reward"],
                       "reward_delta": right["mean_reward"] - left["mean_reward"],
                       "step_a_source_row_indices": left["source_row_indices"],
                       "step_b_source_row_indices": right["source_row_indices"]})
    payload = {
        "schema_version": 1, "kind": "uav_operator_checkpoint_selection",
        "provenance": provenances.pop(),
        "watermark": FIXTURE_WATERMARK if mode == "synthetic_fixture" else None,
        "selection_rule": "fewer hard-safety violations; higher mean reward; more completions; fewer truncations",
        "checkpoints": summaries,
        "paired_per_seed": paired,
        "winner": {"step": winner["step"], "checkpoint_id": winner["checkpoint_id"], "adapter_id": winner["adapter_id"],
                    "metrics": winner, "selection_key": winner_key,
                    "rationale": "lexicographic frozen safety/reward/completion/truncation rule"},
        "loser_selection_key": loser_key,
    }
    _write_json(output_json, payload)
    _plot_selection(payload, output_plot)
    return payload


def _plot_selection(payload: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    items = payload["checkpoints"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([str(item["step"]) for item in items], [item["reward_mean"] for item in items],
           yerr=[[item["reward_mean"] - item["reward_ci95_low"] for item in items],
                 [item["reward_ci95_high"] - item["reward_mean"] for item in items]], capsize=5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean T1 dev reward (95% CI)")
    ax.set_title("uav-operator checkpoint selection")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot(payload: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    milestones = payload["milestones"]
    steps = [item["step"] for item in milestones]
    means = [item["reward_mean"] for item in milestones]
    lower = [mean - item["reward_ci95_low"] for mean, item in zip(means, milestones)]
    upper = [item["reward_ci95_high"] - mean for mean, item in zip(means, milestones)]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.errorbar(steps, means, yerr=[lower, upper], marker="o", capsize=5)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Mean T1 dev reward (95% CI)")
    axis.set_title("uav-operator checkpoint evaluation")
    axis.grid(axis="y", alpha=0.25)
    if payload["provenance"] == "synthetic_fixture":
        axis.text(
            0.5,
            0.5,
            FIXTURE_WATERMARK,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=17,
            color="crimson",
            alpha=0.55,
            rotation=20,
            weight="bold",
        )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, metadata={"Description": payload.get("watermark") or "captured evidence"})
    plt.close(figure)


def _milestone(value: str) -> tuple[int, Path]:
    try:
        step_text, path_text = value.split("=", 1)
        return int(step_text), Path(path_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("milestone must be STEP=RUN_DIR") from exc


def _steps(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected steps must be comma-separated integers") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--base-model", required=True)
    prepare_parser.add_argument("--step", required=True, type=int)
    identity = prepare_parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--adapter-id")
    identity.add_argument("--checkpoint-id")
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    prepare_parser.add_argument(
        "--fixture-metadata-dir",
        type=Path,
        help="Use explicit synthetic Prime metadata snapshots; marks all output as fixture.",
    )

    summarize_parser = commands.add_parser("summarize")
    summarize_parser.add_argument("milestones", nargs="+", type=_milestone)
    summarize_parser.add_argument("--expected-steps", required=True, type=_steps)
    summarize_parser.add_argument("--output-json", required=True, type=Path)
    summarize_parser.add_argument("--output-plot", required=True, type=Path)
    summarize_parser.add_argument("--mode", choices=PROVENANCE_VALUES, default="captured")
    select_parser = commands.add_parser("select-checkpoints")
    select_parser.add_argument("checkpoints", nargs=2, type=_milestone)
    select_parser.add_argument("--output-json", required=True, type=Path)
    select_parser.add_argument("--output-plot", required=True, type=Path)
    select_parser.add_argument("--mode", choices=PROVENANCE_VALUES, default="captured")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "prepare":
            payload = prepare(
                run_id=args.run_id,
                base_model=args.base_model,
                step=args.step,
                output_dir=args.output_dir,
                adapter_id=args.adapter_id,
                checkpoint_id=args.checkpoint_id,
                fixture_metadata_dir=args.fixture_metadata_dir,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.command == "summarize":
            payload = summarize(
                milestones=args.milestones,
                expected_steps=args.expected_steps,
                output_json=args.output_json,
                output_plot=args.output_plot,
                mode=args.mode,
            )
        else:
            payload = select_checkpoints(
                checkpoints=args.checkpoints,
                output_json=args.output_json,
                output_plot=args.output_plot,
                mode=args.mode,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
