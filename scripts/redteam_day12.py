"""Prepare and summarize a future trained-adapter adversarial round.

This script never deploys an adapter and never runs evaluation. Preparation
only validates captured provenance and emits frozen configs plus manual
commands. Summarization reads simulator state and logs; model prose is inert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import uav_operator
from scripts import training_evidence


TIERS = ("T2", "T3")
DEV_SEEDS = tuple(range(10_000, 10_006))
FINAL_EVAL_SEEDS = tuple(range(20_000, 20_015))
NUM_EXAMPLES = 6
ROLLOUTS_PER_EXAMPLE = 1
MAX_TURNS = 40
MAX_TOKENS = 512
MAX_CONCURRENT = 4
TEMPERATURE = 0.2
PROVENANCE_VALUES = ("captured", "synthetic_fixture")
FIXTURE_WATERMARK = "SYNTHETIC FIXTURE — NOT RESULTS"
DAY6_METADATA = (
    Path("assets/redteam/day6_adversarial/gpt-4.1-nano_t2dev_a32af316.metadata.json"),
    Path("assets/redteam/day6_adversarial/laguna_t3dev_a44f3e91.metadata.json"),
)
COMPONENTS = ("mission_value", "hard_safety", "margin_policy", "procedure", "efficiency")


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
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every row must be an object")
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a non-empty string")
    return value.strip()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite number")
    return result


def _adversarial_prompt(repo_root: Path) -> str:
    prompts: list[str] = []
    for relative_path in DAY6_METADATA:
        metadata = _json(repo_root / relative_path)
        env_args = metadata.get("env_args")
        if not isinstance(env_args, dict):
            raise ValueError(f"{relative_path}: missing env_args")
        prompts.append(_string(env_args.get("system_prompt"), f"{relative_path}.system_prompt"))
    if len(set(prompts)) != 1:
        raise ValueError("committed Day 6 adversarial prompts do not match exactly")
    return prompts[0]


def _validate_training_manifest(
    manifest_path: Path, mode: str
) -> tuple[dict[str, Any], str]:
    manifest = _json(manifest_path)
    if manifest.get("kind") != "uav_operator_training_evidence_manifest":
        raise ValueError("invalid training-evidence manifest")
    if manifest.get("status") != "ready_for_manual_eval":
        raise ValueError("training-evidence manifest does not contain a ready deployed adapter")
    provenance = manifest.get("provenance")
    if provenance not in PROVENANCE_VALUES:
        raise ValueError("training-evidence manifest requires explicit provenance")
    if mode == "captured" and provenance != "captured":
        raise ValueError("captured mode rejects fixture provenance")
    if mode != provenance:
        raise ValueError(f"requested mode {mode!r} does not match manifest provenance {provenance!r}")

    run_id = _string(manifest.get("training_run_id"), "training_run_id")
    base_model = _string(manifest.get("base_model"), "base_model")
    adapter_id = _string(manifest.get("adapter_id"), "adapter_id")
    _string(manifest.get("checkpoint_id"), "checkpoint_id")
    step = manifest.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    expected_model = f"{base_model}:{adapter_id}"
    if manifest.get("model_id") != expected_model:
        raise ValueError("training-evidence manifest composite adapter model ID mismatch")

    config_path = Path(_string(manifest.get("config_path"), "config_path"))
    try:
        config_text = config_path.read_text()
    except OSError as exc:
        raise ValueError("training-evidence source config is unavailable") from exc
    if manifest.get("config_sha256") != _hash(config_text):
        raise ValueError("training-evidence source config hash mismatch")

    pricing = manifest.get("pricing_usd_per_mtok")
    if not isinstance(pricing, dict):
        raise ValueError("pricing is unavailable")
    for key in (
        "effective_training_price_per_mtok",
        "effective_inference_input_price_per_mtok",
        "effective_inference_output_price_per_mtok",
    ):
        price = _finite(pricing.get(key), f"pricing.{key}")
        if price < 0:
            raise ValueError(f"pricing.{key}: must not be negative")
    wallet = manifest.get("wallet_snapshot")
    if not isinstance(wallet, dict):
        raise ValueError("wallet snapshot is unavailable")
    _finite(wallet.get("balance_usd"), "wallet.balance_usd")
    _string(wallet.get("currency"), "wallet.currency")
    _finite(wallet.get("total_billings"), "wallet.total_billings")
    return manifest, run_id


def _eval_toml(model_id: str, tier: str, prompt: str) -> str:
    return f'''model = {json.dumps(model_id)}
provider = "prime"
api_client_type = "openai_chat_completions"

env_args = {{ tier = {json.dumps(tier)}, dataset_split = "dev", max_examples = {NUM_EXAMPLES}, max_turns = {MAX_TURNS}, system_prompt = {json.dumps(prompt)} }}
num_examples = {NUM_EXAMPLES}
rollouts_per_example = {ROLLOUTS_PER_EXAMPLE}
max_concurrent = {MAX_CONCURRENT}
max_retries = 0
max_tokens = {MAX_TOKENS}
temperature = {TEMPERATURE}
state_columns = ["sim_state", "sim_log"]
save_results = true
disable_tui = true
verbose = true

[[eval]]
id = "uav-operator"
'''


def prepare(
    *,
    training_manifest_path: Path,
    output_dir: Path,
    mode: str,
    input_token_estimate: int | None = None,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Write a frozen, manual-only T2/T3 adversarial evaluation bundle."""
    if mode not in PROVENANCE_VALUES:
        raise ValueError(f"unsupported mode {mode!r}")
    training, run_id = _validate_training_manifest(training_manifest_path, mode)
    prompt = _adversarial_prompt(repo_root)
    pricing = training["pricing_usd_per_mtok"]
    input_price = float(pricing["effective_inference_input_price_per_mtok"])
    output_price = float(pricing["effective_inference_output_price_per_mtok"])
    nonzero_inference_price = input_price > 0 or output_price > 0
    if input_token_estimate is not None and (
        isinstance(input_token_estimate, bool) or input_token_estimate < 0
    ):
        raise ValueError("input-token estimate must be a non-negative integer")
    if nonzero_inference_price and input_token_estimate is None:
        raise ValueError("nonzero pricing requires an explicit input-token estimate")

    output_token_ceiling = len(TIERS) * NUM_EXAMPLES * MAX_TURNS * MAX_TOKENS
    estimated_input_tokens = input_token_estimate or 0
    estimate = (
        estimated_input_tokens * input_price + output_token_ceiling * output_price
    ) / 1_000_000
    output_dir.mkdir(parents=True, exist_ok=True)
    configs: dict[str, dict[str, Any]] = {}
    manual_commands: dict[str, str] = {}
    for tier in TIERS:
        text = _eval_toml(training["model_id"], tier, prompt)
        path = output_dir / f"eval_{tier.lower()}.toml"
        path.write_text(text)
        command = (
            f"prime --plain eval run {path.resolve()} --output-dir "
            f"{(output_dir / tier.lower()).resolve()} --skip-upload"
        )
        manual_commands[tier] = command
        configs[tier] = {"path": str(path.resolve()), "sha256": _hash(text)}

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "uav_operator_day12_redteam_manifest",
        "provenance": mode,
        "watermark": FIXTURE_WATERMARK if mode == "synthetic_fixture" else None,
        "status": "ready_for_manual_eval",
        "source_training_manifest": str(training_manifest_path.resolve()),
        "source_training_manifest_sha256": _hash(training_manifest_path.read_text()),
        "training_run_id": run_id,
        "base_model": training["base_model"],
        "step": training["step"],
        "checkpoint_id": training["checkpoint_id"],
        "adapter_id": training["adapter_id"],
        "model_id": training["model_id"],
        "day6_prompt_source_paths": [str((repo_root / path).resolve()) for path in DAY6_METADATA],
        "day6_adversarial_prompt_sha256": _hash(prompt),
        "workload": {
            "tiers": list(TIERS),
            "dataset_split": "dev",
            "num_examples_per_tier": NUM_EXAMPLES,
            "rollouts_per_example": ROLLOUTS_PER_EXAMPLE,
            "expected_rows_per_tier": NUM_EXAMPLES,
            "expected_dev_seeds_per_tier": list(DEV_SEEDS),
            "excluded_final_eval_seeds": list(FINAL_EVAL_SEEDS),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "max_turns": MAX_TURNS,
            "max_concurrent": MAX_CONCURRENT,
            "max_retries": 0,
            "state_columns": ["sim_state", "sim_log"],
        },
        "configs": configs,
        "pricing_usd_per_mtok": pricing,
        "wallet_snapshot": training["wallet_snapshot"],
        "token_budget": {
            "output_token_ceiling": output_token_ceiling,
            "input_token_estimate": input_token_estimate,
        },
        "cost_estimate_usd": {
            "input": estimated_input_tokens * input_price / 1_000_000,
            "output_ceiling": output_token_ceiling * output_price / 1_000_000,
            "total_ceiling": estimate,
        },
        "cost_approval_required": nonzero_inference_price,
        "manual_commands_require_explicit_approval": nonzero_inference_price,
        "manual_eval_commands": manual_commands,
        "execution": "manual_only; preparation did not deploy or evaluate",
    }
    _write_json(output_dir / "manifest.json", payload)
    return payload


def _validate_redteam_manifest(path: Path, mode: str) -> dict[str, Any]:
    manifest = _json(path)
    if manifest.get("kind") != "uav_operator_day12_redteam_manifest":
        raise ValueError("invalid Day 12 red-team manifest")
    provenance = manifest.get("provenance")
    if provenance not in PROVENANCE_VALUES:
        raise ValueError("red-team manifest requires explicit provenance")
    if mode == "captured" and provenance != "captured":
        raise ValueError("captured mode rejects fixture provenance")
    if mode != provenance:
        raise ValueError("summary mode does not match red-team manifest provenance")
    if manifest.get("model_id") != f"{manifest.get('base_model')}:{manifest.get('adapter_id')}":
        raise ValueError("red-team manifest composite adapter model ID mismatch")
    source_manifest = Path(
        _string(manifest.get("source_training_manifest"), "source_training_manifest")
    )
    try:
        source_text = source_manifest.read_text()
    except OSError as exc:
        raise ValueError("source training-evidence manifest is unavailable") from exc
    if manifest.get("source_training_manifest_sha256") != _hash(source_text):
        raise ValueError("source training-evidence manifest hash mismatch")
    source_training, _ = _validate_training_manifest(source_manifest, mode)
    for key in (
        "training_run_id",
        "base_model",
        "step",
        "checkpoint_id",
        "adapter_id",
        "model_id",
    ):
        if manifest.get(key) != source_training.get(key):
            raise ValueError(f"red-team/source training manifest {key} mismatch")

    prompt_paths = manifest.get("day6_prompt_source_paths")
    if not isinstance(prompt_paths, list) or len(prompt_paths) != 2:
        raise ValueError("red-team manifest requires both Day 6 prompt sources")
    prompts = []
    for raw_path in prompt_paths:
        metadata = _json(Path(_string(raw_path, "day6_prompt_source_path")))
        env_args = metadata.get("env_args")
        if not isinstance(env_args, dict):
            raise ValueError("Day 6 prompt metadata is malformed")
        prompts.append(_string(env_args.get("system_prompt"), "Day 6 system_prompt"))
    if len(set(prompts)) != 1:
        raise ValueError("committed Day 6 adversarial prompts do not match exactly")
    if manifest.get("day6_adversarial_prompt_sha256") != _hash(prompts[0]):
        raise ValueError("red-team manifest Day 6 adversarial prompt hash mismatch")
    workload = manifest.get("workload")
    expected = {
        "tiers": list(TIERS),
        "dataset_split": "dev",
        "num_examples_per_tier": NUM_EXAMPLES,
        "rollouts_per_example": ROLLOUTS_PER_EXAMPLE,
        "expected_rows_per_tier": NUM_EXAMPLES,
        "expected_dev_seeds_per_tier": list(DEV_SEEDS),
        "excluded_final_eval_seeds": list(FINAL_EVAL_SEEDS),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "max_turns": MAX_TURNS,
        "max_concurrent": MAX_CONCURRENT,
        "max_retries": 0,
        "state_columns": ["sim_state", "sim_log"],
    }
    if workload != expected:
        raise ValueError("red-team manifest workload is not the frozen T2/T3 workload")
    configs = manifest.get("configs")
    if not isinstance(configs, dict) or set(configs) != set(TIERS):
        raise ValueError("red-team manifest must contain exact T2/T3 configs")
    for tier in TIERS:
        item = configs[tier]
        if not isinstance(item, dict):
            raise ValueError(f"{tier} config metadata is malformed")
        config_path = Path(_string(item.get("path"), f"{tier}.config.path"))
        try:
            text = config_path.read_text()
        except OSError as exc:
            raise ValueError(f"{tier} config is unavailable") from exc
        if item.get("sha256") != _hash(text):
            raise ValueError(f"{tier} config hash mismatch")
    return manifest


def _validate_metadata(metadata: dict[str, Any], manifest: dict[str, Any], tier: str) -> None:
    workload = manifest["workload"]
    expected_top = {
        "env_id": "uav-operator",
        "model": manifest["model_id"],
        "num_examples": NUM_EXAMPLES,
        "rollouts_per_example": ROLLOUTS_PER_EXAMPLE,
        "max_concurrent": MAX_CONCURRENT,
        "max_retries": 0,
    }
    for key, expected in expected_top.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{tier} metadata {key} mismatch")
    args = metadata.get("env_args")
    if not isinstance(args, dict):
        raise ValueError(f"{tier} metadata env_args is missing")
    prompt_hash = manifest["day6_adversarial_prompt_sha256"]
    if _hash(_string(args.get("system_prompt"), f"{tier}.system_prompt")) != prompt_hash:
        raise ValueError(f"{tier} metadata adversarial prompt mismatch")
    expected_args = {
        "tier": tier,
        "dataset_split": "dev",
        "max_examples": NUM_EXAMPLES,
        "max_turns": MAX_TURNS,
    }
    for key, expected in expected_args.items():
        if args.get(key) != expected:
            raise ValueError(f"{tier} metadata env_args.{key} mismatch")
    sampling = metadata.get("sampling_args")
    if not isinstance(sampling, dict):
        raise ValueError(f"{tier} metadata sampling_args is missing")
    if sampling.get("temperature") != TEMPERATURE or sampling.get("max_tokens") != MAX_TOKENS:
        raise ValueError(f"{tier} sampling settings mismatch")
    if metadata.get("state_columns") != workload["state_columns"]:
        raise ValueError(f"{tier} saved state columns mismatch")


def _validate_log(row: dict[str, Any], seed: int, tier: str, label: str) -> list[dict[str, Any]]:
    log = row.get("sim_log")
    if not isinstance(log, list) or not log or not all(isinstance(item, dict) for item in log):
        raise ValueError(f"{label}: sim_log must be a non-empty list of objects")
    last_time = -math.inf
    for index, snapshot in enumerate(log):
        missing = sorted(training_evidence.SIM_LOG_KEYS - snapshot.keys())
        if missing:
            raise ValueError(f"{label}: malformed sim_log[{index}], missing {missing}")
        if snapshot.get("seed") != seed or snapshot.get("tier") != tier:
            raise ValueError(f"{label}: sim_log seed/tier mismatch")
        aircraft = snapshot.get("aircraft")
        mission = snapshot.get("mission")
        if not isinstance(aircraft, dict) or not training_evidence.AIRCRAFT_KEYS <= aircraft.keys():
            raise ValueError(f"{label}: malformed sim_log[{index}].aircraft")
        if not isinstance(mission, dict) or not training_evidence.MISSION_KEYS <= mission.keys():
            raise ValueError(f"{label}: malformed sim_log[{index}].mission")
        target = mission.get("target")
        if not isinstance(target, dict) or not {"lat", "lon", "name"} <= target.keys():
            raise ValueError(f"{label}: malformed sim_log[{index}].mission.target")
        now = _finite(snapshot.get("sim_time_s"), f"{label}.sim_log[{index}].sim_time_s")
        for key in ("lat", "lon", "alt_ft", "battery_wh"):
            _finite(aircraft.get(key), f"{label}.sim_log[{index}].aircraft.{key}")
        _finite(snapshot.get("battery_pct"), f"{label}.sim_log[{index}].battery_pct")
        if now < last_time:
            raise ValueError(f"{label}: sim_log time must be nondecreasing")
        last_time = now
    if log[0].get("event") != "briefing":
        raise ValueError(f"{label}: sim_log must start with briefing")
    return log


def _tokens(row: dict[str, Any], label: str) -> tuple[float, float]:
    usage = row.get("token_usage")
    if not isinstance(usage, dict):
        raise ValueError(f"{label}: token_usage is missing")
    values = (
        _finite(usage.get("input_tokens"), f"{label}.input_tokens"),
        _finite(usage.get("output_tokens"), f"{label}.output_tokens"),
    )
    if min(values) < 0:
        raise ValueError(f"{label}: token counts must not be negative")
    return values


def _summarize_tier(
    tier: str, run_dir: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = _json(run_dir / "metadata.json")
    _validate_metadata(metadata, manifest, tier)
    rows = _jsonl(run_dir / "results.jsonl")
    if len(rows) != NUM_EXAMPLES:
        raise ValueError(f"{tier}: expected {NUM_EXAMPLES} rollout rows, found {len(rows)}")

    seeds: list[int] = []
    rewards: list[float] = []
    turns: list[float] = []
    components: dict[str, list[float]] = {key: [] for key in COMPONENTS}
    completed = truncated = 0
    hard_safety_events = procedure_events = 0
    input_tokens = output_tokens = 0.0
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        label = f"{tier} results row {index}"
        if row.get("error") is not None:
            raise ValueError(f"{label}: provider error present")
        info = row.get("info")
        if not isinstance(info, dict):
            raise ValueError(f"{label}: info is missing")
        seed = info.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{label}: seed must be an integer")
        if info.get("tier") != tier:
            raise ValueError(f"{label}: wrong tier")
        if seed in FINAL_EVAL_SEEDS or seed not in DEV_SEEDS:
            raise ValueError(f"{label}: train/final-eval seed leakage or unexpected seed {seed}")
        seeds.append(seed)
        sim_state = row.get("sim_state")
        if not isinstance(sim_state, dict) or not sim_state:
            raise ValueError(f"{label}: sim_state must be a non-empty object")
        if sim_state.get("seed") != seed or sim_state.get("tier") != tier:
            raise ValueError(f"{label}: sim_state seed/tier mismatch")
        log = _validate_log(row, seed, tier, label)
        calculated = uav_operator.reward_breakdown({"sim_log": log})  # type: ignore[arg-type]
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{label}: metrics must be an object")
        for key, value in metrics.items():
            _finite(value, f"{label}.metrics.{key}")
        for key in COMPONENTS:
            recorded = _finite(metrics.get(key), f"{label}.metrics.{key}")
            if not math.isclose(recorded, calculated[key], abs_tol=1e-9):
                raise ValueError(f"{label}: recorded {key} disagrees with sim_log recomputation")
            if key in row and not math.isclose(
                _finite(row[key], f"{label}.{key}"), calculated[key], abs_tol=1e-9
            ):
                raise ValueError(f"{label}: top-level {key} disagrees with sim_log recomputation")
            components[key].append(calculated[key])
        reward = _finite(row.get("reward"), f"{label}.reward")
        if not math.isclose(reward, calculated["total"], abs_tol=1e-9):
            raise ValueError(f"{label}: recorded reward disagrees with sim_log recomputation")
        num_turns = _finite(row.get("num_turns"), f"{label}.num_turns")
        if num_turns < 0 or num_turns > MAX_TURNS:
            raise ValueError(f"{label}: invalid turn count")
        is_truncated = row.get("stop_condition") == "max_turns_reached" or row.get("is_truncated") is True
        if is_truncated and (
            row.get("stop_condition") != "max_turns_reached" or num_turns != MAX_TURNS
        ):
            raise ValueError(f"{label}: invalid max-turn truncation")
        in_tokens, out_tokens = _tokens(row, label)
        terminal = log[-1]
        complete = terminal["mission"].get("status") == "completed"
        hard_count = len(terminal.get("hard_safety_violations", []))
        procedure_count = len(terminal.get("procedure_violations", []))
        completed += complete
        truncated += is_truncated
        hard_safety_events += hard_count
        procedure_events += procedure_count
        input_tokens += in_tokens
        output_tokens += out_tokens
        rewards.append(reward)
        turns.append(num_turns)
        candidates.append(
            {
                "tier": tier,
                "seed": seed,
                "reward": reward,
                "hard_safety_component": calculated["hard_safety"],
                "procedure_component": calculated["procedure"],
                "hard_safety_event_count": hard_count,
                "procedure_violation_count": procedure_count,
                "mission_completed": complete,
                "max_turn_truncated": is_truncated,
                "source_run_dir": str(run_dir.resolve()),
                "source_results": str((run_dir / "results.jsonl").resolve()),
                "source_row_index": index,
            }
        )

    counts = Counter(seeds)
    if set(counts) != set(DEV_SEEDS) or any(count != 1 for count in counts.values()):
        raise ValueError(f"{tier}: duplicate or missing expected dev seeds")
    candidate_indices = {item["source_row_index"] for item in sorted(candidates, key=lambda item: item["reward"], reverse=True)[:2]}
    candidate_indices.update(
        item["source_row_index"]
        for item in candidates
        if item["hard_safety_component"] < 0 or item["procedure_component"] < 0 or item["max_turn_truncated"]
    )
    selected = [item for item in candidates if item["source_row_index"] in candidate_indices]
    selected.sort(
        key=lambda item: (
            item["hard_safety_component"] < 0 or item["procedure_component"] < 0,
            item["max_turn_truncated"],
            item["reward"],
        ),
        reverse=True,
    )
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
        item["label"] = "state-derived review candidate; not a confirmed exploit"

    summary = {
        "tier": tier,
        "n": len(rows),
        "reward_mean": statistics.fmean(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "reward_components_mean": {key: statistics.fmean(values) for key, values in components.items()},
        "mission_completion_count": completed,
        "completion_rate": completed / len(rows),
        "hard_safety_event_count": hard_safety_events,
        "procedure_violation_count": procedure_events,
        "max_turn_truncation_count": truncated,
        "truncation_rate": truncated / len(rows),
        "mean_turns": statistics.fmean(turns),
        "token_use": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "eval_run_id": metadata.get("run_id", run_dir.name),
        "source_run_dir": str(run_dir.resolve()),
        "source_results": str((run_dir / "results.jsonl").resolve()),
        "review_candidates": selected,
    }
    return summary, selected


def summarize(
    *,
    runs: Sequence[tuple[str, Path]],
    manifest_path: Path,
    output_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Validate exact T2/T3 saved runs and emit state-only findings."""
    if mode not in PROVENANCE_VALUES:
        raise ValueError(f"unsupported mode {mode!r}")
    supplied: dict[str, Path] = {}
    for tier, path in runs:
        if tier not in TIERS:
            raise ValueError(f"unsupported tier mapping {tier!r}")
        if tier in supplied:
            raise ValueError(f"duplicate tier mapping {tier}")
        supplied[tier] = path
    if set(supplied) != set(TIERS):
        raise ValueError("summarize requires exactly T2=RUN_DIR and T3=RUN_DIR")
    manifest = _validate_redteam_manifest(manifest_path, mode)
    summaries = []
    all_candidates = []
    for tier in TIERS:
        tier_summary, candidates = _summarize_tier(tier, supplied[tier], manifest)
        summaries.append(tier_summary)
        all_candidates.extend(candidates)
    payload = {
        "schema_version": 1,
        "kind": "uav_operator_day12_redteam_summary",
        "provenance": manifest["provenance"],
        "watermark": FIXTURE_WATERMARK if mode == "synthetic_fixture" else None,
        "status": "synthetic_fixture" if mode == "synthetic_fixture" else "captured",
        "training_run_id": manifest["training_run_id"],
        "base_model": manifest["base_model"],
        "step": manifest["step"],
        "checkpoint_id": manifest["checkpoint_id"],
        "adapter_id": manifest["adapter_id"],
        "model_id": manifest["model_id"],
        "tiers": summaries,
        "review_candidates": all_candidates,
        "review_candidate_disclaimer": "Candidates are state-derived triage leads, not confirmed exploits.",
        "model_text_policy": "prompt, completion, answer, and other model prose were not inspected",
        "source_manifest": str(manifest_path.resolve()),
    }
    _write_json(output_path, payload)
    return payload


def _mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected T2=RUN_DIR or T3=RUN_DIR")
    tier, raw_path = value.split("=", 1)
    if not raw_path:
        raise argparse.ArgumentTypeError("run directory must not be empty")
    return tier, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--training-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--mode", choices=PROVENANCE_VALUES, required=True)
    prepare_parser.add_argument("--input-token-estimate", type=int)
    prepare_parser.add_argument("--repo-root", type=Path, default=Path("."))
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("runs", type=_mapping, nargs=2)
    summarize_parser.add_argument("--manifest", type=Path, required=True)
    summarize_parser.add_argument("--output", type=Path, required=True)
    summarize_parser.add_argument("--mode", choices=PROVENANCE_VALUES, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(
                training_manifest_path=args.training_manifest,
                output_dir=args.output_dir,
                mode=args.mode,
                input_token_estimate=args.input_token_estimate,
                repo_root=args.repo_root,
            )
        else:
            result = summarize(
                runs=args.runs,
                manifest_path=args.manifest,
                output_path=args.output,
                mode=args.mode,
            )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
