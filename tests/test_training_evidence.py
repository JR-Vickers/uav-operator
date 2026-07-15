from __future__ import annotations

import json
import math
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

from scripts import training_evidence as evidence


BASE_MODEL = "example/free-model"
RUN_ID = "training-run-1"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _metadata_fixtures(root: Path, step: int, *, adapter: bool = True) -> Path:
    root.mkdir(parents=True)
    _write_json(
        root / "run.json",
        {"run": {"id": RUN_ID, "base_model": BASE_MODEL, "status": "COMPLETED"}},
    )
    _write_json(
        root / "models.json",
        {
            "models": [
                {
                    "name": BASE_MODEL,
                    "effective_training_price_per_mtok": 0.0,
                    "effective_inference_input_price_per_mtok": 0.25,
                    "effective_inference_output_price_per_mtok": 1.0,
                }
            ]
        },
    )
    _write_json(
        root / "wallet.json",
        {"balance_usd": 50.0, "currency": "USD", "total_billings": 3},
    )
    if adapter:
        _write_json(
            root / "adapters.json",
            {
                "models": [
                    {
                        "id": f"adapter-{step}",
                        "rft_run_id": RUN_ID,
                        "base_model": BASE_MODEL,
                        "step": step,
                        "status": "READY",
                        "deployment_status": "DEPLOYED",
                        "checkpoint_id": f"checkpoint-{step}",
                    }
                ]
            },
        )
    else:
        _write_json(
            root / "checkpoints.json",
            {
                "checkpoints": [
                    {
                        "id": f"checkpoint-{step}",
                        "rft_run_id": RUN_ID,
                        "step": step,
                        "status": "READY",
                    }
                ]
            },
        )
    return root


def _prepare_run(tmp_path: Path, step: int) -> Path:
    fixture_dir = _metadata_fixtures(tmp_path / f"metadata-{step}", step)
    run_dir = tmp_path / f"step-{step}"
    evidence.prepare(
        run_id=RUN_ID,
        base_model=BASE_MODEL,
        step=step,
        adapter_id=f"adapter-{step}",
        output_dir=run_dir,
        fixture_metadata_dir=fixture_dir,
    )
    return run_dir


def _snapshot(seed: int, *, completed: bool, violations: list[str]) -> dict[str, Any]:
    return {
        "event": "briefing",
        "battery_pct": 100.0,
        "mission_distance_to_target_nm": 1.0,
        "seed": seed,
        "scenario_id": f"T1-{seed}",
        "tier": "T1",
        "sim_time_s": 60.0,
        "aircraft": {
            "lat": 37.7,
            "lon": -122.4,
            "alt_ft": 0.0,
            "status": "ground",
            "battery_wh": 400.0,
            "current_site_id": "home",
        },
        "mission": {
            "mission_id": f"T1-{seed}",
            "description": "fixture",
            "launch_site_id": "home",
            "target": {"lat": 37.8, "lon": -122.3, "name": "target"},
            "value": 1.0,
            "sla_min": 30.0,
            "status": "completed" if completed else "pending",
            "completed_time_s": 60.0 if completed else None,
            "failure_reason": None,
        },
        "home_site_id": "home",
        "wind": {},
        "rng_state": {},
        "scenario_par": {},
        "events": [],
        "active_events": [],
        "acknowledged_events": [],
        "closed_sites": [],
        "hard_safety_violations": violations,
        "procedure_violations": [],
        "current_plan": [],
        "lost_link_plan": "return_home",
        "rng_draws": 0,
        "active_failsafe": None,
        "alerts": [],
        "acknowledged_alerts": [],
        "overrides": [],
        "payload_released": False,
        "hold_until_s": None,
        "current_altitude_target_ft": 250.0,
        "current_airspeed_kt": 35.0,
        "ground_no_progress_streak": 0,
        "last_action": "briefing",
        "last_result": {},
        "is_terminal": completed,
        "terminal_reason": "completed" if completed else None,
    }


def _rows(step: int) -> list[dict[str, Any]]:
    rows = []
    for seed_index, seed in enumerate(evidence.DEV_SEEDS):
        for rollout in range(2):
            completed = (seed_index + rollout + step) % 3 != 0
            truncated = seed_index == 0 and rollout == 1
            violations = ["aircraft_loss"] if seed_index == 1 and rollout == 0 else []
            reward = step / 100.0 + seed_index / 1000.0 + rollout / 10000.0
            rows.append(
                {
                    "example_id": seed_index,
                    "info": {"seed": seed, "tier": "T1", "scenario_id": f"T1-{seed}"},
                    "reward": reward,
                    "metrics": {"mission_value": float(completed), "hard_safety": -5.0 * len(violations)},
                    "num_turns": 20.0 if truncated else float(4 + rollout),
                    "error": None,
                    "is_truncated": truncated,
                    "stop_condition": "max_turns_reached" if truncated else "has_final_env_response",
                    "token_usage": {
                        "input_tokens": float(100 + seed_index),
                        "output_tokens": float(10 + rollout),
                    },
                    "sim_state": {
                        "seed": seed,
                        "tier": "T1",
                        "hard_safety_violations": violations,
                    },
                    "sim_log": [_snapshot(seed, completed=completed, violations=violations)],
                }
            )
    return rows


def _complete_run(run_dir: Path, step: int, rows: list[dict[str, Any]] | None = None) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metadata = {
        "run_id": f"eval-{step}",
        "env_id": "uav-operator",
        "env_args": {"tier": "T1", "dataset_split": "dev", "max_examples": 15, "max_turns": 20},
        "model": manifest["model_id"],
        "num_examples": 15,
        "rollouts_per_example": 2,
        "sampling_args": {"max_tokens": 1024, "temperature": 0},
        "state_columns": ["sim_state", "sim_log"],
    }
    _write_json(run_dir / "metadata.json", metadata)
    lines = [json.dumps(row) for row in (rows if rows is not None else _rows(step))]
    (run_dir / "results.jsonl").write_text("\n".join(lines) + "\n")


def test_prepare_adapter_writes_exact_frozen_config_and_manifest(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path, 10)
    config_text = (run_dir / "eval.toml").read_text()
    config = tomllib.loads(config_text)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert config == {
        "model": f"{BASE_MODEL}:adapter-10",
        "provider": "prime",
        "api_client_type": "openai_chat_completions",
        "env_args": {"tier": "T1", "dataset_split": "dev", "max_examples": 15, "max_turns": 20},
        "num_examples": 15,
        "rollouts_per_example": 2,
        "max_concurrent": 4,
        "max_retries": 0,
        "max_tokens": 1024,
        "temperature": 0,
        "state_columns": ["sim_state", "sim_log"],
        "save_results": True,
        "disable_tui": True,
        "verbose": True,
        "eval": [{"id": "uav-operator"}],
    }
    assert manifest["model_id"] == f"{BASE_MODEL}:adapter-10"
    assert manifest["config_sha256"] == evidence._canonical_hash(config_text)
    assert manifest["provenance"] == "synthetic_fixture"
    assert manifest["workload"]["expected_dev_seeds"] == list(evidence.DEV_SEEDS)
    assert manifest["token_ceiling"]["tokens"] == 15 * 2 * 20 * 1024
    assert "--skip-upload" in manifest["manual_eval_command"]


def test_checkpoint_prepare_stops_at_manual_deployment_gate(tmp_path: Path) -> None:
    fixture_dir = _metadata_fixtures(tmp_path / "metadata", 20, adapter=False)
    output = tmp_path / "checkpoint"
    manifest = evidence.prepare(
        run_id=RUN_ID,
        base_model=BASE_MODEL,
        step=20,
        checkpoint_id="checkpoint-20",
        output_dir=output,
        fixture_metadata_dir=fixture_dir,
    )
    assert manifest["status"] == "awaiting_adapter_deployment"
    assert manifest["manual_deployment_command"] == (
        "prime deployments create --checkpoint-id checkpoint-20"
    )
    assert manifest["manual_eval_command"] is None
    assert not (output / "eval.toml").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rft_run_id", "wrong-run", "provenance"),
        ("base_model", "wrong-model", "provenance"),
        ("step", 11, "provenance"),
        ("status", "PENDING", "status READY"),
        ("deployment_status", "DEPLOYING", "deployment status DEPLOYED"),
    ],
)
def test_prepare_rejects_adapter_metadata_mismatch(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    fixture_dir = _metadata_fixtures(tmp_path / "metadata", 10)
    payload = json.loads((fixture_dir / "adapters.json").read_text())
    payload["models"][0][field] = value
    _write_json(fixture_dir / "adapters.json", payload)
    with pytest.raises(ValueError, match=message):
        evidence.prepare(
            run_id=RUN_ID,
            base_model=BASE_MODEL,
            step=10,
            adapter_id="adapter-10",
            output_dir=tmp_path / "output",
            fixture_metadata_dir=fixture_dir,
        )


def test_prepare_fails_closed_without_pricing(tmp_path: Path) -> None:
    fixture_dir = _metadata_fixtures(tmp_path / "metadata", 10)
    payload = json.loads((fixture_dir / "models.json").read_text())
    del payload["models"][0]["effective_inference_output_price_per_mtok"]
    _write_json(fixture_dir / "models.json", payload)
    with pytest.raises(ValueError, match="pricing"):
        evidence.prepare(
            run_id=RUN_ID,
            base_model=BASE_MODEL,
            step=10,
            adapter_id="adapter-10",
            output_dir=tmp_path / "output",
            fixture_metadata_dir=fixture_dir,
        )


def test_captured_prepare_uses_only_read_only_prime_commands(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    payloads = {
        "get": {"run": {"id": RUN_ID, "base_model": BASE_MODEL}},
        "models": {
            "models": [
                {
                    "name": BASE_MODEL,
                    "effective_training_price_per_mtok": 0.0,
                    "effective_inference_input_price_per_mtok": 0.0,
                    "effective_inference_output_price_per_mtok": 0.0,
                }
            ]
        },
        "wallet": {"balance_usd": 50.0, "currency": "USD", "total_billings": 0},
        "list": {
            "models": [
                {
                    "id": "adapter-10",
                    "rft_run_id": RUN_ID,
                    "base_model": BASE_MODEL,
                    "step": 10,
                    "status": "READY",
                    "deployment_status": "DEPLOYED",
                }
            ]
        },
    }

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        key = "wallet" if "wallet" in command else command[3]
        return subprocess.CompletedProcess(command, 0, json.dumps(payloads[key]), "")

    manifest = evidence.prepare(
        run_id=RUN_ID,
        base_model=BASE_MODEL,
        step=10,
        adapter_id="adapter-10",
        output_dir=tmp_path / "output",
        reader=evidence.MetadataReader(runner=runner),
    )
    assert manifest["provenance"] == "captured"
    assert len(calls) == 4
    assert all("create" not in command and "eval" not in command for command in calls)


def test_summarize_counts_metrics_and_ranks_aggregate_same_seed_deltas(tmp_path: Path) -> None:
    baseline = _prepare_run(tmp_path, 0)
    final = _prepare_run(tmp_path, 20)
    baseline_rows = _rows(0)
    final_rows = _rows(20)
    # Seed 10005 improves most only when its two rollout rewards are aggregated.
    for index in (10, 11):
        final_rows[index]["reward"] += 1.0
    _complete_run(baseline, 0, baseline_rows)
    _complete_run(final, 20, final_rows)
    output_json = tmp_path / "summary.json"
    output_plot = tmp_path / "curve.png"

    payload = evidence.summarize(
        milestones=[(0, baseline), (20, final)],
        expected_steps=[0, 20],
        output_json=output_json,
        output_plot=output_plot,
        mode="synthetic_fixture",
    )

    assert payload["provenance"] == "synthetic_fixture"
    assert payload["watermark"] == evidence.FIXTURE_WATERMARK
    assert payload["milestones"][0]["max_turn_truncation_count"] == 1
    assert payload["milestones"][0]["hard_safety_count"] == 1
    assert payload["milestones"][0]["mission_completion_count"] == 20
    assert payload["milestones"][0]["mean_turns"] == pytest.approx(5.0)
    assert payload["milestones"][0]["token_use"] == {
        "input_tokens": 3210.0,
        "output_tokens": 315.0,
        "total_tokens": 3525.0,
    }
    best = payload["same_seed_baseline_to_final"]["ranking"][0]
    assert best["seed"] == 10005
    assert best["baseline_source_row_indices"] == [10, 11]
    assert best["final_source_row_indices"] == [10, 11]
    assert output_json.exists() and output_plot.stat().st_size > 0


def test_summarize_finds_prepare_manifest_above_vf_eval_run(tmp_path: Path) -> None:
    prepared = _prepare_run(tmp_path, 0)
    _complete_run(prepared, 0)
    generated_run = prepared / "evals" / "uav-operator--example--free-model" / "eval-0"
    generated_run.mkdir(parents=True)
    (prepared / "metadata.json").rename(generated_run / "metadata.json")
    (prepared / "results.jsonl").rename(generated_run / "results.jsonl")
    payload = evidence.summarize(
        milestones=[(0, generated_run)],
        expected_steps=[0],
        output_json=tmp_path / "summary.json",
        output_plot=tmp_path / "plot.png",
        mode="synthetic_fixture",
    )
    assert payload["milestones"][0]["source_run_dir"] == str(generated_run.resolve())


def _mutated_run(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    run_dir = _prepare_run(tmp_path, 0)
    rows = _rows(0)
    return run_dir, rows


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[0].update(error={"message": "provider failed"}), "provider error"),
        (lambda rows: rows[0].update(reward=math.nan), "finite number"),
        (lambda rows: rows[0]["info"].update(tier="T2"), "wrong tier"),
        (lambda rows: rows[0]["info"].update(seed=20000), "seed leakage"),
        (lambda rows: rows[0].update(sim_log=[]), "sim_log"),
        (lambda rows: rows.pop(), "expected 30"),
        (
                lambda rows: (
                    rows[2]["info"].update(seed=10000),
                    rows[2]["sim_state"].update(seed=10000),
                    rows[2]["sim_log"][0].update(seed=10000),
                ),
            "duplicate per-seed",
        ),
    ],
)
def test_summarize_rejects_invalid_rollout_evidence(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    run_dir, rows = _mutated_run(tmp_path)
    mutation(rows)
    _complete_run(run_dir, 0, rows)
    with pytest.raises(ValueError, match=message):
        evidence.summarize(
            milestones=[(0, run_dir)],
            expected_steps=[0],
            output_json=tmp_path / "summary.json",
            output_plot=tmp_path / "plot.png",
            mode="synthetic_fixture",
        )


def test_summarize_rejects_missing_duplicate_and_wrong_manifest_steps(tmp_path: Path) -> None:
    baseline = _prepare_run(tmp_path, 0)
    _complete_run(baseline, 0)
    with pytest.raises(ValueError, match=r"missing=\[10\]"):
        evidence.summarize(
            milestones=[(0, baseline)],
            expected_steps=[0, 10],
            output_json=tmp_path / "summary.json",
            output_plot=tmp_path / "plot.png",
            mode="synthetic_fixture",
        )
    with pytest.raises(ValueError, match="duplicate milestone"):
        evidence.summarize(
            milestones=[(0, baseline), (0, baseline)],
            expected_steps=[0],
            output_json=tmp_path / "summary.json",
            output_plot=tmp_path / "plot.png",
            mode="synthetic_fixture",
        )


def test_captured_mode_rejects_fixture_provenance(tmp_path: Path) -> None:
    baseline = _prepare_run(tmp_path, 0)
    _complete_run(baseline, 0)
    with pytest.raises(ValueError, match="rejects fixture provenance"):
        evidence.summarize(
            milestones=[(0, baseline)],
            expected_steps=[0],
            output_json=tmp_path / "summary.json",
            output_plot=tmp_path / "plot.png",
            mode="captured",
        )


def test_summarize_rejects_metadata_and_manifest_mismatch(tmp_path: Path) -> None:
    baseline = _prepare_run(tmp_path, 0)
    _complete_run(baseline, 0)
    metadata = json.loads((baseline / "metadata.json").read_text())
    metadata["env_args"]["dataset_split"] = "eval"
    _write_json(baseline / "metadata.json", metadata)
    with pytest.raises(ValueError, match="dataset_split mismatch"):
        evidence.summarize(
            milestones=[(0, baseline)],
            expected_steps=[0],
            output_json=tmp_path / "summary.json",
            output_plot=tmp_path / "plot.png",
            mode="synthetic_fixture",
        )


def test_summarize_rejects_tampered_eval_config(tmp_path: Path) -> None:
    baseline = _prepare_run(tmp_path, 0)
    _complete_run(baseline, 0)
    with (baseline / "eval.toml").open("a") as config:
        config.write("# changed after preparation\n")
    with pytest.raises(ValueError, match="config hash"):
        evidence._summarize_one(0, baseline, "synthetic_fixture")


def test_malformed_log_and_nonfinite_metric_fail_closed(tmp_path: Path) -> None:
    baseline, rows = _mutated_run(tmp_path)
    del rows[0]["sim_log"][0]["aircraft"]
    _complete_run(baseline, 0, rows)
    with pytest.raises(ValueError, match="malformed sim_log"):
        evidence._summarize_one(0, baseline, "synthetic_fixture")

    rows = _rows(0)
    rows[0]["metrics"]["mission_value"] = math.inf
    _complete_run(baseline, 0, rows)
    with pytest.raises(ValueError, match="finite number"):
        evidence._summarize_one(0, baseline, "synthetic_fixture")
