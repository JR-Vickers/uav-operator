from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

import uav_operator
from scripts import main_training as main


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True))


def _prepare_fixtures(
    root: Path, phase: str, checkpoint_id: str, source_run: str, source_step: int
) -> Path:
    _write(
        root / "source_run.json", {"run": {"id": source_run, "base_model": main.MODEL}}
    )
    _write(
        root / "source_checkpoints.json",
        {
            "checkpoints": [
                {
                    "id": checkpoint_id,
                    "run_id": source_run,
                    "base_model": main.MODEL,
                    "step": source_step,
                    "status": "READY",
                }
            ]
        },
    )
    _write(
        root / "models.json",
        {
            "models": [
                {
                    "name": main.MODEL,
                    "at_capacity": False,
                    "effective_training_price_per_mtok": 0.01,
                    "effective_inference_input_price_per_mtok": 0.01,
                    "effective_inference_output_price_per_mtok": 0.02,
                }
            ]
        },
    )
    _write(
        root / "wallet.json",
        {"balance_usd": 20, "total_billings": 7, "currency": "USD"},
    )
    _write(
        root / "hub.json",
        {
            "latest_version": {"semantic_version": "0.1.1"},
            "action": {"status": "SUCCESS"},
        },
    )
    return root


def _passing_predecessor(
    path: Path, phase: str, run_id: str, checkpoint_id: str
) -> Path:
    _write(
        path,
        {
            "phase": phase,
            "run_id": run_id,
            "summary": {
                "passed": True,
                "run_cost_usd": 1.0,
                "final_checkpoint": {
                    "id": checkpoint_id,
                    "step": main.PHASES[phase].steps,
                    "status": "READY",
                },
            },
        },
    )
    return path


@pytest.mark.parametrize("phase_name", ["A", "B", "C"])
def test_phase_configs_are_exact_and_use_only_train_dev(phase_name: str) -> None:
    phase = main.PHASES[phase_name]
    config = tomllib.loads(main.phase_toml(phase, "checkpoint"))
    main.validate_phase_config(config, phase, "checkpoint")
    assert config["model"] == main.MODEL
    assert config["checkpoint_id"] == "checkpoint"
    assert config["max_steps"] == phase.steps
    assert config["batch_size"] == 16
    assert config["rollouts_per_example"] == 2
    assert config["max_inflight_rollouts"] == 4
    assert config["learning_rate"] == 3e-5
    assert config["lora_alpha"] == 32
    assert config["sampling"] == {
        "max_tokens": 1024,
        "temperature": 0.7,
        "enable_thinking": False,
    }
    assert [(row["args"]["tier"], row["ratio"]) for row in config["env"]] == list(
        phase.mixture
    )
    assert all(
        row["args"]
        == {
            "tier": row["args"]["tier"],
            "dataset_split": "train",
            "max_examples": 75,
            "max_turns": 20,
        }
        for row in config["env"]
    )
    assert all(row["max_retries"] == 0 for row in config["env"])
    assert config["eval"]["interval"] == 10
    assert config["eval"]["skip_first_step"] is False
    assert config["eval"]["num_examples"] == 24
    assert config["eval"]["rollouts_per_example"] == 1
    assert [row["args"]["tier"] for row in config["eval"]["env"]] == [
        "T0",
        "T1",
        "T2",
        "T3",
    ]
    assert all(
        row["num_examples"] == 6
        and row["rollouts_per_example"] == 1
        and row["max_retries"] == 0
        for row in config["eval"]["env"]
    )
    assert config["eval"]["sampling"] == {
        "max_tokens": 1024,
        "temperature": 0.0,
        "enable_thinking": False,
    }
    assert config["checkpoints"] == {
        "interval": phase.artifact_interval,
        "keep_cloud": 2,
    }
    assert config["adapters"] == {"interval": phase.artifact_interval, "keep_last": 2}
    assert 'dataset_split = "eval"' not in main.phase_toml(phase, "checkpoint")


def test_training_and_dev_views_are_deterministic_and_final_eval_is_disjoint() -> None:
    for tier in ("T0", "T1", "T2", "T3"):
        train = list(
            uav_operator.load_environment(
                tier=tier, dataset_split="train", max_examples=75, max_turns=20
            ).get_dataset()
        )
        train_again = list(
            uav_operator.load_environment(
                tier=tier, dataset_split="train", max_examples=75, max_turns=20
            ).get_dataset()
        )
        dev = list(
            uav_operator.load_environment(
                tier=tier, dataset_split="dev", max_examples=6, max_turns=20
            ).get_eval_dataset()
        )
        final = list(
            uav_operator.load_environment(
                tier=tier, dataset_split="eval", max_examples=15, max_turns=20
            ).get_eval_dataset()
        )
        assert [row["info"]["seed"] for row in train] == [
            row["info"]["seed"] for row in train_again
        ]
        sets = [{row["info"]["seed"] for row in rows} for rows in (train, dev, final)]
        assert (
            sets[0].isdisjoint(sets[1])
            and sets[0].isdisjoint(sets[2])
            and sets[1].isdisjoint(sets[2])
        )


def test_prepare_a_writes_hashed_manual_only_bundle(tmp_path: Path) -> None:
    fixtures = _prepare_fixtures(
        tmp_path / "fixtures",
        "A",
        main.SMOKE_CHECKPOINT_ID,
        main.SMOKE_RUN_ID,
        main.SMOKE_STEP,
    )
    artifact = main.prepare("A", tmp_path / "bundle", fixture_dir=fixtures)
    config = tomllib.loads((tmp_path / "bundle" / "train.toml").read_text())
    assert artifact["status"] == "awaiting_explicit_manual_launch_approval"
    assert artifact["input_provenance"] == {
        "run_id": main.SMOKE_RUN_ID,
        "step": 20,
        "checkpoint_id": main.SMOKE_CHECKPOINT_ID,
        "status": "READY",
        "model": main.MODEL,
    }
    assert artifact["config_sha256"] == main._sha256(
        (tmp_path / "bundle" / "train.toml").read_bytes()
    )
    assert artifact["manual_launch_command"].startswith("prime --plain train ")
    assert config["checkpoint_id"] == main.SMOKE_CHECKPOINT_ID
    assert artifact["budget"]["passed"] is True
    assert "optimizer-state restoration is not claimed" in artifact["description"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("id", "wrong", "run/model"), ("base_model", "wrong", "run/model")],
)
def test_prepare_rejects_wrong_source_run_or_model(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    fixtures = _prepare_fixtures(
        tmp_path / "fixtures", "A", main.SMOKE_CHECKPOINT_ID, main.SMOKE_RUN_ID, 20
    )
    payload = json.loads((fixtures / "source_run.json").read_text())
    payload["run"][field] = value
    _write(fixtures / "source_run.json", payload)
    with pytest.raises(ValueError, match=message):
        main.prepare("A", tmp_path / "out", fixture_dir=fixtures)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step", 19, "step"),
        ("status", "PENDING", "READY"),
        ("run_id", "wrong", "run provenance"),
        ("base_model", "wrong", "model provenance"),
    ],
)
def test_prepare_rejects_checkpoint_provenance(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    fixtures = _prepare_fixtures(
        tmp_path / "fixtures", "A", main.SMOKE_CHECKPOINT_ID, main.SMOKE_RUN_ID, 20
    )
    payload = json.loads((fixtures / "source_checkpoints.json").read_text())
    payload["checkpoints"][0][field] = value
    _write(fixtures / "source_checkpoints.json", payload)
    with pytest.raises(ValueError, match=message):
        main.prepare("A", tmp_path / "out", fixture_dir=fixtures)


def test_prepare_b_requires_passing_a_and_exact_final_checkpoint(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Phase-A capture is required"):
        main.resolve_input(main.PHASES["B"], None)
    path = _passing_predecessor(tmp_path / "a.json", "A", "run-a", "checkpoint-a")
    assert main.resolve_input(main.PHASES["B"], path) == ("checkpoint-a", "run-a", 10)
    payload = json.loads(path.read_text())
    payload["summary"]["passed"] = False
    _write(path, payload)
    with pytest.raises(ValueError, match="has not passed"):
        main.resolve_input(main.PHASES["B"], path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("capacity", "capacity"),
        ("price", "pricing"),
        ("wallet", "wallet"),
        ("hub", "Hub"),
    ],
)
def test_prepare_fails_closed_on_live_gates(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixtures = _prepare_fixtures(
        tmp_path / "fixtures", "A", main.SMOKE_CHECKPOINT_ID, main.SMOKE_RUN_ID, 20
    )
    if mutation == "capacity":
        payload = json.loads((fixtures / "models.json").read_text())
        payload["models"][0]["at_capacity"] = True
        _write(fixtures / "models.json", payload)
    elif mutation == "price":
        payload = json.loads((fixtures / "models.json").read_text())
        del payload["models"][0][main.PRICE_FIELDS[0]]
        _write(fixtures / "models.json", payload)
    elif mutation == "wallet":
        _write(fixtures / "wallet.json", {"currency": "USD"})
    else:
        _write(
            fixtures / "hub.json",
            {
                "latest_version": {"semantic_version": "0.1.0"},
                "action": {"status": "FAILED"},
            },
        )
    with pytest.raises(ValueError, match=message):
        main.prepare("A", tmp_path / "out", fixture_dir=fixtures)


def _capture_artifact(phase_name: str = "A") -> dict[str, Any]:
    phase = main.PHASES[phase_name]
    metrics = [
        {"step": step, f"eval/{main.ENVIRONMENT}/T0/avg@1": 0.2}
        for step in range(0, phase.steps + 1, 10)
    ]
    metrics.extend(
        {"step": step, "reward/all/mean": 0.1} for step in range(phase.steps)
    )
    samples = {
        str(step): {
            "samples": [
                {
                    "stop_condition": "has_final_env_response",
                    "metrics": {"hard_safety": 0},
                }
                for _ in range(6)
            ]
        }
        for step in range(phase.steps)
    }
    intervals = [phase.artifact_interval, phase.steps]
    artifact = {
        "phase": phase_name,
        "run_id": f"run-{phase_name}",
        "manifest": {"input_provenance": {"checkpoint_id": "input"}},
        "config": {"parsed": tomllib.loads(main.phase_toml(phase, "input"))},
        "run": {
            "run": {
                "status": "COMPLETED",
                "base_model": main.MODEL,
                "max_steps": phase.steps,
            }
        },
        "progress": {
            "latest_step": phase.steps,
            "steps_with_samples": list(range(phase.steps)),
            "steps_with_distributions": list(range(phase.steps)),
        },
        "metrics": {"metrics": metrics},
        "samples_by_step": samples,
        "checkpoints": {
            "checkpoints": [
                {"id": f"cp-{step}", "step": step, "status": "READY"}
                for step in intervals
            ]
        },
        "adapters": {
            "models": [
                {
                    "id": f"adapter-{step}",
                    "rft_run_id": f"run-{phase_name}",
                    "step": step,
                    "status": "READY",
                }
                for step in intervals
            ]
        },
        "usage": {"total_cost_usd": phase.expected_cost_usd},
    }
    return artifact


def test_capture_validation_accepts_complete_phase_and_final_checkpoint() -> None:
    artifact = _capture_artifact()
    summary = main.validate_capture(artifact)
    assert summary["passed"] is True
    assert summary["final_checkpoint"]["id"] == "cp-10"


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        ("status", "run status"),
        ("steps", "sample steps"),
        ("distribution", "distribution steps"),
        ("eval", "evaluation milestones"),
        ("checkpoint", "checkpoint"),
        ("adapter", "adapter"),
        ("provider", "provider errors"),
        ("cancelled", "cancelled"),
        ("safety", "hard-safety"),
        ("truncation", "truncation"),
        ("cost", "cost ceiling"),
        ("provenance", "provenance"),
    ],
)
def test_capture_gates_fail_closed(mutation: str, failure: str) -> None:
    artifact = _capture_artifact()
    if mutation == "status":
        artifact["run"]["run"]["status"] = "FAILED"
    elif mutation == "steps":
        artifact["progress"]["steps_with_samples"].pop()
    elif mutation == "distribution":
        artifact["progress"]["steps_with_distributions"].pop()
    elif mutation == "eval":
        artifact["metrics"]["metrics"] = [
            row
            for row in artifact["metrics"]["metrics"]
            if not (row["step"] == 10 and any(str(k).startswith("eval/") for k in row))
        ]
    elif mutation == "checkpoint":
        artifact["checkpoints"]["checkpoints"].pop()
    elif mutation == "adapter":
        artifact["adapters"]["models"].pop()
    elif mutation == "provider":
        artifact["samples_by_step"]["0"]["samples"][0]["provider_error"] = "bad"
    elif mutation == "cancelled":
        artifact["samples_by_step"]["0"]["samples"][0]["status"] = "CANCELLED"
    elif mutation == "safety":
        artifact["samples_by_step"]["0"]["samples"][0]["metrics"]["hard_safety"] = -5
    elif mutation == "truncation":
        for row in artifact["samples_by_step"]["0"]["samples"][:4]:
            row["stop_condition"] = "max_turns_reached"
    elif mutation == "cost":
        artifact["usage"]["total_cost_usd"] = 99
    else:
        artifact["config"]["parsed"]["checkpoint_id"] = "wrong"
    summary = main.validate_capture(artifact)
    assert summary["passed"] is False
    assert any(failure in item for item in summary["failures"])


def test_capture_rejects_non_finite_metric() -> None:
    artifact = _capture_artifact()
    artifact["metrics"]["metrics"][0]["bad"] = float("nan")
    assert any(
        "non-finite" in item for item in main.validate_capture(artifact)["failures"]
    )


def test_budget_ledger_and_billing_reconciliation_failures(tmp_path: Path) -> None:
    assert main.budget([])["projected_total_usd"] == pytest.approx(11.5469)
    assert main.budget([])["hard_ceiling_total_usd"] == pytest.approx(12.7969)
    assert main.budget([])["unallocated_after_hard_ceilings_usd"] == pytest.approx(
        2.2031
    )
    capture = tmp_path / "a.json"
    _write(capture, {"phase": "A", "summary": {"passed": True, "run_cost_usd": 1.26}})
    with pytest.raises(ValueError, match="ceiling breached"):
        main.budget([capture])
    with pytest.raises(ValueError, match="unavailable"):
        main.budget([], {"A": None})
    assert main.budget([], {"A": 4.0})["passed"] is False


def test_candidate_selection_is_paired_and_smoke_is_ineligible() -> None:
    rows = [
        {
            "phase": "smoke",
            "paired_seeds": [1, 2],
            "hard_safety_violations": 0,
            "mean_reward": 99,
            "completions": 99,
            "max_turn_truncations": 0,
        },
        {
            "phase": "A",
            "paired_seeds": [1, 2],
            "hard_safety_violations": 0,
            "mean_reward": 0.5,
            "completions": 5,
            "max_turn_truncations": 2,
        },
        {
            "phase": "B",
            "paired_seeds": [1, 2],
            "hard_safety_violations": 1,
            "mean_reward": 1.0,
            "completions": 8,
            "max_turn_truncations": 0,
        },
        {
            "phase": "C",
            "paired_seeds": [1, 2],
            "hard_safety_violations": 0,
            "mean_reward": 0.6,
            "completions": 4,
            "max_turn_truncations": 3,
        },
    ]
    result = main.select_candidates(rows)
    assert result["winner"]["phase"] == "C"
    assert result["eligible_phases"] == ["A", "B", "C"]
    assert result["smoke_is_reference_only"] is True
    with pytest.raises(ValueError, match="paired-seed"):
        main.select_candidates(
            [rows[0], rows[1], {**rows[2], "paired_seeds": [3]}, rows[3]]
        )
    with pytest.raises(ValueError, match="coverage"):
        main.select_candidates(rows[:-1])
