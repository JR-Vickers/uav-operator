from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import uav_operator
import verifiers as vf
from scripts import capture_day8_training as capture_script
from scripts.capture_day8_training import (
    billing_reconciliation,
    find_non_finite,
    summarize_logs,
    summarize_samples,
    validate_capture,
    validate_preflight,
    wallet_evidence,
)
from verifiers.utils.eval_utils import load_toml_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_CONFIG_PATH = REPO_ROOT / "configs" / "day8_llama_1b_t1_diagnostic.toml"
SMOKE_CONFIG_PATH = REPO_ROOT / "configs" / "day8_llama_1b_t1_smoke.toml"
FUNCTIONAL_EVAL_CONFIG_PATH = (
    REPO_ROOT / "configs" / "eval" / "day8_laguna_m1_t1_functional.toml"
)
LLAMA_PREFLIGHT_PATH = (
    REPO_ROOT / "assets" / "training" / "day8_llama_1b_preflight.json"
)
LLAMA_DIAGNOSTIC_PATH = (
    REPO_ROOT / "assets" / "training" / "day8_llama_1b_diagnostic.json"
)
MODEL = "sprints/Llama-3.2-1B-Instruct"


def _load(path: Path) -> dict[str, object]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def _expected_config(*, diagnostic: bool) -> dict[str, object]:
    max_steps = 1 if diagnostic else 50
    train_examples = 8 if diagnostic else 75
    eval_examples = 2 if diagnostic else 15
    interval = 1 if diagnostic else 10
    return {
        "name": (
            "uav-operator-day8-llama-1b-t1-diagnostic"
            if diagnostic
            else "uav-operator-day8-llama-1b-t1-smoke"
        ),
        "model": MODEL,
        "loss": "rl",
        "max_steps": max_steps,
        "batch_size": 16,
        "rollouts_per_example": 2,
        "max_inflight_rollouts": 4,
        "learning_rate": 3e-5,
        "lora_alpha": 32,
        "sampling": {
            "max_tokens": 1024,
            "temperature": 0.7,
            "enable_thinking": False,
        },
        "env": [
            {
                "id": "jarrett/uav-operator@0.1.1",
                "args": {
                    "tier": "T1",
                    "dataset_split": "train",
                    "max_examples": train_examples,
                    "max_turns": 20,
                },
            }
        ],
        "eval": {
            "interval": interval,
            "num_examples": eval_examples,
            "rollouts_per_example": 2,
            "skip_first_step": False,
            "env": [
                {
                    "id": "jarrett/uav-operator@0.1.1",
                    "args": {
                        "tier": "T1",
                        "dataset_split": "dev",
                        "max_examples": eval_examples,
                        "max_turns": 20,
                    },
                }
            ],
            "sampling": {
                "max_tokens": 1024,
                "temperature": 0.0,
                "enable_thinking": False,
            },
        },
        "checkpoints": {
            "interval": interval,
            "keep_cloud": 1 if diagnostic else 2,
        },
        "adapters": {
            "interval": interval,
            "keep_last": 1 if diagnostic else 3,
        },
    }


@pytest.mark.parametrize(
    ("path", "diagnostic"),
    [(DIAGNOSTIC_CONFIG_PATH, True), (SMOKE_CONFIG_PATH, False)],
)
def test_day8_llama_configs_are_exact(path: Path, diagnostic: bool) -> None:
    assert _load(path) == _expected_config(diagnostic=diagnostic)


@pytest.mark.parametrize(
    ("path", "train_count", "dev_count"),
    [(DIAGNOSTIC_CONFIG_PATH, 8, 2), (SMOKE_CONFIG_PATH, 75, 15)],
)
def test_day8_llama_views_are_disjoint_and_withhold_final_eval(
    path: Path, train_count: int, dev_count: int
) -> None:
    config = _load(path)
    train_env = uav_operator.load_environment(**config["env"][0]["args"])
    dev_env = uav_operator.load_environment(**config["eval"]["env"][0]["args"])
    final_eval_env = uav_operator.load_environment(
        tier="T1", dataset_split="eval", max_examples=15, max_turns=20
    )
    train_rows = list(train_env.get_dataset())
    dev_rows = list(dev_env.get_eval_dataset())
    final_eval_rows = list(final_eval_env.get_eval_dataset())
    train_seeds = {row["info"]["seed"] for row in train_rows}
    dev_seeds = {row["info"]["seed"] for row in dev_rows}
    final_eval_seeds = {row["info"]["seed"] for row in final_eval_rows}

    assert len(train_rows) == train_count
    assert len(dev_rows) == dev_count
    assert len(final_eval_rows) == 15
    assert {row["info"]["tier"] for row in train_rows + dev_rows} == {"T1"}
    assert train_seeds.isdisjoint(dev_seeds)
    assert train_seeds.isdisjoint(final_eval_seeds)
    assert dev_seeds.isdisjoint(final_eval_seeds)


def test_day8_laguna_m1_functional_eval_remains_exact_and_renderable() -> None:
    configs = load_toml_config(FUNCTIONAL_EVAL_CONFIG_PATH)
    assert len(configs) == 1
    config = configs[0]
    assert config == {
        "env_id": "uav-operator",
        "model": "poolside/laguna-m.1",
        "provider": "prime",
        "api_client_type": "openai_chat_completions",
        "env_args": {
            "tier": "T1",
            "dataset_split": "dev",
            "max_examples": 2,
            "max_turns": 40,
        },
        "num_examples": 2,
        "rollouts_per_example": 1,
        "max_concurrent": 1,
        "max_retries": 0,
        "max_tokens": 512,
        "temperature": 0.2,
        "state_columns": ["sim_state", "sim_log"],
        "save_results": True,
        "disable_tui": True,
        "verbose": True,
    }
    env = vf.load_environment(config["env_id"], **config["env_args"])
    assert len(list(env.get_eval_dataset())) == 2


def test_preflight_selects_model_from_toml_and_allows_inference_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "example/free-model"\n')
    commands: list[list[str]] = []

    def fake_prime_json(args: list[str]) -> dict[str, object]:
        commands.append(args)
        if args[:2] == ["train", "models"]:
            return {
                "models": [
                    {
                        "name": "example/free-model",
                        "at_capacity": False,
                        **{field: 0.0 for field in capture_script.ZERO_PRICE_FIELDS},
                    },
                    {"name": "poolside/Laguna-XS-2.1"},
                ]
            }
        if args[:2] == ["inference", "models"]:
            return {"data": [{"id": "example/different-id"}]}
        if args[0] == "wallet":
            return {"balance_usd": 12.5, "currency": "USD", "total_billings": 3}
        return {
            "latest_version": {"semantic_version": "0.1.1"},
            "action": {"status": "SUCCESS"},
            "freeTierEligible": True,
        }

    monkeypatch.setattr(capture_script, "_prime_json", fake_prime_json)
    artifact = capture_script.preflight(config_path)

    assert artifact["model"] == "example/free-model"
    assert artifact["hosted_training_model"]["name"] == "example/free-model"
    assert artifact["inference_catalog_exact_id_present"] is False
    assert artifact["summary"]["passed"] is True
    assert ["inference", "models", "--output", "json", "--search", "example/free-model"] in commands


def test_preflight_validation_reports_capacity_pricing_wallet_and_hub() -> None:
    artifact = {
        "hosted_training_model": {
            "at_capacity": True,
            **{field: 0.1 for field in capture_script.ZERO_PRICE_FIELDS},
        },
        "wallet": {},
        "hub_status": {
            "latest_version": {"semantic_version": "0.1.0"},
            "action": {"status": "FAILED"},
        },
        "free_tier_environment_eligibility": {"status": "denied"},
    }
    summary = validate_preflight(artifact)
    assert summary["passed"] is False
    assert len(summary["failures"]) == 8


def test_preflight_fails_closed_without_free_tier_eligibility() -> None:
    artifact = {
        "hosted_training_model": {
            "at_capacity": False,
            **{field: 0.0 for field in capture_script.ZERO_PRICE_FIELDS},
        },
        "wallet": {"balance_usd": 57.9182},
        "hub_status": {
            "latest_version": {"semantic_version": "0.1.1"},
            "action": {"status": "SUCCESS"},
        },
        "free_tier_environment_eligibility": {"status": "unverifiable"},
    }

    summary = validate_preflight(artifact)

    assert summary["passed"] is False
    assert summary["failures"] == [
        "free-tier model/environment eligibility is not affirmatively confirmed"
    ]


def test_recursive_finite_validation_and_capture_summary() -> None:
    assert find_non_finite({"metrics": [{"reward": float("nan")}], "ok": 1.0}) == [
        "$.metrics[0].reward"
    ]
    artifact = {
        "preflight": {
            "summary": {"passed": True},
            "config": {"sha256": "same"},
        },
        "config": {"sha256": "same"},
        "billing_reconciliation": {"run_reports_zero_cost": True},
        "metrics": {"loss": float("inf")},
        "progress": {"latest_step": 1},
        "usage": {"training": {"tokens": 42}},
        "sample_summary": {"provider_errors": 0},
        "log_summary": {"provider_or_context_errors": 0},
    }
    summary = validate_capture(artifact)
    assert summary["passed"] is False
    assert summary["training_tokens"] == 42
    assert "non-finite number at $.metrics.loss" in summary["failures"]


def test_billing_reconciliation_reports_run_cost_and_unrelated_delta() -> None:
    reconciliation = billing_reconciliation(
        {"balance_usd": 57.9},
        {
            "balance_usd": 57.7,
            "run_billings": [{"amount_usd": 0.0}],
        },
        {"total_cost_usd": 0.0},
    )
    assert reconciliation["run_reports_zero_cost"] is True
    assert reconciliation["balance_delta_usd"] == pytest.approx(0.2)
    assert reconciliation["unrelated_or_unreconciled_wallet_delta_usd"] == pytest.approx(0.2)


def test_sample_and_log_summaries_report_truncation_and_provider_errors() -> None:
    summary = summarize_samples(
        {
            "10": {
                "samples": [
                    {"metadata": {"stop_condition": "max_turns_reached"}},
                    {"metadata": {"stop_condition": "completed"}},
                    {"provider_error": "upstream timeout"},
                ]
            }
        }
    )
    assert summary["sample_count"] == 3
    assert summary["max_turn_truncations"] == 1
    assert summary["truncation_rate"] == 1 / 3
    assert summary["provider_errors"] == 1

    logs = "\n".join(
        [
            "Rollout failed in group one — Error: ModelError",
            "Provider error: timed out",
            "Uploaded checkpoint ckpt-1",
            "Uploaded adapter adapter-1",
        ]
    )
    log_summary = summarize_logs(logs)
    assert log_summary["model_errors"] == 1
    assert log_summary["rollout_failures"] == 1
    assert log_summary["provider_or_context_errors"] == 1
    assert log_summary["checkpoint_evidence"] == ["Uploaded checkpoint ckpt-1"]
    assert log_summary["adapter_upload_evidence"] == ["Uploaded adapter adapter-1"]


def test_wallet_evidence_excludes_unrelated_history() -> None:
    wallet = {
        "wallet_id": "private-wallet-id",
        "balance_usd": 57.9182,
        "currency": "USD",
        "total_billings": 10,
        "recent_billings": [
            {"resource_id": "this-run", "amount_usd": 0.0},
            {"resource_id": "unrelated-run", "amount_usd": 1.0},
        ],
    }
    evidence = wallet_evidence(wallet, "this-run")
    assert evidence == {
        "balance_usd": 57.9182,
        "currency": "USD",
        "total_billings": 10,
        "run_billings": [{"resource_id": "this-run", "amount_usd": 0.0}],
    }
    assert "private-wallet-id" not in json.dumps(evidence)


def test_llama_launch_rejection_evidence_is_finite_and_fail_closed() -> None:
    preflight = json.loads(LLAMA_PREFLIGHT_PATH.read_text())
    diagnostic = json.loads(LLAMA_DIAGNOSTIC_PATH.read_text())

    assert preflight["free_tier_environment_eligibility"] == {
        "detail": (
            "HTTP 400: Free-tier model 'sprints/Llama-3.2-1B-Instruct': "
            "'jarrett/uav-operator' does not meet the free-tier environment "
            "requirements."
        ),
        "source": "recorded_run_creation_response",
        "status": "denied",
    }
    assert preflight["summary"]["passed"] is False
    assert diagnostic["run_created"] is False
    assert diagnostic["run_id"] is None
    assert diagnostic["outcome"]["optimizer_steps"] == 0
    assert diagnostic["outcome"]["training_tokens"] == 0
    assert diagnostic["billing_reconciliation"]["balance_delta_usd"] == 0.0
    assert diagnostic["billing_reconciliation"]["run_billing_row"] is None
    assert find_non_finite(preflight) == []
    assert find_non_finite(diagnostic) == []
