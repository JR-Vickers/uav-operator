from __future__ import annotations

import tomllib
from pathlib import Path

import uav_operator
from scripts.capture_day8_training import (
    summarize_logs,
    summarize_samples,
    wallet_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "day8_laguna_t1_smoke.toml"


def _config() -> dict[str, object]:
    with CONFIG_PATH.open("rb") as config_file:
        return tomllib.load(config_file)


def test_day8_hosted_training_config_is_exact() -> None:
    config = _config()

    assert config["model"] == "poolside/Laguna-XS-2.1"
    assert config["loss"] == "rl"
    assert config["max_steps"] == 50
    assert config["batch_size"] == 128
    assert config["rollouts_per_example"] == 8
    assert config["max_inflight_rollouts"] == 96
    assert config["learning_rate"] == 3e-5
    assert config["lora_alpha"] == 32
    assert config["sampling"] == {
        "max_tokens": 1024,
        "temperature": 0.7,
        "enable_thinking": False,
    }
    assert config["env"] == [
        {
            "id": "jarrett/uav-operator@0.1.1",
            "args": {
                "tier": "T1",
                "dataset_split": "train",
                "max_examples": 75,
                "max_turns": 40,
            },
        }
    ]
    assert config["eval"] == {
        "interval": 10,
        "num_examples": 15,
        "rollouts_per_example": 2,
        "skip_first_step": False,
        "env": [
            {
                "id": "jarrett/uav-operator@0.1.1",
                "args": {
                    "tier": "T1",
                    "dataset_split": "dev",
                    "max_examples": 15,
                    "max_turns": 40,
                },
            }
        ],
        "sampling": {
            "max_tokens": 1024,
            "temperature": 0.0,
            "enable_thinking": False,
        },
    }
    assert config["checkpoints"] == {"interval": 10, "keep_cloud": 2}
    assert config["adapters"] == {"interval": 10, "keep_last": 3}
    assert "infrastructure" not in config


def test_day8_configured_t1_splits_are_disjoint_and_withhold_final_eval() -> None:
    config = _config()
    train_args = config["env"][0]["args"]
    dev_args = config["eval"]["env"][0]["args"]
    train_env = uav_operator.load_environment(**train_args)
    dev_env = uav_operator.load_environment(**dev_args)

    train_rows = list(train_env.get_dataset())
    dev_rows = list(dev_env.get_eval_dataset())
    final_eval_rows = list(
        uav_operator.load_environment(
            tier="T1", dataset_split="eval", max_examples=15, max_turns=40
        ).get_eval_dataset()
    )
    train_seeds = {row["info"]["seed"] for row in train_rows}
    dev_seeds = {row["info"]["seed"] for row in dev_rows}
    final_eval_seeds = {row["info"]["seed"] for row in final_eval_rows}

    assert len(train_rows) == 75
    assert len(dev_rows) == len(final_eval_rows) == 15
    assert {row["info"]["tier"] for row in train_rows + dev_rows} == {"T1"}
    assert train_seeds.isdisjoint(dev_seeds)
    assert train_seeds.isdisjoint(final_eval_seeds)
    assert dev_seeds.isdisjoint(final_eval_seeds)


def test_day8_sample_summary_reports_truncation_and_provider_errors() -> None:
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


def test_day8_log_summary_counts_platform_model_errors() -> None:
    logs = "\n".join(
        [
            "Rollout failed in group one — Error: ModelError",
            "Rollout failed in group two — Error: ModelError",
            "ordinary informational line",
        ]
    )

    assert summarize_logs(logs) == {"model_errors": 2, "rollout_failures": 2}


def test_day8_wallet_evidence_excludes_unrelated_history() -> None:
    wallet = {
        "wallet_id": "private-wallet-id",
        "balance_usd": 57.9186,
        "currency": "USD",
        "total_billings": 8,
        "recent_billings": [
            {"resource_id": "this-run", "amount_usd": 0.0},
            {"resource_id": "unrelated-run", "amount_usd": 1.0},
        ],
    }

    assert wallet_evidence(wallet, "this-run") == {
        "balance_usd": 57.9186,
        "currency": "USD",
        "total_billings": 8,
        "run_billings": [{"resource_id": "this-run", "amount_usd": 0.0}],
    }
