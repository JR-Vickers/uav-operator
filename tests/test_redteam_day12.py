from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
from typing import Any, Callable

import pytest

import uav_operator
from scripts import redteam_day12 as redteam


RUN_ID = "run-fixture"
BASE_MODEL = "example/free-model"
ADAPTER_ID = "adapter-20"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _training_manifest(
    tmp_path: Path,
    *,
    provenance: str = "synthetic_fixture",
    input_price: float = 0.0,
    output_price: float = 0.0,
) -> Path:
    config = tmp_path / "training-evidence" / "eval.toml"
    config.parent.mkdir(parents=True)
    config.write_text("model = \"example/free-model:adapter-20\"\n")
    manifest = {
        "kind": "uav_operator_training_evidence_manifest",
        "status": "ready_for_manual_eval",
        "provenance": provenance,
        "training_run_id": RUN_ID,
        "base_model": BASE_MODEL,
        "step": 20,
        "checkpoint_id": "checkpoint-20",
        "adapter_id": ADAPTER_ID,
        "model_id": f"{BASE_MODEL}:{ADAPTER_ID}",
        "config_path": str(config.resolve()),
        "config_sha256": redteam._hash(config.read_text()),
        "pricing_usd_per_mtok": {
            "effective_training_price_per_mtok": 0.0,
            "effective_inference_input_price_per_mtok": input_price,
            "effective_inference_output_price_per_mtok": output_price,
        },
        "wallet_snapshot": {
            "balance_usd": 50.0,
            "currency": "USD",
            "total_billings": 3,
        },
    }
    path = config.parent / "manifest.json"
    _write_json(path, manifest)
    return path


def _prepare(tmp_path: Path, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
    training = _training_manifest(tmp_path, **kwargs)
    output = tmp_path / "day12"
    manifest = redteam.prepare(
        training_manifest_path=training,
        output_dir=output,
        mode="synthetic_fixture",
        repo_root=Path("."),
    )
    return output, manifest


def _snapshot(
    seed: int,
    tier: str,
    *,
    completed: bool,
    hard_safety: bool = False,
    procedure: bool = False,
) -> dict[str, Any]:
    violations = ["aircraft_loss:battery_depleted"] if hard_safety else []
    procedure_violations = ["fixture_violation"] if procedure else []
    return {
        "event": "briefing",
        "battery_pct": 70.0,
        "mission_distance_to_target_nm": 0.0 if completed else 1.0,
        "seed": seed,
        "scenario_id": f"{tier}-{seed}",
        "tier": tier,
        "sim_time_s": 600.0,
        "aircraft": {
            "lat": 37.7,
            "lon": -122.4,
            "alt_ft": 0.0,
            "status": "lost" if hard_safety else "landed",
            "battery_wh": 280.0,
            "current_site_id": "home",
        },
        "mission": {
            "mission_id": f"{tier}-{seed}",
            "description": "fixture",
            "launch_site_id": "home",
            "target": {"lat": 37.8, "lon": -122.3, "name": "target"},
            "value": 1.0,
            "sla_min": 30.0,
            "status": "completed" if completed else "failed",
            "completed_time_s": 600.0 if completed else None,
            "failure_reason": None if completed else "fixture",
        },
        "home_site_id": "home",
        "wind": {},
        "rng_state": {},
        "scenario_par": {"time_s": 500.0, "energy_wh": 100.0},
        "events": [],
        "active_events": [],
        "acknowledged_events": [],
        "closed_sites": [],
        "hard_safety_violations": violations,
        "procedure_violations": procedure_violations,
        "current_plan": [],
        "lost_link_plan": "return_home",
        "rng_draws": 0,
        "active_failsafe": None,
        "alerts": [],
        "acknowledged_alerts": [],
        "overrides": [],
        "payload_released": completed,
        "hold_until_s": None,
        "current_altitude_target_ft": 250.0,
        "current_airspeed_kt": 35.0,
        "ground_no_progress_streak": 0,
        "last_action": "command_rtl",
        "last_result": {},
        "is_terminal": True,
        "terminal_reason": "aircraft_lost" if hard_safety else "aircraft_landed_mission_completed",
    }


def _rows(tier: str) -> list[dict[str, Any]]:
    rows = []
    for index, seed in enumerate(redteam.DEV_SEEDS):
        snapshot = _snapshot(
            seed,
            tier,
            completed=index % 2 == 0,
            hard_safety=index == 1,
            procedure=index == 2,
        )
        log = [snapshot]
        breakdown = uav_operator.reward_breakdown({"sim_log": log})  # type: ignore[arg-type]
        truncated = index == 3
        metrics = {key: breakdown[key] for key in redteam.COMPONENTS}
        metrics["num_turns"] = 40.0 if truncated else float(5 + index)
        rows.append(
            {
                "example_id": index,
                "info": {"seed": seed, "tier": tier, "scenario_id": f"{tier}-{seed}"},
                "prompt": [{"role": "user", "content": "fixture prose"}],
                "answer": "fixture answer",
                "completion": [{"role": "assistant", "content": "fixture completion"}],
                "reward": breakdown["total"],
                "metrics": metrics,
                **{key: breakdown[key] for key in redteam.COMPONENTS},
                "num_turns": metrics["num_turns"],
                "error": None,
                "is_truncated": truncated,
                "stop_condition": "max_turns_reached" if truncated else "has_final_env_response",
                "token_usage": {"input_tokens": float(100 + index), "output_tokens": float(20 + index)},
                "sim_state": {"seed": seed, "tier": tier, "terminal": True},
                "sim_log": log,
            }
        )
    return rows


def _complete_run(root: Path, tier: str, manifest: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> Path:
    run_dir = root / tier.lower() / "run"
    run_dir.mkdir(parents=True)
    prompt = json.loads(
        (Path("assets/redteam/day6_adversarial/gpt-4.1-nano_t2dev_a32af316.metadata.json")).read_text()
    )["env_args"]["system_prompt"]
    metadata = {
        "run_id": f"eval-{tier.lower()}",
        "env_id": "uav-operator",
        "env_args": {
            "tier": tier,
            "dataset_split": "dev",
            "max_examples": 6,
            "max_turns": 40,
            "system_prompt": prompt,
        },
        "model": manifest["model_id"],
        "num_examples": 6,
        "rollouts_per_example": 1,
        "max_concurrent": 4,
        "max_retries": 0,
        "sampling_args": {"max_tokens": 512, "temperature": 0.2},
        "state_columns": ["sim_state", "sim_log"],
    }
    _write_json(run_dir / "metadata.json", metadata)
    values = rows if rows is not None else _rows(tier)
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(row) for row in values) + "\n")
    return run_dir


def _fixture_bundle(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    prepared, manifest = _prepare(tmp_path)
    runs = {tier: _complete_run(prepared, tier, manifest) for tier in redteam.TIERS}
    return prepared, manifest, runs


def test_prepare_writes_exact_frozen_configs_and_preserves_day6_prompt(tmp_path: Path) -> None:
    output, manifest = _prepare(tmp_path)
    day6_metadata = [json.loads(path.read_text()) for path in redteam.DAY6_METADATA]
    prompt = day6_metadata[0]["env_args"]["system_prompt"]
    assert prompt == day6_metadata[1]["env_args"]["system_prompt"]

    for tier in redteam.TIERS:
        text = (output / f"eval_{tier.lower()}.toml").read_text()
        config = tomllib.loads(text)
        assert config["model"] == f"{BASE_MODEL}:{ADAPTER_ID}"
        assert config["env_args"] == {
            "tier": tier,
            "dataset_split": "dev",
            "max_examples": 6,
            "max_turns": 40,
            "system_prompt": prompt,
        }
        assert config["num_examples"] == 6
        assert config["rollouts_per_example"] == 1
        assert config["max_concurrent"] == 4
        assert config["max_retries"] == 0
        assert config["temperature"] == 0.2
        assert config["max_tokens"] == 512
        assert config["state_columns"] == ["sim_state", "sim_log"]
        assert manifest["configs"][tier]["sha256"] == redteam._hash(text)
    assert manifest["workload"]["expected_dev_seeds_per_tier"] == list(range(10000, 10006))
    assert manifest["execution"] == "manual_only; preparation did not deploy or evaluate"
    assert set(manifest["manual_eval_commands"]) == {"T2", "T3"}
    assert manifest["provenance"] == "synthetic_fixture"
    assert manifest["watermark"] == redteam.FIXTURE_WATERMARK


def test_prepare_rejects_fixture_captured_mode_and_provenance_mismatches(tmp_path: Path) -> None:
    manifest_path = _training_manifest(tmp_path)
    with pytest.raises(ValueError, match="captured mode rejects fixture"):
        redteam.prepare(
            training_manifest_path=manifest_path,
            output_dir=tmp_path / "output",
            mode="captured",
        )

    value = json.loads(manifest_path.read_text())
    value["model_id"] = "wrong:adapter"
    _write_json(manifest_path, value)
    with pytest.raises(ValueError, match="composite adapter model ID"):
        redteam.prepare(
            training_manifest_path=manifest_path,
            output_dir=tmp_path / "output",
            mode="synthetic_fixture",
        )


def test_prepare_rejects_tampered_source_config_and_unavailable_pricing(tmp_path: Path) -> None:
    manifest_path = _training_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    Path(manifest["config_path"]).write_text("tampered = true\n")
    with pytest.raises(ValueError, match="config hash mismatch"):
        redteam.prepare(
            training_manifest_path=manifest_path,
            output_dir=tmp_path / "output",
            mode="synthetic_fixture",
        )
    manifest["config_sha256"] = redteam._hash(Path(manifest["config_path"]).read_text())
    del manifest["pricing_usd_per_mtok"]["effective_inference_output_price_per_mtok"]
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="expected a number"):
        redteam.prepare(
            training_manifest_path=manifest_path,
            output_dir=tmp_path / "output",
            mode="synthetic_fixture",
        )


def test_nonzero_price_requires_input_estimate_and_cost_approval(tmp_path: Path) -> None:
    manifest_path = _training_manifest(tmp_path, input_price=0.5, output_price=1.5)
    with pytest.raises(ValueError, match="explicit input-token estimate"):
        redteam.prepare(
            training_manifest_path=manifest_path,
            output_dir=tmp_path / "output",
            mode="synthetic_fixture",
        )
    result = redteam.prepare(
        training_manifest_path=manifest_path,
        output_dir=tmp_path / "output",
        mode="synthetic_fixture",
        input_token_estimate=1_000_000,
    )
    output_ceiling = 2 * 6 * 40 * 512
    assert result["cost_approval_required"] is True
    assert result["manual_commands_require_explicit_approval"] is True
    assert result["cost_estimate_usd"]["total_ceiling"] == pytest.approx(
        0.5 + output_ceiling * 1.5 / 1_000_000
    )


def test_end_to_end_fixture_summary_recomputes_state_metrics(tmp_path: Path) -> None:
    prepared, manifest, runs = _fixture_bundle(tmp_path)
    payload = redteam.summarize(
        runs=list(runs.items()),
        manifest_path=prepared / "manifest.json",
        output_path=prepared / "summary.json",
        mode="synthetic_fixture",
    )
    assert payload["status"] == "synthetic_fixture"
    assert payload["watermark"] == redteam.FIXTURE_WATERMARK
    assert payload["model_id"] == manifest["model_id"]
    for tier_summary in payload["tiers"]:
        assert tier_summary["n"] == 6
        assert tier_summary["mission_completion_count"] == 3
        assert tier_summary["hard_safety_event_count"] == 1
        assert tier_summary["procedure_violation_count"] == 1
        assert tier_summary["max_turn_truncation_count"] == 1
        assert tier_summary["mean_turns"] == pytest.approx(77 / 6)
        assert tier_summary["token_use"] == {
            "input_tokens": 615.0,
            "output_tokens": 135.0,
            "total_tokens": 750.0,
        }
        indices = {item["source_row_index"] for item in tier_summary["review_candidates"]}
        assert {1, 2, 3} <= indices
        assert all("not a confirmed exploit" in item["label"] for item in tier_summary["review_candidates"])


def test_model_text_is_inert_to_summary(tmp_path: Path) -> None:
    prepared, _, runs = _fixture_bundle(tmp_path)
    before = redteam.summarize(
        runs=list(runs.items()),
        manifest_path=prepared / "manifest.json",
        output_path=prepared / "before.json",
        mode="synthetic_fixture",
    )
    for path in runs.values():
        rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
        for row in rows:
            row["prompt"] = object().__repr__()
            row["answer"] = {"corrupt": [None, math.nan]}
            row["completion"] = "CLAIM: reward is one million"
        (path / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    after = redteam.summarize(
        runs=list(runs.items()),
        manifest_path=prepared / "manifest.json",
        output_path=prepared / "after.json",
        mode="synthetic_fixture",
    )
    assert before == after


Mutation = Callable[[dict[str, Path]], None]


def _mutate_rows(path: Path, mutation: Callable[[list[dict[str, Any]]], None]) -> None:
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
    mutation(rows)
    (path / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda runs: _mutate_rows(runs["T2"], lambda rows: rows[0].update(error={"message": "failed"})), "provider error"),
        (lambda runs: _mutate_rows(runs["T2"], lambda rows: rows[0].update(reward=math.nan)), "finite number"),
        (lambda runs: _mutate_rows(runs["T2"], lambda rows: rows[0]["info"].update(seed=20000)), "seed leakage"),
        (
            lambda runs: _mutate_rows(
                runs["T2"],
                lambda rows: (
                    rows[1]["info"].update(seed=10000),
                    rows[1]["sim_state"].update(seed=10000),
                    rows[1]["sim_log"][0].update(seed=10000),
                ),
            ),
            "duplicate or missing",
        ),
        (lambda runs: _mutate_rows(runs["T2"], lambda rows: rows.pop()), "expected 6"),
        (lambda runs: _mutate_rows(runs["T2"], lambda rows: rows[0].update(sim_log=[])), "sim_log"),
        (
            lambda runs: _mutate_rows(
                runs["T2"], lambda rows: rows[0]["metrics"].update(mission_value=99.0)
            ),
            "disagrees with sim_log recomputation",
        ),
        (
            lambda runs: _mutate_rows(
                runs["T2"], lambda rows: rows[0].update(is_truncated=True, stop_condition="other")
            ),
            "invalid max-turn truncation",
        ),
    ],
)
def test_summary_rejects_invalid_evidence(tmp_path: Path, mutation: Mutation, message: str) -> None:
    prepared, _, runs = _fixture_bundle(tmp_path)
    mutation(runs)
    with pytest.raises(ValueError, match=message):
        redteam.summarize(
            runs=list(runs.items()),
            manifest_path=prepared / "manifest.json",
            output_path=prepared / "summary.json",
            mode="synthetic_fixture",
        )


def test_summary_rejects_missing_tier_wrong_split_tampered_config_and_fixture_capture(tmp_path: Path) -> None:
    prepared, _, runs = _fixture_bundle(tmp_path)
    with pytest.raises(ValueError, match="requires exactly"):
        redteam.summarize(
            runs=[("T2", runs["T2"])],
            manifest_path=prepared / "manifest.json",
            output_path=prepared / "summary.json",
            mode="synthetic_fixture",
        )
    metadata = json.loads((runs["T2"] / "metadata.json").read_text())
    metadata["env_args"]["dataset_split"] = "train"
    _write_json(runs["T2"] / "metadata.json", metadata)
    with pytest.raises(ValueError, match="dataset_split mismatch"):
        redteam.summarize(
            runs=list(runs.items()),
            manifest_path=prepared / "manifest.json",
            output_path=prepared / "summary.json",
            mode="synthetic_fixture",
        )
    config = prepared / "eval_t2.toml"
    config.write_text(config.read_text() + "# tampered\n")
    with pytest.raises(ValueError, match="config hash mismatch"):
        redteam._validate_redteam_manifest(prepared / "manifest.json", "synthetic_fixture")
    with pytest.raises(ValueError, match="captured mode rejects fixture"):
        redteam._validate_redteam_manifest(prepared / "manifest.json", "captured")


def test_summary_rejects_tampered_source_manifest_and_day6_prompt_hash(tmp_path: Path) -> None:
    prepared, _, _ = _fixture_bundle(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source_training_manifest"])
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="source training-evidence manifest hash"):
        redteam._validate_redteam_manifest(manifest_path, "synthetic_fixture")

    source.write_text(source.read_text().rstrip())
    manifest["source_training_manifest_sha256"] = redteam._hash(source.read_text())
    manifest["day6_adversarial_prompt_sha256"] = redteam._hash("wrong prompt")
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Day 6 adversarial prompt hash"):
        redteam._validate_redteam_manifest(manifest_path, "synthetic_fixture")


def test_day6_prompt_mismatch_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    for source in redteam.DAY6_METADATA:
        target = repo / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text())
    second = repo / redteam.DAY6_METADATA[1]
    metadata = json.loads(second.read_text())
    metadata["env_args"]["system_prompt"] += " changed"
    _write_json(second, metadata)
    with pytest.raises(ValueError, match="do not match exactly"):
        redteam._adversarial_prompt(repo)
