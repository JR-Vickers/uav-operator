from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_day5_scores import _summarize_run
from scripts.run_frontier_calibration import _compact_results, _parse_resume_tiers


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_compact_results_archives_errors_and_preserves_per_example_quota(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_rows(
        run_dir / "results.jsonl",
        [
            {"example_id": 0, "reward": 1.0, "error": None},
            {"example_id": 0, "reward": 0.0, "error": {"error": "ModelError"}},
            {"example_id": 0, "reward": 0.5, "error": None},
            {"example_id": 0, "reward": 0.4, "error": None},
            {"example_id": 1, "reward": -0.1, "error": None, "is_truncated": True},
        ],
    )

    counts = _compact_results(run_dir, rollouts_per_example=2)

    assert counts == {
        "clean": 3,
        "failed_this_attempt": 1,
        "overflow_this_attempt": 1,
    }
    retained = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert [row["example_id"] for row in retained] == [0, 0, 1]
    assert retained[-1]["is_truncated"] is True
    assert len((run_dir / "failed_results.jsonl").read_text().splitlines()) == 1
    assert len((run_dir / "overflow_results.jsonl").read_text().splitlines()) == 1


def test_parse_resume_tiers() -> None:
    assert _parse_resume_tiers(["T0=/tmp/t0", "T3=relative/t3"]) == {
        "T0": Path("/tmp/t0"),
        "T3": Path("relative/t3"),
    }


def test_plot_summary_rejects_provider_errors(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "env_args": {"tier": "T2"},
                "model": "test/model",
                "run_id": "run",
            }
        )
    )
    _write_rows(
        run_dir / "results.jsonl",
        [
            {
                "example_id": 0,
                "reward": 0.0,
                "error": {"error": "ModelError"},
                "num_turns": 1,
            }
        ],
    )

    with pytest.raises(ValueError, match="1 provider-error rollout"):
        _summarize_run(run_dir)
