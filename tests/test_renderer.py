from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.render import (
    RenderConfig,
    RenderError,
    activated_events,
    battery_level,
    event_activations_at,
    render_artifact,
    segment_points,
    terminal_label,
    trajectory_through,
    validate_artifact,
    wind_at,
)


def _snapshot(*, event: str, time_s: float, terminal: bool = False) -> dict:
    tfr = {
        "event_id": "tfr-1",
        "type": "TFR_POPUP",
        "trigger_time_s": 30.0,
        "status": "active" if time_s >= 30.0 else "pending",
        "message": "Pop-up restriction at mission intake.",
        "params": {
            "zone_id": "test-tfr",
            "name": "Test TFR",
            "polygon": [
                [37.78, -122.40],
                [37.79, -122.40],
                [37.79, -122.38],
                [37.78, -122.38],
            ],
            "floor_ft": 0.0,
            "ceiling_ft": 2000.0,
        },
    }
    if time_s >= 30.0:
        tfr["activated_time_s"] = 30.0
    lat = 37.7707 if time_s < 60.0 else 37.823
    lon = -122.3869 if time_s < 60.0 else -122.371
    return {
        "event": event,
        "battery_pct": 100.0 if time_s == 0.0 else 55.0,
        "mission_distance_to_target_nm": 3.2 if time_s < 60.0 else 0.0,
        "seed": 7,
        "scenario_id": "T3-test-treasure-island",
        "tier": "T3",
        "sim_time_s": time_s,
        "aircraft": {
            "lat": lat,
            "lon": lon,
            "alt_ft": 0.0 if terminal or time_s == 0.0 else 200.0,
            "status": "landed" if terminal else "ground" if time_s == 0.0 else "holding",
            "battery_wh": 400.0 if time_s == 0.0 else 220.0,
            "current_site_id": "mission_bay" if terminal or time_s == 0.0 else None,
        },
        "mission": {
            "mission_id": "test-mission",
            "description": "Inspect Treasure Island.",
            "launch_site_id": "mission_bay",
            "target": {"lat": 37.823, "lon": -122.371, "name": "Treasure Island"},
            "value": 1.0,
            "sla_min": 30.0,
            "status": "completed" if terminal else "pending",
            "completed_time_s": 60.0 if terminal else None,
            "failure_reason": None,
        },
        "home_site_id": "mission_bay",
        "wind": {
            "base_dir_deg_from": 240.0,
            "base_speed_kt": 12.0,
            "blobs": [],
            "gust_front": None,
        },
        "rng_state": {},
        "scenario_par": {"time_s": 900.0, "energy_wh": 80.0},
        "events": [tfr],
        "active_events": ["tfr-1"] if time_s >= 30.0 else [],
        "acknowledged_events": [],
        "closed_sites": [],
        "hard_safety_violations": [],
        "procedure_violations": [],
        "current_plan": [{"lat": 37.823, "lon": -122.371, "name": "target"}],
        "lost_link_plan": "RTL",
        "rng_draws": 0,
        "active_failsafe": None,
        "alerts": (
            [
                {
                    "alert_id": "tfr-1",
                    "sim_time_s": 30.0,
                    "message": "TFR active",
                    "metadata": {"event_type": "TFR_POPUP"},
                }
            ]
            if time_s >= 30.0
            else []
        ),
        "acknowledged_alerts": [],
        "overrides": [],
        "payload_released": terminal,
        "hold_until_s": None,
        "current_altitude_target_ft": 200.0,
        "current_airspeed_kt": 35.0,
        "ground_no_progress_streak": 0,
        "last_action": event,
        "last_result": {},
        "is_terminal": terminal,
        "terminal_reason": "aircraft_landed_mission_completed" if terminal else None,
    }


@pytest.fixture
def artifact() -> dict:
    first = _snapshot(event="briefing", time_s=0.0)
    second = _snapshot(event="file_flight_plan", time_s=30.0)
    second["last_result"] = {
        "segments": [
            {
                "from": {"lat": 37.7707, "lon": -122.3869, "alt_ft": 0.0},
                "to": {"lat": 37.79, "lon": -122.38, "alt_ft": 200.0},
                "distance_nm": 1.2,
                "track_deg": 10.0,
                "groundspeed_kt": 30.0,
                "wind": {"dir_deg_from": 240.0, "speed_kt": 12.0},
                "execution_multiplier": 1.0,
                "duration_s": 30.0,
                "energy_wh": 5.0,
                "interrupted_by_event": True,
                "progress_fraction": 0.5,
            }
        ]
    }
    final = _snapshot(event="command_rtl", time_s=90.0, terminal=True)
    return {"reward": 0.865, "metrics": {"hard_safety": 0.0}, "sim_log": [first, second, final]}


def test_schema_validation_returns_structured_paths(artifact: dict) -> None:
    assert validate_artifact(artifact) is artifact
    broken = deepcopy(artifact)
    del broken["sim_log"][1]["aircraft"]
    with pytest.raises(RenderError) as exc:
        validate_artifact(broken)
    assert exc.value.code == "invalid_snapshot"
    assert exc.value.path == "$.sim_log[1]"


def test_segment_and_trajectory_reconstruction(artifact: dict) -> None:
    assert segment_points(artifact["sim_log"][1]) == [
        (-122.3869, 37.7707),
        (-122.38, 37.79),
    ]
    assert trajectory_through(artifact["sim_log"], 1)[-1] == (-122.38, 37.79)


def test_event_activation_drives_tfr_visibility(artifact: dict) -> None:
    log = artifact["sim_log"]
    assert activated_events(log, 0, event_type="TFR_POPUP") == []
    assert [item["event_id"] for item in activated_events(log, 1, event_type="TFR_POPUP")] == [
        "tfr-1"
    ]
    assert event_activations_at(log, 1) == ["TFR_POPUP"]


def test_wind_sampling_and_battery_thresholds(artifact: dict) -> None:
    direction, speed = wind_at(37.8, -122.38, 0.0, 0.0, artifact["sim_log"][0]["wind"])
    assert direction == pytest.approx(240.0)
    assert speed == pytest.approx(12.0)
    assert battery_level(31.0)[0] == "NOMINAL"
    assert battery_level(30.0)[0] == "LOW / RTL"
    assert battery_level(12.0)[0] == "CRITICAL"


def test_terminal_state_labels_success_and_loss(artifact: dict) -> None:
    assert terminal_label(artifact["sim_log"][-1])[0].startswith("MISSION COMPLETE")
    lost = deepcopy(artifact["sim_log"][-1])
    lost["aircraft"]["status"] = "lost"
    lost["terminal_reason"] = "aircraft_lost_battery_depleted"
    assert terminal_label(lost)[0] == "AIRCRAFT LOST"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_no_basemap_low_resolution_mp4_and_gif(tmp_path: Path, artifact: dict) -> None:
    config = RenderConfig(
        width=320,
        height=180,
        fps=2,
        briefing_s=0.5,
        body_s=1.0,
        terminal_s=0.5,
        dpi=80,
    )
    mp4 = tmp_path / "rollout.mp4"
    gif = tmp_path / "rollout.gif"
    render_artifact(artifact, mp4, no_basemap=True, config=config)
    render_artifact(artifact, gif, no_basemap=True, config=config)
    assert mp4.stat().st_size > 1_000
    assert gif.stat().st_size > 1_000
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height",
            "-of",
            "json",
            str(mp4),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {"codec_name": "h264", "width": 320, "height": 180, "pix_fmt": "yuv420p"}
