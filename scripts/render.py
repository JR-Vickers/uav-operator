#!/usr/bin/env python3
"""Render a frozen uav-operator rollout without replaying the simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
WORLD_PATH = ROOT / "data" / "world.json"
DEFAULT_CACHE = ROOT / ".cache" / "uav-renderer" / "mission-dark.tif"

BG = "#071019"
PANEL = "#0d1924"
GRID = "#294052"
MUTED = "#8ba0ae"
TEXT = "#e5f1f7"
CYAN = "#41d9ff"
GREEN = "#5ef2a4"
AMBER = "#ffc857"
RED = "#ff5263"


class RenderError(ValueError):
    """A structured, user-correctable rollout rendering error."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path

    def payload(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class RenderConfig:
    """Output controls; production defaults match the public artifact contract."""

    width: int = 1920
    height: int = 1080
    fps: int = 30
    briefing_s: float = 2.0
    body_s: float = 18.0
    terminal_s: float = 2.0
    dpi: int = 100


REQUIRED_SNAPSHOT_KEYS = {
    "event",
    "battery_pct",
    "scenario_id",
    "tier",
    "sim_time_s",
    "aircraft",
    "mission",
    "home_site_id",
    "wind",
    "events",
    "active_events",
    "current_plan",
    "active_failsafe",
    "alerts",
    "last_action",
    "last_result",
    "is_terminal",
    "terminal_reason",
}
REQUIRED_AIRCRAFT_KEYS = {"lat", "lon", "alt_ft", "status", "battery_wh"}
REQUIRED_MISSION_KEYS = {"mission_id", "target", "status"}


def load_artifact(path: Path) -> dict[str, Any]:
    """Load and validate one compact rollout artifact."""

    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RenderError("input_not_found", f"Input does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RenderError(
            "invalid_json", f"Invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    return validate_artifact(data)


def validate_artifact(data: Any) -> dict[str, Any]:
    """Validate the renderer-facing subset of frozen sim_log schema v1."""

    if not isinstance(data, dict):
        raise RenderError("invalid_artifact", "Top-level JSON value must be an object")
    log = data.get("sim_log")
    if not isinstance(log, list) or not log:
        raise RenderError("missing_sim_log", "sim_log must be a non-empty list", "$.sim_log")
    last_time = -math.inf
    for index, snapshot in enumerate(log):
        path = f"$.sim_log[{index}]"
        if not isinstance(snapshot, dict):
            raise RenderError("invalid_snapshot", "Snapshot must be an object", path)
        missing = sorted(REQUIRED_SNAPSHOT_KEYS - snapshot.keys())
        if missing:
            raise RenderError(
                "invalid_snapshot", f"Missing required keys: {', '.join(missing)}", path
            )
        aircraft = snapshot["aircraft"]
        if not isinstance(aircraft, dict) or not REQUIRED_AIRCRAFT_KEYS <= aircraft.keys():
            raise RenderError("invalid_aircraft", "Malformed aircraft state", f"{path}.aircraft")
        mission = snapshot["mission"]
        if not isinstance(mission, dict) or not REQUIRED_MISSION_KEYS <= mission.keys():
            raise RenderError("invalid_mission", "Malformed mission state", f"{path}.mission")
        target = mission.get("target")
        if not isinstance(target, dict) or not {"lat", "lon", "name"} <= target.keys():
            raise RenderError("invalid_mission", "Malformed mission target", f"{path}.mission.target")
        try:
            now = float(snapshot["sim_time_s"])
            float(aircraft["lat"])
            float(aircraft["lon"])
            float(snapshot["battery_pct"])
        except (TypeError, ValueError) as exc:
            raise RenderError("invalid_number", "State coordinates/time must be numeric", path) from exc
        if now < last_time:
            raise RenderError("nonmonotonic_log", "sim_time_s must be nondecreasing", path)
        last_time = now
    if log[0]["event"] != "briefing":
        raise RenderError("invalid_log_start", "First snapshot must be briefing", "$.sim_log[0]")
    return data


def load_world() -> dict[str, Any]:
    """Read bundled static world data."""

    return json.loads(WORLD_PATH.read_text())


def segment_points(snapshot: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Return flown points recorded in one snapshot's result."""

    result = snapshot.get("last_result")
    if not isinstance(result, Mapping):
        return []
    raw: list[Any] = []
    segments = result.get("segments")
    if isinstance(segments, list):
        raw.extend(segments)
    segment = result.get("segment")
    if isinstance(segment, Mapping):
        raw.append(segment)
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, Mapping) or "failsafe" in item or "error" in item:
            continue
        for endpoint in (item.get("from"), item.get("to")):
            if isinstance(endpoint, Mapping) and "lat" in endpoint and "lon" in endpoint:
                point = (float(endpoint["lon"]), float(endpoint["lat"]))
                if not points or point != points[-1]:
                    points.append(point)
    return points


def trajectory_through(log: Sequence[Mapping[str, Any]], index: int) -> list[tuple[float, float]]:
    """Reconstruct the high-fidelity flown breadcrumb through a snapshot."""

    first = log[0]["aircraft"]
    points = [(float(first["lon"]), float(first["lat"]))]
    for snapshot in log[1 : index + 1]:
        additions = segment_points(snapshot)
        if additions:
            for point in additions:
                if point != points[-1]:
                    points.append(point)
        else:
            aircraft = snapshot["aircraft"]
            point = (float(aircraft["lon"]), float(aircraft["lat"]))
            if point != points[-1]:
                points.append(point)
    return points


def event_catalog(log: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge event records so activation metadata survives acknowledgements."""

    catalog: dict[str, dict[str, Any]] = {}
    for snapshot in log:
        for event in snapshot.get("events", []):
            if not isinstance(event, Mapping) or "event_id" not in event:
                continue
            merged = catalog.setdefault(str(event["event_id"]), {})
            merged.update(event)
    return catalog


def activated_events(
    log: Sequence[Mapping[str, Any]], index: int, *, event_type: str | None = None
) -> list[dict[str, Any]]:
    """Return events whose logged activation time is visible at this snapshot."""

    now = float(log[index]["sim_time_s"])
    active: list[dict[str, Any]] = []
    for event in event_catalog(log).values():
        activated = event.get("activated_time_s")
        if activated is None:
            continue
        if float(activated) <= now and (event_type is None or event.get("type") == event_type):
            active.append(event)
    return active


def event_activations_at(log: Sequence[Mapping[str, Any]], index: int) -> list[str]:
    """List event types that became visible in exactly this snapshot."""

    current = {item["event_id"]: item for item in activated_events(log, index)}
    previous = (
        {item["event_id"]: item for item in activated_events(log, index - 1)}
        if index > 0
        else {}
    )
    return [str(item.get("type", "EVENT")) for key, item in current.items() if key not in previous]


def battery_level(pct: float) -> tuple[str, str]:
    """Return threshold label and console color for battery percentage."""

    if pct <= 12.0:
        return "CRITICAL", RED
    if pct <= 30.0:
        return "LOW / RTL", AMBER
    return "NOMINAL", GREEN


def terminal_label(snapshot: Mapping[str, Any]) -> tuple[str, str]:
    """Map logged terminal state to a concise outcome."""

    reason = str(snapshot.get("terminal_reason") or "")
    mission = str(snapshot["mission"].get("status", "pending"))
    if "lost" in reason or snapshot["aircraft"].get("status") == "lost":
        return "AIRCRAFT LOST", RED
    if mission == "completed":
        return "MISSION COMPLETE · AIRCRAFT SAFE", GREEN
    if mission == "failed":
        return "MISSION FAILED · AIRCRAFT SAFE", AMBER
    return reason.replace("_", " ").upper() or "EPISODE ENDED", TEXT


def _move_latlon(lat: float, lon: float, bearing_deg: float, distance_nm: float) -> tuple[float, float]:
    bearing = math.radians(bearing_deg)
    dlat = distance_nm * math.cos(bearing) / 60.0
    cos_lat = max(0.1, math.cos(math.radians(lat)))
    dlon = distance_nm * math.sin(bearing) / (60.0 * cos_lat)
    return lat + dlat, lon + dlon


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    value = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 3440.065 * 2.0 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1.0 - value)))


def wind_at(
    lat: float, lon: float, alt_ft: float, t_s: float, wind: Mapping[str, Any]
) -> tuple[float, float]:
    """Reconstruct deterministic logged wind without importing simulator code."""

    def vector(direction: float, speed: float) -> tuple[float, float]:
        toward = math.radians((direction + 180.0) % 360.0)
        return speed * math.sin(toward), speed * math.cos(toward)

    east, north = vector(float(wind["base_dir_deg_from"]), float(wind["base_speed_kt"]))
    for blob in wind.get("blobs", []):
        clat, clon = _move_latlon(
            float(blob["center_lat"]),
            float(blob["center_lon"]),
            float(blob["drift_bearing_deg"]),
            float(blob["drift_speed_kt"]) * t_s / 3600.0,
        )
        radius = float(blob["radius_nm"])
        weight = math.exp(-0.5 * (_haversine_nm(clat, clon, lat, lon) / radius) ** 2)
        de, dn = vector(
            float(wind["base_dir_deg_from"]) + float(blob["dir_delta_deg"]),
            float(blob["speed_delta_kt"]),
        )
        east += weight * de
        north += weight * dn
    gust = wind.get("gust_front")
    if isinstance(gust, Mapping):
        bearing = math.radians(float(gust["movement_bearing_deg"]))
        dlat = (lat - float(gust["anchor_lat"])) * 60.0
        dlon = (lon - float(gust["anchor_lon"])) * 60.0 * math.cos(math.radians(lat))
        signed = dlat * math.cos(bearing) + dlon * math.sin(bearing)
        arrival = float(gust["forecast_eta_s"]) + float(gust["eta_error_s"])
        moved = float(gust["speed_kt"]) * (t_s - arrival) / 3600.0
        if signed <= moved:
            de, dn = vector(
                float(wind["base_dir_deg_from"]) + float(gust["dir_shift_deg"]),
                float(gust["speed_delta_kt"]),
            )
            east += de
            north += dn
    speed = math.hypot(east, north)
    toward = math.degrees(math.atan2(east, north)) % 360.0
    direction = (toward + 180.0) % 360.0
    alt_kft = max(0.0, alt_ft) / 1000.0
    return (direction + 5.0 * alt_kft) % 360.0, speed * (1.0 + 0.15 * alt_kft)


def wind_grid(snapshot: Mapping[str, Any], extent: tuple[float, float, float, float]) -> tuple[np.ndarray, ...]:
    """Sample the logged wind field on a small display grid."""

    west, east, south, north = extent
    xs, ys = np.meshgrid(np.linspace(west, east, 8), np.linspace(south, north, 6))
    us = np.zeros_like(xs)
    vs = np.zeros_like(ys)
    for row, col in np.ndindex(xs.shape):
        direction, speed = wind_at(
            float(ys[row, col]),
            float(xs[row, col]),
            float(snapshot["aircraft"]["alt_ft"]),
            float(snapshot["sim_time_s"]),
            snapshot["wind"],
        )
        toward = math.radians((direction + 180.0) % 360.0)
        us[row, col] = math.sin(toward) * speed
        vs[row, col] = math.cos(toward) * speed
    return xs, ys, us, vs


def frame_schedule(log_length: int, config: RenderConfig) -> list[int]:
    """Build briefing/body/terminal holds with event-driven snapshot timing."""

    briefing = max(1, round(config.briefing_s * config.fps))
    terminal = max(1, round(config.terminal_s * config.fps))
    body = max(1, round(config.body_s * config.fps))
    if log_length == 1:
        return [0] * (briefing + body + terminal)
    indices = np.linspace(1, log_length - 1, body).astype(int).tolist()
    return [0] * briefing + indices + [log_length - 1] * terminal


def _map_extent(log: Sequence[Mapping[str, Any]]) -> tuple[float, float, float, float]:
    points: list[tuple[float, float]] = []
    for snapshot in log:
        aircraft = snapshot["aircraft"]
        points.append((float(aircraft["lon"]), float(aircraft["lat"])))
        target = snapshot["mission"]["target"]
        points.append((float(target["lon"]), float(target["lat"])))
        for waypoint in snapshot.get("current_plan", []):
            if isinstance(waypoint, Mapping) and "lon" in waypoint and "lat" in waypoint:
                points.append((float(waypoint["lon"]), float(waypoint["lat"])))
    for event in event_catalog(log).values():
        if event.get("type") == "TFR_POPUP":
            for lat, lon in event.get("params", {}).get("polygon", []):
                points.append((float(lon), float(lat)))
    lons, lats = zip(*points, strict=True)
    lon_pad = max(0.007, (max(lons) - min(lons)) * 0.12)
    lat_pad = max(0.006, (max(lats) - min(lats)) * 0.12)
    return min(lons) - lon_pad, max(lons) + lon_pad, min(lats) - lat_pad, max(lats) + lat_pad


def _add_basemap(ax: Axes, extent: tuple[float, float, float, float], cache: Path) -> None:
    try:
        import contextily as cx
    except ImportError as exc:
        raise RenderError(
            "basemap_dependency_missing",
            "Contextily is required for basemaps; run with --no-basemap or install dev dependencies",
        ) from exc
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(
        ",".join(f"{coordinate:.5f}" for coordinate in extent).encode()
    ).hexdigest()[:12]
    raster_cache = cache.with_name(f"{cache.stem}-{cache_key}{cache.suffix}")
    if not raster_cache.exists():
        west, east, south, north = extent
        try:
            cx.bounds2raster(
                west,
                south,
                east,
                north,
                str(raster_cache),
                zoom=12,
                source=cx.providers.CartoDB.DarkMatterNoLabels,
                ll=True,
                use_cache=True,
            )
        except Exception as exc:  # network/provider failures become actionable CLI errors
            raise RenderError(
                "basemap_unavailable",
                f"Could not download basemap tiles ({exc}); retry with --no-basemap",
            ) from exc
    cx.add_basemap(
        ax,
        source=str(raster_cache),
        crs="EPSG:4326",
        attribution="© OpenStreetMap contributors © CARTO",
        attribution_size=5,
        reset_extent=False,
    )


def _style_axis(ax: Axes) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)


def _draw_static_map(
    ax: Axes,
    world: Mapping[str, Any],
    log: Sequence[Mapping[str, Any]],
    extent: tuple[float, float, float, float],
    *,
    no_basemap: bool,
    basemap_cache: Path,
) -> dict[str, Any]:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / max(0.1, math.cos(math.radians(sum(extent[2:]) / 2.0))))
    if not no_basemap:
        _add_basemap(ax, extent, basemap_cache)
        # A reusable full-Bay raster must not dictate the mission viewport.
        # Contextily expands axes to its source bounds, so restore the tight
        # rollout-derived extent after drawing it.
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        ax.grid(color=GRID, alpha=0.35, linewidth=0.5)
        ax.text(
            0.012,
            0.018,
            "OFFLINE VECTOR MODE · WGS84",
            transform=ax.transAxes,
            color=MUTED,
            fontsize=6,
        )
    for zone in world["airspace"]:
        coords = [(lon, lat) for lat, lon in zone["polygon"]]
        auth = bool(zone["authorization_required"])
        ax.add_patch(
            Polygon(
                coords,
                closed=True,
                facecolor=RED if auth else CYAN,
                edgecolor=RED if auth else CYAN,
                alpha=0.055 if auth else 0.035,
                linewidth=0.7,
                linestyle="--",
                zorder=2,
            )
        )
    first = log[0]
    home_id = first["home_site_id"]
    for site in world["sites"]:
        if site["site_id"] == home_id:
            ax.scatter(site["lon"], site["lat"], marker="s", s=65, color=CYAN, zorder=8)
            ax.text(site["lon"], site["lat"], "  HOME", color=CYAN, fontsize=7, zorder=8)
    target = first["mission"]["target"]
    ax.scatter(target["lon"], target["lat"], marker="*", s=150, color=AMBER, zorder=8)
    ax.text(target["lon"], target["lat"], f"  {target['name']}", color=AMBER, fontsize=7, zorder=8)
    tfr_patches: dict[str, Polygon] = {}
    for event_id, event in event_catalog(log).items():
        if event.get("type") != "TFR_POPUP":
            continue
        coords = [(lon, lat) for lat, lon in event.get("params", {}).get("polygon", [])]
        patch = Polygon(
            coords,
            closed=True,
            facecolor=RED,
            edgecolor="#ff9aa5",
            alpha=0.3,
            linewidth=2.0,
            hatch="////",
            visible=False,
            zorder=6,
        )
        ax.add_patch(patch)
        tfr_patches[event_id] = patch
    planned_line, = ax.plot([], [], color=AMBER, linewidth=1.3, linestyle="--", alpha=0.8, zorder=5)
    trail_line, = ax.plot([], [], color=CYAN, linewidth=3.0, alpha=0.95, zorder=7)
    aircraft, = ax.plot([], [], marker="^", markersize=10, color=TEXT, markeredgecolor=CYAN, zorder=9)
    xs, ys, us, vs = wind_grid(first, extent)
    quiver = ax.quiver(xs, ys, us, vs, color=CYAN, alpha=0.38, scale=220, width=0.002, zorder=3)
    ax.set_title("LIVE OPERATIONS MAP", loc="left", color=TEXT, fontsize=10, fontweight="bold")
    return {
        "planned": planned_line,
        "trail": trail_line,
        "aircraft": aircraft,
        "quiver": quiver,
        "tfrs": tfr_patches,
    }


def _event_summary(snapshot: Mapping[str, Any]) -> str:
    active = []
    for alert in snapshot.get("alerts", []):
        if not isinstance(alert, Mapping):
            continue
        event_type = alert.get("metadata", {}).get("event_type")
        if event_type and event_type not in active:
            active.append(str(event_type))
    return "  ·  ".join(active[-4:]) if active else "MONITORING · NO ACTIVE DISRUPTIONS"


def _reward_text(artifact: Mapping[str, Any]) -> str:
    reward = artifact.get("reward")
    if not isinstance(reward, (int, float)):
        return "REWARD  —"
    return f"REWARD  {float(reward):+.3f}"


def _decision_label(log: Sequence[Mapping[str, Any]]) -> str:
    actions = [str(item.get("last_action", "")) for item in log]
    override_count = actions.count("override_failsafe")
    filing_count = actions.count("file_flight_plan")
    if "lost" in str(log[-1].get("terminal_reason")) and "land_now" in actions:
        return "LAND NOW · TOO LATE"
    if override_count >= 2:
        return f"{override_count}× OVERRIDE + {filing_count}× REFILE"
    if "amend_route" in actions:
        return "AMEND ROUTE"
    if "land_now" in actions:
        return "LAND NOW"
    if "command_rtl" in actions:
        return "COMMAND RTL"
    return "MONITOR"


def _cold_summary(log: Sequence[Mapping[str, Any]]) -> str:
    events = [str(item.get("type", "EVENT")).replace("_", " ") for item in event_catalog(log).values()]
    outcome, _ = terminal_label(log[-1])
    return f"MISSION  →  {' + '.join(events)}  →  {_decision_label(log)}  →  {outcome}"


def render_artifact(
    artifact: Mapping[str, Any],
    output: Path,
    *,
    no_basemap: bool = False,
    config: RenderConfig | None = None,
    basemap_cache: Path = DEFAULT_CACHE,
) -> None:
    """Render one validated artifact to MP4 or GIF."""

    data = validate_artifact(dict(artifact))
    config = config or (
        RenderConfig(width=960, height=540, fps=12)
        if output.suffix.lower() == ".gif"
        else RenderConfig()
    )
    if output.suffix.lower() not in {".mp4", ".gif"}:
        raise RenderError("unsupported_output", "Output extension must be .mp4 or .gif", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, Any]] = data["sim_log"]
    world = load_world()
    extent = _map_extent(log)
    fig = plt.figure(
        figsize=(config.width / config.dpi, config.height / config.dpi),
        dpi=config.dpi,
        facecolor=BG,
    )
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(3.2, 1.15),
        height_ratios=(0.18, 1.0, 0.14),
        left=0.025,
        right=0.98,
        bottom=0.045,
        top=0.96,
        wspace=0.025,
        hspace=0.035,
    )
    header = fig.add_subplot(grid[0, :])
    map_ax = fig.add_subplot(grid[1, 0])
    hud = fig.add_subplot(grid[1, 1])
    footer = fig.add_subplot(grid[2, :])
    for ax in (header, map_ax, hud, footer):
        _style_axis(ax)
    for ax in (header, hud, footer):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    artists = _draw_static_map(
        map_ax,
        world,
        log,
        extent,
        no_basemap=no_basemap,
        basemap_cache=basemap_cache,
    )
    scenario = log[0]["scenario_id"]
    header.text(0.018, 0.62, "UAV OPS // BAY CONTROL", color=CYAN, fontsize=14, fontweight="bold")
    header.text(0.018, 0.2, f"{scenario}  ·  {log[0]['tier']} COMPOSED EVENTS", color=MUTED, fontsize=8)
    header_status = header.text(0.97, 0.5, "MISSION BRIEFING", color=AMBER, fontsize=12, ha="right", va="center", fontweight="bold")

    hud.text(0.06, 0.95, "AIRCRAFT", color=MUTED, fontsize=8, fontweight="bold")
    clock_text = hud.text(0.06, 0.89, "T+00:00", color=TEXT, fontsize=20, fontweight="bold")
    state_text = hud.text(0.06, 0.82, "GROUND", color=CYAN, fontsize=10, fontweight="bold")
    hud.text(0.06, 0.75, "BATTERY", color=MUTED, fontsize=8)
    battery_bg = Rectangle((0.06, 0.695), 0.87, 0.045, facecolor="#182b38", edgecolor=GRID)
    battery_bar = Rectangle((0.06, 0.695), 0.87, 0.045, facecolor=GREEN, edgecolor="none")
    hud.add_patch(battery_bg)
    hud.add_patch(battery_bar)
    battery_text = hud.text(0.06, 0.66, "100.0% · NOMINAL", color=GREEN, fontsize=10, fontweight="bold")
    flight_text = hud.text(0.06, 0.56, "ALT     0 FT\nSPEED  35 KT", color=TEXT, fontsize=10, linespacing=1.7)
    hud.plot([0.06, 0.94], [0.49, 0.49], color=GRID, linewidth=1)
    hud.text(0.06, 0.45, "LAST OPERATOR ACTION", color=MUTED, fontsize=8)
    action_text = hud.text(0.06, 0.39, "BRIEFING", color=CYAN, fontsize=11, fontweight="bold", wrap=True)
    hud.text(0.06, 0.31, "ALERT / FAILSAFE", color=MUTED, fontsize=8)
    alert_text = hud.text(0.06, 0.24, "NONE", color=GREEN, fontsize=9, fontweight="bold", wrap=True)
    hud.text(0.06, 0.13, "SCORE", color=MUTED, fontsize=8)
    reward_text = hud.text(0.06, 0.065, _reward_text(data), color=TEXT, fontsize=14, fontweight="bold")

    footer_label = footer.text(0.018, 0.58, "MISSION INTAKE · REVIEW AIRSPACE / WEATHER / RESERVE", color=AMBER, fontsize=11, fontweight="bold", va="center")
    footer_detail = footer.text(0.018, 0.18, _cold_summary(log), color=MUTED, fontsize=7)
    terminal_box = footer.text(
        0.98,
        0.5,
        "",
        color=TEXT,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="center",
        bbox={"facecolor": PANEL, "edgecolor": GRID, "boxstyle": "round,pad=0.5"},
    )
    terminal_box.set_path_effects([path_effects.withStroke(linewidth=1.0, foreground=BG)])

    schedule = frame_schedule(len(log), config)

    def update(frame: int) -> Iterable[Any]:
        index = schedule[frame]
        snapshot = log[index]
        aircraft_state = snapshot["aircraft"]
        trail = trajectory_through(log, index)
        artists["trail"].set_data([item[0] for item in trail], [item[1] for item in trail])
        artists["aircraft"].set_data([aircraft_state["lon"]], [aircraft_state["lat"]])
        plan = snapshot.get("current_plan", [])
        plan_points = [(float(aircraft_state["lon"]), float(aircraft_state["lat"]))]
        plan_points.extend(
            (float(item["lon"]), float(item["lat"]))
            for item in plan
            if isinstance(item, Mapping) and "lon" in item and "lat" in item
        )
        artists["planned"].set_data(
            [item[0] for item in plan_points], [item[1] for item in plan_points]
        )
        active_tfr_ids = {item["event_id"] for item in activated_events(log, index, event_type="TFR_POPUP")}
        for event_id, patch in artists["tfrs"].items():
            patch.set_visible(event_id in active_tfr_ids)
        _, _, us, vs = wind_grid(snapshot, extent)
        artists["quiver"].set_UVC(us, vs)

        now = int(round(float(snapshot["sim_time_s"])))
        clock_text.set_text(f"T+{now // 60:02d}:{now % 60:02d}")
        status = str(aircraft_state["status"]).upper()
        state_text.set_text(status)
        state_text.set_color(RED if status == "LOST" else CYAN)
        pct = max(0.0, min(100.0, float(snapshot["battery_pct"])))
        level, color = battery_level(pct)
        battery_bar.set_width(0.87 * pct / 100.0)
        battery_bar.set_facecolor(color)
        battery_text.set_text(f"{pct:05.1f}% · {level}")
        battery_text.set_color(color)
        flight_text.set_text(
            f"ALT   {float(aircraft_state['alt_ft']):4.0f} FT\nSPEED  {float(snapshot['current_airspeed_kt']):2.0f} KT"
        )
        action = str(snapshot["last_action"]).replace("_", " ").upper()
        action_text.set_text(action)
        failsafe = snapshot.get("active_failsafe")
        alert_text.set_text(str(failsafe) if failsafe else _event_summary(snapshot))
        alert_text.set_color(RED if failsafe else AMBER if snapshot.get("alerts") else GREEN)

        activations = event_activations_at(log, index)
        if "TFR_POPUP" in activations:
            footer_label.set_text("POP-UP RESTRICTION ACTIVE · ROUTE REVIEW REQUIRED")
            footer_label.set_color(RED)
        elif activations:
            footer_label.set_text(f"OPS INTERRUPT · {' + '.join(activations)}")
            footer_label.set_color(AMBER)
        elif snapshot["is_terminal"]:
            label, outcome_color = terminal_label(snapshot)
            footer_label.set_text(label)
            footer_label.set_color(outcome_color)
        elif index == 0:
            footer_label.set_text("MISSION INTAKE · REVIEW AIRSPACE / WEATHER / RESERVE")
            footer_label.set_color(AMBER)
        else:
            footer_label.set_text("OPERATOR DECISION LOGGED · AUTOPILOT ADVANCING")
            footer_label.set_color(CYAN)
        footer_detail.set_text(
            _cold_summary(log)
            if index == 0
            else f"EVENTS: {_event_summary(snapshot)}  ·  MISSION: {str(snapshot['mission']['status']).upper()}"
        )
        if snapshot["is_terminal"]:
            label, outcome_color = terminal_label(snapshot)
            terminal_box.set_text(label)
            terminal_box.set_color(outcome_color)
            header_status.set_text("TERMINAL")
            header_status.set_color(outcome_color)
        else:
            terminal_box.set_text("")
            header_status.set_text("LIVE" if index else "MISSION BRIEFING")
            header_status.set_color(GREEN if index else AMBER)
        reward_text.set_color(GREEN if float(data.get("reward", 0.0)) > 0 else RED)
        return (
            artists["trail"],
            artists["aircraft"],
            artists["planned"],
            artists["quiver"],
            clock_text,
            battery_bar,
            battery_text,
            action_text,
            alert_text,
            footer_label,
            footer_detail,
            terminal_box,
            header_status,
        )

    movie = animation.FuncAnimation(fig, update, frames=len(schedule), interval=1000 / config.fps, blit=False)
    try:
        if output.suffix.lower() == ".gif":
            movie.save(str(output), writer=animation.PillowWriter(fps=config.fps), dpi=config.dpi)
        else:
            writer = animation.FFMpegWriter(
                fps=config.fps,
                codec="libx264",
                bitrate=3600,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"],
            )
            movie.save(str(output), writer=writer, dpi=config.dpi)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RenderError("encoder_failed", f"Could not encode {output.name}: {exc}") from exc
    finally:
        plt.close(fig)


def render_contact_sheet(artifacts: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Render a static five-episode trigger/decision/outcome contact sheet."""

    if not artifacts:
        raise RenderError("empty_contact_sheet", "At least one artifact is required")
    fig, axes = plt.subplots(len(artifacts), 1, figsize=(16, 2.15 * len(artifacts)), dpi=120)
    if len(artifacts) == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)
    for number, (artifact, ax) in enumerate(zip(artifacts, axes, strict=True), start=1):
        validate_artifact(dict(artifact))
        log = artifact["sim_log"]
        final = log[-1]
        outcome, color = terminal_label(final)
        events = [str(item.get("type", "EVENT")) for item in event_catalog(log).values()]
        decisive = _decision_label(log)
        ax.set_facecolor(PANEL)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.text(0.02, 0.68, f"0{number}  {log[0]['scenario_id']}", transform=ax.transAxes, color=TEXT, fontsize=13, fontweight="bold")
        ax.text(0.02, 0.28, "TRIGGER", transform=ax.transAxes, color=MUTED, fontsize=7, fontweight="bold")
        ax.text(0.14, 0.28, " + ".join(events), transform=ax.transAxes, color=AMBER, fontsize=9)
        ax.text(0.47, 0.28, "DECISION", transform=ax.transAxes, color=MUTED, fontsize=7, fontweight="bold")
        ax.text(0.57, 0.28, decisive, transform=ax.transAxes, color=CYAN, fontsize=9)
        ax.text(0.98, 0.62, outcome, transform=ax.transAxes, color=color, fontsize=11, fontweight="bold", ha="right")
        ax.text(0.98, 0.2, _reward_text(artifact), transform=ax.transAxes, color=TEXT, fontsize=9, ha="right")
    fig.suptitle("DAY 7 · FIVE FROZEN T3 OPERATIONS", color=CYAN, fontsize=17, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.95), h_pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON object containing frozen v1 sim_log")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output .mp4 or .gif")
    parser.add_argument("--no-basemap", action="store_true", help="Render vector layers without downloading or reading map tiles")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = load_artifact(args.input)
        render_artifact(artifact, args.output, no_basemap=args.no_basemap)
    except RenderError as exc:
        print(json.dumps(exc.payload(), sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "input": str(args.input), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
