"""uav-operator Verifiers environment module."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import verifiers as vf
from datasets import Dataset

ENV_ID = "uav-operator"

MIN_LAT = 37.55
MAX_LAT = 37.95
MIN_LON = -122.60
MAX_LON = -122.15
EARTH_RADIUS_NM = 3440.065
BATTERY_WH = 400.0
DEFAULT_AIRSPEED_KT = 35.0
TOOL_LATENCY_S = 20.0
READ_TOOL_LATENCY_S = 10.0
HIGH_IMPACT_TOOL_LATENCY_S = 30.0
INVALID_ACTION_LATENCY_S = 30.0
LOW_BATT_RTL_THRESHOLD_PCT = 30.0
LOW_BATT_RTL_THRESHOLD_WH = BATTERY_WH * LOW_BATT_RTL_THRESHOLD_PCT / 100.0
BRIEFED_RESERVE_PCT = 20.0
DEFAULT_WIND_DIR_FROM_DEG = 270.0
DEFAULT_WIND_SPEED_KT = 0.0
MIN_GROUNDSPEED_KT = 1.0
WIND_ALT_SPEED_SCALE_PER_1000_FT = 0.15
WIND_ALT_VEER_DEG_PER_1000_FT = 5.0
DAY5_TRAIN_EXAMPLES = 300
DAY5_DEV_EXAMPLES = 60
DAY5_EVAL_EXAMPLES = 60
DAY5_TIERS = ("T0", "T1", "T2", "T3")
EVENT_BLOCKED_TOOLS = {
    "file_flight_plan",
    "amend_route",
    "set_altitude",
    "set_speed",
    "resume",
    "land_now",
    "release_payload",
    "override_failsafe",
}
VALID_OVERRIDE_JUSTIFICATIONS = {
    "MISSION_CRITICAL_MARGIN_OK",
    "SAFE_LANDING_ASSURED",
    "AIRSPACE_AUTH_CONFIRMED",
    "RECOVERY_SITE_ASSURED",
}

AirspaceClass = Literal["B", "C", "D", "NO_FLY"]


@dataclass(frozen=True)
class Site:
    """Launch, recovery, or charging site."""

    site_id: str
    name: str
    lat: float
    lon: float
    kind: str = "recovery"


@dataclass(frozen=True)
class Waypoint:
    """A route point supplied by the model or scenario."""

    lat: float
    lon: float
    name: str = ""


@dataclass
class Mission:
    """Single Day 1 T0 mission."""

    mission_id: str
    description: str
    launch_site_id: str
    target: Waypoint
    value: float = 1.0
    sla_min: float = 45.0
    status: Literal["pending", "completed", "failed"] = "pending"
    completed_time_s: float | None = None
    failure_reason: str | None = None


@dataclass
class Aircraft:
    """Analytic aircraft state owned by the simulator."""

    lat: float
    lon: float
    alt_ft: float
    status: Literal["ground", "airborne", "holding", "landed", "lost"] = "ground"
    battery_wh: float = BATTERY_WH
    current_site_id: str | None = None


@dataclass(frozen=True)
class AirspaceZone:
    """Simplified static airspace or no-fly polygon."""

    zone_id: str
    name: str
    polygon: tuple[tuple[float, float], ...]
    floor_ft: float
    ceiling_ft: float
    airspace_class: AirspaceClass
    authorization_required: bool = True


@dataclass(frozen=True)
class ObstacleCell:
    """Coarse obstacle proxy cell."""

    cell_id: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    elevation_ft: float


@dataclass(frozen=True)
class WindBlob:
    """Smooth seeded wind perturbation."""

    center_lat: float
    center_lon: float
    radius_nm: float
    dir_delta_deg: float
    speed_delta_kt: float
    drift_bearing_deg: float
    drift_speed_kt: float


@dataclass(frozen=True)
class GustFront:
    """Seeded moving gust-front line."""

    anchor_lat: float
    anchor_lon: float
    orientation_deg: float
    movement_bearing_deg: float
    speed_kt: float
    dir_shift_deg: float
    speed_delta_kt: float
    forecast_eta_s: float
    eta_error_s: float


@dataclass(frozen=True)
class WindField:
    """Serializable per-episode wind configuration."""

    base_dir_deg_from: float
    base_speed_kt: float
    blobs: list[WindBlob]
    gust_front: GustFront | None = None


@dataclass
class SimState:
    """Serializable simulator state for one episode."""

    seed: int
    scenario_id: str
    tier: str
    sim_time_s: float
    aircraft: Aircraft
    mission: Mission
    home_site_id: str
    wind: WindField
    rng_state: dict[str, Any]
    scenario_par: dict[str, float]
    events: list[dict[str, Any]] = field(default_factory=list)
    active_events: list[str] = field(default_factory=list)
    acknowledged_events: list[str] = field(default_factory=list)
    closed_sites: list[str] = field(default_factory=list)
    hard_safety_violations: list[str] = field(default_factory=list)
    procedure_violations: list[str] = field(default_factory=list)
    current_plan: list[Waypoint] = field(default_factory=list)
    lost_link_plan: str = "return_home"
    rng_draws: int = 0
    active_failsafe: str | None = None
    alerts: list[dict[str, Any]] = field(default_factory=list)
    acknowledged_alerts: list[str] = field(default_factory=list)
    overrides: list[dict[str, Any]] = field(default_factory=list)
    payload_released: bool = False
    hold_until_s: float | None = None
    current_altitude_target_ft: float = 250.0
    current_airspeed_kt: float = DEFAULT_AIRSPEED_KT
    last_action: str = "briefing"
    last_result: dict[str, Any] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_reason: str | None = None


SITES: dict[str, Site] = {
    "mission_bay": Site("mission_bay", "Mission Bay Pad", 37.7707, -122.3869),
    "alameda": Site("alameda", "Alameda Depot", 37.7866, -122.3080),
    "san_mateo": Site("san_mateo", "San Mateo Bayfront", 37.5670, -122.3200),
    "berkeley": Site("berkeley", "Berkeley Marina Pad", 37.8646, -122.3130),
    "redwood": Site("redwood", "Redwood Shores Pad", 37.5530, -122.2580),
    "daly_city": Site("daly_city", "Daly City Ridge Pad", 37.6879, -122.4702),
    "oakland_port": Site("oakland_port", "Oakland Port Pad", 37.7955, -122.2856),
    "hayward": Site("hayward", "Hayward Executive Pad", 37.6597, -122.1219),
    "palo_alto": Site("palo_alto", "Palo Alto Baylands Pad", 37.5535, -122.1510),
    "richmond": Site("richmond", "Richmond Harbor Pad", 37.9104, -122.3630),
    "skyline": Site("skyline", "Skyline Ridge Recovery", 37.5520, -122.5010),
    "fremont": Site("fremont", "Fremont Warm Springs Pad", 37.5550, -122.1510),
}


TARGET_POINTS: tuple[Waypoint, ...] = (
    Waypoint(37.8230, -122.3710, "Treasure Island inspection point"),
    Waypoint(37.8087, -122.4098, "Pier 39 delivery point"),
    Waypoint(37.7350, -122.2390, "Bay Farm pump station"),
    Waypoint(37.5908, -122.3257, "Coyote Point marker"),
    Waypoint(37.9130, -122.3570, "Richmond Channel buoy"),
    Waypoint(37.6152, -122.3899, "SFO perimeter sensor"),
    Waypoint(37.7516, -122.2005, "Coliseum parking sensor"),
    Waypoint(37.6890, -122.4010, "San Bruno ridge camera"),
    Waypoint(37.8022, -122.4484, "Crissy Field shoreline marker"),
    Waypoint(37.7793, -122.2455, "Alameda East dock"),
)


AIRSPACE_ZONES: tuple[AirspaceZone, ...] = (
    AirspaceZone(
        zone_id="sfo_b_core",
        name="SFO Class B Core",
        polygon=(
            (37.5850, -122.4250),
            (37.6500, -122.4250),
            (37.6650, -122.3650),
            (37.6200, -122.3350),
            (37.5700, -122.3650),
        ),
        floor_ft=0.0,
        ceiling_ft=3000.0,
        airspace_class="B",
    ),
    AirspaceZone(
        zone_id="sfo_b_north_shelf",
        name="SFO Class B North Shelf",
        polygon=(
            (37.6500, -122.5000),
            (37.7350, -122.5000),
            (37.7350, -122.3600),
            (37.6650, -122.3650),
        ),
        floor_ft=1500.0,
        ceiling_ft=3000.0,
        airspace_class="B",
    ),
    AirspaceZone(
        zone_id="oak_c_core",
        name="OAK Class C Core",
        polygon=(
            (37.6900, -122.2450),
            (37.7600, -122.2450),
            (37.7750, -122.1850),
            (37.7150, -122.1550),
            (37.6750, -122.1900),
        ),
        floor_ft=0.0,
        ceiling_ft=2500.0,
        airspace_class="C",
        authorization_required=False,
    ),
    AirspaceZone(
        zone_id="sql_d",
        name="San Carlos Class D",
        polygon=(
            (37.4950, -122.2900),
            (37.5550, -122.2900),
            (37.5750, -122.2350),
            (37.5300, -122.1900),
            (37.4850, -122.2200),
        ),
        floor_ft=0.0,
        ceiling_ft=2500.0,
        airspace_class="D",
        authorization_required=False,
    ),
    AirspaceZone(
        zone_id="oakland_coliseum_no_fly",
        name="Oakland Coliseum event no-fly polygon",
        polygon=(
            (37.7465, -122.2075),
            (37.7570, -122.2075),
            (37.7570, -122.1930),
            (37.7465, -122.1930),
        ),
        floor_ft=0.0,
        ceiling_ft=2000.0,
        airspace_class="NO_FLY",
    ),
)


OBSTACLE_CELLS: tuple[ObstacleCell, ...] = (
    ObstacleCell("bay_flat", 37.55, 37.95, -122.60, -122.15, 40.0),
    ObstacleCell("san_bruno_hills", 37.62, 37.72, -122.50, -122.40, 850.0),
    ObstacleCell("oakland_hills", 37.76, 37.90, -122.25, -122.15, 1050.0),
    ObstacleCell("san_mateo_ridge", 37.52, 37.62, -122.48, -122.34, 1200.0),
)


WORLD_DATA: dict[str, Any] = {
    "bbox": {
        "min_lat": MIN_LAT,
        "max_lat": MAX_LAT,
        "min_lon": MIN_LON,
        "max_lon": MAX_LON,
    },
    "sites": [asdict(site) for site in SITES.values()],
    "targets": [asdict(target) for target in TARGET_POINTS],
    "airspace": [asdict(zone) for zone in AIRSPACE_ZONES],
    "obstacle_cells": [asdict(cell) for cell in OBSTACLE_CELLS],
}


SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "scenario_id": "T0-treasure-island",
        "launch_site_id": "mission_bay",
        "target": Waypoint(37.8230, -122.3710, "Treasure Island inspection point"),
        "description": "Inspect the Treasure Island shoreline marker.",
        "sla_min": 35.0,
    },
    {
        "scenario_id": "T0-pier-39",
        "launch_site_id": "mission_bay",
        "target": Waypoint(37.8087, -122.4098, "Pier 39 delivery point"),
        "description": "Deliver a lightweight sensor tag near Pier 39.",
        "sla_min": 30.0,
    },
    {
        "scenario_id": "T0-bay-farm",
        "launch_site_id": "alameda",
        "target": Waypoint(37.7350, -122.2390, "Bay Farm pump station"),
        "description": "Inspect the Bay Farm pump station perimeter.",
        "sla_min": 35.0,
    },
    {
        "scenario_id": "T0-coyote-point",
        "launch_site_id": "san_mateo",
        "target": Waypoint(37.5908, -122.3257, "Coyote Point marker"),
        "description": "Photograph the Coyote Point shoreline marker.",
        "sla_min": 25.0,
    },
    {
        "scenario_id": "T0-richmond-channel",
        "launch_site_id": "berkeley",
        "target": Waypoint(37.9130, -122.3570, "Richmond Channel buoy"),
        "description": "Inspect the Richmond Channel buoy from standoff.",
        "sla_min": 40.0,
    },
)


SYSTEM_PROMPT = """You are the remote pilot in command for uav-operator.

The simulator owns the aircraft, autopilot, energy model, and reward. Your job
is operational judgment at decision points. Use tools to inspect telemetry,
weather, airspace, sites, and mission status; then file or amend plans, hold,
return, land, release payloads, abort, acknowledge alerts, or override active
failsafes only when the enum justification is supported by the sim state.
Wind is seeded, spatially varying, altitude-dependent, and may include a gust
front forecast with error. The simulator validates route geometry, energy,
airspace, and autopilot failsafes.

You control the operation exclusively through tool calls.  Describing an action 
does not execute it. At every nonterminal turn, call an appropriate tool. If the 
route is blocked or unsafe, use the available tools to hold, reroute, return, or 
abort; do not merely state your intention.

Reward comes only from simulator state: complete the mission value and bring
the aircraft down. Natural-language claims do not score.
"""


def _tool_defs() -> list[dict[str, Any]]:
    waypoint_schema = {
        "type": "object",
        "properties": {
            "lat": {"type": "number"},
            "lon": {"type": "number"},
            "name": {"type": "string"},
        },
        "required": ["lat", "lon"],
        "additionalProperties": False,
    }
    waypoints_schema = {
        "type": "array",
        "minItems": 1,
        "items": waypoint_schema,
    }
    return [
        {
            "name": "get_telemetry",
            "description": "Return current aircraft, mission, weather, alerts, and active failsafe telemetry.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "get_weather",
            "description": "Return current wind and coarse gust-front forecast at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "alt_ft": {"type": "number"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_airspace",
            "description": "Return static airspace zones and optional route conflicts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "route": waypoints_schema,
                    "alt_ft": {"type": "number"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_mission_status",
            "description": "Return mission status and target/SLA details.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "get_sites",
            "description": "Return launch/recovery site information.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "file_flight_plan",
            "description": "File a waypoint route for the autopilot to fly analytically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "waypoints": waypoints_schema,
                    "alt_ft": {"type": "number"},
                    "airspeed_kt": {"type": "number"},
                    "lost_link_plan": {"type": "string"},
                },
                "required": ["waypoints", "alt_ft", "airspeed_kt", "lost_link_plan"],
                "additionalProperties": False,
            },
        },
        {
            "name": "amend_route",
            "description": "Replace the remaining route from the current aircraft position.",
            "parameters": {
                "type": "object",
                "properties": {"waypoints_from_current": waypoints_schema},
                "required": ["waypoints_from_current"],
                "additionalProperties": False,
            },
        },
        {
            "name": "set_altitude",
            "description": "Set commanded altitude for subsequent route execution.",
            "parameters": {
                "type": "object",
                "properties": {"ft": {"type": "number"}},
                "required": ["ft"],
                "additionalProperties": False,
            },
        },
        {
            "name": "set_speed",
            "description": "Set commanded airspeed for subsequent route execution.",
            "parameters": {
                "type": "object",
                "properties": {"kt": {"type": "number"}},
                "required": ["kt"],
                "additionalProperties": False,
            },
        },
        {
            "name": "hold",
            "description": "Hold position for a number of simulated minutes.",
            "parameters": {
                "type": "object",
                "properties": {"minutes": {"type": "number"}},
                "required": ["minutes"],
                "additionalProperties": False,
            },
        },
        {
            "name": "resume",
            "description": "Resume after a hold or acknowledged alert.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "command_rtl",
            "description": "Command autonomous return to a recovery site and land.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site_id": {
                        "type": "string",
                        "description": "Optional site id. Defaults to the nearest recovery site.",
                    }
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "land_now",
            "description": "Land immediately at current position or route to a named recovery site and land.",
            "parameters": {
                "type": "object",
                "properties": {"site_id": {"type": "string"}},
                "additionalProperties": False,
            },
        },
        {
            "name": "release_payload",
            "description": "Release payload at the delivery/inspection target if within tolerance.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "abort_mission",
            "description": "Abort the active mission and mark it failed.",
            "parameters": {
                "type": "object",
                "properties": {"mission_id": {"type": "string"}},
                "required": ["mission_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "override_failsafe",
            "description": "Override an active overridable failsafe using an enum justification code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "justification_code": {
                        "type": "string",
                        "enum": sorted(VALID_OVERRIDE_JUSTIFICATIONS),
                    },
                },
                "required": ["id", "justification_code"],
                "additionalProperties": False,
            },
        },
        {
            "name": "acknowledge",
            "description": "Acknowledge an alert by alert_id.",
            "parameters": {
                "type": "object",
                "properties": {"alert_id": {"type": "string"}},
                "required": ["alert_id"],
                "additionalProperties": False,
            },
        },
    ]


def _available_tool_names() -> list[str]:
    return [str(tool["name"]) for tool in _tool_defs()]


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_NM * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees true."""

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _in_bounds(lat: float, lon: float) -> bool:
    return MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON


def _point_in_polygon(lat: float, lon: float, polygon: Sequence[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, point in enumerate(polygon):
        lat_i, lon_i = point
        lat_j, lon_j = polygon[j]
        if (lon_i > lon) != (lon_j > lon):
            slope_lat = (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i) + lat_i
            if lat < slope_lat:
                inside = not inside
        j = i
    return inside


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-12:
        return 0
    return 1 if value > 0.0 else 2


def _on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _on_segment(a1, b1, a2))
        or (o2 == 0 and _on_segment(a1, b2, a2))
        or (o3 == 0 and _on_segment(b1, a1, b2))
        or (o4 == 0 and _on_segment(b1, a2, b2))
    )


def _route_crosses_polygon(
    start: tuple[float, float],
    end: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    if _point_in_polygon(start[0], start[1], polygon) or _point_in_polygon(end[0], end[1], polygon):
        return True
    return any(
        _segments_intersect(start, end, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _vertical_overlap(alt_ft: float, floor_ft: float, ceiling_ft: float) -> bool:
    return floor_ft <= alt_ft <= ceiling_ft


def _event_tfr_zone(event: Mapping[str, Any]) -> AirspaceZone | None:
    if event.get("type") != "TFR_POPUP" or event.get("status") != "active":
        return None
    params = event.get("params")
    if not isinstance(params, Mapping):
        return None
    polygon = params.get("polygon")
    if not isinstance(polygon, Sequence):
        return None
    return AirspaceZone(
        zone_id=str(params.get("zone_id", event.get("event_id", "tfr_popup"))),
        name=str(params.get("name", "Pop-up TFR")),
        polygon=tuple((float(point[0]), float(point[1])) for point in polygon),
        floor_ft=float(params.get("floor_ft", 0.0)),
        ceiling_ft=float(params.get("ceiling_ft", 2000.0)),
        airspace_class="NO_FLY",
        authorization_required=True,
    )


def _all_airspace_zones(sim: SimState | None = None) -> tuple[AirspaceZone, ...]:
    zones = list(AIRSPACE_ZONES)
    if sim is not None:
        for event in sim.events:
            zone = _event_tfr_zone(event)
            if zone is not None:
                zones.append(zone)
    return tuple(zones)


def _airspace_conflicts_for_segment(
    sim: SimState,
    start: Waypoint,
    end: Waypoint,
    alt_ft: float,
) -> list[AirspaceZone]:
    start_point = (start.lat, start.lon)
    end_point = (end.lat, end.lon)
    return [
        zone
        for zone in _all_airspace_zones(sim)
        if _vertical_overlap(alt_ft, zone.floor_ft, zone.ceiling_ft)
        and _route_crosses_polygon(start_point, end_point, zone.polygon)
    ]


def min_safe_altitude_ft(lat: float, lon: float) -> float:
    """Coarse terrain/structure clearance proxy in feet MSL."""

    matching = [
        cell.elevation_ft + 100.0
        for cell in OBSTACLE_CELLS
        if cell.lat_min <= lat <= cell.lat_max and cell.lon_min <= lon <= cell.lon_max
    ]
    return max(matching) if matching else 100.0


def _nearest_site(lat: float, lon: float, closed_site_ids: Sequence[str] = ()) -> Site:
    open_sites = [
        site
        for site in SITES.values()
        if site.site_id not in set(closed_site_ids)
    ] or list(SITES.values())
    return min(
        open_sites,
        key=lambda site: haversine_nm(lat, lon, site.lat, site.lon),
    )


def _scenario_par(launch_site: Site, target: Waypoint, sla_min: float) -> dict[str, float]:
    distance_to_target = haversine_nm(launch_site.lat, launch_site.lon, target.lat, target.lon)
    direct_time_s = 2.0 * distance_to_target / DEFAULT_AIRSPEED_KT * 3600.0 + 2.0 * TOOL_LATENCY_S
    direct_energy_wh = 2.0 * _segment_energy_wh(
        distance_nm=distance_to_target,
        airspeed_kt=DEFAULT_AIRSPEED_KT,
        start_alt_ft=0.0,
        end_alt_ft=250.0,
    )
    return {
        "time_s": max(direct_time_s, sla_min * 60.0 * 0.65),
        "energy_wh": max(direct_energy_wh, BATTERY_WH * 0.25),
    }


def _tfr_event_polygon(
    target: Waypoint,
    *,
    lat_delta: float = 0.018,
    lon_delta: float = 0.022,
) -> tuple[tuple[float, float], ...]:
    return (
        (target.lat - lat_delta, target.lon - lon_delta),
        (target.lat + lat_delta, target.lon - lon_delta),
        (target.lat + lat_delta, target.lon + lon_delta),
        (target.lat - lat_delta, target.lon + lon_delta),
    )


def _day4_t1_event(seed: int, index: int, base: Mapping[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(seed * 1009 + index * 17)
    event_type = ("WIND_SHIFT", "TFR_POPUP", "BATT_DEGRADE", "SITE_CLOSED")[index % 4]
    event_id = f"{base['scenario_id']}-{event_type.lower()}"
    target = cast(Waypoint, base["target"])
    if event_type == "WIND_SHIFT":
        params = {
            "dir_delta_deg": float(rng.choice([-70.0, -55.0, 55.0, 70.0])),
            "speed_delta_kt": float(rng.uniform(8.0, 14.0)),
        }
        message = "WIND_SHIFT: observed winds have shifted; recompute route energy before continuing."
    elif event_type == "TFR_POPUP":
        params = {
            "zone_id": f"tfr_{index}",
            "name": "Pop-up Bay Area TFR",
            "polygon": _tfr_event_polygon(target),
            "floor_ft": 0.0,
            "ceiling_ft": 2000.0,
        }
        message = "TFR_POPUP: NOTAM active near the target, surface to 2000 ft; avoid the polygon."
    elif event_type == "BATT_DEGRADE":
        params = {"capacity_loss_pct": float(rng.uniform(12.0, 22.0))}
        message = "BATT_DEGRADE: pack health degraded; reserve margin is reduced."
    else:
        launch_site = SITES[cast(str, base["launch_site_id"])]
        candidates = [
            site
            for site in SITES.values()
            if site.site_id != launch_site.site_id
        ]
        closed_site = min(
            candidates,
            key=lambda site: haversine_nm(target.lat, target.lon, site.lat, site.lon),
        )
        params = {"site_id": closed_site.site_id, "site_name": closed_site.name}
        message = f"SITE_CLOSED: {closed_site.name} is unavailable for recovery."
    return {
        "event_id": event_id,
        "type": event_type,
        "trigger_time_s": 20.0,
        "status": "pending",
        "message": message,
        "params": params,
    }


def _day5_composed_events(seed: int, index: int, base: Mapping[str, Any], tier: str) -> list[dict[str, Any]]:
    """Generate T2/T3 event compositions using only the active Day 4 taxonomy."""

    rng = np.random.default_rng(seed * 2029 + index * 31)
    event_count = 2 if tier == "T2" else int(rng.integers(2, 5))
    event_types = ["WIND_SHIFT", "BATT_DEGRADE", "SITE_CLOSED", "TFR_POPUP"]
    chosen = list(rng.choice(event_types, size=event_count, replace=False))
    launch = SITES[cast(str, base["launch_site_id"])]
    target = cast(Waypoint, base["target"])
    midpoint = Waypoint(
        (launch.lat + target.lat) / 2.0,
        (launch.lon + target.lon) / 2.0,
        "mid-route TFR reference",
    )
    events: list[dict[str, Any]] = []
    for event_index, event_type in enumerate(chosen):
        event_id = f"{base['scenario_id']}-{event_type.lower()}-{event_index}"
        trigger_time_s = 20.0 + event_index * 45.0
        if event_type == "WIND_SHIFT":
            params = {
                "dir_delta_deg": float(rng.choice([-75.0, -60.0, 60.0, 75.0])),
                "speed_delta_kt": float(rng.uniform(9.0, 16.0)),
            }
            message = "WIND_SHIFT: observed winds changed; recompute energy and groundspeed."
        elif event_type == "BATT_DEGRADE":
            params = {"capacity_loss_pct": float(rng.uniform(10.0, 18.0))}
            message = "BATT_DEGRADE: pack health reduced; landing reserve must be recomputed."
        elif event_type == "SITE_CLOSED":
            candidates = [site for site in SITES.values() if site.site_id != launch.site_id]
            closed_site = min(
                candidates,
                key=lambda site: haversine_nm(target.lat, target.lon, site.lat, site.lon),
            )
            params = {"site_id": closed_site.site_id, "site_name": closed_site.name}
            message = f"SITE_CLOSED: {closed_site.name} is unavailable for recovery."
        else:
            params = {
                "zone_id": f"tfr_day5_{index}_{event_index}",
                "name": "Pop-up Bay Area TFR",
                # The TFR blocks the direct leg, not the delivery point itself.
                "polygon": _tfr_event_polygon(midpoint, lat_delta=0.008, lon_delta=0.010),
                "floor_ft": 0.0,
                "ceiling_ft": 2000.0,
            }
            message = (
                "TFR_POPUP: NOTAM active across the direct corridor, surface to 2000 ft; "
                "route around the polygon."
            )
        events.append(
            {
                "event_id": event_id,
                "type": event_type,
                "trigger_time_s": trigger_time_s,
                "status": "pending",
                "message": message,
                "params": params,
            }
        )
    return events


def _solver_route_candidates(launch: Site, target: Waypoint, events: Sequence[Mapping[str, Any]]) -> list[list[Waypoint]]:
    """Return direct and deterministic TFR-detour candidates for feasibility checks."""

    candidates = [[target]]
    for event in events:
        if event.get("type") != "TFR_POPUP":
            continue
        params = event.get("params", {})
        if not isinstance(params, Mapping):
            continue
        polygon = params.get("polygon")
        if not isinstance(polygon, Sequence) or not polygon:
            continue
        max_lat = max(float(point[0]) for point in polygon)
        min_lat = min(float(point[0]) for point in polygon)
        max_lon = max(float(point[1]) for point in polygon)
        min_lon = min(float(point[1]) for point in polygon)
        clearance = 0.012
        candidates.extend(
            [
                [
                    Waypoint(max_lat + clearance, min_lon - clearance, "TFR north-west detour"),
                    Waypoint(max_lat + clearance, max_lon + clearance, "TFR north-east detour"),
                    target,
                ],
                [
                    Waypoint(min_lat - clearance, min_lon - clearance, "TFR south-west detour"),
                    Waypoint(min_lat - clearance, max_lon + clearance, "TFR south-east detour"),
                    target,
                ],
            ]
        )
    return candidates


def _scenario_has_feasible_resolution(base: Mapping[str, Any], seed: int) -> bool:
    """Check analytic route/recovery candidates without mutating simulator state."""

    launch = SITES[cast(str, base["launch_site_id"])]
    target = cast(Waypoint, base["target"])
    events = cast(Sequence[Mapping[str, Any]], base.get("events", []))
    closed_sites = {
        str(cast(Mapping[str, Any], event.get("params", {})).get("site_id", ""))
        for event in events
        if event.get("type") == "SITE_CLOSED" and isinstance(event.get("params"), Mapping)
    }
    capacity_loss = sum(
        float(cast(Mapping[str, Any], event.get("params", {})).get("capacity_loss_pct", 0.0))
        for event in events
        if event.get("type") == "BATT_DEGRADE" and isinstance(event.get("params"), Mapping)
    )
    wind = _build_wind_field(np.random.default_rng(seed), gust_front_probability=0.0)
    dynamic_zones = [
        _event_tfr_zone({**event, "status": "active"})
        for event in events
        if event.get("type") == "TFR_POPUP"
    ]
    zones = tuple(zone for zone in (*AIRSPACE_ZONES, *dynamic_zones) if zone is not None)
    for route in _solver_route_candidates(launch, target, events):
        points = [Waypoint(launch.lat, launch.lon, launch.name), *route]
        recovery = _nearest_site(target.lat, target.lon, tuple(closed_sites))
        points.append(Waypoint(recovery.lat, recovery.lon, recovery.name))
        energy_wh = BATTERY_WH * capacity_loss / 100.0 + BATTERY_WH * 0.08
        feasible = True
        for start, end in zip(points, points[1:]):
            track = bearing_deg(start.lat, start.lon, end.lat, end.lon)
            wind_dir, wind_speed = wind_at(start.lat, start.lon, 300.0, 120.0, wind)
            groundspeed = _groundspeed_kt(DEFAULT_AIRSPEED_KT, track, wind_dir, wind_speed)
            if groundspeed <= MIN_GROUNDSPEED_KT:
                feasible = False
                break
            if any(
                zone.authorization_required
                and _vertical_overlap(300.0, zone.floor_ft, zone.ceiling_ft)
                and _route_crosses_polygon((start.lat, start.lon), (end.lat, end.lon), zone.polygon)
                for zone in zones
            ):
                feasible = False
                break
            distance = haversine_nm(start.lat, start.lon, end.lat, end.lon)
            energy_wh += _segment_energy_wh(
                distance_nm=distance,
                airspeed_kt=DEFAULT_AIRSPEED_KT,
                start_alt_ft=300.0,
                end_alt_ft=300.0,
                track_deg=track,
                wind_dir_from_deg=wind_dir,
                wind_speed_kt=wind_speed,
            )
        if feasible and BATTERY_WH - energy_wh >= BATTERY_WH * BRIEFED_RESERVE_PCT / 100.0:
            return True
    return False


def _scenario_tier(index: int, requested_tier: str) -> str:
    tier = requested_tier.upper()
    if tier in DAY5_TIERS:
        return tier
    if tier in {"MIXED_DAY5", "MIXED"}:
        return DAY5_TIERS[index % len(DAY5_TIERS)]
    if tier in {"MIXED_DAY4", "DAY4"}:
        return "T1" if index % 2 else "T0"
    return "T0"


def _scenario_for_index(index: int, seed: int = 0, tier: str = "mixed_day5") -> dict[str, Any]:
    base = dict(SCENARIOS[index % len(SCENARIOS)])
    scenario_tier = _scenario_tier(index, tier)

    base["tier"] = scenario_tier
    base["scenario_id"] = f"{scenario_tier}-{index:03d}-{base['scenario_id'][3:]}"
    base["events"] = []
    if scenario_tier == "T1":
        base["events"] = [_day4_t1_event(seed, index, base)]
        base["description"] = f"{base['description']} Expect one operational interrupt."
        base["sla_min"] = float(base["sla_min"]) + 10.0
    elif scenario_tier in {"T2", "T3"}:
        base["events"] = _day5_composed_events(seed, index, base, scenario_tier)
        base["description"] = f"{base['description']} Resolve composed operational interrupts safely."
        base["sla_min"] = float(base["sla_min"]) + (8.0 if scenario_tier == "T2" else 3.0)
        if not _scenario_has_feasible_resolution(base, seed):
            # Keep the intended decision density while replacing an infeasible TFR geometry.
            for event in base["events"]:
                if event["type"] == "TFR_POPUP":
                    event["type"] = "WIND_SHIFT"
                    event["params"] = {"dir_delta_deg": 55.0, "speed_delta_kt": 10.0}
                    event["message"] = "WIND_SHIFT: observed winds changed; recompute energy and groundspeed."
            if not _scenario_has_feasible_resolution(base, seed):
                raise RuntimeError(f"scenario has no feasible resolution: {base['scenario_id']}")

    launch_site = SITES[cast(str, base["launch_site_id"])]
    base["par"] = _scenario_par(
        launch_site=launch_site,
        target=cast(Waypoint, base["target"]),
        sla_min=float(base["sla_min"]),
    )
    return base


def _move_latlon(lat: float, lon: float, bearing: float, distance_nm: float) -> tuple[float, float]:
    bearing_rad = math.radians(bearing)
    d_lat = math.cos(bearing_rad) * distance_nm / 60.0
    cos_lat = max(0.1, math.cos(math.radians(lat)))
    d_lon = math.sin(bearing_rad) * distance_nm / (60.0 * cos_lat)
    return lat + d_lat, lon + d_lon


def _wind_to_vector(dir_from_deg: float, speed_kt: float) -> tuple[float, float]:
    wind_to_rad = math.radians((dir_from_deg + 180.0) % 360.0)
    east = speed_kt * math.sin(wind_to_rad)
    north = speed_kt * math.cos(wind_to_rad)
    return east, north


def _vector_to_wind(east_kt: float, north_kt: float) -> tuple[float, float]:
    speed = math.hypot(east_kt, north_kt)
    if speed < 1e-9:
        return DEFAULT_WIND_DIR_FROM_DEG, 0.0
    wind_to_deg = math.degrees(math.atan2(east_kt, north_kt)) % 360.0
    return (wind_to_deg + 180.0) % 360.0, speed


def _signed_distance_from_line_nm(
    lat: float,
    lon: float,
    anchor_lat: float,
    anchor_lon: float,
    normal_bearing_deg: float,
) -> float:
    distance = haversine_nm(anchor_lat, anchor_lon, lat, lon)
    bearing = bearing_deg(anchor_lat, anchor_lon, lat, lon)
    angle = math.radians((bearing - normal_bearing_deg + 540.0) % 360.0 - 180.0)
    return distance * math.cos(angle)


def _build_wind_field(
    rng: np.random.Generator,
    *,
    wind_enabled: bool = True,
    gust_front_probability: float = 0.5,
) -> WindField:
    if not wind_enabled:
        return WindField(
            base_dir_deg_from=DEFAULT_WIND_DIR_FROM_DEG,
            base_speed_kt=DEFAULT_WIND_SPEED_KT,
            blobs=[],
            gust_front=None,
        )
    base_dir = float(rng.uniform(210.0, 300.0))
    base_speed = float(rng.uniform(6.0, 16.0))
    blobs = [
        WindBlob(
            center_lat=float(rng.uniform(MIN_LAT, MAX_LAT)),
            center_lon=float(rng.uniform(MIN_LON, MAX_LON)),
            radius_nm=float(rng.uniform(6.0, 14.0)),
            dir_delta_deg=float(rng.uniform(-60.0, 60.0)),
            speed_delta_kt=float(rng.uniform(-5.0, 8.0)),
            drift_bearing_deg=float(rng.uniform(0.0, 360.0)),
            drift_speed_kt=float(rng.uniform(4.0, 18.0)),
        )
        for _ in range(int(rng.integers(2, 5)))
    ]
    gust_front = None
    if bool(rng.random() < gust_front_probability):
        gust_front = GustFront(
            anchor_lat=(MIN_LAT + MAX_LAT) / 2.0,
            anchor_lon=(MIN_LON + MAX_LON) / 2.0,
            orientation_deg=float(rng.uniform(330.0, 390.0) % 360.0),
            movement_bearing_deg=float(rng.uniform(60.0, 120.0)),
            speed_kt=float(rng.uniform(20.0, 35.0)),
            dir_shift_deg=float(rng.uniform(40.0, 90.0)),
            speed_delta_kt=float(rng.uniform(15.0, 25.0)),
            forecast_eta_s=float(rng.uniform(10.0, 45.0) * 60.0),
            eta_error_s=float(rng.normal(0.0, 5.0 * 60.0)),
        )
    return WindField(
        base_dir_deg_from=base_dir,
        base_speed_kt=base_speed,
        blobs=blobs,
        gust_front=gust_front,
    )


def _wind_field_from_mapping(value: Mapping[str, Any]) -> WindField:
    gust_payload = value.get("gust_front")
    return WindField(
        base_dir_deg_from=float(value["base_dir_deg_from"]),
        base_speed_kt=float(value["base_speed_kt"]),
        blobs=[WindBlob(**blob) for blob in value.get("blobs", [])],
        gust_front=GustFront(**gust_payload) if isinstance(gust_payload, Mapping) else None,
    )


def _pack_rng_state(rng_state: Mapping[str, Any]) -> dict[str, Any]:
    """Make NumPy bit-generator state msgpack-safe for Verifiers transport."""

    packed = dict(rng_state)
    inner = packed.get("state")
    if isinstance(inner, Mapping):
        packed["state"] = {
            key: str(value) if key in {"state", "inc"} else value
            for key, value in inner.items()
        }
    return packed


def _unpack_rng_state(rng_state: Mapping[str, Any]) -> dict[str, Any]:
    unpacked = dict(rng_state)
    inner = unpacked.get("state")
    if isinstance(inner, Mapping):
        unpacked["state"] = {
            key: int(value) if key in {"state", "inc"} else value
            for key, value in inner.items()
        }
    return unpacked


def wind_at(
    lat: float,
    lon: float,
    alt_ft: float,
    t_s: float,
    wind_field: WindField | Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    """Return deterministic wind as (direction-from degrees, speed kt)."""

    wind = (
        _build_wind_field(np.random.default_rng(0))
        if wind_field is None
        else _wind_field_from_mapping(wind_field)
        if isinstance(wind_field, Mapping)
        else wind_field
    )
    east, north = _wind_to_vector(wind.base_dir_deg_from, wind.base_speed_kt)

    for blob in wind.blobs:
        center_lat, center_lon = _move_latlon(
            blob.center_lat,
            blob.center_lon,
            blob.drift_bearing_deg,
            blob.drift_speed_kt * t_s / 3600.0,
        )
        distance = haversine_nm(center_lat, center_lon, lat, lon)
        weight = math.exp(-0.5 * (distance / blob.radius_nm) ** 2)
        blob_east, blob_north = _wind_to_vector(
            wind.base_dir_deg_from + blob.dir_delta_deg,
            blob.speed_delta_kt,
        )
        east += weight * blob_east
        north += weight * blob_north

    if wind.gust_front is not None:
        front = wind.gust_front
        normal_bearing = front.movement_bearing_deg
        actual_arrival_s = front.forecast_eta_s + front.eta_error_s
        moved_nm = front.speed_kt * (t_s - actual_arrival_s) / 3600.0
        signed_nm = _signed_distance_from_line_nm(
            lat,
            lon,
            front.anchor_lat,
            front.anchor_lon,
            normal_bearing,
        )
        if signed_nm <= moved_nm:
            gust_east, gust_north = _wind_to_vector(
                wind.base_dir_deg_from + front.dir_shift_deg,
                front.speed_delta_kt,
            )
            east += gust_east
            north += gust_north

    dir_from, speed = _vector_to_wind(east, north)
    alt_kft = max(0.0, alt_ft) / 1000.0
    return (
        (dir_from + WIND_ALT_VEER_DEG_PER_1000_FT * alt_kft) % 360.0,
        speed * (1.0 + WIND_ALT_SPEED_SCALE_PER_1000_FT * alt_kft),
    )


def _build_sim(
    seed: int,
    scenario_index: int,
    *,
    tier: str = "mixed_day5",
    wind_enabled: bool = True,
    gust_front_probability: float = 0.5,
) -> SimState:
    rng = np.random.default_rng(seed)
    wind = _build_wind_field(
        rng,
        wind_enabled=wind_enabled,
        gust_front_probability=gust_front_probability,
    )
    scenario = _scenario_for_index(scenario_index, seed=seed, tier=tier)
    launch_site = SITES[cast(str, scenario["launch_site_id"])]
    mission = Mission(
        mission_id=cast(str, scenario["scenario_id"]),
        description=cast(str, scenario["description"]),
        launch_site_id=launch_site.site_id,
        target=cast(Waypoint, scenario["target"]),
        sla_min=float(scenario["sla_min"]),
    )
    return SimState(
        seed=seed,
        scenario_id=mission.mission_id,
        tier=str(scenario["tier"]),
        sim_time_s=0.0,
        aircraft=Aircraft(
            lat=launch_site.lat,
            lon=launch_site.lon,
            alt_ft=0.0,
            current_site_id=launch_site.site_id,
        ),
        mission=mission,
        home_site_id=launch_site.site_id,
        wind=wind,
        rng_state=_pack_rng_state(cast(Mapping[str, Any], rng.bit_generator.state)),
        scenario_par=dict(cast(Mapping[str, float], scenario["par"])),
        events=[dict(event) for event in scenario.get("events", [])],
    )


def _scenario_briefing(scenario: dict[str, Any]) -> str:
    launch_site = SITES[cast(str, scenario["launch_site_id"])]
    target = cast(Waypoint, scenario["target"])
    events = list(scenario.get("events", []))
    event_line = (
        "- Airspace/events: no active pop-up restrictions in this T0 scenario; "
        "static airspace/geofence checks still apply\n"
        if not events
        else f"- Airspace/events: {scenario.get('tier', 'T1')} scenario with seeded operational interrupts; "
        "acknowledge alerts and re-check affected constraints before continuing\n"
    )
    return (
        "Mission briefing\n"
        f"- Scenario: {scenario['scenario_id']}\n"
        f"- Tier: {scenario.get('tier', 'T0')}\n"
        f"- Launch/recovery: {launch_site.name} ({launch_site.site_id}) "
        f"{launch_site.lat:.4f}, {launch_site.lon:.4f}\n"
        f"- Task: {scenario['description']}\n"
        f"- Target: {target.name} at {target.lat:.4f}, {target.lon:.4f}\n"
        f"- SLA: complete within {float(scenario['sla_min']):.0f} simulated minutes\n"
        "- Weather: seeded Bay Area wind field; query get_weather for current "
        "winds, altitude shear, and gust-front forecast\n"
        f"{event_line}"
        "- Expected flow: query the console as needed, file a plan to the target, "
        "release payload/confirm work if appropriate, then command RTL after "
        "mission completion.\n"
    )


def _split_seed(base_seed: int, split: str, index: int) -> int:
    offsets = {"train": 0, "dev": 10_000, "eval": 20_000}
    return base_seed + offsets[split] + index


def _dataset(
    seed: int,
    max_examples: int,
    tier: str = "mixed_day5",
    split: str = "eval",
) -> Dataset:
    default_counts = {"train": DAY5_TRAIN_EXAMPLES, "dev": DAY5_DEV_EXAMPLES, "eval": DAY5_EVAL_EXAMPLES}
    if split not in default_counts:
        raise ValueError(f"unknown dataset split: {split}")
    count = default_counts[split] if max_examples < 0 else max_examples
    rows = []
    for index in range(count):
        scenario_seed = _split_seed(seed, split, index)
        scenario = _scenario_for_index(index, seed=scenario_seed, tier=tier)
        rows.append(
            {
                "question": _scenario_briefing(scenario),
                "answer": scenario["scenario_id"],
                "info": {
                    "seed": scenario_seed,
                    "scenario_index": index,
                    "scenario_id": scenario["scenario_id"],
                    "tier": scenario["tier"],
                },
            }
        )
    return Dataset.from_list(rows)


def _as_waypoint(value: Mapping[str, Any]) -> Waypoint:
    lat = float(value["lat"])
    lon = float(value["lon"])
    name = value.get("name", "")
    return Waypoint(lat=lat, lon=lon, name=str(name) if name is not None else "")


def _mission_distance_to_target(sim: SimState) -> float:
    return haversine_nm(
        sim.aircraft.lat,
        sim.aircraft.lon,
        sim.mission.target.lat,
        sim.mission.target.lon,
    )


def _wind_component_along_track_kt(
    wind_dir_from_deg: float,
    wind_speed_kt: float,
    track_deg: float,
) -> float:
    wind_to_deg = (wind_dir_from_deg + 180.0) % 360.0
    angle = math.radians((wind_to_deg - track_deg + 540.0) % 360.0 - 180.0)
    return wind_speed_kt * math.cos(angle)


def _groundspeed_kt(
    airspeed_kt: float,
    track_deg: float,
    wind_dir_from_deg: float = DEFAULT_WIND_DIR_FROM_DEG,
    wind_speed_kt: float = DEFAULT_WIND_SPEED_KT,
) -> float:
    return airspeed_kt + _wind_component_along_track_kt(
        wind_dir_from_deg=wind_dir_from_deg,
        wind_speed_kt=wind_speed_kt,
        track_deg=track_deg,
    )


def _segment_wind(sim: SimState, start: Waypoint, end: Waypoint, alt_ft: float) -> tuple[float, float]:
    return wind_at(
        lat=(start.lat + end.lat) / 2.0,
        lon=(start.lon + end.lon) / 2.0,
        alt_ft=alt_ft,
        t_s=sim.sim_time_s,
        wind_field=sim.wind,
    )


def _segment_energy_wh(
    distance_nm: float,
    airspeed_kt: float,
    start_alt_ft: float,
    end_alt_ft: float,
    landing: bool = False,
    *,
    track_deg: float | None = None,
    wind_dir_from_deg: float = DEFAULT_WIND_DIR_FROM_DEG,
    wind_speed_kt: float = DEFAULT_WIND_SPEED_KT,
    payload_kg: float = 0.0,
) -> float:
    if track_deg is None:
        groundspeed_kt = airspeed_kt
    else:
        groundspeed_kt = _groundspeed_kt(
            airspeed_kt=airspeed_kt,
            track_deg=track_deg,
            wind_dir_from_deg=wind_dir_from_deg,
            wind_speed_kt=wind_speed_kt,
        )
    if groundspeed_kt <= MIN_GROUNDSPEED_KT:
        return math.inf

    time_h = distance_nm / groundspeed_kt if groundspeed_kt > 0.0 else math.inf
    cruise_power_w = 180.0 + 0.0016 * airspeed_kt**3 + 22.0 * payload_kg
    climb_wh = max(0.0, end_alt_ft - start_alt_ft) * 0.003
    landing_wh = BATTERY_WH * 0.08 if landing else 0.0
    return cruise_power_w * time_h + climb_wh + landing_wh


def _execution_multiplier(sim: SimState) -> float:
    rng = np.random.default_rng()
    rng.bit_generator.state = _unpack_rng_state(sim.rng_state)
    multiplier = float(rng.normal(1.0, 0.03))
    sim.rng_state = _pack_rng_state(cast(Mapping[str, Any], rng.bit_generator.state))
    sim.rng_draws += 1
    return min(1.10, max(0.90, multiplier))


def _advance_segment(
    sim: SimState,
    waypoint: Waypoint,
    alt_ft: float,
    airspeed_kt: float,
    *,
    landing: bool = False,
) -> dict[str, Any]:
    start_lat = sim.aircraft.lat
    start_lon = sim.aircraft.lon
    start_alt = sim.aircraft.alt_ft
    distance_nm = haversine_nm(start_lat, start_lon, waypoint.lat, waypoint.lon)
    track_deg = bearing_deg(start_lat, start_lon, waypoint.lat, waypoint.lon)
    wind_dir_from_deg, wind_speed_kt = _segment_wind(
        sim,
        Waypoint(start_lat, start_lon),
        waypoint,
        alt_ft,
    )
    groundspeed_kt = _groundspeed_kt(
        airspeed_kt=airspeed_kt,
        track_deg=track_deg,
        wind_dir_from_deg=wind_dir_from_deg,
        wind_speed_kt=wind_speed_kt,
    )
    if groundspeed_kt <= MIN_GROUNDSPEED_KT:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.aircraft.status = "holding"
        return {
            "from": {"lat": start_lat, "lon": start_lon, "alt_ft": start_alt},
            "to": {"lat": waypoint.lat, "lon": waypoint.lon, "alt_ft": alt_ft},
            "distance_nm": round(distance_nm, 3),
            "track_deg": round(track_deg, 1),
            "groundspeed_kt": round(groundspeed_kt, 2),
            "wind": {
                "dir_deg_from": round(wind_dir_from_deg, 1),
                "speed_kt": round(wind_speed_kt, 1),
            },
            "error": "segment_infeasible_groundspeed",
        }
    multiplier = _execution_multiplier(sim)
    duration_s = (distance_nm / groundspeed_kt) * 3600.0 * multiplier
    energy_wh = _segment_energy_wh(
        distance_nm=distance_nm,
        airspeed_kt=airspeed_kt,
        start_alt_ft=start_alt,
        end_alt_ft=0.0 if landing else alt_ft,
        landing=landing,
        track_deg=track_deg,
        wind_dir_from_deg=wind_dir_from_deg,
        wind_speed_kt=wind_speed_kt,
    ) * multiplier

    sim.sim_time_s += duration_s
    sim.aircraft.battery_wh -= energy_wh
    sim.aircraft.lat = waypoint.lat
    sim.aircraft.lon = waypoint.lon
    sim.aircraft.alt_ft = 0.0 if landing else alt_ft
    sim.aircraft.status = "landed" if landing else "holding"
    sim.aircraft.current_site_id = None

    if sim.aircraft.battery_wh <= 0.0:
        sim.aircraft.battery_wh = 0.0
        sim.aircraft.status = "lost"
        sim.is_terminal = True
        sim.terminal_reason = "aircraft_lost_battery_depleted"
        _record_unique(sim.hard_safety_violations, "aircraft_loss:battery_depleted")

    return {
        "from": {"lat": start_lat, "lon": start_lon, "alt_ft": start_alt},
        "to": {"lat": waypoint.lat, "lon": waypoint.lon, "alt_ft": sim.aircraft.alt_ft},
        "distance_nm": round(distance_nm, 3),
        "track_deg": round(track_deg, 1),
        "groundspeed_kt": round(groundspeed_kt, 2),
        "wind": {
            "dir_deg_from": round(wind_dir_from_deg, 1),
            "speed_kt": round(wind_speed_kt, 1),
        },
        "execution_multiplier": round(multiplier, 4),
        "duration_s": round(duration_s, 1),
        "energy_wh": round(energy_wh, 2),
    }


def _snapshot(sim: SimState, event: str) -> dict[str, Any]:
    data = asdict(sim)
    data["event"] = event
    data["battery_pct"] = round(100.0 * sim.aircraft.battery_wh / BATTERY_WH, 2)
    data["mission_distance_to_target_nm"] = round(_mission_distance_to_target(sim), 3)
    return data


def _last_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Compact previous tool result for telemetry without recursive payloads."""

    summary: dict[str, Any] = {}
    for key in ("ok", "error", "warnings", "errors", "active_failsafe", "landing_site"):
        if key in result:
            summary[key] = result[key]
    if "segments" in result and isinstance(result["segments"], Sequence):
        summary["segment_count"] = len(result["segments"])
    if "segment" in result and isinstance(result["segment"], Mapping):
        segment = cast(Mapping[str, Any], result["segment"])
        summary["segment"] = {
            key: segment[key]
            for key in ("distance_nm", "duration_s", "energy_wh")
            if key in segment
        }
    if "conflicts" in result and isinstance(result["conflicts"], Sequence):
        summary["conflict_count"] = len(result["conflicts"])
    return summary


def _gust_front_summary(sim: SimState) -> dict[str, Any] | None:
    if sim.wind.gust_front is None:
        return None
    front = sim.wind.gust_front
    return {
        "forecast_eta_min": round(front.forecast_eta_s / 60.0, 1),
        "forecast_error_min": round(front.eta_error_s / 60.0, 1),
        "speed_kt": round(front.speed_kt, 1),
        "speed_delta_kt": round(front.speed_delta_kt, 1),
        "dir_shift_deg": round(front.dir_shift_deg, 1),
    }


def _telemetry(sim: SimState) -> dict[str, Any]:
    wind_dir, wind_speed = wind_at(
        sim.aircraft.lat,
        sim.aircraft.lon,
        sim.aircraft.alt_ft,
        sim.sim_time_s,
        sim.wind,
    )
    return {
        "ok": True,
        "sim_time_min": round(sim.sim_time_s / 60.0, 2),
        "aircraft": {
            "lat": round(sim.aircraft.lat, 5),
            "lon": round(sim.aircraft.lon, 5),
            "alt_ft": round(sim.aircraft.alt_ft, 1),
            "status": sim.aircraft.status,
            "battery_wh": round(sim.aircraft.battery_wh, 2),
            "battery_pct": round(100.0 * sim.aircraft.battery_wh / BATTERY_WH, 2),
        },
        "mission": {
            "mission_id": sim.mission.mission_id,
            "description": sim.mission.description,
            "target": asdict(sim.mission.target),
            "status": sim.mission.status,
            "sla_min": sim.mission.sla_min,
            "completed_time_min": (
                round(sim.mission.completed_time_s / 60.0, 2)
                if sim.mission.completed_time_s is not None
                else None
            ),
        },
        "weather": {
            "wind_dir_deg_from": round(wind_dir, 1),
            "wind_speed_kt": round(wind_speed, 1),
            "gust_front": _gust_front_summary(sim),
        },
        "active_failsafe": sim.active_failsafe,
        "active_events": sim.active_events,
        "alerts": sim.alerts,
        "acknowledged_alerts": sim.acknowledged_alerts,
        "closed_site_ids": sim.closed_sites,
        "overrides": sim.overrides,
        "payload_released": sim.payload_released,
        "last_action": sim.last_action,
        "last_result_summary": _last_result_summary(sim.last_result),
    }


def _append_alert(
    sim: SimState,
    alert_id: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    sim.alerts.append(
        {
            "alert_id": alert_id,
            "sim_time_s": round(sim.sim_time_s, 1),
            "message": message,
            "metadata": dict(metadata or {}),
        }
    )


def _record_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _apply_event_effect(sim: SimState, event: dict[str, Any]) -> None:
    params = event.get("params", {})
    if not isinstance(params, Mapping):
        params = {}
    event_type = str(event.get("type", ""))
    if event_type == "WIND_SHIFT":
        sim.wind = WindField(
            base_dir_deg_from=(
                sim.wind.base_dir_deg_from + float(params.get("dir_delta_deg", 0.0))
            )
            % 360.0,
            base_speed_kt=max(
                0.0,
                sim.wind.base_speed_kt + float(params.get("speed_delta_kt", 0.0)),
            ),
            blobs=sim.wind.blobs,
            gust_front=sim.wind.gust_front,
        )
    elif event_type == "BATT_DEGRADE":
        capacity_loss_pct = float(params.get("capacity_loss_pct", 0.0))
        sim.aircraft.battery_wh = max(
            0.0,
            sim.aircraft.battery_wh - BATTERY_WH * capacity_loss_pct / 100.0,
        )
        if sim.aircraft.battery_wh <= 0.0:
            sim.aircraft.status = "lost"
            sim.is_terminal = True
            sim.terminal_reason = "aircraft_lost_battery_depleted"
            _record_unique(sim.hard_safety_violations, "aircraft_loss:battery_depleted")
    elif event_type == "SITE_CLOSED":
        site_id = str(params.get("site_id", ""))
        if site_id in SITES:
            _record_unique(sim.closed_sites, site_id)


def _apply_due_events(sim: SimState) -> list[dict[str, Any]]:
    activated: list[dict[str, Any]] = []
    for event in sim.events:
        if event.get("status") != "pending":
            continue
        if float(event.get("trigger_time_s", math.inf)) > sim.sim_time_s:
            continue
        event["status"] = "active"
        event["activated_time_s"] = round(sim.sim_time_s, 1)
        event_id = str(event.get("event_id", event.get("type", "event")))
        _record_unique(sim.active_events, event_id)
        _apply_event_effect(sim, event)
        _append_alert(
            sim,
            event_id,
            str(event.get("message", event.get("type", "Operational event"))),
            {
                "event_type": event.get("type"),
                "params": event.get("params", {}),
            },
        )
        activated.append(event)
    return activated


def _event_interrupt_result(sim: SimState, tool_name: str) -> dict[str, Any] | None:
    if not sim.active_events:
        return None
    sim.last_action = tool_name
    sim.last_result = {
        "ok": False,
        "error": "event_interrupt",
        "active_events": list(sim.active_events),
        "alerts": sim.alerts,
        "telemetry": _telemetry(sim),
    }
    return sim.last_result


def _charge_latency(sim: SimState, seconds: float, tool_name: str) -> dict[str, Any] | None:
    sim.sim_time_s += seconds
    _apply_due_events(sim)
    if tool_name in EVENT_BLOCKED_TOOLS:
        return _event_interrupt_result(sim, tool_name)
    return None


def _route_validation(
    sim: SimState,
    waypoints: Sequence[Waypoint],
    alt_ft: float,
    airspeed_kt: float,
) -> tuple[list[str], list[str], list[AirspaceZone]]:
    errors: list[str] = []
    warnings: list[str] = []
    geofence_conflicts: list[AirspaceZone] = []

    if not 100.0 <= alt_ft <= 400.0:
        errors.append("altitude_outside_day3_policy_100_400_ft")
    if not 20.0 <= airspeed_kt <= 45.0:
        errors.append("airspeed_outside_airframe_envelope_20_45_kt")
    if any(not _in_bounds(wp.lat, wp.lon) for wp in waypoints):
        errors.append("route_leaves_san_francisco_bay_area_box")

    start = Waypoint(sim.aircraft.lat, sim.aircraft.lon, "current_position")
    for waypoint in waypoints:
        track_deg = bearing_deg(start.lat, start.lon, waypoint.lat, waypoint.lon)
        wind_dir_from_deg, wind_speed_kt = _segment_wind(sim, start, waypoint, alt_ft)
        if (
            _groundspeed_kt(
                airspeed_kt=airspeed_kt,
                track_deg=track_deg,
                wind_dir_from_deg=wind_dir_from_deg,
                wind_speed_kt=wind_speed_kt,
            )
            <= MIN_GROUNDSPEED_KT
        ):
            errors.append("segment_infeasible_groundspeed")

        safe_alt_ft = max(
            min_safe_altitude_ft(start.lat, start.lon),
            min_safe_altitude_ft(waypoint.lat, waypoint.lon),
        )
        if alt_ft < safe_alt_ft:
            warnings.append(f"min_safe_altitude_violation:{safe_alt_ft:.0f}ft_required")

        for zone in _airspace_conflicts_for_segment(sim, start, waypoint, alt_ft):
            if zone.authorization_required:
                geofence_conflicts.append(zone)
            else:
                warnings.append(f"airspace_advisory:{zone.zone_id}")
        start = waypoint

    dedup_errors = list(dict.fromkeys(errors))
    dedup_warnings = list(dict.fromkeys(warnings))
    dedup_geofence = list({zone.zone_id: zone for zone in geofence_conflicts}.values())
    return dedup_errors, dedup_warnings, dedup_geofence


def _trigger_geofence_hold(sim: SimState, zones: Sequence[AirspaceZone]) -> dict[str, Any]:
    sim.active_failsafe = "GEOFENCE_HOLD"
    if sim.aircraft.status == "airborne":
        sim.aircraft.status = "holding"
    zone_payload = [
        {
            "zone_id": zone.zone_id,
            "name": zone.name,
            "floor_ft": zone.floor_ft,
            "ceiling_ft": zone.ceiling_ft,
            "class": zone.airspace_class,
        }
        for zone in zones
    ]
    _append_alert(
        sim,
        "GEOFENCE_HOLD",
        "Predicted unauthorized airspace incursion; autopilot holding before entry.",
        {"zones": zone_payload},
    )
    for zone in zones:
        # Preserve each repeat: deduplication made a file/hold loop nearly free.
        sim.procedure_violations.append(f"filed_known_conflict:{zone.zone_id}")
    return {
        "ok": False,
        "error": "failsafe:GEOFENCE_HOLD",
        "active_failsafe": sim.active_failsafe,
        "conflicts": zone_payload,
        "telemetry": _telemetry(sim),
    }


def _trigger_low_batt_rtl(sim: SimState) -> dict[str, Any]:
    sim.active_failsafe = "LOW_BATT_RTL"
    site = _nearest_site(sim.aircraft.lat, sim.aircraft.lon, sim.closed_sites)
    _append_alert(
        sim,
        "LOW_BATT_RTL",
        "Battery at or below 30%; autopilot returning to nearest recovery site.",
        {
            "battery_pct": round(100.0 * sim.aircraft.battery_wh / BATTERY_WH, 2),
            "site_id": site.site_id,
        },
    )
    segment = _advance_segment(
        sim=sim,
        waypoint=Waypoint(site.lat, site.lon, site.name),
        alt_ft=max(sim.aircraft.alt_ft, 250.0),
        airspeed_kt=DEFAULT_AIRSPEED_KT,
        landing=True,
    )
    sim.aircraft.current_site_id = site.site_id
    if sim.mission.status == "pending" and sim.aircraft.status != "lost":
        sim.mission.status = "failed"
        sim.mission.failure_reason = "low_batt_rtl_before_mission_completion"
    _mark_terminal_if_done(sim)
    return {
        "active_failsafe": sim.active_failsafe,
        "landing_site": asdict(site),
        "segment": segment,
    }


def _mark_terminal_if_done(sim: SimState) -> None:
    if sim.aircraft.status == "lost":
        sim.is_terminal = True
        sim.terminal_reason = sim.terminal_reason or "aircraft_lost"
    elif sim.aircraft.status == "landed" and sim.mission.status in {
        "completed",
        "failed",
    }:
        sim.is_terminal = True
        sim.terminal_reason = f"aircraft_landed_mission_{sim.mission.status}"


def _airspace_zone_payload(zone: AirspaceZone) -> dict[str, Any]:
    return {
        "zone_id": zone.zone_id,
        "name": zone.name,
        "floor_ft": zone.floor_ft,
        "ceiling_ft": zone.ceiling_ft,
        "class": zone.airspace_class,
        "authorization_required": zone.authorization_required,
        "polygon": zone.polygon,
    }


def _execute_get_telemetry(sim: SimState, _args: Mapping[str, Any]) -> dict[str, Any]:
    _charge_latency(sim, READ_TOOL_LATENCY_S, "get_telemetry")
    sim.last_action = "get_telemetry"
    sim.last_result = _telemetry(sim)
    return sim.last_result


def _execute_get_weather(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    _charge_latency(sim, READ_TOOL_LATENCY_S, "get_weather")
    lat = float(args.get("lat", sim.aircraft.lat))
    lon = float(args.get("lon", sim.aircraft.lon))
    alt_ft = float(args.get("alt_ft", sim.aircraft.alt_ft))
    current = wind_at(lat, lon, alt_ft, sim.sim_time_s, sim.wind)
    forecast_15 = wind_at(lat, lon, alt_ft, sim.sim_time_s + 15.0 * 60.0, sim.wind)
    forecast_30 = wind_at(lat, lon, alt_ft, sim.sim_time_s + 30.0 * 60.0, sim.wind)
    sim.last_action = "get_weather"
    sim.last_result = {
        "ok": True,
        "location": {"lat": round(lat, 5), "lon": round(lon, 5), "alt_ft": round(alt_ft, 1)},
        "current": {
            "dir_deg_from": round(current[0], 1),
            "speed_kt": round(current[1], 1),
        },
        "forecast": [
            {
                "minutes_ahead": 15,
                "dir_deg_from": round(forecast_15[0], 1),
                "speed_kt": round(forecast_15[1], 1),
            },
            {
                "minutes_ahead": 30,
                "dir_deg_from": round(forecast_30[0], 1),
                "speed_kt": round(forecast_30[1], 1),
            },
        ],
        "gust_front": _gust_front_summary(sim),
    }
    return sim.last_result


def _execute_get_airspace(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    _charge_latency(sim, READ_TOOL_LATENCY_S, "get_airspace")
    conflicts: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    if "route" in args:
        try:
            raw_route = args["route"]
            if not isinstance(raw_route, Sequence) or isinstance(raw_route, str):
                raise ValueError("route must be an array of waypoint objects")
            waypoints = [_as_waypoint(cast(Mapping[str, Any], item)) for item in raw_route]
            alt_ft = float(args.get("alt_ft", sim.current_altitude_target_ft))
            errors, warnings, zones = _route_validation(sim, waypoints, alt_ft, sim.current_airspeed_kt)
            conflicts = [_airspace_zone_payload(zone) for zone in zones]
        except (TypeError, ValueError, KeyError) as exc:
            sim.sim_time_s += INVALID_ACTION_LATENCY_S
            errors = [f"invalid_arguments:{exc}"]
    sim.last_action = "get_airspace"
    sim.last_result = {
        "ok": not errors,
        "zones": [_airspace_zone_payload(zone) for zone in _all_airspace_zones(sim)],
        "route_errors": errors,
        "route_warnings": warnings,
        "route_conflicts": conflicts,
    }
    return sim.last_result


def _execute_get_mission_status(sim: SimState, _args: Mapping[str, Any]) -> dict[str, Any]:
    _charge_latency(sim, READ_TOOL_LATENCY_S, "get_mission_status")
    sim.last_action = "get_mission_status"
    sim.last_result = {
        "ok": True,
        "mission": asdict(sim.mission),
        "distance_to_target_nm": round(_mission_distance_to_target(sim), 3),
        "payload_released": sim.payload_released,
    }
    return sim.last_result


def _execute_get_sites(sim: SimState, _args: Mapping[str, Any]) -> dict[str, Any]:
    _charge_latency(sim, READ_TOOL_LATENCY_S, "get_sites")
    sim.last_action = "get_sites"
    sim.last_result = {
        "ok": True,
        "sites": [asdict(site) for site in SITES.values()],
        "nearest_site_id": _nearest_site(sim.aircraft.lat, sim.aircraft.lon, sim.closed_sites).site_id,
        "closed_site_ids": list(sim.closed_sites),
    }
    return sim.last_result


def _execute_file_flight_plan(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    interrupt = _charge_latency(sim, TOOL_LATENCY_S, "file_flight_plan")
    if interrupt is not None:
        return interrupt

    try:
        raw_waypoints = args["waypoints"]
        if not isinstance(raw_waypoints, Sequence) or isinstance(raw_waypoints, str):
            raise ValueError("waypoints must be a non-empty array of objects")
        waypoints = [_as_waypoint(cast(Mapping[str, Any], item)) for item in raw_waypoints]
        if not waypoints:
            raise ValueError("waypoints must contain at least one point")
        alt_ft = float(args["alt_ft"])
        airspeed_kt = float(args["airspeed_kt"])
        lost_link_plan = str(args["lost_link_plan"])
    except (KeyError, TypeError, ValueError) as exc:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "file_flight_plan"
        sim.last_result = {"ok": False, "error": f"invalid_arguments: {exc}"}
        return sim.last_result

    if sim.aircraft.status not in {"ground", "holding"}:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "file_flight_plan"
        sim.last_result = {
            "ok": False,
            "error": f"cannot_file_plan_from_status:{sim.aircraft.status}",
        }
        return sim.last_result

    errors, warnings, geofence_conflicts = _route_validation(
        sim=sim,
        waypoints=waypoints,
        alt_ft=alt_ft,
        airspeed_kt=airspeed_kt,
    )
    if errors:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "file_flight_plan"
        sim.last_result = {"ok": False, "errors": errors, "warnings": warnings}
        return sim.last_result
    if geofence_conflicts:
        sim.last_action = "file_flight_plan"
        sim.last_result = _trigger_geofence_hold(sim, geofence_conflicts)
        return sim.last_result

    sim.active_failsafe = None
    sim.aircraft.status = "airborne"
    sim.aircraft.current_site_id = None
    sim.current_plan = waypoints
    sim.lost_link_plan = lost_link_plan
    sim.current_altitude_target_ft = alt_ft
    sim.current_airspeed_kt = airspeed_kt
    segments = []
    for waypoint in waypoints:
        segments.append(
            _advance_segment(
                sim=sim,
                waypoint=waypoint,
                alt_ft=alt_ft,
                airspeed_kt=airspeed_kt,
            )
        )
        if sim.is_terminal:
            break
        if sim.mission.status == "pending" and _mission_distance_to_target(sim) <= 0.2:
            sim.mission.status = "completed"
            sim.mission.completed_time_s = sim.sim_time_s
        if sim.aircraft.battery_wh <= LOW_BATT_RTL_THRESHOLD_WH:
            segments.append({"failsafe": _trigger_low_batt_rtl(sim)})
            break

    sim.last_action = "file_flight_plan"
    sim.last_result = {
        "ok": not sim.is_terminal,
        "warnings": warnings,
        "segments": segments,
        "telemetry": _telemetry(sim),
    }
    _mark_terminal_if_done(sim)
    return sim.last_result


def _execute_amend_route(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return _execute_file_flight_plan(
            sim,
            {
                "waypoints": args["waypoints_from_current"],
                "alt_ft": sim.current_altitude_target_ft,
                "airspeed_kt": sim.current_airspeed_kt,
                "lost_link_plan": sim.lost_link_plan,
            },
        )
    finally:
        sim.last_action = "amend_route"


def _execute_set_altitude(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    interrupt = _charge_latency(sim, TOOL_LATENCY_S, "set_altitude")
    if interrupt is not None:
        return interrupt
    try:
        alt_ft = float(args["ft"])
    except (KeyError, TypeError, ValueError) as exc:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "set_altitude"
        sim.last_result = {"ok": False, "error": f"invalid_arguments:{exc}"}
        return sim.last_result
    if not 100.0 <= alt_ft <= 400.0:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "set_altitude"
        sim.last_result = {"ok": False, "error": "altitude_outside_day3_policy_100_400_ft"}
        return sim.last_result
    sim.current_altitude_target_ft = alt_ft
    if sim.aircraft.status in {"airborne", "holding"}:
        sim.aircraft.alt_ft = alt_ft
    sim.last_action = "set_altitude"
    sim.last_result = {"ok": True, "alt_ft": alt_ft, "telemetry": _telemetry(sim)}
    return sim.last_result


def _execute_set_speed(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    interrupt = _charge_latency(sim, TOOL_LATENCY_S, "set_speed")
    if interrupt is not None:
        return interrupt
    try:
        airspeed_kt = float(args["kt"])
    except (KeyError, TypeError, ValueError) as exc:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "set_speed"
        sim.last_result = {"ok": False, "error": f"invalid_arguments:{exc}"}
        return sim.last_result
    if not 20.0 <= airspeed_kt <= 45.0:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "set_speed"
        sim.last_result = {"ok": False, "error": "airspeed_outside_airframe_envelope_20_45_kt"}
        return sim.last_result
    sim.current_airspeed_kt = airspeed_kt
    sim.last_action = "set_speed"
    sim.last_result = {"ok": True, "airspeed_kt": airspeed_kt, "telemetry": _telemetry(sim)}
    return sim.last_result


def _execute_hold(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    _charge_latency(sim, TOOL_LATENCY_S, "hold")
    try:
        minutes = float(args["minutes"])
    except (KeyError, TypeError, ValueError) as exc:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "hold"
        sim.last_result = {"ok": False, "error": f"invalid_arguments:{exc}"}
        return sim.last_result
    if minutes <= 0.0 or minutes > 60.0:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "hold"
        sim.last_result = {"ok": False, "error": "hold_minutes_outside_0_60"}
        return sim.last_result
    sim.sim_time_s += minutes * 60.0
    _apply_due_events(sim)
    sim.hold_until_s = sim.sim_time_s
    if sim.aircraft.status in {"airborne", "holding"}:
        sim.aircraft.status = "holding"
    sim.last_action = "hold"
    sim.last_result = {"ok": True, "held_minutes": minutes, "telemetry": _telemetry(sim)}
    return sim.last_result


def _execute_resume(sim: SimState, _args: Mapping[str, Any]) -> dict[str, Any]:
    interrupt = _charge_latency(sim, TOOL_LATENCY_S, "resume")
    if interrupt is not None:
        return interrupt
    if sim.active_failsafe is not None:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "resume"
        sim.last_result = {
            "ok": False,
            "error": f"cannot_resume_active_failsafe:{sim.active_failsafe}",
        }
        return sim.last_result
    if sim.aircraft.status == "holding":
        sim.aircraft.status = "airborne"
    sim.hold_until_s = None
    sim.last_action = "resume"
    sim.last_result = {"ok": True, "telemetry": _telemetry(sim)}
    return sim.last_result


def _execute_command_rtl(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    _charge_latency(sim, TOOL_LATENCY_S, "command_rtl")
    site_id = args.get("site_id")
    if site_id is None or site_id == "":
        site = _nearest_site(sim.aircraft.lat, sim.aircraft.lon, sim.closed_sites)
    elif str(site_id) in SITES and str(site_id) not in sim.closed_sites:
        site = SITES[str(site_id)]
    elif str(site_id) in sim.closed_sites:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "command_rtl"
        sim.last_result = {"ok": False, "error": f"closed_site_id:{site_id}"}
        return sim.last_result
    else:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "command_rtl"
        sim.last_result = {"ok": False, "error": f"unknown_site_id:{site_id}"}
        return sim.last_result

    if sim.aircraft.status == "lost":
        sim.last_action = "command_rtl"
        sim.last_result = {"ok": False, "error": "aircraft_already_lost"}
        _mark_terminal_if_done(sim)
        return sim.last_result

    sim.active_failsafe = None
    target = Waypoint(site.lat, site.lon, site.name)
    segment = _advance_segment(
        sim=sim,
        waypoint=target,
        alt_ft=max(sim.aircraft.alt_ft, 250.0),
        airspeed_kt=DEFAULT_AIRSPEED_KT,
        landing=True,
    )
    sim.aircraft.current_site_id = site.site_id
    if sim.mission.status == "pending" and sim.aircraft.status != "lost":
        sim.mission.status = "failed"
        sim.mission.failure_reason = "returned_to_land_before_mission_completion"

    sim.last_action = "command_rtl"
    sim.last_result = {
        "ok": sim.aircraft.status != "lost",
        "landing_site": asdict(site),
        "segment": segment,
        "telemetry": _telemetry(sim),
    }
    _mark_terminal_if_done(sim)
    return sim.last_result


def _execute_land_now(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    interrupt = _charge_latency(sim, HIGH_IMPACT_TOOL_LATENCY_S, "land_now")
    if interrupt is not None:
        return interrupt
    site_id = args.get("site_id")
    if site_id is None or site_id == "":
        target = Waypoint(sim.aircraft.lat, sim.aircraft.lon, "current_position")
        site: Site | None = None
    elif str(site_id) in SITES and str(site_id) not in sim.closed_sites:
        site = SITES[str(site_id)]
        target = Waypoint(site.lat, site.lon, site.name)
    elif str(site_id) in sim.closed_sites:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "land_now"
        sim.last_result = {"ok": False, "error": f"closed_site_id:{site_id}"}
        return sim.last_result
    else:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "land_now"
        sim.last_result = {"ok": False, "error": f"unknown_site_id:{site_id}"}
        return sim.last_result
    segment = _advance_segment(
        sim=sim,
        waypoint=target,
        alt_ft=max(sim.aircraft.alt_ft, sim.current_altitude_target_ft),
        airspeed_kt=sim.current_airspeed_kt,
        landing=True,
    )
    sim.aircraft.current_site_id = site.site_id if site is not None else None
    if sim.mission.status == "pending" and sim.aircraft.status != "lost":
        sim.mission.status = "failed"
        sim.mission.failure_reason = "land_now_before_mission_completion"
    sim.last_action = "land_now"
    sim.last_result = {
        "ok": sim.aircraft.status != "lost",
        "landing_site": asdict(site) if site is not None else None,
        "segment": segment,
        "telemetry": _telemetry(sim),
    }
    _mark_terminal_if_done(sim)
    return sim.last_result


def _execute_release_payload(sim: SimState, _args: Mapping[str, Any]) -> dict[str, Any]:
    interrupt = _charge_latency(sim, TOOL_LATENCY_S, "release_payload")
    if interrupt is not None:
        return interrupt
    if _mission_distance_to_target(sim) > 0.2:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "release_payload"
        sim.last_result = {
            "ok": False,
            "error": "not_at_payload_release_point",
            "distance_to_target_nm": round(_mission_distance_to_target(sim), 3),
        }
        return sim.last_result
    sim.payload_released = True
    if sim.mission.status == "pending":
        sim.mission.status = "completed"
        sim.mission.completed_time_s = sim.sim_time_s
    sim.last_action = "release_payload"
    sim.last_result = {"ok": True, "mission": asdict(sim.mission), "telemetry": _telemetry(sim)}
    return sim.last_result


def _execute_abort_mission(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    _charge_latency(sim, HIGH_IMPACT_TOOL_LATENCY_S, "abort_mission")
    mission_id = str(args.get("mission_id", ""))
    if mission_id != sim.mission.mission_id:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "abort_mission"
        sim.last_result = {"ok": False, "error": f"unknown_mission_id:{mission_id}"}
        return sim.last_result
    if sim.mission.status == "pending":
        sim.mission.status = "failed"
        sim.mission.failure_reason = "operator_abort"
    sim.last_action = "abort_mission"
    sim.last_result = {"ok": True, "mission": asdict(sim.mission), "telemetry": _telemetry(sim)}
    _mark_terminal_if_done(sim)
    return sim.last_result


def _execute_override_failsafe(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    interrupt = _charge_latency(sim, HIGH_IMPACT_TOOL_LATENCY_S, "override_failsafe")
    if interrupt is not None:
        return interrupt
    failsafe_id = str(args.get("id", ""))
    justification = str(args.get("justification_code", ""))
    if justification not in VALID_OVERRIDE_JUSTIFICATIONS:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.procedure_violations.append("invalid_override_justification")
        sim.last_action = "override_failsafe"
        sim.last_result = {"ok": False, "error": f"invalid_justification_code:{justification}"}
        return sim.last_result
    if sim.active_failsafe != failsafe_id:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.procedure_violations.append(f"override_inactive_failsafe:{failsafe_id}")
        sim.last_action = "override_failsafe"
        sim.last_result = {
            "ok": False,
            "error": f"failsafe_not_active:{failsafe_id}",
            "active_failsafe": sim.active_failsafe,
        }
        return sim.last_result
    sim.overrides.append(
        {
            "id": failsafe_id,
            "justification_code": justification,
            "sim_time_s": round(sim.sim_time_s, 1),
        }
    )
    sim.active_failsafe = None
    sim.last_action = "override_failsafe"
    sim.last_result = {"ok": True, "overrides": sim.overrides, "telemetry": _telemetry(sim)}
    return sim.last_result


def _execute_acknowledge(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    _charge_latency(sim, READ_TOOL_LATENCY_S, "acknowledge")
    alert_id = str(args.get("alert_id", ""))
    if alert_id not in {str(alert["alert_id"]) for alert in sim.alerts}:
        sim.sim_time_s += INVALID_ACTION_LATENCY_S
        sim.last_action = "acknowledge"
        sim.last_result = {"ok": False, "error": f"unknown_alert_id:{alert_id}"}
        return sim.last_result
    if alert_id not in sim.acknowledged_alerts:
        sim.acknowledged_alerts.append(alert_id)
    if alert_id in sim.active_events:
        sim.active_events.remove(alert_id)
        _record_unique(sim.acknowledged_events, alert_id)
    sim.last_action = "acknowledge"
    sim.last_result = {"ok": True, "acknowledged_alerts": sim.acknowledged_alerts}
    return sim.last_result


def _parse_tool_args(raw_args: str) -> dict[str, Any]:
    if raw_args == "":
        return {}
    parsed = json.loads(raw_args)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to a JSON object")
    return parsed


def _message_role(message: Any) -> str | None:
    if isinstance(message, Mapping):
        value = message.get("role")
    else:
        value = getattr(message, "role", None)
    return value if isinstance(value, str) else None


def _assistant_tool_calls(messages: Sequence[Any]) -> list[Any]:
    for message in reversed(messages):
        if _message_role(message) != "assistant":
            continue
        tool_calls = (
            message.get("tool_calls")
            if isinstance(message, Mapping)
            else getattr(message, "tool_calls", None)
        )
        if tool_calls:
            return list(tool_calls)
        return []
    return []


def _tool_call_field(tool_call: Any, field_name: str) -> Any:
    if isinstance(tool_call, Mapping):
        return tool_call.get(field_name)
    return getattr(tool_call, field_name)


def _state_sim(state: vf.State) -> SimState:
    wind_payload = state["sim_state"].get("wind")
    wind = (
        _wind_field_from_mapping(cast(Mapping[str, Any], wind_payload))
        if isinstance(wind_payload, Mapping)
        else _build_wind_field(np.random.default_rng(int(state["sim_state"]["seed"])))
    )
    return SimState(
        seed=int(state["sim_state"]["seed"]),
        scenario_id=str(state["sim_state"]["scenario_id"]),
        tier=str(state["sim_state"].get("tier", "T0")),
        sim_time_s=float(state["sim_state"]["sim_time_s"]),
        aircraft=Aircraft(**state["sim_state"]["aircraft"]),
        mission=Mission(
            **{
                **state["sim_state"]["mission"],
                "target": Waypoint(**state["sim_state"]["mission"]["target"]),
            }
        ),
        home_site_id=str(state["sim_state"]["home_site_id"]),
        wind=wind,
        rng_state=_pack_rng_state(
            cast(
                Mapping[str, Any],
                state["sim_state"].get("rng_state", np.random.default_rng(0).bit_generator.state),
            )
        ),
        scenario_par=dict(state["sim_state"].get("scenario_par", {})),
        events=[dict(event) for event in state["sim_state"].get("events", [])],
        active_events=list(state["sim_state"].get("active_events", [])),
        acknowledged_events=list(state["sim_state"].get("acknowledged_events", [])),
        closed_sites=list(state["sim_state"].get("closed_sites", [])),
        hard_safety_violations=list(state["sim_state"].get("hard_safety_violations", [])),
        procedure_violations=list(state["sim_state"].get("procedure_violations", [])),
        current_plan=[
            Waypoint(**waypoint) for waypoint in state["sim_state"].get("current_plan", [])
        ],
        lost_link_plan=str(state["sim_state"].get("lost_link_plan", "return_home")),
        rng_draws=int(state["sim_state"].get("rng_draws", 0)),
        active_failsafe=state["sim_state"].get("active_failsafe"),
        alerts=list(state["sim_state"].get("alerts", [])),
        acknowledged_alerts=list(state["sim_state"].get("acknowledged_alerts", [])),
        overrides=list(state["sim_state"].get("overrides", [])),
        payload_released=bool(state["sim_state"].get("payload_released", False)),
        hold_until_s=state["sim_state"].get("hold_until_s"),
        current_altitude_target_ft=float(
            state["sim_state"].get("current_altitude_target_ft", 250.0)
        ),
        current_airspeed_kt=float(
            state["sim_state"].get("current_airspeed_kt", DEFAULT_AIRSPEED_KT)
        ),
        last_action=str(state["sim_state"].get("last_action", "briefing")),
        last_result=dict(state["sim_state"].get("last_result", {})),
        is_terminal=bool(state["sim_state"].get("is_terminal", False)),
        terminal_reason=state["sim_state"].get("terminal_reason"),
    )


def _store_sim(state: vf.State, sim: SimState, event: str) -> None:
    state["sim_state"] = asdict(sim)
    state.setdefault("sim_log", []).append(_snapshot(sim, event))


def _final_log_snapshot(state: vf.State) -> Mapping[str, Any] | None:
    sim_log = state.get("sim_log")
    if not isinstance(sim_log, Sequence) or isinstance(sim_log, (str, bytes)) or not sim_log:
        return None
    final = sim_log[-1]
    return final if isinstance(final, Mapping) else None


def mission_value(state: vf.State) -> float:
    """Mission value reward, computed only from logged simulator state."""

    snapshot = _final_log_snapshot(state)
    if snapshot is None:
        return 0.0
    mission = snapshot.get("mission")
    if not isinstance(mission, Mapping) or mission.get("status") != "completed":
        return 0.0
    completed_time = mission.get("completed_time_s")
    sla_min = float(mission.get("sla_min", 45.0))
    value = float(mission.get("value", 1.0))
    if completed_time is None:
        return 0.0
    completed_min = float(completed_time) / 60.0
    if completed_min <= sla_min:
        return value
    decay = max(0.0, 1.0 - (completed_min - sla_min) / sla_min)
    return value * decay


def hard_safety(state: vf.State) -> float:
    """Hard safety penalty from logged sim outcomes only."""

    snapshot = _final_log_snapshot(state)
    if snapshot is None:
        return 0.0
    violations = set(snapshot.get("hard_safety_violations", []))
    aircraft = snapshot.get("aircraft", {})
    if isinstance(aircraft, Mapping) and aircraft.get("status") == "lost":
        violations.add("aircraft_loss")
    terminal_reason = str(snapshot.get("terminal_reason", ""))
    if terminal_reason.startswith("aircraft_lost"):
        violations.add("aircraft_loss")
    battery_pct = float(snapshot.get("battery_pct", 100.0))
    if battery_pct <= 12.0 and isinstance(aircraft, Mapping) and aircraft.get("status") != "landed":
        violations.add("crit_batt_airborne")
    return -5.0 * float(len(violations))


def margin_policy(state: vf.State) -> float:
    """Reserve, override, and clearance policy penalty from sim logs."""

    snapshot = _final_log_snapshot(state)
    if snapshot is None:
        return 0.0
    penalty = 0.0
    aircraft = snapshot.get("aircraft", {})
    battery_pct = float(snapshot.get("battery_pct", 100.0))
    if isinstance(aircraft, Mapping) and aircraft.get("status") == "landed":
        reserve_shortfall = max(0.0, BRIEFED_RESERVE_PCT - battery_pct) / BRIEFED_RESERVE_PCT
        penalty -= min(1.0, reserve_shortfall)
    overrides = snapshot.get("overrides", [])
    if isinstance(overrides, Sequence) and not isinstance(overrides, (str, bytes)):
        poor_outcome = battery_pct < BRIEFED_RESERVE_PCT or hard_safety(state) < 0.0
        if poor_outcome:
            penalty -= min(1.0, 0.25 * len(overrides))

    sim_log = state.get("sim_log", [])
    min_safe_hits = 0
    if isinstance(sim_log, Sequence) and not isinstance(sim_log, (str, bytes)):
        for entry in sim_log:
            if not isinstance(entry, Mapping):
                continue
            result = entry.get("last_result", {})
            if not isinstance(result, Mapping):
                continue
            warnings = result.get("warnings", [])
            if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
                min_safe_hits += sum(
                    1 for warning in warnings if str(warning).startswith("min_safe_altitude_violation")
                )
    penalty -= min(1.0, 0.2 * min_safe_hits)
    return max(-2.0, penalty)


def procedure(state: vf.State) -> float:
    """Procedural compliance penalty from alerts and logged violations."""

    snapshot = _final_log_snapshot(state)
    if snapshot is None:
        return 0.0
    alerts = snapshot.get("alerts", [])
    acknowledged = set(snapshot.get("acknowledged_alerts", []))
    unacked = 0
    if isinstance(alerts, Sequence) and not isinstance(alerts, (str, bytes)):
        for alert in alerts:
            if isinstance(alert, Mapping) and str(alert.get("alert_id", "")) not in acknowledged:
                unacked += 1
    violations = snapshot.get("procedure_violations", [])
    violation_count = len(violations) if isinstance(violations, Sequence) else 0
    return -0.1 * float(unacked + violation_count)


def efficiency(state: vf.State) -> float:
    """Efficiency penalty normalized by scenario par from the sim log."""

    snapshot = _final_log_snapshot(state)
    if snapshot is None:
        return 0.0
    par = snapshot.get("scenario_par", {})
    aircraft = snapshot.get("aircraft", {})
    if not isinstance(par, Mapping) or not isinstance(aircraft, Mapping):
        return 0.0
    par_time_s = max(1.0, float(par.get("time_s", 1.0)))
    par_energy_wh = max(1.0, float(par.get("energy_wh", 1.0)))
    time_ratio = max(0.0, float(snapshot.get("sim_time_s", 0.0)) / par_time_s - 1.0)
    energy_used_wh = max(0.0, BATTERY_WH - float(aircraft.get("battery_wh", BATTERY_WH)))
    energy_ratio = max(0.0, energy_used_wh / par_energy_wh - 1.0)
    return -0.3 * min(1.0, (time_ratio + energy_ratio) / 2.0)


def reward_breakdown(state: vf.State) -> dict[str, float]:
    """Return component scores using only saved simulator logs."""

    components = {
        "mission_value": mission_value(state),
        "hard_safety": hard_safety(state),
        "margin_policy": margin_policy(state),
        "procedure": procedure(state),
        "efficiency": efficiency(state),
    }
    components["total"] = sum(components.values())
    return components


class UAVOperatorEnv(vf.MultiTurnEnv):
    """Verifiers MultiTurnEnv for small-UAS operator decisions."""

    def __init__(
        self,
        sim_time_cap_min: int = 90,
        tier: str = "mixed_day5",
        wind_enabled: bool = True,
        gust_front_probability: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.sim_time_cap_s = float(sim_time_cap_min) * 60.0
        self.tier = tier
        self.wind_enabled = wind_enabled
        self.gust_front_probability = gust_front_probability
        super().__init__(**kwargs)

    async def setup_state(self, state: vf.State) -> vf.State:
        info = state.get("info", {})
        if not isinstance(info, Mapping):
            info = {}
        seed = int(info.get("seed", 0))
        scenario_index = int(info.get("scenario_index", 0))
        tier = str(info.get("tier", self.tier))
        sim = _build_sim(
            seed=seed,
            scenario_index=scenario_index,
            tier=tier,
            wind_enabled=self.wind_enabled,
            gust_front_probability=self.gust_front_probability,
        )
        _store_sim(state, sim, "briefing")
        return state

    @vf.stop
    async def sim_terminal(self, state: vf.State) -> bool:
        sim_state = state.get("sim_state")
        return isinstance(sim_state, Mapping) and bool(sim_state.get("is_terminal"))

    @vf.stop
    async def sim_time_cap_reached(self, state: vf.State) -> bool:
        sim_state = state.get("sim_state")
        if not isinstance(sim_state, Mapping):
            return False
        if float(sim_state.get("sim_time_s", 0.0)) < self.sim_time_cap_s:
            return False
        sim = _state_sim(state)
        sim.is_terminal = True
        sim.terminal_reason = "sim_time_cap_reached"
        if sim.mission.status == "pending":
            sim.mission.status = "failed"
            sim.mission.failure_reason = "sim_time_cap_reached"
        _store_sim(state, sim, "sim_time_cap_reached")
        return True

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **_kwargs: Any,
    ) -> vf.Messages:
        sim = _state_sim(state)
        tool_calls = _assistant_tool_calls(messages)
        if not tool_calls:
            sim.sim_time_s += INVALID_ACTION_LATENCY_S
            sim.last_action = "no_tool_call"
            sim.last_result = {
                "ok": False,
                "error": "No tool call detected. Use the operator console tools.",
                "available_tools": _available_tool_names(),
            }
            _store_sim(state, sim, "no_tool_call")
            return [vf.UserMessage(content=json.dumps(sim.last_result, sort_keys=True))]

        responses: list[Any] = []
        for tool_call in tool_calls:
            call_id = str(_tool_call_field(tool_call, "id"))
            name = str(_tool_call_field(tool_call, "name"))
            raw_args = str(_tool_call_field(tool_call, "arguments") or "{}")
            try:
                args = _parse_tool_args(raw_args)
                if name == "get_telemetry":
                    result = _execute_get_telemetry(sim, args)
                elif name == "get_weather":
                    result = _execute_get_weather(sim, args)
                elif name == "get_airspace":
                    result = _execute_get_airspace(sim, args)
                elif name == "get_mission_status":
                    result = _execute_get_mission_status(sim, args)
                elif name == "get_sites":
                    result = _execute_get_sites(sim, args)
                elif name == "file_flight_plan":
                    result = _execute_file_flight_plan(sim, args)
                elif name == "amend_route":
                    result = _execute_amend_route(sim, args)
                elif name == "set_altitude":
                    result = _execute_set_altitude(sim, args)
                elif name == "set_speed":
                    result = _execute_set_speed(sim, args)
                elif name == "hold":
                    result = _execute_hold(sim, args)
                elif name == "resume":
                    result = _execute_resume(sim, args)
                elif name == "command_rtl":
                    result = _execute_command_rtl(sim, args)
                elif name == "land_now":
                    result = _execute_land_now(sim, args)
                elif name == "release_payload":
                    result = _execute_release_payload(sim, args)
                elif name == "abort_mission":
                    result = _execute_abort_mission(sim, args)
                elif name == "override_failsafe":
                    result = _execute_override_failsafe(sim, args)
                elif name == "acknowledge":
                    result = _execute_acknowledge(sim, args)
                else:
                    sim.sim_time_s += INVALID_ACTION_LATENCY_S
                    sim.last_action = name
                    result = {
                        "ok": False,
                        "error": f"unknown_tool:{name}",
                        "available_tools": _available_tool_names(),
                    }
                    sim.last_result = result
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                sim.sim_time_s += INVALID_ACTION_LATENCY_S
                sim.last_action = name
                result = {"ok": False, "error": f"tool_error:{exc}"}
                sim.last_result = result

            _store_sim(state, sim, name)
            responses.append(
                vf.ToolMessage(
                    tool_call_id=call_id,
                    content=json.dumps(result, sort_keys=True),
                )
            )
            if sim.is_terminal:
                break

        if sim.is_terminal:
            final_summary = {
                "terminal": True,
                "terminal_reason": sim.terminal_reason,
                "reward_preview": reward_breakdown(state)["total"],
                "reward_breakdown": reward_breakdown(state),
                "telemetry": _telemetry(sim),
            }
            responses.append(vf.UserMessage(content=json.dumps(final_summary, sort_keys=True)))
            state["final_env_response"] = responses

        return cast(vf.Messages, responses)


def load_environment(**kwargs: object) -> vf.Environment:
    """Load the UAV operator environment."""

    seed = int(kwargs.pop("seed", 0))
    max_examples = int(kwargs.pop("max_examples", -1))
    max_turns = int(kwargs.pop("max_turns", 40))
    sim_time_cap_min = int(kwargs.pop("sim_time_cap_min", 90))
    tier = str(kwargs.pop("tier", "mixed_day5"))
    dataset_split = str(kwargs.pop("dataset_split", "default"))
    wind_enabled = bool(kwargs.pop("wind_enabled", True))
    gust_front_probability = float(kwargs.pop("gust_front_probability", 0.5))
    if dataset_split == "default":
        dataset = _dataset(seed=seed, max_examples=max_examples, tier=tier, split="train")
        eval_dataset = _dataset(seed=seed, max_examples=max_examples, tier=tier, split="eval")
    elif dataset_split in {"train", "dev", "eval"}:
        dataset = _dataset(seed=seed, max_examples=max_examples, tier=tier, split=dataset_split)
        eval_dataset = dataset
    else:
        raise ValueError("dataset_split must be default, train, dev, or eval")
    rubric = vf.Rubric(
        funcs=[mission_value, hard_safety, margin_policy, procedure, efficiency],
        weights=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    return UAVOperatorEnv(
        env_id=ENV_ID,
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        tool_defs=_tool_defs(),
        rubric=rubric,
        max_turns=max_turns,
        sim_time_cap_min=sim_time_cap_min,
        tier=tier,
        wind_enabled=wind_enabled,
        gust_front_probability=gust_front_probability,
        **kwargs,
    )
