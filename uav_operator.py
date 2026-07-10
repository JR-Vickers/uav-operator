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
INVALID_ACTION_LATENCY_S = 30.0
LOW_BATT_RTL_THRESHOLD_PCT = 30.0
LOW_BATT_RTL_THRESHOLD_WH = BATTERY_WH * LOW_BATT_RTL_THRESHOLD_PCT / 100.0
DEFAULT_WIND_DIR_FROM_DEG = 270.0
DEFAULT_WIND_SPEED_KT = 0.0
MIN_GROUNDSPEED_KT = 1.0

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


@dataclass
class SimState:
    """Serializable simulator state for one episode."""

    seed: int
    scenario_id: str
    sim_time_s: float
    aircraft: Aircraft
    mission: Mission
    home_site_id: str
    current_plan: list[Waypoint] = field(default_factory=list)
    lost_link_plan: str = "return_home"
    rng_draws: int = 0
    active_failsafe: str | None = None
    alerts: list[dict[str, Any]] = field(default_factory=list)
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
file a flight plan to the mission target, and command return-to-launch when the
mission is complete. Current T0 scenarios have flat wind; the simulator still
validates route geometry, energy, airspace, and autopilot failsafes.

Reward comes only from simulator state: complete the mission value and bring
the aircraft down. Natural-language claims do not score.
"""


def _tool_defs() -> list[dict[str, Any]]:
    return [
        {
            "name": "get_telemetry",
            "description": "Return current aircraft, mission, and flat-weather telemetry.",
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
                    "waypoints": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "lat": {"type": "number"},
                                "lon": {"type": "number"},
                                "name": {"type": "string"},
                            },
                            "required": ["lat", "lon"],
                            "additionalProperties": False,
                        },
                    },
                    "alt_ft": {"type": "number"},
                    "airspeed_kt": {"type": "number"},
                    "lost_link_plan": {"type": "string"},
                },
                "required": ["waypoints", "alt_ft", "airspeed_kt", "lost_link_plan"],
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
    ]


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


def _airspace_conflicts_for_segment(
    start: Waypoint,
    end: Waypoint,
    alt_ft: float,
) -> list[AirspaceZone]:
    start_point = (start.lat, start.lon)
    end_point = (end.lat, end.lon)
    return [
        zone
        for zone in AIRSPACE_ZONES
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


def _nearest_site(lat: float, lon: float) -> Site:
    return min(
        SITES.values(),
        key=lambda site: haversine_nm(lat, lon, site.lat, site.lon),
    )


def _scenario_for_index(index: int) -> dict[str, Any]:
    return SCENARIOS[index % len(SCENARIOS)]


def _build_sim(seed: int, scenario_index: int) -> SimState:
    # Keep one seeded generator per episode even while Day 1 scenarios are static.
    _ = np.random.default_rng(seed)
    scenario = _scenario_for_index(scenario_index)
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
        sim_time_s=0.0,
        aircraft=Aircraft(
            lat=launch_site.lat,
            lon=launch_site.lon,
            alt_ft=0.0,
            current_site_id=launch_site.site_id,
        ),
        mission=mission,
        home_site_id=launch_site.site_id,
    )


def _scenario_briefing(scenario: dict[str, Any]) -> str:
    launch_site = SITES[cast(str, scenario["launch_site_id"])]
    target = cast(Waypoint, scenario["target"])
    return (
        "Mission briefing\n"
        f"- Scenario: {scenario['scenario_id']}\n"
        f"- Launch/recovery: {launch_site.name} ({launch_site.site_id}) "
        f"{launch_site.lat:.4f}, {launch_site.lon:.4f}\n"
        f"- Task: {scenario['description']}\n"
        f"- Target: {target.name} at {target.lat:.4f}, {target.lon:.4f}\n"
        f"- SLA: complete within {float(scenario['sla_min']):.0f} simulated minutes\n"
        "- Weather: flat wind, 0 kt at all altitudes\n"
        "- Airspace: no active pop-up restrictions in this T0 scenario; static "
        "airspace/geofence checks still apply\n"
        "- Expected flow: query telemetry if needed, file a plan to the target, "
        "then command RTL after mission completion.\n"
    )


def _dataset(seed: int, max_examples: int) -> Dataset:
    count = len(SCENARIOS) if max_examples < 0 else min(max_examples, len(SCENARIOS))
    rows = []
    for index in range(count):
        scenario = _scenario_for_index(index)
        rows.append(
            {
                "question": _scenario_briefing(scenario),
                "answer": scenario["scenario_id"],
                "info": {
                    "seed": seed + index,
                    "scenario_index": index,
                    "scenario_id": scenario["scenario_id"],
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
    rng = np.random.default_rng(sim.seed)
    multiplier = 1.0
    for _ in range(sim.rng_draws + 1):
        multiplier = float(rng.normal(1.0, 0.03))
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
    groundspeed_kt = _groundspeed_kt(airspeed_kt=airspeed_kt, track_deg=track_deg)
    multiplier = _execution_multiplier(sim)
    duration_s = (distance_nm / groundspeed_kt) * 3600.0 * multiplier
    energy_wh = _segment_energy_wh(
        distance_nm=distance_nm,
        airspeed_kt=airspeed_kt,
        start_alt_ft=start_alt,
        end_alt_ft=0.0 if landing else alt_ft,
        landing=landing,
        track_deg=track_deg,
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

    return {
        "from": {"lat": start_lat, "lon": start_lon, "alt_ft": start_alt},
        "to": {"lat": waypoint.lat, "lon": waypoint.lon, "alt_ft": sim.aircraft.alt_ft},
        "distance_nm": round(distance_nm, 3),
        "track_deg": round(track_deg, 1),
        "groundspeed_kt": round(groundspeed_kt, 2),
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


def _telemetry(sim: SimState) -> dict[str, Any]:
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
        "weather": {"wind_dir_deg_from": 0, "wind_speed_kt": 0},
        "active_failsafe": sim.active_failsafe,
        "alerts": sim.alerts,
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
        errors.append("altitude_outside_day2_policy_100_400_ft")
    if not 20.0 <= airspeed_kt <= 45.0:
        errors.append("airspeed_outside_airframe_envelope_20_45_kt")
    if any(not _in_bounds(wp.lat, wp.lon) for wp in waypoints):
        errors.append("route_leaves_san_francisco_bay_area_box")

    start = Waypoint(sim.aircraft.lat, sim.aircraft.lon, "current_position")
    for waypoint in waypoints:
        track_deg = bearing_deg(start.lat, start.lon, waypoint.lat, waypoint.lon)
        if _groundspeed_kt(airspeed_kt=airspeed_kt, track_deg=track_deg) <= MIN_GROUNDSPEED_KT:
            errors.append("segment_infeasible_groundspeed")

        safe_alt_ft = max(
            min_safe_altitude_ft(start.lat, start.lon),
            min_safe_altitude_ft(waypoint.lat, waypoint.lon),
        )
        if alt_ft < safe_alt_ft:
            warnings.append(f"min_safe_altitude_violation:{safe_alt_ft:.0f}ft_required")

        for zone in _airspace_conflicts_for_segment(start, waypoint, alt_ft):
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
    return {
        "ok": False,
        "error": "failsafe:GEOFENCE_HOLD",
        "active_failsafe": sim.active_failsafe,
        "conflicts": zone_payload,
        "telemetry": _telemetry(sim),
    }


def _trigger_low_batt_rtl(sim: SimState) -> dict[str, Any]:
    sim.active_failsafe = "LOW_BATT_RTL"
    site = _nearest_site(sim.aircraft.lat, sim.aircraft.lon)
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


def _execute_get_telemetry(sim: SimState, _args: Mapping[str, Any]) -> dict[str, Any]:
    sim.sim_time_s += TOOL_LATENCY_S
    sim.last_action = "get_telemetry"
    sim.last_result = _telemetry(sim)
    return sim.last_result


def _execute_file_flight_plan(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    sim.sim_time_s += TOOL_LATENCY_S

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


def _execute_command_rtl(sim: SimState, args: Mapping[str, Any]) -> dict[str, Any]:
    sim.sim_time_s += TOOL_LATENCY_S
    site_id = args.get("site_id")
    if site_id is None or site_id == "":
        site = _nearest_site(sim.aircraft.lat, sim.aircraft.lon)
    elif str(site_id) in SITES:
        site = SITES[str(site_id)]
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
    return SimState(
        seed=int(state["sim_state"]["seed"]),
        scenario_id=str(state["sim_state"]["scenario_id"]),
        sim_time_s=float(state["sim_state"]["sim_time_s"]),
        aircraft=Aircraft(**state["sim_state"]["aircraft"]),
        mission=Mission(
            **{
                **state["sim_state"]["mission"],
                "target": Waypoint(**state["sim_state"]["mission"]["target"]),
            }
        ),
        home_site_id=str(state["sim_state"]["home_site_id"]),
        current_plan=[
            Waypoint(**waypoint) for waypoint in state["sim_state"].get("current_plan", [])
        ],
        lost_link_plan=str(state["sim_state"].get("lost_link_plan", "return_home")),
        rng_draws=int(state["sim_state"].get("rng_draws", 0)),
        active_failsafe=state["sim_state"].get("active_failsafe"),
        alerts=list(state["sim_state"].get("alerts", [])),
        last_action=str(state["sim_state"].get("last_action", "briefing")),
        last_result=dict(state["sim_state"].get("last_result", {})),
        is_terminal=bool(state["sim_state"].get("is_terminal", False)),
        terminal_reason=state["sim_state"].get("terminal_reason"),
    )


def _store_sim(state: vf.State, sim: SimState, event: str) -> None:
    state["sim_state"] = asdict(sim)
    state.setdefault("sim_log", []).append(_snapshot(sim, event))


def mission_value(state: vf.State) -> float:
    """Mission value reward, computed only from simulator state."""

    sim_state = state.get("sim_state")
    if not isinstance(sim_state, Mapping):
        return 0.0
    mission = sim_state.get("mission")
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


class UAVOperatorEnv(vf.MultiTurnEnv):
    """Day 1 Verifiers MultiTurnEnv for small-UAS operator decisions."""

    def __init__(self, sim_time_cap_min: int = 90, **kwargs: Any) -> None:
        self.sim_time_cap_s = float(sim_time_cap_min) * 60.0
        super().__init__(**kwargs)

    async def setup_state(self, state: vf.State) -> vf.State:
        info = state.get("info", {})
        if not isinstance(info, Mapping):
            info = {}
        seed = int(info.get("seed", 0))
        scenario_index = int(info.get("scenario_index", 0))
        sim = _build_sim(seed=seed, scenario_index=scenario_index)
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
                "error": "No tool call detected. Use get_telemetry, file_flight_plan, or command_rtl.",
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
                elif name == "file_flight_plan":
                    result = _execute_file_flight_plan(sim, args)
                elif name == "command_rtl":
                    result = _execute_command_rtl(sim, args)
                else:
                    sim.sim_time_s += INVALID_ACTION_LATENCY_S
                    sim.last_action = name
                    result = {
                        "ok": False,
                        "error": f"unknown_tool:{name}",
                        "available_tools": [
                            "get_telemetry",
                            "file_flight_plan",
                            "command_rtl",
                        ],
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
                "reward_preview": mission_value(state),
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
    dataset = _dataset(seed=seed, max_examples=max_examples)
    rubric = vf.Rubric(funcs=[mission_value], weights=[1.0])
    return UAVOperatorEnv(
        env_id=ENV_ID,
        dataset=dataset,
        eval_dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        tool_defs=_tool_defs(),
        rubric=rubric,
        max_turns=max_turns,
        sim_time_cap_min=sim_time_cap_min,
        **kwargs,
    )
