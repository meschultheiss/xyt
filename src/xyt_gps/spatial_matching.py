"""Map matching helpers for low-quality GPS legs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import warnings
from xml.etree import ElementTree

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry


DEFAULT_OSRM_SERVICES: Mapping[str, tuple[str, str]] = {
    "driving": ("http://127.0.0.1:5500", "driving"),
    "cycling": ("http://127.0.0.1:5501", "cycling"),
    "walking": ("http://127.0.0.1:5502", "walking"),
}
GOOGLE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_ROADS_SNAP_URL = "https://roads.googleapis.com/v1/snapToRoads"
GRAPHHOPPER_MATCH_URL = "https://graphhopper.com/api/1/match"

_MODE_PROFILES = {
    "Mode::Car": "driving",
    "Mode::Motorbike": "driving",
    "Mode::Bus": "driving",
    "Mode::TaxiUber": "driving",
    "Mode::Carsharing": "driving",
    "Mode::Ecar": "driving",
    "Mode::Ridepooling": "driving",
    "Mode::CarPassenger": "driving",
    "Mode::Bicycle": "cycling",
    "Mode::Ebicycle": "cycling",
    "Mode::KickScooter": "cycling",
    "Mode::Walk": "walking",
    "Car": "driving",
    "Bicycle": "cycling",
    "Walk": "walking",
    "Marche": "walking",
    "Velo": "cycling",
    "Vélo": "cycling",
}
_GOOGLE_TRAVEL_MODES = {
    "driving": "DRIVE",
    "cycling": "BICYCLE",
    "walking": "WALK",
}
_GRAPHHOPPER_PROFILES = {
    "driving": "car",
    "cycling": "bike",
    "walking": "foot",
}

JsonTransport = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class LegMapMatchingResult:
    """Clean legs and the original/clean geometry correspondence table."""

    legs: gpd.GeoDataFrame
    geometry_correspondence: gpd.GeoDataFrame


@dataclass(frozen=True)
class _ProviderResult:
    geometry: LineString
    source: str
    confidence: float | None
    waypoint_count: int
    request_count: int = 1
    provider_distance_m: float | None = None


@dataclass(frozen=True)
class _CandidateEvaluation:
    accepted: bool
    geometry: LineString
    length_m: float
    metrics: Mapping[str, float | bool]
    failures: tuple[str, ...]


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 10.0,
) -> Mapping[str, Any]:
    if payload is not None and body is not None:
        raise ValueError("Provide either payload or body, not both")
    request_body = json.dumps(payload).encode("utf-8") if payload is not None else body
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    request = Request(url, data=request_body, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {message[:300]}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise ConnectionError(str(getattr(error, "reason", error))) from error


def _mode_profile(mode: object) -> str | None:
    if mode is None or pd.isna(mode):
        return None
    return _MODE_PROFILES.get(str(mode))


def _geometry_in_wgs84(geometry: BaseGeometry, crs: object) -> BaseGeometry:
    if crs is None or str(crs).upper() == "EPSG:4326":
        return geometry
    return gpd.GeoSeries([geometry], crs=crs).to_crs("EPSG:4326").iloc[0]


def _geometry_from_wgs84(geometry: LineString, crs: object) -> LineString:
    if crs is None or str(crs).upper() == "EPSG:4326":
        return geometry
    return gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(crs).iloc[0]


def _metric_length(geometry: BaseGeometry, crs: object, operations_crs: str) -> float:
    source_crs = crs or "EPSG:4326"
    return float(gpd.GeoSeries([geometry], crs=source_crs).to_crs(operations_crs).length.iloc[0])


def _chunk_coordinates(
    coordinates: list[tuple[float, float]],
    *,
    max_points: int,
    overlap: int = 2,
) -> list[list[tuple[float, float]]]:
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    if len(coordinates) <= max_points:
        return [coordinates]
    step = max(1, max_points - min(overlap, max_points - 1))
    chunks = []
    for start in range(0, len(coordinates) - 1, step):
        chunk = coordinates[start : start + max_points]
        if len(chunk) >= 2:
            chunks.append(chunk)
        if start + max_points >= len(coordinates):
            break
    return chunks


def _append_coordinates(
    target: list[tuple[float, float]],
    coordinates: list[tuple[float, float]],
) -> None:
    if not coordinates:
        return
    if target:
        first_distance = LineString([target[-1], coordinates[0]]).length
        last_distance = LineString([target[-1], coordinates[-1]]).length
        if last_distance + 1e-12 < first_distance:
            coordinates = list(reversed(coordinates))
        max_overlap = min(len(target), len(coordinates))
        for overlap in range(max_overlap, 0, -1):
            if target[-overlap:] == coordinates[:overlap]:
                coordinates = coordinates[overlap:]
                break
    target.extend(coordinates)


def _ordered_geometry_coordinates(
    geometry: BaseGeometry | None,
) -> list[tuple[float, float]]:
    """Return observed vertices in traversal order for line geometries."""

    if isinstance(geometry, LineString):
        return [(float(value[0]), float(value[1])) for value in geometry.coords]
    if isinstance(geometry, MultiLineString):
        coordinates: list[tuple[float, float]] = []
        for part in geometry.geoms:
            _append_coordinates(
                coordinates,
                [(float(value[0]), float(value[1])) for value in part.coords],
            )
        return coordinates
    return []


def _graphhopper_gpx(coordinates: list[tuple[float, float]]) -> bytes:
    root = ElementTree.Element(
        "gpx",
        {
            "version": "1.1",
            "creator": "xyt-gps",
            "xmlns": "http://www.topografix.com/GPX/1/1",
        },
    )
    segment = ElementTree.SubElement(ElementTree.SubElement(root, "trk"), "trkseg")
    for longitude, latitude in coordinates:
        ElementTree.SubElement(
            segment,
            "trkpt",
            {"lat": f"{latitude:.7f}", "lon": f"{longitude:.7f}"},
        )
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _graphhopper_match(
    coordinates: list[tuple[float, float]],
    *,
    profile: str,
    api_key: str,
    endpoint: str,
    max_points: int,
    timeout: float,
    transport: JsonTransport,
) -> _ProviderResult:
    matched_coordinates: list[tuple[float, float]] = []
    chunks = _chunk_coordinates(coordinates, max_points=max_points)
    provider_distance_m = 0.0
    has_provider_distance = False
    for chunk in chunks:
        params = urlencode(
            {
                "profile": _GRAPHHOPPER_PROFILES[profile],
                "points_encoded": "false",
                "instructions": "false",
                "calc_points": "true",
                "elevation": "false",
                "key": api_key,
            }
        )
        response = transport(
            f"{endpoint}?{params}",
            method="POST",
            body=_graphhopper_gpx(chunk),
            headers={"Content-Type": "application/gpx+xml"},
            timeout=timeout,
        )
        paths = response.get("paths") or []
        if not paths:
            raise RuntimeError(str(response.get("message") or "GraphHopper returned no path"))
        values = paths[0].get("points", {}).get("coordinates", [])
        chunk_coordinates = [tuple(value[:2]) for value in values]
        if len(chunk_coordinates) < 2:
            raise RuntimeError("GraphHopper returned no usable geometry")
        _append_coordinates(matched_coordinates, chunk_coordinates)
        if paths[0].get("distance") is not None:
            provider_distance_m += float(paths[0]["distance"])
            has_provider_distance = True
    if len(matched_coordinates) < 2:
        raise RuntimeError("GraphHopper returned no usable geometry")
    return _ProviderResult(
        geometry=LineString(matched_coordinates),
        source="graphhopper",
        confidence=None,
        waypoint_count=len(coordinates),
        request_count=len(chunks),
        provider_distance_m=provider_distance_m if has_provider_distance else None,
    )


def _osrm_match(
    coordinates: list[tuple[float, float]],
    *,
    service: tuple[str, str],
    max_points: int,
    timeout: float,
    transport: JsonTransport,
) -> _ProviderResult:
    endpoint, route_profile = service
    matched_coordinates: list[tuple[float, float]] = []
    confidences = []
    for chunk in _chunk_coordinates(coordinates, max_points=max_points):
        coordinate_path = ";".join(f"{lon:.7f},{lat:.7f}" for lon, lat in chunk)
        params = urlencode(
            {
                "geometries": "geojson",
                "overview": "full",
                "steps": "false",
                "annotations": "false",
                "gaps": "ignore",
                "tidy": "true",
            }
        )
        url = f"{endpoint.rstrip('/')}/match/v1/{route_profile}/{coordinate_path}?{params}"
        response = transport(url, method="GET", timeout=timeout)
        if response.get("code") != "Ok" or not response.get("matchings"):
            raise RuntimeError(str(response.get("message") or response.get("code") or "OSRM NoMatch"))
        chunk_coordinates: list[tuple[float, float]] = []
        for matching in response["matchings"]:
            geometry = matching.get("geometry", {})
            values = geometry.get("coordinates", [])
            _append_coordinates(chunk_coordinates, [tuple(value[:2]) for value in values])
            if matching.get("confidence") is not None:
                confidences.append(float(matching["confidence"]))
        if len(chunk_coordinates) < 2:
            raise RuntimeError("OSRM returned no usable geometry")
        _append_coordinates(matched_coordinates, chunk_coordinates)
    if len(matched_coordinates) < 2:
        raise RuntimeError("OSRM returned no usable geometry")
    confidence = min(confidences) if confidences else None
    return _ProviderResult(
        geometry=LineString(matched_coordinates),
        source="osrm",
        confidence=confidence,
        waypoint_count=len(coordinates),
    )


def _sample_intermediate_waypoints(
    coordinates: list[tuple[float, float]],
    *,
    max_waypoints: int,
) -> list[tuple[float, float]]:
    if len(coordinates) <= 2 or max_waypoints <= 0:
        return []
    segment_lengths = [
        LineString([start, end]).length
        for start, end in zip(coordinates[:-1], coordinates[1:])
    ]
    total_length = sum(segment_lengths)
    if total_length <= 0:
        return []
    waypoint_count = min(max_waypoints, len(coordinates) - 2)
    targets = [total_length * step / (waypoint_count + 1) for step in range(1, waypoint_count + 1)]
    cumulative = 0.0
    indices = []
    target_index = 0
    for index, segment_length in enumerate(segment_lengths, start=1):
        cumulative += segment_length
        while target_index < len(targets) and cumulative >= targets[target_index]:
            candidate = min(max(index, 1), len(coordinates) - 2)
            indices.append(candidate)
            target_index += 1
    return [coordinates[index] for index in sorted(set(indices))]


def _decode_google_polyline(encoded: str) -> list[tuple[float, float]]:
    coordinates = []
    index = 0
    latitude = 0
    longitude = 0
    while index < len(encoded):
        deltas = []
        for _ in range(2):
            result = 0
            shift = 0
            while True:
                value = ord(encoded[index]) - 63
                index += 1
                result |= (value & 0x1F) << shift
                shift += 5
                if value < 0x20:
                    break
            deltas.append(~(result >> 1) if result & 1 else result >> 1)
        latitude += deltas[0]
        longitude += deltas[1]
        coordinates.append((longitude / 100_000.0, latitude / 100_000.0))
    return coordinates


ApiKeyProvider = Callable[[], str]


def _google_roads_snap(
    coordinates: list[tuple[float, float]],
    *,
    api_key_provider: ApiKeyProvider,
    max_points: int,
    timeout: float,
    transport: JsonTransport,
) -> _ProviderResult:
    matched_coordinates: list[tuple[float, float]] = []
    chunks = _chunk_coordinates(coordinates, max_points=max_points)
    for chunk in chunks:
        path = "|".join(f"{lat:.7f},{lon:.7f}" for lon, lat in chunk)
        params = urlencode(
            {
                "path": path,
                "interpolate": "true",
                "key": api_key_provider(),
            }
        )
        response = transport(
            f"{GOOGLE_ROADS_SNAP_URL}?{params}",
            method="GET",
            timeout=timeout,
        )
        snapped_points = response.get("snappedPoints") or []
        chunk_coordinates = [
            (
                float(point["location"]["longitude"]),
                float(point["location"]["latitude"]),
            )
            for point in snapped_points
            if isinstance(point, Mapping)
            and isinstance(point.get("location"), Mapping)
            and point["location"].get("longitude") is not None
            and point["location"].get("latitude") is not None
        ]
        if len(chunk_coordinates) < 2:
            raise RuntimeError(
                str(response.get("warningMessage") or "Google Roads returned no usable geometry")
            )
        _append_coordinates(matched_coordinates, chunk_coordinates)
    if len(matched_coordinates) < 2:
        raise RuntimeError("Google Roads returned no usable geometry")
    return _ProviderResult(
        geometry=LineString(matched_coordinates),
        source="google_roads",
        confidence=None,
        waypoint_count=len(coordinates),
        request_count=len(chunks),
    )


def _google_routes_match(
    coordinates: list[tuple[float, float]],
    *,
    profile: str,
    api_key_provider: ApiKeyProvider,
    max_waypoints: int,
    timeout: float,
    transport: JsonTransport,
) -> _ProviderResult:
    travel_mode = _GOOGLE_TRAVEL_MODES[profile]

    def waypoint(coordinate: tuple[float, float], *, via: bool = False) -> dict[str, Any]:
        lon, lat = coordinate
        value: dict[str, Any] = {
            "location": {"latLng": {"latitude": lat, "longitude": lon}}
        }
        if via:
            value["via"] = True
        return value

    intermediates = _sample_intermediate_waypoints(
        coordinates,
        max_waypoints=max_waypoints,
    )
    payload = {
        "origin": waypoint(coordinates[0]),
        "destination": waypoint(coordinates[-1]),
        "intermediates": [waypoint(value, via=True) for value in intermediates],
        "travelMode": travel_mode,
        "polylineQuality": "HIGH_QUALITY",
        "optimizeWaypointOrder": False,
    }
    response = transport(
        GOOGLE_ROUTES_URL,
        method="POST",
        payload=payload,
        headers={
            "X-Goog-Api-Key": api_key_provider(),
            "X-Goog-FieldMask": "routes.polyline.encodedPolyline,routes.distanceMeters",
        },
        timeout=timeout,
    )
    routes = response.get("routes") or []
    if not routes:
        raise RuntimeError("Google Routes returned no route")
    encoded = routes[0].get("polyline", {}).get("encodedPolyline")
    if not encoded:
        raise RuntimeError("Google Routes returned no polyline")
    matched_coordinates = _decode_google_polyline(encoded)
    if len(matched_coordinates) < 2:
        raise RuntimeError("Google Routes returned no usable geometry")
    provider_distance = routes[0].get("distanceMeters")
    return _ProviderResult(
        geometry=LineString(matched_coordinates),
        source="google",
        confidence=None,
        waypoint_count=len(intermediates),
        provider_distance_m=float(provider_distance) if provider_distance is not None else None,
    )


def _google_match(
    coordinates: list[tuple[float, float]],
    *,
    profile: str,
    action: str,
    api_key_provider: ApiKeyProvider,
    max_waypoints: int,
    roads_max_points: int,
    timeout: float,
    transport: JsonTransport,
) -> _ProviderResult:
    if action == "snap" and profile == "driving":
        return _google_roads_snap(
            coordinates,
            api_key_provider=api_key_provider,
            max_points=roads_max_points,
            timeout=timeout,
            transport=transport,
        )
    return _google_routes_match(
        coordinates,
        profile=profile,
        api_key_provider=api_key_provider,
        max_waypoints=max_waypoints,
        timeout=timeout,
        transport=transport,
    )


def _normalize_api_keys(
    primary_key: str | None,
    keys: Sequence[str] | None,
) -> tuple[str, ...]:
    values = []
    if keys is not None:
        values.extend(str(value).strip() for value in keys)
    elif primary_key is not None:
        values.append(str(primary_key).strip())
    return tuple(value for value in values if value)


def _google_planned_requests(
    coordinates: list[tuple[float, float]],
    *,
    action: str,
    profile: str,
    roads_max_points: int,
) -> int:
    if action == "snap" and profile == "driving":
        return len(_chunk_coordinates(coordinates, max_points=roads_max_points))
    return 1


def _short_reason(error: Exception) -> str:
    message = " ".join(str(error).split())
    return message[:240] or error.__class__.__name__


def _graphhopper_error_reason(error: Exception) -> str:
    message = " ".join(str(error).split())
    if (
        "Too many points for Map Matching API" in message
        and "allowed: 0" in message
    ):
        return "graphhopper_map_matching_subscription_unavailable"
    return f"graphhopper_failed:{_short_reason(error)}"


def _safe_metric_distance(left: BaseGeometry, right: BaseGeometry) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        value = float(left.distance(right))
    return value if math.isfinite(value) else float("nan")


def _safe_hausdorff_distance(left: BaseGeometry, right: BaseGeometry) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        value = float(left.hausdorff_distance(right))
    return value if math.isfinite(value) else float("nan")


def _safe_metric_length(geometry: BaseGeometry) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        value = float(geometry.length)
    return value if math.isfinite(value) else float("nan")


def _candidate_metrics(
    original_geometry: LineString,
    matched_geometry: LineString,
    *,
    crs: object,
    operations_crs: str,
    point_distance_tolerance_m: float,
    loop_endpoint_distance_m: float,
    loop_min_length_m: float,
) -> dict[str, float | bool]:
    source_crs = crs or "EPSG:4326"
    projected = gpd.GeoSeries(
        [original_geometry, matched_geometry],
        crs=source_crs,
    ).to_crs(operations_crs)
    original_metric = projected.iloc[0]
    matched_metric = projected.iloc[1]
    original_coordinates = list(original_metric.coords)
    matched_coordinates = list(matched_metric.coords)
    point_distances = [
        _safe_metric_distance(Point(value), matched_metric)
        for value in original_coordinates
    ]
    observed_gaps = [
        _safe_metric_distance(Point(start), Point(end))
        for start, end in zip(original_coordinates[:-1], original_coordinates[1:])
    ]
    original_length = _safe_metric_length(original_metric)
    matched_length = _safe_metric_length(matched_metric)
    endpoint_distance = _safe_metric_distance(
        Point(original_coordinates[0]), Point(original_coordinates[-1])
    )
    loop_detected = (
        endpoint_distance <= loop_endpoint_distance_m and original_length >= loop_min_length_m
    )
    original_extent = max(
        _safe_metric_distance(Point(original_coordinates[0]), Point(value))
        for value in original_coordinates
    )
    matched_extent = max(
        _safe_metric_distance(Point(original_coordinates[0]), Point(value))
        for value in matched_coordinates
    )
    finite_point_distances = [value for value in point_distances if math.isfinite(value)]
    point_coverage_ratio = (
        sum(value <= point_distance_tolerance_m for value in finite_point_distances)
        / len(point_distances)
        if point_distances
        else float("nan")
    )
    finite_observed_gaps = [value for value in observed_gaps if math.isfinite(value)]
    return {
        "length_ratio": matched_length / original_length if original_length > 0 else float("nan"),
        "origin_shift_m": _safe_metric_distance(
            Point(original_coordinates[0]), Point(matched_coordinates[0])
        ),
        "destination_shift_m": _safe_metric_distance(
            Point(original_coordinates[-1]),
            Point(matched_coordinates[-1])
        ),
        "point_coverage_ratio": point_coverage_ratio,
        "max_observed_point_distance_m": max(
            finite_point_distances,
            default=float("nan"),
        ),
        "hausdorff_distance_m": _safe_hausdorff_distance(original_metric, matched_metric),
        "observed_max_gap_m": max(finite_observed_gaps, default=0.0),
        "loop_detected": loop_detected,
        "loop_extent_ratio": (
            matched_extent / original_extent if original_extent > 0 else float("nan")
        ),
        "reference_length_m": original_length,
        "matched_length_m": matched_length,
    }


def _evaluate_candidate(
    candidate_geometry: LineString,
    original_geometry: LineString,
    *,
    crs: object,
    operations_crs: str,
    min_length_ratio: float,
    max_length_ratio: float,
    max_endpoint_distance_m: float,
    point_distance_tolerance_m: float,
    min_point_coverage_ratio: float,
    dense_trace_max_gap_m: float,
    max_dense_hausdorff_distance_m: float,
    loop_endpoint_distance_m: float,
    loop_min_length_m: float,
    min_loop_extent_ratio: float,
) -> _CandidateEvaluation:
    metrics = _candidate_metrics(
        original_geometry,
        candidate_geometry,
        crs=crs,
        operations_crs=operations_crs,
        point_distance_tolerance_m=point_distance_tolerance_m,
        loop_endpoint_distance_m=loop_endpoint_distance_m,
        loop_min_length_m=loop_min_length_m,
    )
    failures = []
    critical_metric_names = (
        "length_ratio",
        "origin_shift_m",
        "destination_shift_m",
        "point_coverage_ratio",
        "hausdorff_distance_m",
        "matched_length_m",
    )
    if any(
        not math.isfinite(float(metrics[name]))
        for name in critical_metric_names
    ):
        failures.append("non_finite_metric")
    length_ratio = float(metrics["length_ratio"])
    if not math.isfinite(length_ratio) or not min_length_ratio <= length_ratio <= max_length_ratio:
        failures.append("length_ratio")
    if float(metrics["origin_shift_m"]) > max_endpoint_distance_m:
        failures.append("origin_shift")
    if float(metrics["destination_shift_m"]) > max_endpoint_distance_m:
        failures.append("destination_shift")
    if float(metrics["point_coverage_ratio"]) < min_point_coverage_ratio:
        failures.append("point_coverage")
    if (
        float(metrics["observed_max_gap_m"]) <= dense_trace_max_gap_m
        and float(metrics["hausdorff_distance_m"]) > max_dense_hausdorff_distance_m
    ):
        failures.append("dense_trace_hausdorff")
    if bool(metrics["loop_detected"]) and (
        not math.isfinite(float(metrics["loop_extent_ratio"]))
        or float(metrics["loop_extent_ratio"]) < min_loop_extent_ratio
    ):
        failures.append("loop_extent")
    return _CandidateEvaluation(
        accepted=not failures,
        geometry=candidate_geometry,
        length_m=float(metrics["matched_length_m"]),
        metrics=metrics,
        failures=tuple(failures),
    )


def map_match_legs(
    legs: gpd.GeoDataFrame,
    *,
    operations_crs: str = "EPSG:2056",
    quality_flag_col: str = "low_quality_signal",
    mode_col: str = "mode",
    match_all_supported_modes: bool = False,
    graphhopper_enabled: bool = False,
    graphhopper_api_key: str | None = None,
    graphhopper_match_url: str = GRAPHHOPPER_MATCH_URL,
    graphhopper_max_points_per_request: int = 100,
    graphhopper_max_requests: int | None = 100,
    osrm_enabled: bool = True,
    osrm_services: Mapping[str, tuple[str, str]] | None = None,
    google_fallback_enabled: bool = False,
    google_api_key: str | None = None,
    google_api_keys: Sequence[str] | None = None,
    google_roads_max_points_per_request: int = 100,
    osrm_max_points_per_chunk: int = 99,
    google_max_intermediate_waypoints: int = 10,
    google_max_requests: int | None = 100,
    min_length_ratio: float = 0.50,
    max_length_ratio: float = 1.75,
    max_endpoint_distance_m: float = 250.0,
    point_distance_tolerance_m: float = 100.0,
    min_point_coverage_ratio: float = 0.80,
    dense_trace_max_gap_m: float = 300.0,
    max_dense_hausdorff_distance_m: float = 250.0,
    loop_endpoint_distance_m: float = 100.0,
    loop_min_length_m: float = 1_000.0,
    min_loop_extent_ratio: float = 0.50,
    repair_gap_threshold_m: float = 300.0,
    request_timeout_seconds: float = 10.0,
    transport: JsonTransport | None = None,
) -> LegMapMatchingResult:
    """Map-match selected road legs and retain an auditable original geometry.

    GraphHopper receives GPX traces before OSRM and the optional Google fallback.
    The API key alone never enables a paid provider. Set
    `match_all_supported_modes=True` to process every car, bike and foot leg;
    otherwise only rows selected by `quality_flag_col` are processed.
    """

    required = {"leg_id", "geometry", mode_col}
    if not match_all_supported_modes:
        required.add(quality_flag_col)
    missing = required - set(legs.columns)
    if missing:
        raise KeyError(f"legs missing required column(s): {sorted(missing)}")
    if min_length_ratio <= 0 or max_length_ratio < min_length_ratio:
        raise ValueError("Invalid map-matching length ratio bounds")
    if google_max_requests is not None and google_max_requests < 0:
        raise ValueError("google_max_requests must be non-negative or None")
    if google_roads_max_points_per_request < 2:
        raise ValueError("google_roads_max_points_per_request must be at least 2")
    if graphhopper_enabled and not graphhopper_api_key:
        raise ValueError("graphhopper_api_key is required when graphhopper_enabled=True")
    if graphhopper_max_points_per_request < 2:
        raise ValueError("graphhopper_max_points_per_request must be at least 2")
    if graphhopper_max_requests is not None and graphhopper_max_requests < 0:
        raise ValueError("graphhopper_max_requests must be non-negative or None")
    if not 0 <= min_point_coverage_ratio <= 1:
        raise ValueError("min_point_coverage_ratio must be between 0 and 1")
    if not 0 <= min_loop_extent_ratio <= 1:
        raise ValueError("min_loop_extent_ratio must be between 0 and 1")

    out = legs.copy()
    if not isinstance(out, gpd.GeoDataFrame):
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=getattr(legs, "crs", None))
    services = dict(DEFAULT_OSRM_SERVICES if osrm_services is None else osrm_services)
    request_json = transport or _request_json
    google_keys = _normalize_api_keys(google_api_key, google_api_keys)

    out["map_match_source"] = "original"
    out["map_match_status"] = "not_requested"
    out["map_match_reason"] = "not_selected"
    out["map_match_method"] = "not_requested"
    out["map_match_profile"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["map_match_confidence"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["map_match_length_ratio"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["map_match_waypoints_count"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["map_match_request_count"] = pd.Series(0, index=out.index, dtype="Int64")
    out["map_match_origin_shift_m"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["map_match_destination_shift_m"] = pd.Series(
        pd.NA, index=out.index, dtype="Float64"
    )
    out["map_match_point_coverage_ratio"] = pd.Series(
        pd.NA, index=out.index, dtype="Float64"
    )
    out["map_match_hausdorff_distance_m"] = pd.Series(
        pd.NA, index=out.index, dtype="Float64"
    )
    out["map_match_observed_max_gap_m"] = pd.Series(
        pd.NA, index=out.index, dtype="Float64"
    )
    out["map_match_loop_detected"] = pd.Series(False, index=out.index, dtype="boolean")
    out["map_match_loop_extent_ratio"] = pd.Series(
        pd.NA, index=out.index, dtype="Float64"
    )
    out["map_match_original_length_m"] = pd.Series(
        pd.NA, index=out.index, dtype="Float64"
    )
    out["map_match_clean_length_m"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["map_match_provider_distance_m"] = pd.Series(
        pd.NA, index=out.index, dtype="Float64"
    )

    correspondence_rows: list[dict[str, Any]] = []
    unavailable_profiles: set[str] = set()
    graphhopper_unavailable_reason: str | None = None
    graphhopper_request_count = 0
    google_request_count = 0
    if match_all_supported_modes:
        requested_mask = out[mode_col].map(_mode_profile).notna()
        out.loc[~requested_mask, "map_match_reason"] = "unsupported_mode"
    else:
        requested_mask = out[quality_flag_col].fillna(False).astype(bool)

    def counted_graphhopper_transport(url: str, **kwargs: Any) -> Mapping[str, Any]:
        nonlocal graphhopper_request_count
        graphhopper_request_count += 1
        return request_json(url, **kwargs)

    def next_google_api_key() -> str:
        nonlocal google_request_count
        if not google_keys:
            raise ValueError("google_api_key is required when google_fallback_enabled=True")
        key = google_keys[google_request_count % len(google_keys)]
        google_request_count += 1
        return key

    for index, row in out.loc[requested_mask].iterrows():
        original_geometry = row.geometry
        clean_geometry = original_geometry
        source = "original"
        status = "kept_original"
        method = "keep_original"
        reasons = []
        confidence = None
        waypoint_count = 0
        request_count = 0
        provider_distance_m = None
        profile = _mode_profile(row[mode_col])
        metrics: Mapping[str, float | bool] = {}
        original_length = (
            _metric_length(original_geometry, out.crs, operations_crs)
            if isinstance(original_geometry, (LineString, MultiLineString))
            and not original_geometry.is_empty
            else float("nan")
        )
        clean_length = original_length
        ratio = 1.0 if original_length > 0 else float("nan")

        if profile is None:
            reasons.append(f"unsupported_mode:{row[mode_col]}")
        else:
            original_wgs84 = _geometry_in_wgs84(original_geometry, out.crs)
            coordinates = _ordered_geometry_coordinates(original_wgs84)
            if len(coordinates) < 2:
                reasons.append("invalid_line_geometry")
            else:
                reference_geometry = LineString(coordinates)
                reference_metrics = _candidate_metrics(
                    reference_geometry,
                    reference_geometry,
                    crs="EPSG:4326",
                    operations_crs=operations_crs,
                    point_distance_tolerance_m=point_distance_tolerance_m,
                    loop_endpoint_distance_m=loop_endpoint_distance_m,
                    loop_min_length_m=loop_min_length_m,
                )
                metrics = reference_metrics
                action = (
                    "repair"
                    if isinstance(original_geometry, MultiLineString)
                    or float(reference_metrics["observed_max_gap_m"]) > repair_gap_threshold_m
                    else "snap"
                )
                accepted = False

                def evaluate(candidate: _ProviderResult) -> bool:
                    nonlocal clean_geometry, clean_length, confidence, metrics
                    nonlocal method, ratio, source, status, waypoint_count
                    nonlocal provider_distance_m
                    evaluation = _evaluate_candidate(
                        candidate.geometry,
                        reference_geometry,
                        crs="EPSG:4326",
                        operations_crs=operations_crs,
                        min_length_ratio=min_length_ratio,
                        max_length_ratio=max_length_ratio,
                        max_endpoint_distance_m=max_endpoint_distance_m,
                        point_distance_tolerance_m=point_distance_tolerance_m,
                        min_point_coverage_ratio=min_point_coverage_ratio,
                        dense_trace_max_gap_m=dense_trace_max_gap_m,
                        max_dense_hausdorff_distance_m=max_dense_hausdorff_distance_m,
                        loop_endpoint_distance_m=loop_endpoint_distance_m,
                        loop_min_length_m=loop_min_length_m,
                        min_loop_extent_ratio=min_loop_extent_ratio,
                    )
                    metrics = evaluation.metrics
                    ratio = float(metrics["length_ratio"])
                    waypoint_count = candidate.waypoint_count
                    provider_distance_m = candidate.provider_distance_m
                    if not evaluation.accepted:
                        reasons.append(
                            f"{candidate.source}_rejected_{','.join(evaluation.failures)}"
                        )
                        return False
                    clean_geometry = _geometry_from_wgs84(evaluation.geometry, out.crs)
                    clean_length = evaluation.length_m
                    confidence = candidate.confidence
                    method = f"{candidate.source}_{action}"
                    source = candidate.source
                    status = "matched"
                    reasons.append(f"{candidate.source}_accepted")
                    return True

                if graphhopper_enabled:
                    planned_requests = len(
                        _chunk_coordinates(
                            coordinates,
                            max_points=graphhopper_max_points_per_request,
                        )
                    )
                    if graphhopper_unavailable_reason:
                        reasons.append(graphhopper_unavailable_reason)
                    elif (
                        graphhopper_max_requests is not None
                        and graphhopper_request_count + planned_requests
                        > graphhopper_max_requests
                    ):
                        reasons.append("graphhopper_request_limit_reached")
                    else:
                        request_count_before = graphhopper_request_count
                        try:
                            candidate = _graphhopper_match(
                                coordinates,
                                profile=profile,
                                api_key=str(graphhopper_api_key),
                                endpoint=graphhopper_match_url,
                                max_points=graphhopper_max_points_per_request,
                                timeout=request_timeout_seconds,
                                transport=counted_graphhopper_transport,
                            )
                            accepted = evaluate(candidate)
                        except Exception as error:
                            reason = _graphhopper_error_reason(error)
                            reasons.append(reason)
                            if reason == "graphhopper_map_matching_subscription_unavailable":
                                graphhopper_unavailable_reason = reason
                        request_count += graphhopper_request_count - request_count_before

                if not accepted and osrm_enabled:
                    service = services.get(profile)
                    if service is None:
                        reasons.append(f"osrm_not_configured:{profile}")
                    elif profile in unavailable_profiles:
                        reasons.append(f"osrm_unavailable:{profile}")
                    else:
                        try:
                            candidate = _osrm_match(
                                coordinates,
                                service=service,
                                max_points=osrm_max_points_per_chunk,
                                timeout=request_timeout_seconds,
                                transport=request_json,
                            )
                            request_count += candidate.request_count
                            accepted = evaluate(candidate)
                        except ConnectionError as error:
                            unavailable_profiles.add(profile)
                            reasons.append(f"osrm_unavailable:{_short_reason(error)}")
                        except Exception as error:
                            reasons.append(f"osrm_failed:{_short_reason(error)}")
                elif not accepted and not osrm_enabled:
                    reasons.append("osrm_disabled")

                if (
                    not accepted
                    and google_fallback_enabled
                    and google_keys
                    and profile in _GOOGLE_TRAVEL_MODES
                ):
                    planned_google_requests = _google_planned_requests(
                        coordinates,
                        action=action,
                        profile=profile,
                        roads_max_points=google_roads_max_points_per_request,
                    )
                    if (
                        google_max_requests is not None
                        and google_request_count + planned_google_requests
                        > google_max_requests
                    ):
                        reasons.append("google_request_limit_reached")
                    else:
                        request_count_before = google_request_count
                        try:
                            candidate = _google_match(
                                coordinates,
                                profile=profile,
                                action=action,
                                api_key_provider=next_google_api_key,
                                max_waypoints=google_max_intermediate_waypoints,
                                roads_max_points=google_roads_max_points_per_request,
                                timeout=request_timeout_seconds,
                                transport=request_json,
                            )
                            accepted = evaluate(candidate)
                        except Exception as error:
                            reasons.append(f"google_failed:{_short_reason(error)}")
                        request_count += google_request_count - request_count_before
                elif not accepted and not google_fallback_enabled:
                    reasons.append("google_fallback_disabled")
                elif not accepted and not google_keys:
                    reasons.append("google_api_key_missing")
                elif (
                    not accepted
                    and google_max_requests is not None
                    and google_request_count >= google_max_requests
                ):
                    reasons.append("google_request_limit_reached")

        out.at[index, "geometry"] = clean_geometry
        out.at[index, "map_match_source"] = source
        out.at[index, "map_match_status"] = status
        out.at[index, "map_match_reason"] = ";".join(reasons)
        out.at[index, "map_match_method"] = method
        out.at[index, "map_match_profile"] = (
            _GRAPHHOPPER_PROFILES.get(profile, profile) if profile is not None else pd.NA
        )
        out.at[index, "map_match_confidence"] = confidence
        out.at[index, "map_match_length_ratio"] = ratio
        out.at[index, "map_match_waypoints_count"] = waypoint_count
        out.at[index, "map_match_request_count"] = request_count
        out.at[index, "map_match_origin_shift_m"] = metrics.get("origin_shift_m", pd.NA)
        out.at[index, "map_match_destination_shift_m"] = metrics.get(
            "destination_shift_m", pd.NA
        )
        out.at[index, "map_match_point_coverage_ratio"] = metrics.get(
            "point_coverage_ratio", pd.NA
        )
        out.at[index, "map_match_hausdorff_distance_m"] = metrics.get(
            "hausdorff_distance_m", pd.NA
        )
        out.at[index, "map_match_observed_max_gap_m"] = metrics.get(
            "observed_max_gap_m", pd.NA
        )
        out.at[index, "map_match_loop_detected"] = metrics.get("loop_detected", False)
        out.at[index, "map_match_loop_extent_ratio"] = metrics.get(
            "loop_extent_ratio", pd.NA
        )
        out.at[index, "map_match_original_length_m"] = original_length
        out.at[index, "map_match_clean_length_m"] = clean_length
        out.at[index, "map_match_provider_distance_m"] = (
            provider_distance_m if provider_distance_m is not None else pd.NA
        )
        if status == "matched":
            for length_col in ("length", "length_leg"):
                if length_col in out.columns:
                    out.at[index, length_col] = clean_length

        shared = {
            "leg_id": row["leg_id"],
            "user_id": row.get("user_id", pd.NA),
            "map_match_status": status,
            "map_match_reason": ";".join(reasons),
            "map_match_method": method,
            "map_match_profile": _GRAPHHOPPER_PROFILES.get(profile, profile),
            "map_match_confidence": confidence,
            "map_match_length_ratio": ratio,
            "map_match_request_count": request_count,
            "map_match_origin_shift_m": metrics.get("origin_shift_m", pd.NA),
            "map_match_destination_shift_m": metrics.get("destination_shift_m", pd.NA),
            "map_match_point_coverage_ratio": metrics.get("point_coverage_ratio", pd.NA),
            "map_match_hausdorff_distance_m": metrics.get("hausdorff_distance_m", pd.NA),
            "map_match_observed_max_gap_m": metrics.get("observed_max_gap_m", pd.NA),
            "map_match_loop_detected": metrics.get("loop_detected", False),
            "map_match_loop_extent_ratio": metrics.get("loop_extent_ratio", pd.NA),
            "reference_length_m": metrics.get("reference_length_m", original_length),
            "original_length_m": original_length,
            "clean_length_m": clean_length,
            "provider_distance_m": provider_distance_m,
        }
        correspondence_rows.extend(
            [
                {
                    **shared,
                    "geometry_version": "original",
                    "map_match_source": "original",
                    "geometry": original_geometry,
                },
                {
                    **shared,
                    "geometry_version": "clean",
                    "map_match_source": source,
                    "geometry": clean_geometry,
                },
            ]
        )

    correspondence_columns = [
        "leg_id",
        "user_id",
        "geometry_version",
        "map_match_source",
        "map_match_status",
        "map_match_reason",
        "map_match_method",
        "map_match_profile",
        "map_match_confidence",
        "map_match_length_ratio",
        "map_match_request_count",
        "map_match_origin_shift_m",
        "map_match_destination_shift_m",
        "map_match_point_coverage_ratio",
        "map_match_hausdorff_distance_m",
        "map_match_observed_max_gap_m",
        "map_match_loop_detected",
        "map_match_loop_extent_ratio",
        "reference_length_m",
        "original_length_m",
        "clean_length_m",
        "provider_distance_m",
        "geometry",
    ]
    correspondence = gpd.GeoDataFrame(
        correspondence_rows,
        columns=correspondence_columns,
        geometry="geometry",
        crs=out.crs,
    )
    return LegMapMatchingResult(legs=out, geometry_correspondence=correspondence)


__all__ = [
    "DEFAULT_OSRM_SERVICES",
    "GOOGLE_ROADS_SNAP_URL",
    "GOOGLE_ROUTES_URL",
    "GRAPHHOPPER_MATCH_URL",
    "LegMapMatchingResult",
    "map_match_legs",
]
