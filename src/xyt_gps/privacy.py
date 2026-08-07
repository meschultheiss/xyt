"""Privacy helpers for GPS demonstration datasets.

The functions in this module are designed for preparing shareable test data.
They keep the middle of observed GPS tracks intact, but remove the parts that
start or end inside frequent staypoint clusters for each user.
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass, field
import re
from typing import Mapping
import unicodedata
import warnings

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkb, wkt
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring, transform as shapely_transform, unary_union


DEFAULT_PURPOSE_COLUMNS = (
    "purpose",
    "purpose_niv1",
    "purpose_niv2",
    "purpose_mapped",
    "motif",
    "motif_niv1",
    "motif_niv2",
)

DEFAULT_DIRECT_IDENTIFIER_COLUMNS = (
    "email",
    "e_mail",
    "mail",
    "courriel",
    "invite_code",
    "registration_code",
    "participant_code",
    "questionnaire_match_value",
)

DEFAULT_SENSITIVE_PURPOSE_GROUPS = {
    "home": (
        "home",
        "domicile",
        "maison",
        "chez_soi",
        "residence",
        "résidence",
    ),
    "work": (
        "work",
        "travail",
        "job",
        "bureau",
        "workplace",
        "lieu_de_travail",
    ),
    "education": (
        "education",
        "formation",
        "school",
        "ecole",
        "école",
        "university",
        "universite",
        "université",
        "studies",
        "etudes",
        "études",
    ),
}

DEFAULT_SENSITIVE_PURPOSE_REPLACEMENTS = {
    "home": "work",
    "work": "education",
    "education": "home",
}

DEFAULT_SENSITIVE_PURPOSE_OUTPUT_VALUES = {
    "home": "home",
    "work": "work",
    "education": "formation",
}


@dataclass(frozen=True)
class GeofenceAnonymizationConfig:
    """Configuration for route-preserving GPS anonymization.

    The method detects the main staypoint clusters for each user, builds
    geofences around those places, then truncates track geometries only where
    they enter or leave a protected geofence. It avoids random endpoint
    displacement, which tends to create circular artifacts on maps.
    """

    top_staypoint_clusters: int = 4
    dbscan_eps_m: float = 250.0
    dbscan_min_samples: int = 2
    geofence_buffer_m: float = 120.0
    geofence_min_radius_m: float = 180.0
    geofence_max_radius_m: float = 650.0
    trim_buffer_min_m: float = 20.0
    trim_buffer_max_m: float = 120.0
    min_remaining_leg_length_m: float = 120.0
    short_leg_policy: str = "remove_geometry"
    metric_crs: str = "EPSG:2056"
    source_crs: str = "EPSG:4326"
    random_state: int = 20260615
    pseudonym_prefix: str = "demo_user_"
    sample_user_count: int | None = None
    sample_random_state: int = 20260616
    swap_sensitive_purpose_values: bool = True
    purpose_columns: tuple[str, ...] = DEFAULT_PURPOSE_COLUMNS
    sensitive_purpose_groups: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: DEFAULT_SENSITIVE_PURPOSE_GROUPS
    )
    sensitive_purpose_replacements: Mapping[str, str] = field(
        default_factory=lambda: DEFAULT_SENSITIVE_PURPOSE_REPLACEMENTS
    )
    sensitive_purpose_output_values: Mapping[str, str] = field(
        default_factory=lambda: DEFAULT_SENSITIVE_PURPOSE_OUTPUT_VALUES
    )
    drop_columns: tuple[str, ...] = DEFAULT_DIRECT_IDENTIFIER_COLUMNS
    user_id_column: str = "user_id"
    geometry_column: str = "geometry"
    type_column: str = "type"
    started_at_column: str = "started_at"


@dataclass
class GpsAnonymizationResult:
    """Result returned by `anonymize_landed_gps_tables`."""

    tables: dict[str, pd.DataFrame]
    geofences: pd.DataFrame
    report: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)


def _require_dbscan():
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as error:
        raise ImportError(
            "Geofence anonymization requires scikit-learn. "
            "Install the optional privacy dependency with `pip install xyt-gps[privacy]`."
        ) from error
    return DBSCAN


def _geometry_output_format(series: pd.Series) -> str:
    sample = series.dropna()
    if sample.empty:
        return "wkb_hex"
    first = sample.iloc[0]
    if isinstance(first, BaseGeometry):
        return "shapely"
    if isinstance(first, bytes):
        return "wkb_bytes"
    text = str(first).strip()
    if text.upper().startswith(("POINT", "LINESTRING", "MULTI")):
        return "wkt"
    return "wkb_hex"


def _parse_geometry(value) -> BaseGeometry | None:
    if value is None:
        return None
    if isinstance(value, BaseGeometry):
        return value
    if isinstance(value, bytes):
        return wkb.loads(value)
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith(("POINT", "LINESTRING", "MULTI")):
        return wkt.loads(text)
    return wkb.loads(binascii.unhexlify(text))


def _format_geometry(geometry: BaseGeometry | None, output_format: str):
    if geometry is None or geometry.is_empty:
        return pd.NA
    if output_format == "shapely":
        return geometry
    if output_format == "wkb_bytes":
        return geometry.wkb
    if output_format == "wkt":
        return geometry.wkt
    return geometry.wkb_hex


def _as_linestring(geometry: BaseGeometry | None) -> LineString | None:
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, LineString):
        return geometry
    if isinstance(geometry, MultiLineString):
        coordinates = []
        for part in geometry.geoms:
            coordinates.extend(list(part.coords))
        if len(coordinates) >= 2:
            return LineString(coordinates)
    return None


def _point_from_geometry(geometry: BaseGeometry | None) -> Point | None:
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, Point):
        return geometry
    return geometry.centroid


def _inside(geometry: BaseGeometry | None, point: Point) -> bool:
    return bool(geometry is not None and not geometry.is_empty and geometry.covers(point))


def _boundary_distance_from_start(
    line: LineString,
    geofence_union: BaseGeometry,
    *,
    samples: int = 128,
) -> float:
    length = float(line.length)
    previous = 0.0
    for distance in np.linspace(0, length, samples):
        if not _inside(geofence_union, line.interpolate(float(distance))):
            low = previous
            high = float(distance)
            for _ in range(28):
                midpoint = (low + high) / 2
                if _inside(geofence_union, line.interpolate(midpoint)):
                    low = midpoint
                else:
                    high = midpoint
            return high
        previous = float(distance)
    return length


def _boundary_distance_from_end(
    line: LineString,
    geofence_union: BaseGeometry,
    *,
    samples: int = 128,
) -> float:
    length = float(line.length)
    previous = length
    for distance in np.linspace(length, 0, samples):
        if not _inside(geofence_union, line.interpolate(float(distance))):
            low = float(distance)
            high = previous
            for _ in range(28):
                midpoint = (low + high) / 2
                if _inside(geofence_union, line.interpolate(midpoint)):
                    high = midpoint
                else:
                    low = midpoint
            return high
        previous = float(distance)
    return 0.0


def _draw_trim_buffer(config: GeofenceAnonymizationConfig, rng: np.random.Generator) -> float:
    low = float(config.trim_buffer_min_m)
    high = float(config.trim_buffer_max_m)
    if low < 0 or high < 0:
        raise ValueError("trim_buffer_min_m and trim_buffer_max_m must be non-negative.")
    if high < low:
        raise ValueError("trim_buffer_max_m must be greater than or equal to trim_buffer_min_m.")
    if high == low:
        return low
    return float(rng.uniform(low, high))


def _handle_short_leg(
    line: LineString,
    trimmed_line: LineString | None,
    status: str,
    start_inside: bool,
    end_inside: bool,
    config: GeofenceAnonymizationConfig,
) -> tuple[LineString | None, str, bool, bool]:
    if config.short_leg_policy == "remove_geometry":
        return None, status, start_inside, end_inside
    if config.short_leg_policy == "keep_original_flagged":
        return line, "short_leg_kept_original_after_geofence", start_inside, end_inside
    if config.short_leg_policy == "keep_trimmed_flagged" and trimmed_line is not None:
        return trimmed_line, "short_leg_kept_trimmed_after_geofence", start_inside, end_inside
    raise ValueError(
        "short_leg_policy must be one of: "
        "'remove_geometry', 'keep_original_flagged', 'keep_trimmed_flagged'."
    )


def _trim_line_by_endpoint_geofences(
    line: LineString,
    geofence_union: BaseGeometry | None,
    config: GeofenceAnonymizationConfig,
    rng: np.random.Generator,
) -> tuple[LineString | None, str, bool, bool, float, float]:
    if line is None or line.length <= 0:
        return None, "missing_or_empty_geometry", False, False, 0.0, 0.0
    if geofence_union is None or geofence_union.is_empty:
        return line, "unchanged_no_user_geofence", False, False, 0.0, 0.0

    start_inside = _inside(geofence_union, line.interpolate(0))
    end_inside = _inside(geofence_union, line.interpolate(line.length))

    start_distance = 0.0
    end_distance = float(line.length)
    start_trim_buffer_m = 0.0
    end_trim_buffer_m = 0.0
    if start_inside:
        start_trim_buffer_m = _draw_trim_buffer(config, rng)
        start_distance = min(
            line.length,
            _boundary_distance_from_start(line, geofence_union) + start_trim_buffer_m,
        )
    if end_inside:
        end_trim_buffer_m = _draw_trim_buffer(config, rng)
        end_distance = max(
            0.0,
            _boundary_distance_from_end(line, geofence_union) - end_trim_buffer_m,
        )

    if end_distance - start_distance < config.min_remaining_leg_length_m:
        line_out, status, start_flag, end_flag = _handle_short_leg(
            line,
            None,
            "short_leg_geometry_removed_after_geofence",
            start_inside,
            end_inside,
            config,
        )
        return line_out, status, start_flag, end_flag, start_trim_buffer_m, end_trim_buffer_m

    if not start_inside and not end_inside:
        return line, "unchanged_outside_geofence", False, False, 0.0, 0.0

    trimmed = substring(line, start_distance, end_distance)
    trimmed_line = _as_linestring(trimmed)
    if trimmed_line is None or trimmed_line.length < config.min_remaining_leg_length_m:
        line_out, status, start_flag, end_flag = _handle_short_leg(
            line,
            trimmed_line,
            "short_leg_geometry_removed_after_geofence",
            start_inside,
            end_inside,
            config,
        )
        return line_out, status, start_flag, end_flag, start_trim_buffer_m, end_trim_buffer_m
    return trimmed_line, "trimmed_by_geofence", start_inside, end_inside, start_trim_buffer_m, end_trim_buffer_m


def detect_staypoint_geofences(
    storyline: pd.DataFrame,
    config: GeofenceAnonymizationConfig | None = None,
) -> pd.DataFrame:
    """Detect the main staypoint geofences for each user.

    Returns one row per protected cluster, with user id, cluster rank, number
    of staypoints, WGS84 center coordinates and radius in meters.
    """

    config = config or GeofenceAnonymizationConfig()
    required = {config.user_id_column, config.geometry_column, config.type_column}
    missing = sorted(required - set(storyline.columns))
    if missing:
        raise KeyError(f"Missing required column(s) for geofence detection: {missing}")

    DBSCAN = _require_dbscan()
    to_metric = Transformer.from_crs(config.source_crs, config.metric_crs, always_xy=True).transform
    to_source = Transformer.from_crs(config.metric_crs, config.source_crs, always_xy=True).transform

    stay_mask = storyline[config.type_column].astype("string").eq("Stay")
    stays = storyline.loc[stay_mask, [config.user_id_column, config.geometry_column]].copy()
    if stays.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "geofence_id",
                "cluster_rank",
                "staypoints_count",
                "radius_m",
                "center_lon",
                "center_lat",
            ]
        )

    stays["_point_metric"] = (
        stays[config.geometry_column]
        .apply(_parse_geometry)
        .apply(_point_from_geometry)
        .apply(lambda point: shapely_transform(to_metric, point) if point is not None else None)
    )

    rows: list[dict[str, object]] = []
    for user_id, group in stays.dropna(subset=["_point_metric"]).groupby(config.user_id_column, dropna=False):
        points = list(group["_point_metric"])
        if len(points) < config.dbscan_min_samples:
            continue
        coordinates = np.array([[point.x, point.y] for point in points], dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r"Could not find the number of physical cores.*",
            )
            labels = DBSCAN(eps=config.dbscan_eps_m, min_samples=config.dbscan_min_samples).fit_predict(coordinates)
        cluster_rows = []
        for label in sorted(set(labels)):
            if label < 0:
                continue
            cluster_points = [point for point, point_label in zip(points, labels) if point_label == label]
            if not cluster_points:
                continue
            cluster_coordinates = np.array([[point.x, point.y] for point in cluster_points], dtype=float)
            center_x = float(cluster_coordinates[:, 0].mean())
            center_y = float(cluster_coordinates[:, 1].mean())
            distances = np.sqrt((cluster_coordinates[:, 0] - center_x) ** 2 + (cluster_coordinates[:, 1] - center_y) ** 2)
            radius = float(np.quantile(distances, 0.9) + config.geofence_buffer_m)
            radius = min(config.geofence_max_radius_m, max(config.geofence_min_radius_m, radius))
            cluster_rows.append(
                {
                    "user_id": user_id,
                    "label": int(label),
                    "staypoints_count": int(len(cluster_points)),
                    "radius_m": radius,
                    "center_metric": Point(center_x, center_y),
                }
            )

        cluster_rows = sorted(cluster_rows, key=lambda row: row["staypoints_count"], reverse=True)
        for rank, row in enumerate(cluster_rows[: config.top_staypoint_clusters], start=1):
            center_wgs84 = shapely_transform(to_source, row["center_metric"])
            rows.append(
                {
                    "user_id": row["user_id"],
                    "geofence_id": f"{row['user_id']}__geofence_{rank}",
                    "cluster_rank": rank,
                    "staypoints_count": row["staypoints_count"],
                    "radius_m": row["radius_m"],
                    "center_lon": center_wgs84.x,
                    "center_lat": center_wgs84.y,
                }
            )

    return pd.DataFrame(rows)


def anonymize_storyline_with_geofencing(
    storyline: pd.DataFrame,
    config: GeofenceAnonymizationConfig | None = None,
    *,
    geofences: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Anonymize storyline geometries by geofencing frequent staypoints.

    Track rows are truncated when their start or end lies inside one of the
    user's detected geofences. Staypoints inside a geofence are reattached to
    neighboring truncated track endpoints when possible.
    """

    config = config or GeofenceAnonymizationConfig()
    if config.geometry_column not in storyline.columns:
        raise KeyError(f"Missing geometry column: {config.geometry_column}")

    geofences = detect_staypoint_geofences(storyline, config) if geofences is None else geofences.copy()

    to_metric = Transformer.from_crs(config.source_crs, config.metric_crs, always_xy=True).transform
    to_source = Transformer.from_crs(config.metric_crs, config.source_crs, always_xy=True).transform
    output_format = _geometry_output_format(storyline[config.geometry_column])

    geofence_unions: dict[object, BaseGeometry] = {}
    for user_id, group in geofences.groupby("user_id", dropna=False):
        circles = []
        for row in group.itertuples(index=False):
            center = Point(float(row.center_lon), float(row.center_lat))
            center_metric = shapely_transform(to_metric, center)
            circles.append(center_metric.buffer(float(row.radius_m)))
        if circles:
            geofence_unions[user_id] = unary_union(circles)

    out = storyline.copy()
    out["_started_at_sort"] = pd.to_datetime(
        out.get(config.started_at_column),
        errors="coerce",
        utc=True,
        format="mixed",
    )
    out["_geometry_metric_original"] = (
        out[config.geometry_column]
        .apply(_parse_geometry)
        .apply(lambda geom: shapely_transform(to_metric, geom) if geom is not None else None)
    )
    out["_geometry_metric_private"] = out["_geometry_metric_original"]
    out["privacy_geofenced"] = False
    out["privacy_geometry_status"] = "unchanged"

    report_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(config.random_state)

    for user_id, group in out.sort_values([config.user_id_column, "_started_at_sort"]).groupby(config.user_id_column, dropna=False):
        user_union = geofence_unions.get(user_id)
        group_indices = list(group.index)
        track_endpoints: dict[int, tuple[Point | None, Point | None]] = {}

        for index in group_indices:
            row_type = str(out.at[index, config.type_column]) if config.type_column in out.columns else ""
            if row_type != "Track":
                continue
            line = _as_linestring(out.at[index, "_geometry_metric_original"])
            (
                trimmed,
                status,
                start_geofenced,
                end_geofenced,
                start_trim_buffer_m,
                end_trim_buffer_m,
            ) = _trim_line_by_endpoint_geofences(
                line,
                user_union,
                config,
                rng,
            )
            out.at[index, "_geometry_metric_private"] = trimmed
            out.at[index, "privacy_geofenced"] = bool(start_geofenced or end_geofenced)
            out.at[index, "privacy_geometry_status"] = status
            track_endpoints[index] = (
                Point(trimmed.coords[0]) if trimmed is not None and len(trimmed.coords) else None,
                Point(trimmed.coords[-1]) if trimmed is not None and len(trimmed.coords) else None,
            )
            report_rows.append(
                {
                    "user_id": user_id,
                    "row_index": index,
                    "type": row_type,
                    "privacy_geometry_status": status,
                    "start_in_geofence": bool(start_geofenced),
                    "end_in_geofence": bool(end_geofenced),
                    "start_trim_buffer_m": start_trim_buffer_m,
                    "end_trim_buffer_m": end_trim_buffer_m,
                    "original_length_m": None if line is None else float(line.length),
                    "private_length_m": None if trimmed is None else float(trimmed.length),
                }
            )

        for position, index in enumerate(group_indices):
            row_type = str(out.at[index, config.type_column]) if config.type_column in out.columns else ""
            if row_type != "Stay":
                continue
            point = _point_from_geometry(out.at[index, "_geometry_metric_original"])
            if point is None or not _inside(user_union, point):
                continue

            previous_endpoint = None
            next_endpoint = None
            for previous_index in reversed(group_indices[:position]):
                if previous_index in track_endpoints:
                    previous_endpoint = track_endpoints[previous_index][1]
                    break
            for next_index in group_indices[position + 1 :]:
                if next_index in track_endpoints:
                    next_endpoint = track_endpoints[next_index][0]
                    break

            candidates = [candidate for candidate in (previous_endpoint, next_endpoint) if candidate is not None]
            if candidates:
                if len(candidates) == 1:
                    private_point = candidates[0]
                else:
                    private_point = Point(
                        float(np.mean([candidate.x for candidate in candidates])),
                        float(np.mean([candidate.y for candidate in candidates])),
                    )
                status = "reattached_to_trimmed_leg_endpoint"
            else:
                private_point = point
                status = "unchanged_no_adjacent_trimmed_leg"

            out.at[index, "_geometry_metric_private"] = private_point
            out.at[index, "privacy_geofenced"] = True
            out.at[index, "privacy_geometry_status"] = status
            report_rows.append(
                {
                    "user_id": user_id,
                    "row_index": index,
                    "type": row_type,
                    "privacy_geometry_status": status,
                    "start_in_geofence": None,
                    "end_in_geofence": None,
                    "start_trim_buffer_m": None,
                    "end_trim_buffer_m": None,
                    "original_length_m": None,
                    "private_length_m": None,
                }
            )

    out[config.geometry_column] = (
        out["_geometry_metric_private"]
        .apply(lambda geom: shapely_transform(to_source, geom) if geom is not None else None)
        .apply(lambda geom: _format_geometry(geom, output_format))
    )

    out = out.drop(
        columns=["_started_at_sort", "_geometry_metric_original", "_geometry_metric_private"],
        errors="ignore",
    )
    report = pd.DataFrame(report_rows)
    return out, geofences, report


def build_user_pseudonym_mapping(
    tables: Mapping[str, pd.DataFrame],
    config: GeofenceAnonymizationConfig | None = None,
) -> dict[str, str]:
    """Build deterministic pseudonymous user ids from all tables."""

    config = config or GeofenceAnonymizationConfig()
    user_ids = sorted(
        {
            str(value)
            for table in tables.values()
            if table is not None and config.user_id_column in table.columns
            for value in table[config.user_id_column].dropna().astype(str)
        }
    )
    rng = np.random.default_rng(config.random_state)
    ordered_positions = list(rng.permutation(len(user_ids))) if user_ids else []
    mapping = {}
    for pseudonym_index, source_position in enumerate(ordered_positions, start=1):
        mapping[user_ids[int(source_position)]] = f"{config.pseudonym_prefix}{pseudonym_index:04d}"
    return mapping


def apply_user_pseudonym_mapping(
    tables: Mapping[str, pd.DataFrame],
    user_id_mapping: Mapping[str, str],
    config: GeofenceAnonymizationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply a user-id pseudonym mapping to all tables containing `user_id`."""

    config = config or GeofenceAnonymizationConfig()
    out: dict[str, pd.DataFrame] = {}
    for table_name, table in tables.items():
        frame = table.copy()
        columns = [config.user_id_column]
        if table_name == "user_statistics":
            columns.append("user_statistics_id")
        for column in columns:
            if column in frame.columns:
                frame[column] = frame[column].apply(
                    lambda value: user_id_mapping.get(str(value), value) if pd.notna(value) else value
                )
        out[table_name] = frame
    return out


def _normalize_purpose_value(value) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).strip().lower())
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^0-9a-z]+", "_", normalized)
    return normalized.strip("_")


def _purpose_group(value, config: GeofenceAnonymizationConfig) -> str | None:
    if pd.isna(value):
        return None
    normalized = _normalize_purpose_value(value)
    for group_name, group_values in config.sensitive_purpose_groups.items():
        normalized_group_values = {_normalize_purpose_value(group_value) for group_value in group_values}
        if normalized in normalized_group_values:
            return group_name
    return None


def _replacement_sensitive_purpose_value(value, config: GeofenceAnonymizationConfig):
    group = _purpose_group(value, config)
    if group is None:
        return value
    replacement_group = config.sensitive_purpose_replacements.get(group)
    if replacement_group is None:
        return value
    return config.sensitive_purpose_output_values.get(replacement_group, replacement_group)


def _derange_sensitive_values(values: pd.Series, config: GeofenceAnonymizationConfig, rng: np.random.Generator):
    if len(values) == 0:
        return values
    if len(values) == 1:
        return pd.Series([_replacement_sensitive_purpose_value(values.iloc[0], config)], index=values.index)

    original = values.reset_index(drop=True)
    original_groups = original.map(lambda value: _purpose_group(value, config))

    for _ in range(50):
        permutation = rng.permutation(len(original))
        candidate = original.iloc[permutation].reset_index(drop=True)
        candidate_groups = candidate.map(lambda value: _purpose_group(value, config))
        if not candidate_groups.eq(original_groups).any():
            return pd.Series(candidate.to_numpy(), index=values.index)

    rotated = original.shift(-1).fillna(original.iloc[0])
    rotated_groups = rotated.map(lambda value: _purpose_group(value, config))
    adjusted = rotated.copy()
    same_group_mask = rotated_groups.eq(original_groups)
    for position in same_group_mask[same_group_mask].index:
        adjusted.iloc[position] = _replacement_sensitive_purpose_value(original.iloc[position], config)
    return pd.Series(adjusted.to_numpy(), index=values.index)


def swap_sensitive_purpose_columns(
    tables: Mapping[str, pd.DataFrame],
    config: GeofenceAnonymizationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Swap only sensitive purpose values in existing purpose columns.

    Non-sensitive values are left unchanged. Sensitive values are values such
    as home, work or education/training, configured in
    `GeofenceAnonymizationConfig.sensitive_purpose_groups`.
    """

    config = config or GeofenceAnonymizationConfig()
    rng = np.random.default_rng(config.random_state)
    out: dict[str, pd.DataFrame] = {}
    for table_name, table in tables.items():
        frame = table.copy()
        for column in [column for column in config.purpose_columns if column in frame.columns]:
            sensitive_mask = frame[column].map(lambda value: _purpose_group(value, config) is not None)
            if sensitive_mask.any():
                frame.loc[sensitive_mask, column] = _derange_sensitive_values(
                    frame.loc[sensitive_mask, column],
                    config,
                    rng,
                ).to_numpy()
        out[table_name] = frame
    return out


def sample_tables_by_user_count(
    tables: Mapping[str, pd.DataFrame],
    config: GeofenceAnonymizationConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], set[str] | None]:
    """Keep a deterministic user sample after pseudonymization."""

    config = config or GeofenceAnonymizationConfig()
    if config.sample_user_count is None:
        return {name: table.copy() for name, table in tables.items()}, None
    if config.sample_user_count <= 0:
        raise ValueError("sample_user_count must be None or a positive integer.")

    user_ids = sorted(
        {
            str(value)
            for table in tables.values()
            if table is not None and config.user_id_column in table.columns
            for value in table[config.user_id_column].dropna().astype(str)
        }
    )
    if len(user_ids) <= config.sample_user_count:
        selected = set(user_ids)
    else:
        rng = np.random.default_rng(config.sample_random_state)
        selected = set(rng.choice(user_ids, size=config.sample_user_count, replace=False).tolist())

    out: dict[str, pd.DataFrame] = {}
    for table_name, table in tables.items():
        frame = table.copy()
        if config.user_id_column in frame.columns:
            frame = frame.loc[frame[config.user_id_column].astype(str).isin(selected)].copy()
        out[table_name] = frame
    return out, selected


def drop_direct_identifier_columns(
    tables: Mapping[str, pd.DataFrame],
    config: GeofenceAnonymizationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Drop direct identifier and linkage columns from all tables."""

    config = config or GeofenceAnonymizationConfig()
    columns = set(config.drop_columns)
    return {
        table_name: table.drop(columns=[column for column in table.columns if column in columns], errors="ignore").copy()
        for table_name, table in tables.items()
    }


def anonymize_landed_gps_tables(
    tables: Mapping[str, pd.DataFrame],
    config: GeofenceAnonymizationConfig | None = None,
) -> GpsAnonymizationResult:
    """Anonymize landed GPS tables for demos or controlled sharing.

    The input tables are expected to be the landed tables of a single
    experiment before user-id shuffling. The returned tables do not include
    the private user-id mapping.
    """

    config = config or GeofenceAnonymizationConfig()
    if "storyline" not in tables:
        raise KeyError("`tables` must contain a `storyline` table.")

    anonymized_tables = {name: table.copy() for name, table in tables.items()}
    storyline, geofences, report = anonymize_storyline_with_geofencing(
        anonymized_tables["storyline"],
        config,
    )
    anonymized_tables["storyline"] = storyline

    if config.swap_sensitive_purpose_values:
        anonymized_tables = swap_sensitive_purpose_columns(anonymized_tables, config)

    user_id_mapping = build_user_pseudonym_mapping(anonymized_tables, config)
    anonymized_tables = apply_user_pseudonym_mapping(anonymized_tables, user_id_mapping, config)
    anonymized_tables, selected_user_ids = sample_tables_by_user_count(anonymized_tables, config)
    anonymized_tables = drop_direct_identifier_columns(anonymized_tables, config)

    private_geofences = geofences.copy()
    if "user_id" in private_geofences.columns:
        private_geofences["user_id"] = private_geofences["user_id"].apply(
            lambda value: user_id_mapping.get(str(value), value) if pd.notna(value) else value
        )
        if selected_user_ids is not None:
            private_geofences = private_geofences.loc[
                private_geofences["user_id"].astype(str).isin(selected_user_ids)
            ].copy()
        if {"user_id", "cluster_rank"}.issubset(private_geofences.columns):
            private_geofences["geofence_id"] = private_geofences.apply(
                lambda row: f"{row['user_id']}__geofence_{int(row['cluster_rank'])}",
                axis=1,
            )

    private_report = report.copy()
    if "user_id" in private_report.columns:
        private_report["user_id"] = private_report["user_id"].apply(
            lambda value: user_id_mapping.get(str(value), value) if pd.notna(value) else value
        )
        if selected_user_ids is not None:
            private_report = private_report.loc[
                private_report["user_id"].astype(str).isin(selected_user_ids)
            ].copy()

    metadata = {
        "method": "staypoint_dbscan_geofence_trim",
        "top_staypoint_clusters": config.top_staypoint_clusters,
        "dbscan_eps_m": config.dbscan_eps_m,
        "dbscan_min_samples": config.dbscan_min_samples,
        "geofence_buffer_m": config.geofence_buffer_m,
        "geofence_min_radius_m": config.geofence_min_radius_m,
        "geofence_max_radius_m": config.geofence_max_radius_m,
        "trim_buffer_min_m": config.trim_buffer_min_m,
        "trim_buffer_max_m": config.trim_buffer_max_m,
        "min_remaining_leg_length_m": config.min_remaining_leg_length_m,
        "short_leg_policy": config.short_leg_policy,
        "swap_sensitive_purpose_values": config.swap_sensitive_purpose_values,
        "sensitive_purpose_groups": sorted(config.sensitive_purpose_groups),
        "user_id_policy": "generated_pseudonyms",
        "private_user_id_mapping_exported": False,
        "sample_user_count": config.sample_user_count,
        "sample_user_ids_exported": False,
        "selected_user_count": None if selected_user_ids is None else len(selected_user_ids),
    }
    return GpsAnonymizationResult(
        tables=anonymized_tables,
        geofences=private_geofences,
        report=private_report,
        metadata=metadata,
    )
