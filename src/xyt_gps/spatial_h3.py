"""H3 point extraction and aggregation helpers."""

from __future__ import annotations

import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

DEFAULT_H3_METADATA_COLUMNS = (
    "leg_id",
    "storyline_id",
    "user_id",
    "user_id_day",
    "trip_id",
    "journey_id",
    "experiment_name",
    "phase",
    "started_at",
    "finished_at",
    "started_on",
    "type",
    "mode",
    "detected_mode",
    "mode_niv1",
    "mode_niv2",
    "mode_mrmt",
    "purpose",
    "purpose_niv1",
    "purpose_niv2",
    "purpose_mrmt",
    "trip_purpose",
    "trip_purpose_mrmt",
    "main_mode_niv1",
    "main_mode_niv2",
    "main_mode_mrmt",
    "trip_main_mode_niv1",
    "trip_main_mode_niv2",
    "trip_main_mode_mrmt",
    "time_slice",
    "length",
    "duration_seconds",
    "is_excursion",
)

DEFAULT_H3_COUNT_DIMENSION_SETS = (
    ("mode_niv1",),
    ("purpose_mrmt",),
    ("time_slice",),
    ("mode_niv1", "time_slice"),
    ("purpose_mrmt", "time_slice"),
)

DEFAULT_H3_COUNT_METRICS = ("point_count", "trip_count")



def _import_h3():
    try:
        import h3
    except ImportError as error:
        raise ImportError(
            "H3 spatial indexes require the optional analytics dependencies. "
            "Install them with `python -m pip install -r requirements-analytics.txt`."
        ) from error
    return h3


def _latlng_to_h3(lat: float, lon: float, resolution: int) -> str:
    h3 = _import_h3()
    return _latlng_to_h3_with_module(h3, lat, lon, resolution)


def _latlng_to_h3_with_module(h3_module, lat: float, lon: float, resolution: int) -> str:
    if hasattr(h3_module, "latlng_to_cell"):
        return h3_module.latlng_to_cell(lat, lon, resolution)
    return h3_module.geo_to_h3(lat, lon, resolution)


def _ensure_h3_resolution(h3_resolution: int) -> int:
    resolution = int(h3_resolution)
    if not 0 <= resolution <= 15:
        raise ValueError("h3_resolution must be between 0 and 15.")
    return resolution


def _ensure_h3_resolutions(h3_resolution: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(h3_resolution, (int, str)):
        resolutions = (_ensure_h3_resolution(int(h3_resolution)),)
    else:
        resolutions = tuple(_ensure_h3_resolution(int(value)) for value in h3_resolution)
    if not resolutions:
        raise ValueError("At least one H3 resolution is required.")
    return tuple(dict.fromkeys(resolutions))


def _ensure_geodataframe(table: pd.DataFrame, *, geometry_col: str = "geometry") -> gpd.GeoDataFrame:
    if geometry_col not in table.columns:
        raise KeyError(f"Table must contain a `{geometry_col}` column.")
    if isinstance(table, gpd.GeoDataFrame):
        gdf = table.copy()
        if gdf.geometry.name != geometry_col:
            gdf = gdf.set_geometry(geometry_col)
    else:
        gdf = gpd.GeoDataFrame(table.copy(), geometry=geometry_col, crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    if str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _iter_line_parts(geometry) -> Iterable[LineString]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def _line_vertices(geometry) -> list[Point]:
    points: list[Point] = []
    for line in _iter_line_parts(geometry):
        points.extend(Point(lon, lat) for lon, lat, *_ in line.coords)
    return points


def _sample_line_regularly(geometry, sample_distance_m: float) -> list[Point]:
    points: list[Point] = []
    for line in _iter_line_parts(geometry):
        length = float(line.length)
        if length == 0:
            points.append(line.interpolate(0))
            continue
        distances = [0.0]
        distance = float(sample_distance_m)
        while distance < length:
            distances.append(distance)
            distance += float(sample_distance_m)
        if distances[-1] != length:
            distances.append(length)
        points.extend(line.interpolate(distance) for distance in distances)
    return points


def _sample_legs_as_points(
    legs: gpd.GeoDataFrame,
    *,
    geometry_col: str,
    sample_distance_m: float | None,
) -> list[tuple[int, list[Point]]]:
    if sample_distance_m is None:
        return [
            (index, _line_vertices(geometry))
            for index, geometry in legs[geometry_col].items()
        ]

    if sample_distance_m <= 0:
        raise ValueError("sample_distance_m must be positive when provided.")
    projected = legs.to_crs("EPSG:3857")
    sampled_projected = [
        (index, _sample_line_regularly(geometry, sample_distance_m))
        for index, geometry in projected.geometry.items()
    ]
    rows = []
    for index, points in sampled_projected:
        if not points:
            rows.append((index, []))
            continue
        sampled_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:3857").to_crs("EPSG:4326")
        rows.append((index, sampled_gdf.geometry.tolist()))
    return rows


def _chunked(items: list, chunk_size: int) -> Iterable[list]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _rows_for_h3_chunk(
    sampled_by_leg: list[tuple[int, list[Point]]],
    *,
    metadata_by_leg: dict,
    resolutions: tuple[int, ...],
    sample_distance_m: float | None,
    max_points_per_leg: int | None,
    h3_module,
) -> list[dict]:
    rows = []
    point_source = "leg_geometry_regular_sample" if sample_distance_m is not None else "leg_geometry_vertex"

    for leg_index, points in sampled_by_leg:
        if max_points_per_leg is not None:
            points = points[:max_points_per_leg]
        metadata = metadata_by_leg.get(leg_index, {})
        for point_sequence, point in enumerate(points):
            lon = float(point.x)
            lat = float(point.y)
            for resolution in resolutions:
                rows.append(
                    {
                        **metadata,
                        "source_leg_index": leg_index,
                        "point_sequence": point_sequence,
                        "point_source": point_source,
                        "lon": lon,
                        "lat": lat,
                        "h3_resolution": resolution,
                        "h3_cell": _latlng_to_h3_with_module(h3_module, lat, lon, resolution),
                        "geometry": point,
                    }
                )
    return rows


def legs_to_h3_points(
    legs: pd.DataFrame,
    *,
    h3_resolution: int | Iterable[int] = 9,
    geometry_col: str = "geometry",
    metadata_cols: Iterable[str] | None = None,
    sample_distance_m: float | None = None,
    max_points_per_leg: int | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    chunk_size: int = 250,
) -> gpd.GeoDataFrame:
    """Convert leg geometries into H3-indexed point observations.

    The function creates one row per point extracted from each leg geometry.
    By default, points are the vertices already present in the leg `LineString`.
    When `sample_distance_m` is provided, the line is sampled at a regular
    interval before H3 indexing. This is useful for fréquentation maps, but it
    should be documented as a sampled representation rather than raw GPS
    observations.

    Args:
        legs: Leg table, usually `dataset.legs`, with line geometries.
        h3_resolution: H3 resolution or list of resolutions, from 0 to 15.
            Higher values produce smaller hexagons and larger output tables.
        geometry_col: Geometry column containing `LineString` or
            `MultiLineString` objects.
        metadata_cols: Columns to copy from legs to each point row. When
            omitted, common mobility identifiers, dates, modes and phases are
            copied when present.
        sample_distance_m: Optional regular sampling distance in metres. If
            omitted, existing line vertices are used.
        max_points_per_leg: Optional cap after vertex extraction or regular
            sampling. This is a safeguard for very dense geometries.
        parallel: If true, split the H3 indexing step by leg chunks and process
            chunks with a thread pool. This is useful for large exports.
        max_workers: Optional number of worker threads when `parallel=True`.
            When omitted, Python chooses a sensible default.
        chunk_size: Number of legs per H3 indexing chunk when `parallel=True`.

    Returns:
        GeoDataFrame in EPSG:4326 with `lon`, `lat`, `h3_cell`,
        `h3_resolution`, `point_sequence` and copied metadata columns.

    Raises:
        ImportError: If the optional `h3` dependency is missing.
        KeyError: If the geometry column is missing.
        ValueError: If the H3 resolution or sampling distance is invalid.
    """

    resolutions = _ensure_h3_resolutions(h3_resolution)
    h3_module = _import_h3()
    if max_points_per_leg is not None:
        max_points_per_leg = int(max_points_per_leg)
        if max_points_per_leg < 0:
            raise ValueError("max_points_per_leg must be non-negative when provided.")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if max_workers is not None:
        max_workers = int(max_workers)
        if max_workers <= 0:
            raise ValueError("max_workers must be positive when provided.")

    legs_gdf = _ensure_geodataframe(legs, geometry_col=geometry_col)
    copied_cols = [
        column
        for column in (tuple(metadata_cols) if metadata_cols is not None else DEFAULT_H3_METADATA_COLUMNS)
        if column in legs_gdf.columns and column != geometry_col
    ]
    metadata_by_leg = legs_gdf[copied_cols].to_dict(orient="index") if copied_cols else {index: {} for index in legs_gdf.index}

    sampled_by_leg = _sample_legs_as_points(
        legs_gdf,
        geometry_col=geometry_col,
        sample_distance_m=sample_distance_m,
    )
    if not parallel or len(sampled_by_leg) <= chunk_size:
        rows = _rows_for_h3_chunk(
            sampled_by_leg,
            metadata_by_leg=metadata_by_leg,
            resolutions=resolutions,
            sample_distance_m=sample_distance_m,
            max_points_per_leg=max_points_per_leg,
            h3_module=h3_module,
        )
    else:
        worker_count = max_workers if max_workers is not None else min(32, (os.cpu_count() or 1) + 4)
        rows = []
        chunks = list(_chunked(sampled_by_leg, chunk_size))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for chunk_rows in executor.map(
                lambda chunk: _rows_for_h3_chunk(
                    chunk,
                    metadata_by_leg=metadata_by_leg,
                    resolutions=resolutions,
                    sample_distance_m=sample_distance_m,
                    max_points_per_leg=max_points_per_leg,
                    h3_module=h3_module,
                ),
                chunks,
            ):
                rows.extend(chunk_rows)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def aggregate_h3_frequencies(
    h3_points: pd.DataFrame,
    *,
    h3_col: str = "h3_cell",
    group_cols: Iterable[str] | None = None,
    user_col: str = "user_id",
    leg_col: str = "leg_id",
    trip_col: str = "trip_id",
) -> pd.DataFrame:
    """Aggregate H3-indexed points into fréquentation counts.

    Args:
        h3_points: Output of `legs_to_h3_points()` or another table containing
            an H3 cell column.
        h3_col: Column containing H3 cell identifiers.
        group_cols: Additional grouping columns. The H3 cell is always kept.
            Add columns such as `mode_niv1`, `phase` or `experiment_name` to
            build dashboard-ready slices.
        user_col: User identifier column used for `user_count` when present.
        leg_col: Leg identifier column used for `leg_count` when present.
        trip_col: Trip identifier column used for `trip_count` when present.

    Returns:
        DataFrame with counts by H3 cell and optional grouping columns.
    """

    if h3_col not in h3_points.columns:
        raise KeyError(f"Table must contain a `{h3_col}` column.")
    groups = []
    if "h3_resolution" in h3_points.columns:
        groups.append("h3_resolution")
    groups.append(h3_col)
    if group_cols is not None:
        groups.extend(column for column in group_cols if column != h3_col)
    missing = [column for column in groups if column not in h3_points.columns]
    if missing:
        raise KeyError(f"Missing group column(s): {missing}")

    aggregations: dict[str, tuple[str, str]] = {"point_count": (h3_col, "size")}
    if user_col in h3_points.columns:
        aggregations["user_count"] = (user_col, "nunique")
    if leg_col in h3_points.columns:
        aggregations["leg_count"] = (leg_col, "nunique")
    if trip_col in h3_points.columns:
        aggregations["trip_count"] = (trip_col, "nunique")
    if "started_at" in h3_points.columns:
        aggregations["first_started_at"] = ("started_at", "min")
        aggregations["last_started_at"] = ("started_at", "max")

    return (
        h3_points.groupby(groups, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["point_count", *groups], ascending=[False, *([True] * len(groups))])
        .reset_index(drop=True)
    )


def _metric_aggregations(
    table: pd.DataFrame,
    *,
    h3_col: str,
    user_col: str,
    leg_col: str,
    trip_col: str,
    metrics: Iterable[str],
) -> dict[str, tuple[str, str]]:
    requested = tuple(metrics)
    aggregations: dict[str, tuple[str, str]] = {}
    if "point_count" in requested:
        aggregations["point_count"] = (h3_col, "size")
    if "user_count" in requested and user_col in table.columns:
        aggregations["user_count"] = (user_col, "nunique")
    if "leg_count" in requested and leg_col in table.columns:
        aggregations["leg_count"] = (leg_col, "nunique")
    if "trip_count" in requested and trip_col in table.columns:
        aggregations["trip_count"] = (trip_col, "nunique")
    if not aggregations:
        raise ValueError("No requested count metric can be computed from this table.")
    return aggregations


def _safe_column_part(value) -> str:
    text = "missing" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return text or "value"


def _dimension_label(dimension_set: tuple[str, ...], values) -> str:
    if not isinstance(values, tuple):
        values = (values,)
    return "__".join(
        f"{_safe_column_part(column)}__{_safe_column_part(value)}"
        for column, value in zip(dimension_set, values, strict=False)
    )


def build_h3_count_matrix(
    h3_points: pd.DataFrame,
    *,
    h3_col: str = "h3_cell",
    dimension_sets: Iterable[Iterable[str]] = DEFAULT_H3_COUNT_DIMENSION_SETS,
    metrics: Iterable[str] = DEFAULT_H3_COUNT_METRICS,
    user_col: str = "user_id",
    leg_col: str = "leg_id",
    trip_col: str = "trip_id",
    fill_value: int = 0,
) -> pd.DataFrame:
    """Build a wide H3 count table for dashboards.

    The output keeps one row per H3 cell and creates explicit metric columns
    for requested dimensions, for example
    `point_count__mode_niv1__marche` or `trip_count__time_slice__hpm`.

    Args:
        h3_points: H3-indexed points produced by `legs_to_h3_points()`.
        h3_col: H3 cell column.
        dimension_sets: Dimension sets used to create count columns. Each set
            is aggregated independently. Missing dimension columns are skipped.
        metrics: Count metrics to compute: `point_count`, `user_count`,
            `leg_count`, `trip_count`.
        user_col: User identifier column.
        leg_col: Leg identifier column.
        trip_col: Trip identifier column.
        fill_value: Value used for absent combinations.

    Returns:
        Wide table keyed by `h3_resolution` when present and `h3_cell`.
    """

    if h3_col not in h3_points.columns:
        raise KeyError(f"Table must contain a `{h3_col}` column.")

    index_cols = []
    if "h3_resolution" in h3_points.columns:
        index_cols.append("h3_resolution")
    index_cols.append(h3_col)

    base_aggs = _metric_aggregations(
        h3_points,
        h3_col=h3_col,
        user_col=user_col,
        leg_col=leg_col,
        trip_col=trip_col,
        metrics=metrics,
    )
    out = h3_points.groupby(index_cols, dropna=False).agg(**base_aggs).reset_index()
    out = out.rename(columns={metric: f"{metric}__total" for metric in base_aggs})

    for raw_dimension_set in dimension_sets:
        dimension_set = tuple(raw_dimension_set)
        if not dimension_set or any(column not in h3_points.columns for column in dimension_set):
            continue

        grouped = (
            h3_points.groupby([*index_cols, *dimension_set], dropna=False)
            .agg(**base_aggs)
            .reset_index()
        )
        for metric in base_aggs:
            pivot = grouped.pivot_table(
                index=index_cols,
                columns=list(dimension_set),
                values=metric,
                fill_value=fill_value,
                aggfunc="sum",
            )
            pivot.columns = [
                f"{metric}__{_dimension_label(dimension_set, values)}"
                for values in pivot.columns.to_flat_index()
            ]
            out = out.merge(pivot.reset_index(), on=index_cols, how="left")

    count_cols = [column for column in out.columns if column not in index_cols]
    out[count_cols] = out[count_cols].fillna(fill_value)
    for column in count_cols:
        if pd.api.types.is_float_dtype(out[column]) and out[column].dropna().mod(1).eq(0).all():
            out[column] = out[column].astype("int64")
    return out.sort_values(index_cols).reset_index(drop=True)
