"""Geometry helpers for mobility traces."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import ops
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry


def _as_geodataframe(
    table: pd.DataFrame,
    *,
    geometry_col: str,
    default_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    if geometry_col not in table.columns:
        raise KeyError(f"table must contain {geometry_col}")
    if isinstance(table, gpd.GeoDataFrame):
        gdf = table.copy()
        if gdf.crs is None:
            gdf = gdf.set_crs(default_crs)
        return gdf
    return gpd.GeoDataFrame(table.copy(), geometry=geometry_col, crs=default_crs)


def _read_zones(
    zones: str | Path | gpd.GeoDataFrame,
    *,
    default_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    if isinstance(zones, (str, Path)):
        zones_gdf = gpd.read_file(zones)
    else:
        zones_gdf = zones.copy()
    if zones_gdf.crs is None:
        zones_gdf = zones_gdf.set_crs(default_crs)
    return zones_gdf


def _area_from_inputs(
    *,
    area: BaseGeometry | gpd.GeoDataFrame | None,
    area_path: str | None,
    center: tuple[float, float] | None,
    radius_m: float | None,
    target_crs: str,
    operations_crs: str,
):
    provided = sum(value is not None for value in (area, area_path, center))
    if provided != 1:
        raise ValueError("Provide exactly one of `area`, `area_path` or `center`.")
    if center is not None and radius_m is None:
        raise ValueError("`radius_m` is required when `center` is provided.")

    if area_path is not None:
        area_gdf = gpd.read_file(area_path)
        if area_gdf.crs is None:
            area_gdf = area_gdf.set_crs(target_crs)
        return area_gdf.to_crs(operations_crs).geometry.union_all()

    if isinstance(area, gpd.GeoDataFrame):
        area_gdf = area.copy()
        if area_gdf.crs is None:
            area_gdf = area_gdf.set_crs(target_crs)
        return area_gdf.to_crs(operations_crs).geometry.union_all()

    if area is not None:
        area_gdf = gpd.GeoDataFrame({"geometry": [area]}, geometry="geometry", crs=target_crs)
        return area_gdf.to_crs(operations_crs).geometry.union_all()

    lon, lat = center
    center_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(float(lon), float(lat))]},
        geometry="geometry",
        crs=target_crs,
    ).to_crs(operations_crs)
    return center_gdf.geometry.iloc[0].buffer(float(radius_m))


def _line_endpoint(geometry: BaseGeometry | None, *, position: str) -> Point | None:
    if geometry is None or pd.isna(geometry):
        return None
    if isinstance(geometry, Point):
        return geometry
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        if not coords:
            return None
        return Point(coords[0] if position == "origin" else coords[-1])
    if isinstance(geometry, MultiLineString):
        parts = [part for part in geometry.geoms if len(part.coords)]
        if not parts:
            return None
        part = parts[0] if position == "origin" else parts[-1]
        coords = list(part.coords)
        return Point(coords[0] if position == "origin" else coords[-1])
    return None


def clean_leg_geometries(legs: gpd.GeoDataFrame, *, drop_discontinuous: bool = True) -> gpd.GeoDataFrame:
    """Convert continuous MultiLineString legs to LineString."""

    out = legs.copy()
    if "geometry" not in out.columns:
        return out
    out["geometry"] = out["geometry"].apply(
        lambda geom: ops.linemerge(geom) if isinstance(geom, MultiLineString) else geom
    )
    if drop_discontinuous:
        out = out.loc[out.geometry.geom_type.ne("MultiLineString")].copy()
    out["point_per_linestring"] = out["geometry"].apply(
        lambda geom: len(geom.coords) if hasattr(geom, "coords") else pd.NA
    )
    return out


def max_consecutive_point_distance(geometry: BaseGeometry | None) -> float:
    """Return the maximum distance between consecutive points."""

    if geometry is None or pd.isna(geometry):
        return 0.0
    if isinstance(geometry, MultiLineString):
        distances = [max_consecutive_point_distance(part) for part in geometry.geoms]
        return max(distances, default=0.0)
    if not isinstance(geometry, LineString) or len(geometry.coords) < 2:
        return 0.0

    max_distance = 0.0
    coords = list(geometry.coords)
    for start, end in zip(coords[:-1], coords[1:]):
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        if distance > max_distance:
            max_distance = distance
    return float(max_distance)


__all__ = [
    "clean_leg_geometries",
    "max_consecutive_point_distance",
]
