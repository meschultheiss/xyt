"""Small mobility-table transformations used by the preparation workflow."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .parsing import parse_date_columns


def split_storyline(storyline: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split a parsed storyline into staypoints and legs."""

    if "storyline_id" not in storyline.columns:
        raise KeyError("storyline must contain `storyline_id` before splitting staypoints and legs.")

    staypoints = storyline[storyline["type"].isin(["Resampled_stay", "Stay"])].copy()
    legs = storyline[storyline["type"].eq("Track")].copy()

    staypoints = staypoints.reset_index(drop=True).copy()
    legs = legs.reset_index(drop=True).copy()
    staypoints["activity_id"] = staypoints["storyline_id"]
    legs["leg_id"] = legs["storyline_id"]
    staypoints = staypoints.drop(columns=["id"], errors="ignore")
    legs = legs.drop(columns=["id"], errors="ignore")
    staypoints = parse_date_columns(
        staypoints,
        ("started_at_in_timezone", "finished_at_in_timezone", "started_at", "finished_at", "confirmed_at"),
    )
    if {"started_at_in_timezone", "finished_at_in_timezone"}.issubset(staypoints.columns):
        staypoints["duration"] = (
            staypoints["finished_at_in_timezone"] - staypoints["started_at_in_timezone"]
        ).dt.total_seconds()
    if "geometry" in staypoints.columns:
        staypoints["lon"] = staypoints.geometry.x
        staypoints["lat"] = staypoints.geometry.y
    staypoints = staypoints.drop(
        columns=["mode_niv1", "mode_niv2", "mode_mrmt", "detected_mode", "mode", "length"],
        errors="ignore",
    )
    legs = legs.drop(columns=["purpose", "purpose_niv1"], errors="ignore")
    if {"started_at", "finished_at"}.issubset(legs.columns):
        duration_seconds = (legs["finished_at"] - legs["started_at"]).dt.total_seconds()
        legs["duration"] = pd.to_numeric(duration_seconds, errors="coerce").round().astype("Int64")
    legs["track_count"] = 1
    return staypoints, legs


def add_user_id_day(legs: pd.DataFrame) -> pd.DataFrame:
    """Add the person-day id used by downstream indicators and notebooks."""

    out = legs.copy()
    out["legs_date"] = pd.to_datetime(out["started_at"].dt.date)
    out["user_id_day"] = out["user_id"].astype(str) + "_" + out["started_at"].dt.strftime("%Y%m%d")
    return out


def add_length_quantile_flags(
    legs: pd.DataFrame,
    *,
    group_col: str = "mode",
    length_col: str = "length",
    quantiles: tuple[float, ...] = (0.98, 0.99),
) -> pd.DataFrame:
    """Flag unusually long legs within each mode group."""

    out = legs.copy()
    if group_col not in out.columns or length_col not in out.columns:
        return out
    for quantile in quantiles:
        suffix = int(quantile * 100)
        threshold = out.groupby(group_col)[length_col].transform(lambda series: series.quantile(quantile))
        out[f"extreme{suffix}_length_{group_col}"] = out[length_col] > threshold
    return out
