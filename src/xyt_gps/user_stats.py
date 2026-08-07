"""User-level statistics derived from prepared GPS tables."""

from __future__ import annotations

import pandas as pd

from .config import ProjectConfig
from .quality import calculate_tracking_periods, calculate_user_tracking_stats, flag_tracking_quality
from .quality_tracking import build_user_day_coverage


def add_excursion_stats_to_user_stats(
    user_stats: pd.DataFrame,
    legs: pd.DataFrame,
    *,
    excursion_col: str = "excursion",
) -> pd.DataFrame:
    """Merge user-level excursion counts from leg-level flags.

    Args:
        user_stats: User-level table containing `user_id`.
        legs: Leg table containing `user_id` and an excursion flag column.
        excursion_col: Name of the leg-level excursion flag.

    Returns:
        A copy of `user_stats` with `excursion_legs_count`,
        `total_legs_count` and `excursion_leg_ratio`.
    """

    out = user_stats.drop(
        columns=["excursion_legs_count", "total_legs_count", "excursion_leg_ratio"],
        errors="ignore",
    ).copy()
    if "user_id" not in out.columns:
        return out
    if {"user_id", excursion_col} - set(legs.columns):
        out["excursion_legs_count"] = 0
        out["total_legs_count"] = 0
        out["excursion_leg_ratio"] = 0.0
        return out

    leg_flags = legs[["user_id", excursion_col]].copy()
    leg_flags[excursion_col] = leg_flags[excursion_col].fillna(False).astype(bool)
    stats = (
        leg_flags.groupby("user_id", dropna=False)[excursion_col]
        .agg(excursion_legs_count="sum", total_legs_count="count")
        .reset_index()
    )
    stats["excursion_legs_count"] = stats["excursion_legs_count"].astype(int)
    stats["total_legs_count"] = stats["total_legs_count"].astype(int)
    stats["excursion_leg_ratio"] = stats["excursion_legs_count"].div(stats["total_legs_count"]).fillna(0.0)
    out = pd.merge(out, stats, on="user_id", how="left")
    out["excursion_legs_count"] = out["excursion_legs_count"].fillna(0).astype(int)
    out["total_legs_count"] = out["total_legs_count"].fillna(0).astype(int)
    out["excursion_leg_ratio"] = out["excursion_leg_ratio"].fillna(0.0)
    return out


def build_user_stats(
    storyline: pd.DataFrame,
    config: ProjectConfig,
    *,
    user_statistics: pd.DataFrame | None = None,
    sociodemo: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    weight_col: str = "weight",
    default_weight: float = 1.0,
    user_day_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build user-level tracking stats and merge optional side tables.

    The output keeps the GPS database readable at user level. When the
    ``days_in_range_count`` covers the inclusive interval between first and
    last real observations. ``tracked_days_count`` counts non-resampled GPS
    days and ``active_days_count`` counts days containing a real movement.
    If no weights are provided, `weight_col` is set to `default_weight`.
    """

    out = calculate_user_tracking_stats(storyline)
    coverage = (
        user_day_coverage.copy()
        if isinstance(user_day_coverage, pd.DataFrame)
        else build_user_day_coverage(storyline, config)
    )
    out["observed_tracking_days_count"] = out["tracked_days_count"]
    out["observed_active_days_count"] = out["active_days_count"]
    out["observed_tracking_completeness"] = out["tracking_completeness"]

    if user_statistics is not None:
        mt = user_statistics.copy()
        if "user_id" not in mt.columns:
            raise KeyError("user_statistics must contain `user_id`; `id` is not accepted as an alias.")
        mt = mt.drop(columns=["id"], errors="ignore")
        mt = mt.loc[:, ~mt.columns.duplicated()].copy()
        drop_cols = [
            "email",
            "first_tracked_at",
            "latest_tracked_at",
            "days_in_range_count",
            "tracked_days_count",
            "active_days_count",
            "zero_movement_tracked_days_count",
            "untracked_days_count",
            "inactive_days_count",
            "max_gap",
            "tracking_completeness",
            "active_days_share",
            "resampled_days_count",
            "observed_tracking_days_count",
            "observed_active_days_count",
            "observed_tracking_completeness",
        ]
        mt = mt.drop(columns=drop_cols, errors="ignore")
        if "user_id" in mt.columns:
            mt = mt.drop_duplicates(subset=["user_id"])
            out = pd.merge(out, mt, on="user_id", how="outer")

    zero_fill_columns = [
        "days_in_range_count",
        "tracked_days_count",
        "active_days_count",
        "zero_movement_tracked_days_count",
        "untracked_days_count",
        "inactive_days_count",
        "max_gap",
        "resampled_days_count",
        "observed_active_days_count",
        "observed_tracking_days_count",
        "excursion_legs_count",
        "total_legs_count",
    ]
    for column in zero_fill_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    ratio_fill_columns = [
        "tracking_completeness",
        "active_days_share",
        "observed_tracking_completeness",
        "excursion_leg_ratio",
    ]
    for column in ratio_fill_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    if sociodemo is not None:
        out = pd.merge(out, sociodemo, on="user_id", how="left")
        household_cols = ["structmen 65+", "structmen 26-64", "structmen 18-25", "structmen 16-17", "structmen 0-15"]
        if set(household_cols).issubset(out.columns):
            out["structmen"] = out[household_cols].sum(axis=1)
            out["enfants"] = out["structmen 0-15"].gt(0).astype(int)
        tim_cols = ["equipement menage motos", "equipement menage voitures"]
        if set(tim_cols).issubset(out.columns):
            out["equipement menage TIM"] = out[tim_cols].sum(axis=1)
        if config.reference_year is not None and "annee_naissance" in out.columns:
            out["age"] = config.reference_year - out["annee_naissance"]
        if config.reference_year is not None and "année_naissance" in out.columns:
            out["age"] = config.reference_year - out["année_naissance"]

    if weights is not None:
        weight_table = weights.copy()
        if weight_col in weight_table.columns and "user_id" not in weight_table.columns:
            raise KeyError("weights must contain `user_id`; `id` is not accepted as an alias.")
        weight_table = weight_table.drop(columns=["id"], errors="ignore")
        if {"user_id", weight_col}.issubset(weight_table.columns):
            out = pd.merge(
                out.drop(columns=[weight_col], errors="ignore"),
                weight_table[["user_id", weight_col]].drop_duplicates(subset=["user_id"]),
                on="user_id",
                how="left",
            )
    if weight_col not in out.columns:
        out[weight_col] = default_weight
    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce").fillna(default_weight)

    out = calculate_tracking_periods(out, config, user_day_coverage=coverage)
    out = flag_tracking_quality(out, config)
    return out
