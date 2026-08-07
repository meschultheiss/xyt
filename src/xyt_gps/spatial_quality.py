"""GPS signal-quality metrics and filters."""

from __future__ import annotations

from collections.abc import Mapping

import geopandas as gpd
import pandas as pd

from .config import ProjectConfig, SignalLossThreshold
from .spatial_geometry import max_consecutive_point_distance


MAX_SIGNAL_LOSS_COL = "max_signal_loss_m"
RELATIVE_SIGNAL_LOSS_COL = "relative_signal_loss"

SIGNAL_QUALITY_REQUIRED_COLUMNS = {
    "low_quality_legs_1",
    RELATIVE_SIGNAL_LOSS_COL,
    MAX_SIGNAL_LOSS_COL,
    "bad_signal_user",
}


def add_signal_loss_metrics(
    legs: gpd.GeoDataFrame,
    config: ProjectConfig,
    *,
    geometry_col: str = "geometry",
) -> gpd.GeoDataFrame:
    """Add absolute and relative GPS signal-loss metrics to legs."""

    if geometry_col not in legs.columns:
        raise KeyError(f"legs must contain {geometry_col}")

    out = legs.copy()
    metric_legs = out
    if isinstance(out, gpd.GeoDataFrame) and out.crs is not None and config.operations_crs:
        metric_legs = out.to_crs(config.operations_crs)

    max_loss = metric_legs[geometry_col].apply(max_consecutive_point_distance)
    length = metric_legs[geometry_col].apply(
        lambda geom: float(geom.length) if geom is not None and not pd.isna(geom) else 0.0
    )
    relative = max_loss.div(length.where(length.gt(0))).fillna(0.0)

    out[MAX_SIGNAL_LOSS_COL] = max_loss.to_numpy()
    out["length_leg"] = length.to_numpy()
    out[RELATIVE_SIGNAL_LOSS_COL] = relative.to_numpy()
    return out


def _iter_signal_thresholds(
    thresholds: Mapping[int, Mapping[str, SignalLossThreshold]],
):
    for level, categories in sorted(thresholds.items()):
        for category_name, threshold in categories.items():
            yield int(level), category_name, threshold


def flag_low_quality_legs_by_mode(
    legs: gpd.GeoDataFrame,
    config: ProjectConfig,
    *,
    mode_col: str | None = None,
) -> gpd.GeoDataFrame:
    """Flag legs with mode-specific signal-loss thresholds."""

    mode_column = mode_col or config.spatial_quality_thresholds.signal_loss_mode_column
    if mode_column not in legs.columns:
        raise KeyError(f"legs must contain {mode_column}")

    required = {MAX_SIGNAL_LOSS_COL, RELATIVE_SIGNAL_LOSS_COL}
    if not required.issubset(legs.columns):
        legs = add_signal_loss_metrics(legs, config)

    out = legs.copy()
    thresholds = config.spatial_quality_thresholds.signal_loss_thresholds_by_level
    levels = sorted(int(level) for level in thresholds)
    for level in levels:
        out[f"low_quality_legs_{level}"] = 0
    out["signal_quality_reason"] = "ok"

    for level, category_name, threshold in _iter_signal_thresholds(thresholds):
        flag_col = f"low_quality_legs_{level}"
        mode_mask = out[mode_column].isin(threshold.modes)
        quality_mask = out[MAX_SIGNAL_LOSS_COL].gt(threshold.max_gap_m) & out[
            RELATIVE_SIGNAL_LOSS_COL
        ].gt(threshold.max_relative_gap)
        final_mask = mode_mask & quality_mask
        out.loc[final_mask, flag_col] = 1
        out.loc[
            final_mask & out["signal_quality_reason"].eq("ok"),
            "signal_quality_reason",
        ] = f"{category_name}_signal_loss_level_{level}"

    if levels:
        out["low_quality_signal"] = out[[f"low_quality_legs_{level}" for level in levels]].any(axis=1)
    else:
        out["low_quality_signal"] = False
    return out


def identify_bad_signal_users(
    legs: pd.DataFrame,
    *,
    quantile_threshold: float = 0.995,
    user_id_col: str = "user_id",
) -> pd.Index:
    """Identify users with unusually high average signal loss."""

    required = {user_id_col, MAX_SIGNAL_LOSS_COL, RELATIVE_SIGNAL_LOSS_COL}
    missing = required - set(legs.columns)
    if missing:
        raise KeyError(f"legs missing required column(s): {sorted(missing)}")
    if legs.empty:
        return pd.Index([], name=user_id_col)

    user_signal = (
        legs.groupby(user_id_col, dropna=False)[[MAX_SIGNAL_LOSS_COL, RELATIVE_SIGNAL_LOSS_COL]]
        .mean()
        .reset_index()
    )
    rel_threshold = user_signal[RELATIVE_SIGNAL_LOSS_COL].quantile(quantile_threshold)
    abs_threshold = user_signal[MAX_SIGNAL_LOSS_COL].quantile(quantile_threshold)
    bad_users = user_signal.loc[
        user_signal[RELATIVE_SIGNAL_LOSS_COL].gt(rel_threshold)
        | user_signal[MAX_SIGNAL_LOSS_COL].gt(abs_threshold),
        user_id_col,
    ]
    return pd.Index(bad_users, name=user_id_col)


def add_signal_quality_flags(
    legs: gpd.GeoDataFrame,
    config: ProjectConfig,
    *,
    mode_col: str | None = None,
) -> gpd.GeoDataFrame:
    """Add signal-loss metrics, leg flags and user signal-quality flags."""

    out = add_signal_loss_metrics(legs, config)
    out = flag_low_quality_legs_by_mode(out, config, mode_col=mode_col)
    bad_users = identify_bad_signal_users(
        out,
        quantile_threshold=config.spatial_quality_thresholds.bad_signal_user_quantile,
    )
    out["bad_signal_user"] = out["user_id"].isin(bad_users).astype(int) if "user_id" in out.columns else 0
    return out


def build_user_signal_quality_stats(legs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate signal-quality flags at user level."""

    if "user_id" not in legs.columns:
        return pd.DataFrame()
    if legs.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "legs_count",
                "low_quality_legs_count",
                "low_quality_legs_share",
                "mean_relative_signal_loss",
                "mean_max_signal_loss_m",
                "bad_signal_user",
                "signal_quality_computed",
            ]
        )

    out = legs.copy()
    signal_quality_computed = SIGNAL_QUALITY_REQUIRED_COLUMNS.issubset(out.columns)
    if not signal_quality_computed:
        stats = out.groupby("user_id", dropna=False).size().reset_index(name="legs_count")
        stats["low_quality_legs_count"] = pd.NA
        stats["low_quality_legs_share"] = pd.NA
        stats["mean_relative_signal_loss"] = pd.NA
        stats["mean_max_signal_loss_m"] = pd.NA
        stats["bad_signal_user"] = pd.NA
        stats["signal_quality_computed"] = False
        return stats

    stats = (
        out.groupby("user_id", dropna=False)
        .agg(
            legs_count=("user_id", "size"),
            low_quality_legs_count=("low_quality_legs_1", "sum"),
            mean_relative_signal_loss=(RELATIVE_SIGNAL_LOSS_COL, "mean"),
            mean_max_signal_loss_m=(MAX_SIGNAL_LOSS_COL, "mean"),
            bad_signal_user=("bad_signal_user", "max"),
        )
        .reset_index()
    )
    stats["low_quality_legs_share"] = stats["low_quality_legs_count"].div(stats["legs_count"])
    stats["signal_quality_computed"] = True
    return stats


def add_signal_quality_to_user_stats(
    user_stats: pd.DataFrame,
    legs: pd.DataFrame,
    *,
    signal_quality_computed: bool | None = None,
) -> pd.DataFrame:
    """Merge user-level signal-quality metrics into `user_stats`."""

    computed = (
        SIGNAL_QUALITY_REQUIRED_COLUMNS.issubset(legs.columns)
        if signal_quality_computed is None
        else bool(signal_quality_computed)
    )
    signal_columns = [
        "legs_count",
        "low_quality_legs_count",
        "low_quality_legs_share",
        "mean_relative_signal_loss",
        "mean_max_signal_loss_m",
        "bad_signal_user",
        "signal_quality_computed",
    ]

    if "user_id" not in user_stats.columns:
        return user_stats.copy()

    signal_stats = build_user_signal_quality_stats(legs)
    if signal_stats.empty:
        out = user_stats.drop(columns=signal_columns, errors="ignore").copy()
        out["legs_count"] = 0
        out["signal_quality_computed"] = computed
        for column in signal_columns:
            if column not in {"legs_count", "signal_quality_computed"}:
                out[column] = 0 if computed else pd.NA
        if computed:
            out["bad_signal_user"] = out["bad_signal_user"].astype("Int64")
        return out

    out = user_stats.drop(columns=signal_columns, errors="ignore")
    out = pd.merge(out, signal_stats, on="user_id", how="left")
    out["legs_count"] = out["legs_count"].fillna(0).astype("Int64")
    out["signal_quality_computed"] = computed
    if computed:
        fill_zero = [
            "low_quality_legs_count",
            "low_quality_legs_share",
            "mean_relative_signal_loss",
            "mean_max_signal_loss_m",
            "bad_signal_user",
        ]
        for column in fill_zero:
            if column in out.columns:
                out[column] = out[column].fillna(0)
        out["bad_signal_user"] = out["bad_signal_user"].astype("Int64")
    return out


__all__ = [
    "MAX_SIGNAL_LOSS_COL",
    "RELATIVE_SIGNAL_LOSS_COL",
    "SIGNAL_QUALITY_REQUIRED_COLUMNS",
    "add_signal_loss_metrics",
    "flag_low_quality_legs_by_mode",
    "identify_bad_signal_users",
    "add_signal_quality_flags",
    "build_user_signal_quality_stats",
    "add_signal_quality_to_user_stats",
]
