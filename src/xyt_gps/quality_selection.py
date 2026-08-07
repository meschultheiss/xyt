"""User selection and consistent filtering of mobility datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import geopandas as gpd
import pandas as pd

from .models import MobilityDataset


def select_valid_tracking_users(
    user_stats: pd.DataFrame,
    *,
    quality_column: str = "tracking_quality_ok",
) -> pd.Index:
    """Return user ids that pass one quality column.

    This helper is intentionally narrow and remains useful for quick checks.
    For analysis filters that combine tracking quality and GPS signal flags,
    prefer `build_user_selection_table` followed by `select_analysis_users`.

    Args:
        user_stats: User-level table containing `user_id` and the requested
            quality column.
        quality_column: Boolean column used as the selection criterion.

    Returns:
        User ids for which `quality_column` is true.

    Raises:
        KeyError: If the requested quality column is missing.
    """

    if quality_column not in user_stats.columns:
        raise KeyError(f"user_stats must contain {quality_column}")
    return pd.Index(user_stats.loc[user_stats[quality_column].fillna(False), "user_id"])


def filter_table_by_users(
    df: pd.DataFrame,
    user_ids: Iterable,
    *,
    user_id_column: str = "user_id",
) -> pd.DataFrame:
    """Filter any mobility table by selected users while keeping its schema.

    Args:
        df: Table to filter. If `user_id_column` is absent, the table is
            returned unchanged.
        user_ids: Iterable of user identifiers to keep.
        user_id_column: Name of the user identifier column in `df`.

    Returns:
        A copy of `df` restricted to selected users, with the index reset.
    """

    if user_id_column not in df.columns:
        return df.copy()
    return df[df[user_id_column].isin(set(user_ids))].copy().reset_index(drop=True)


def build_user_selection_table(
    user_stats: pd.DataFrame,
    *,
    require_tracking_quality: bool = False,
    exclude_bad_signal_users: bool = True,
    max_low_quality_legs_share: float | None = None,
) -> pd.DataFrame:
    """Build an explicit user-level table for selecting analysis users.

    The function does not remove rows. It adds one boolean column per rule,
    a final `analysis_user_ok` flag and an `analysis_user_reason` column.
    This keeps exclusions inspectable before any table is filtered. The name
    deliberately uses `selection_table` rather than `filter_matrix` because
    the object is meant to be read by humans before being applied.

    Args:
        user_stats: User-level table produced by `prepare_mobility_dataset`.
            Expected columns depend on enabled rules: `tracking_quality_ok`,
            `bad_signal_user` and `low_quality_legs_share`.
        require_tracking_quality: When true, keep only users with
            `tracking_quality_ok == True`. Leave false when tracking quality
            is used as a diagnostic rather than an exclusion rule, for example
            when the current dump does not cover all configured phases.
        exclude_bad_signal_users: When true, exclude users flagged by the
            signal-quality step.
        max_low_quality_legs_share: Optional maximum share of level-1
            low-quality legs tolerated per user.

    Returns:
        A copy of `user_stats` with `*_filter_ok`, `analysis_user_ok` and
        `analysis_user_reason` columns.

    Raises:
        KeyError: If a column required by an enabled rule is missing.
        ValueError: If `max_low_quality_legs_share` is outside `[0, 1]`.
    """

    if "user_id" not in user_stats.columns:
        raise KeyError("user_stats must contain user_id")
    if max_low_quality_legs_share is not None and not 0 <= max_low_quality_legs_share <= 1:
        raise ValueError("max_low_quality_legs_share must be between 0 and 1.")

    out = user_stats.copy()
    rule_columns = []

    if require_tracking_quality:
        if "tracking_quality_ok" not in out.columns:
            raise KeyError("user_stats must contain tracking_quality_ok")
        out["tracking_filter_ok"] = out["tracking_quality_ok"].fillna(False).astype(bool)
    else:
        out["tracking_filter_ok"] = True
    rule_columns.append("tracking_filter_ok")

    if exclude_bad_signal_users:
        if "bad_signal_user" not in out.columns:
            raise KeyError("user_stats must contain bad_signal_user")
        if "signal_quality_computed" in out.columns:
            computed = out["signal_quality_computed"].fillna(False).astype(bool)
            if not computed.all():
                n_missing = int((~computed).sum())
                raise ValueError(
                    "Signal-quality filtering was requested, but GPS signal "
                    f"quality was not computed for {n_missing} user(s). "
                    "Run prepare_mobility_dataset with add_signal_quality_flags=True "
                    "or call build_user_selection_table(..., "
                    "exclude_bad_signal_users=False)."
                )
        if out["bad_signal_user"].isna().any():
            raise ValueError(
                "Signal-quality filtering was requested, but bad_signal_user "
                "contains missing values."
            )
        out["bad_signal_user_filter_ok"] = ~out["bad_signal_user"].astype(int).astype(bool)
    else:
        out["bad_signal_user_filter_ok"] = True
    rule_columns.append("bad_signal_user_filter_ok")

    if max_low_quality_legs_share is not None:
        if "low_quality_legs_share" not in out.columns:
            raise KeyError("user_stats must contain low_quality_legs_share")
        out["low_quality_legs_share_filter_ok"] = (
            out["low_quality_legs_share"].fillna(0).le(max_low_quality_legs_share)
        )
        rule_columns.append("low_quality_legs_share_filter_ok")

    out["analysis_user_ok"] = out[rule_columns].all(axis=1)
    out["analysis_user_reason"] = "ok"

    if require_tracking_quality:
        tracking_reason = out.get("tracking_quality_reason", "insufficient_tracking_quality")
        out.loc[~out["tracking_filter_ok"], "analysis_user_reason"] = tracking_reason
    if exclude_bad_signal_users:
        mask = out["analysis_user_reason"].eq("ok") & ~out["bad_signal_user_filter_ok"]
        out.loc[mask, "analysis_user_reason"] = "bad_signal_user"
    if max_low_quality_legs_share is not None:
        mask = out["analysis_user_reason"].eq("ok") & ~out["low_quality_legs_share_filter_ok"]
        out.loc[mask, "analysis_user_reason"] = "too_many_low_quality_legs"
    return out


def select_analysis_users(
    selection_table: pd.DataFrame,
    *,
    quality_column: str = "analysis_user_ok",
) -> pd.Index:
    """Return user ids selected by a user selection table.

    Args:
        selection_table: Table produced by `build_user_selection_table`.
        quality_column: Boolean column used as the final selection flag.

    Returns:
        User ids for which `quality_column` is true.

    Raises:
        KeyError: If `user_id` or `quality_column` is missing.
    """

    missing = {"user_id", quality_column} - set(selection_table.columns)
    if missing:
        raise KeyError(f"selection_table missing required column(s): {sorted(missing)}")
    return pd.Index(selection_table.loc[selection_table[quality_column].fillna(False), "user_id"])


def _values(df: pd.DataFrame, column: str) -> set:
    if column not in df.columns:
        return set()
    return set(df[column].dropna())


def _filter_track_trip_journey_map(
    mapping: pd.DataFrame,
    *,
    legs: pd.DataFrame,
    trips: pd.DataFrame,
    journeys: pd.DataFrame,
) -> pd.DataFrame:
    out = mapping.copy()
    valid_leg_ids = _values(legs, "leg_id")
    valid_trip_ids = _values(trips, "trip_id")
    valid_journey_ids = _values(journeys, "journey_id")

    if "leg_id" in out.columns and valid_leg_ids:
        out = out[out["leg_id"].isin(valid_leg_ids)]
    if "trip_id" in out.columns and valid_trip_ids:
        out = out[out["trip_id"].isna() | out["trip_id"].isin(valid_trip_ids)]
    if "journey_id" in out.columns and valid_journey_ids:
        out = out[out["journey_id"].isna() | out["journey_id"].isin(valid_journey_ids)]
    return out.reset_index(drop=True)


def _filter_legs_staypoints_map(
    mapping: pd.DataFrame,
    *,
    legs: pd.DataFrame,
    staypoints: pd.DataFrame,
) -> pd.DataFrame:
    out = mapping.copy()
    valid_leg_ids = _values(legs, "leg_id")
    valid_activity_ids = _values(staypoints, "activity_id")

    if "activity_id" in out.columns and valid_activity_ids:
        out = out[out["activity_id"].isin(valid_activity_ids)]
    for column in ("previous_leg_id", "next_leg_id"):
        if column in out.columns and valid_leg_ids:
            out = out[out[column].isna() | out[column].isin(valid_leg_ids)]
    return out.reset_index(drop=True)


def filter_mobility_dataset_by_users(
    dataset: MobilityDataset,
    user_ids: Iterable,
) -> MobilityDataset:
    """Filter every user-indexed table of a `MobilityDataset`.

    Mapping tables are filtered after the core tables so they only reference
    retained legs, trips, journeys and staypoints. The function does not
    recompute indicators; it only preserves relational consistency after a
    user selection.

    Args:
        dataset: Transformed mobility dataset.
        user_ids: Iterable of user identifiers to keep.

    Returns:
        A new `MobilityDataset` with filtered tables and unchanged validation
        reports.
    """

    selected = set(user_ids)
    storyline = filter_table_by_users(dataset.storyline, selected)
    staypoints = filter_table_by_users(dataset.staypoints, selected)
    legs = filter_table_by_users(dataset.legs, selected)
    trips = filter_table_by_users(dataset.trips, selected)
    journeys = filter_table_by_users(dataset.journeys, selected)
    user_stats = filter_table_by_users(dataset.user_stats, selected)
    user_day_coverage = filter_table_by_users(dataset.user_day_coverage, selected)
    map_track_trip_journey = _filter_track_trip_journey_map(
        dataset.map_track_trip_journey,
        legs=legs,
        trips=trips,
        journeys=journeys,
    )
    map_legs_staypoints = _filter_legs_staypoints_map(
        dataset.map_legs_staypoints,
        legs=legs,
        staypoints=staypoints,
    )
    source_user_ids = _values(dataset.user_stats, "user_id") or _values(dataset.legs, "user_id")
    filtered_user_ids = _values(user_stats, "user_id") or _values(legs, "user_id")
    metadata = dict(dataset.metadata or {})
    metadata["filtered_dataset"] = True
    metadata.setdefault("source_user_count", len(source_user_ids) if source_user_ids else len(selected))
    metadata["filtered_user_count"] = len(filtered_user_ids) if filtered_user_ids else len(selected)
    return MobilityDataset(
        storyline=storyline,
        staypoints=staypoints,
        legs=legs,
        trips=trips,
        journeys=journeys,
        user_stats=user_stats,
        map_track_trip_journey=map_track_trip_journey,
        map_legs_staypoints=map_legs_staypoints,
        user_day_coverage=user_day_coverage,
        validation=dataset.validation,
        metadata=metadata,
    )
