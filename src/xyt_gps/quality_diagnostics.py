"""Diagnostic reports for leg lengths, confirmations and mode detection."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def summarize_leg_lengths_by_mode(
    legs: pd.DataFrame,
    *,
    mode_col: str = "mode_niv2",
    length_col: str = "length",
    quantiles: Iterable[float] = (0.95, 0.98, 0.99),
) -> pd.DataFrame:
    """Summarize leg-distance distributions by mode.

    Args:
        legs: Leg table.
        mode_col: Mode column used for grouping.
        length_col: Length column, expected in meters.
        quantiles: Quantiles to add to the summary.

    Returns:
        One row per mode with distances expressed in kilometers.
    """

    missing = {mode_col, length_col} - set(legs.columns)
    if missing:
        raise KeyError(f"legs missing required column(s): {sorted(missing)}")
    data = legs[[mode_col, length_col]].copy()
    data["length_km"] = pd.to_numeric(data[length_col], errors="coerce") / 1000
    grouped = data.groupby(mode_col, dropna=False)["length_km"]
    summary = grouped.agg(
        legs_count="count",
        mean_km="mean",
        median_km="median",
        std_km="std",
        max_km="max",
    ).reset_index()
    for quantile in quantiles:
        suffix = int(float(quantile) * 100)
        summary[f"p{suffix}_km"] = grouped.quantile(float(quantile)).to_numpy()
    return summary


def get_extreme_legs_by_mode(
    legs: pd.DataFrame,
    *,
    mode_col: str = "mode_niv1",
    length_col: str = "length",
    duration_col: str = "duration",
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the longest legs within each mode.

    Args:
        legs: Leg table.
        mode_col: Mode column used for grouping.
        length_col: Length column, expected in meters.
        duration_col: Optional duration column in seconds.
        top_n: Number of longest legs retained per mode.

    Returns:
        A table sorted by mode and descending distance, with distance in
        kilometers and speed when duration is available.
    """

    missing = {mode_col, length_col} - set(legs.columns)
    if missing:
        raise KeyError(f"legs missing required column(s): {sorted(missing)}")
    out = legs.copy()
    out["distance_km"] = pd.to_numeric(out[length_col], errors="coerce") / 1000
    if duration_col in out.columns:
        duration_hours = pd.to_numeric(out[duration_col], errors="coerce") / 3600
        out["speed_kmh"] = out["distance_km"].div(duration_hours.where(duration_hours.gt(0)))
    else:
        out["speed_kmh"] = pd.NA

    parts = []
    for mode, group in out.dropna(subset=["distance_km"]).groupby(mode_col, dropna=False):
        top = group.nlargest(top_n, "distance_km").copy()
        top.insert(0, "rank_in_mode", range(1, len(top) + 1))
        top[mode_col] = mode
        parts.append(top)
    if not parts:
        return pd.DataFrame()
    keep = [
        column
        for column in (
            "rank_in_mode",
            mode_col,
            "leg_id",
            "user_id",
            length_col,
            "distance_km",
            duration_col,
            "speed_kmh",
        )
        if column in pd.concat(parts, ignore_index=True).columns
    ]
    return pd.concat(parts, ignore_index=True)[keep]


def build_user_confirmation_rates(
    storyline: pd.DataFrame,
    *,
    user_id_col: str = "user_id",
    type_col: str = "type",
    confirmed_at_col: str = "confirmed_at",
) -> pd.DataFrame:
    """Calculate user-level confirmation rates for stays and tracks.

    Args:
        storyline: Parsed storyline table.
        user_id_col: User identifier column.
        type_col: Column distinguishing `Stay` and `Track`.
        confirmed_at_col: Confirmation timestamp column.

    Returns:
        One row per user with stay and track confirmation counts and rates.
    """

    missing = {user_id_col, type_col} - set(storyline.columns)
    if missing:
        raise KeyError(f"storyline missing required column(s): {sorted(missing)}")
    data = storyline.copy()
    if confirmed_at_col not in data.columns:
        data[confirmed_at_col] = pd.NA
    data["_is_confirmed"] = data[confirmed_at_col].notna()
    rows = []
    for user_id, group in data.groupby(user_id_col, dropna=False):
        stays = group.loc[group[type_col].eq("Stay")]
        tracks = group.loc[group[type_col].eq("Track")]
        rows.append(
            {
                "user_id": user_id,
                "stays_count": int(len(stays)),
                "confirmed_stays_count": int(stays["_is_confirmed"].sum()),
                "stay_confirmation_rate": float(stays["_is_confirmed"].mean()) if len(stays) else pd.NA,
                "tracks_count": int(len(tracks)),
                "confirmed_tracks_count": int(tracks["_is_confirmed"].sum()),
                "track_confirmation_rate": float(tracks["_is_confirmed"].mean()) if len(tracks) else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def build_mode_detection_precision(
    storyline: pd.DataFrame,
    *,
    mode_col: str = "mode",
    detected_mode_col: str = "detected_mode",
    type_col: str = "type",
    confirmed_at_col: str = "confirmed_at",
    confirmed_only: bool = True,
    group_cols: Iterable[str] = ("mode", "mode_niv1", "mode_mrmt"),
) -> pd.DataFrame:
    """Compare detected and confirmed transport modes.

    Args:
        storyline: Parsed storyline table.
        mode_col: Confirmed or corrected mode column.
        detected_mode_col: Detected mode column.
        type_col: Column used to keep only track rows.
        confirmed_at_col: Confirmation timestamp column.
        confirmed_only: When true, keep only rows with a confirmation
            timestamp when the column exists.
        group_cols: Mode columns to summarize when present.

    Returns:
        A long table with one precision row per available grouping column and
        label.
    """

    missing = {mode_col, detected_mode_col} - set(storyline.columns)
    if missing:
        raise KeyError(f"storyline missing required column(s): {sorted(missing)}")
    tracks = storyline.copy()
    if type_col in tracks.columns:
        tracks = tracks.loc[tracks[type_col].eq("Track")].copy()
    if confirmed_only and confirmed_at_col in tracks.columns:
        tracks = tracks.loc[tracks[confirmed_at_col].notna()].copy()
    tracks = tracks.dropna(subset=[mode_col, detected_mode_col]).copy()
    tracks["detected_mode_matches"] = tracks[mode_col].eq(tracks[detected_mode_col])

    rows = []
    for group_col in group_cols:
        if group_col not in tracks.columns:
            continue
        grouped = tracks.groupby(group_col, dropna=False)["detected_mode_matches"]
        for label, values in grouped:
            rows.append(
                {
                    "grouping": group_col,
                    "label": label,
                    "tracks_count": int(values.count()),
                    "matched_tracks_count": int(values.sum()),
                    "precision": float(values.mean()) if values.count() else pd.NA,
                }
            )
    return pd.DataFrame(rows)
