"""Temporal resampling helpers for missing tracking days."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .config import ProjectConfig
from .parsing import assign_phase, assign_week_sequence


def resample_missing_stays(
    storyline: gpd.GeoDataFrame,
    config: ProjectConfig,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Create transparent placeholder stays for missing tracking dates.

    The generated rows are labelled `Resampled_stay`. They are useful when
    the analysis requires a continuous user calendar, but they remain
    identifiable through their `type` and `comment_feedback` values.

    Args:
        storyline: Parsed storyline table with `user_id`, `started_at` and
            geometry columns.
        config: Project configuration used to assign experimental phases to
            inserted days.

    Returns:
        A tuple `(storyline, missing_days)` where `storyline` includes the
        optional placeholder rows and `missing_days` lists one row per
        inserted user-day.
    """

    out = storyline.copy()
    missing_rows = []
    for user_id, group in out.groupby("user_id", dropna=False):
        dates = pd.to_datetime(group["started_at"].dt.date).drop_duplicates().sort_values()
        if dates.empty:
            continue
        complete_range = pd.date_range(dates.min(), dates.max(), freq="D")
        existing = set(dates)
        for date in complete_range:
            if date not in existing:
                missing_rows.append({"user_id": user_id, "missing_date": date})

    missing = pd.DataFrame(missing_rows)
    if missing.empty:
        return out, missing

    records = []
    template_columns = list(out.columns)
    for row in missing.itertuples(index=False):
        resampled_id = f"resampled_{row.user_id}_{row.missing_date.strftime('%Y%m%d')}"
        record = {column: pd.NA for column in template_columns}
        record.update(
            {
                "storyline_id": resampled_id,
                "user_id": row.user_id,
                "type": "Resampled_stay",
                "started_at": pd.Timestamp(row.missing_date.date()).tz_localize("UTC"),
                "finished_at": (
                    pd.Timestamp(row.missing_date.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                ).tz_localize("UTC"),
                "started_on": pd.Timestamp(row.missing_date.date()).tz_localize("UTC"),
                "length": 0,
                "purpose": None,
                "purpose_niv1": None,
                "geometry": None,
                "comment_feedback": "Artificial stay added to keep a continuous user timeline.",
            }
        )
        if config.phases or "phase" in out.columns:
            phase = assign_phase(row.missing_date, config.phases)
            record["phase"] = phase
        if config.experiment_start_date is not None or "n_week_sequence" in out.columns:
            record["n_week_sequence"] = assign_week_sequence(
                row.missing_date,
                config.experiment_start_date,
            )
        if "started_at_in_timezone" in record:
            record["started_at_in_timezone"] = record["started_at"]
        if "finished_at_in_timezone" in record:
            record["finished_at_in_timezone"] = record["finished_at"]
        records.append(record)

    resampled = gpd.GeoDataFrame(records, geometry="geometry", crs=out.crs)
    non_empty_columns = [
        column
        for column in resampled.columns
        if column == "geometry" or resampled[column].notna().any()
    ]
    resampled = resampled[non_empty_columns]
    combined = pd.concat([out, resampled], ignore_index=True, sort=False)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=out.crs)
    combined = combined.drop(columns=["id"], errors="ignore")
    combined = combined.drop_duplicates(subset=["storyline_id", "user_id"], keep="first")
    combined = combined.sort_values(["user_id", "started_at"]).reset_index(drop=True)
    return combined, missing
