"""Temporal correspondence tables for cross-experiment analysis."""

from __future__ import annotations

from collections.abc import Mapping
import warnings

import pandas as pd


FRENCH_WEEKDAYS = {
    0: "lundi",
    1: "mardi",
    2: "mercredi",
    3: "jeudi",
    4: "vendredi",
    5: "samedi",
    6: "dimanche",
}


def _local_datetime(
    table: pd.DataFrame,
    timestamp_col: str,
    timezone_col: str | None,
    *,
    default_timezone: str = "Europe/Zurich",
) -> pd.Series:
    """Return naive local datetimes, using row timezones when available."""

    utc_values = pd.to_datetime(table[timestamp_col], errors="coerce", utc=True)
    local_values = pd.Series(pd.NaT, index=table.index, dtype="datetime64[ns]")

    if timezone_col and timezone_col in table.columns:
        timezones = table[timezone_col].fillna(default_timezone).astype(str)
    else:
        timezones = pd.Series(default_timezone, index=table.index, dtype="string")

    for timezone, index in timezones.groupby(timezones).groups.items():
        try:
            local_values.loc[index] = utc_values.loc[index].dt.tz_convert(timezone).dt.tz_localize(None)
        except Exception:
            local_values.loc[index] = utc_values.loc[index].dt.tz_convert(default_timezone).dt.tz_localize(None)

    return local_values


def _relative_started_at(local_started: pd.Series, sequence_week: pd.Series, relative_anchor: str) -> pd.Series:
    """Build comparable datetimes from week sequence, weekday and local time."""

    anchor = pd.Timestamp(relative_anchor)
    week = pd.to_numeric(sequence_week, errors="coerce").astype("Float64")
    weekday = local_started.dt.weekday.astype("Float64")
    day_offset = (week - 1) * 7 + weekday
    relative_date = anchor + pd.to_timedelta(day_offset.astype("float"), unit="D")
    return relative_date + (local_started - local_started.dt.normalize())


def _relative_finished_at(
    local_started: pd.Series,
    local_finished: pd.Series,
    relative_started: pd.Series,
) -> pd.Series:
    """Build comparable finish datetimes while preserving overnight events."""

    day_delta = (local_finished.dt.normalize() - local_started.dt.normalize()).dt.days
    relative_date = relative_started.dt.normalize() + pd.to_timedelta(day_delta, unit="D")
    return relative_date + (local_finished - local_finished.dt.normalize())


def _format_time(values: pd.Series) -> pd.Series:
    return values.dt.strftime("%H:%M:%S").astype("string")


def _sequence_key(sequence_week: pd.Series, weekday_number: pd.Series) -> pd.Series:
    week = pd.to_numeric(sequence_week, errors="coerce")
    weekday = pd.to_numeric(weekday_number, errors="coerce")
    valid = week.notna() & weekday.notna()
    out = pd.Series(pd.NA, index=sequence_week.index, dtype="string")
    out.loc[valid] = [
        f"S{int(week_value):02d}-J{int(weekday_value) + 1}"
        for week_value, weekday_value in zip(week.loc[valid], weekday.loc[valid])
    ]
    return out


def _fill_columns_from_mapping(
    table: pd.DataFrame,
    mapping: pd.DataFrame | None,
    *,
    key: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if mapping is None or mapping.empty or key not in mapping.columns:
        return table

    available_columns = [column for column in columns if column in mapping.columns]
    if not available_columns:
        return table

    mapping_table = mapping[[key, *available_columns]].copy()
    mapping_table[key] = mapping_table[key].astype("string")
    mapping_table = mapping_table.drop_duplicates(subset=[key])

    out = table.copy()
    out[key] = out[key].astype("string")
    merged = out.merge(mapping_table, on=key, how="left", suffixes=("", "__mapped"))
    for column in available_columns:
        mapped_column = f"{column}__mapped"
        if mapped_column not in merged.columns:
            continue
        if column in out.columns:
            merged[column] = merged[column].combine_first(merged[mapped_column])
        else:
            merged[column] = merged[mapped_column]
        merged = merged.drop(columns=[mapped_column])
    return merged


def build_relative_time_index(
    tables: Mapping[str, pd.DataFrame],
    *,
    default_timezone: str = "Europe/Zurich",
    relative_anchor: str = "2000-01-03",
) -> pd.DataFrame:
    """Build a storyline-level time correspondence table.

    The output has one row per `storyline` row. It keeps real local timestamps
    (`absolute_started_at`, `absolute_finished_at`) and adds relative timestamps
    (`relative_started_at`, `relative_finished_at`) that align shifted waves on a
    common protocol calendar. Available correspondence identifiers are attached
    from `map_track_trip_journey` and `map_legs_staypoints`.

    Args:
        tables: Dictionary containing at least `storyline`. When available,
            `map_track_trip_journey` provides `trip_id` and `journey_id`, and
            `map_legs_staypoints` provides adjacent leg IDs around staypoints.
        default_timezone: Fallback timezone used only when no row-level timezone
            column exists. When `started_at_timezone` or `finished_at_timezone`
            exists, absolute local timestamps use the original row timezone.
        relative_anchor: Date used as week 1 day 1 in the synthetic comparison
            calendar. With the current convention, sequence weeks start on
            Monday, so this should be a Monday date such as `2000-01-03`.
            Using another weekday shifts the relative dates and emits a warning.

    Returns:
        DataFrame with one row per storyline event and explicit absolute and
        relative temporal columns.
    """

    relative_anchor_timestamp = pd.Timestamp(relative_anchor)
    if relative_anchor_timestamp.weekday() != 0:
        warnings.warn(
            "`relative_anchor` should be a Monday with the current sequence-week "
            "convention. Otherwise relative dates keep local times but shift "
            "weekdays in the synthetic calendar.",
            UserWarning,
            stacklevel=2,
        )

    storyline = tables.get("storyline")
    if storyline is None or storyline.empty or "started_at" not in storyline.columns:
        return pd.DataFrame(
            columns=[
                "storyline_id",
                "leg_id",
                "activity_id",
                "previous_leg_id",
                "next_leg_id",
                "trip_id",
                "journey_id",
                "experiment_name",
                "user_id",
                "type",
                "phase",
                "phase_number",
                "sequence_week_number",
                "weekday_number",
                "weekday_label",
                "sequence_day_key",
                "absolute_started_at",
                "absolute_finished_at",
                "absolute_started_date",
                "absolute_finished_date",
                "started_time",
                "started_hour",
                "started_minute_of_day",
                "started_timezone",
                "finished_time",
                "finished_hour",
                "finished_minute_of_day",
                "finished_timezone",
                "relative_started_at",
                "relative_finished_at",
            ]
        )

    work = storyline.copy()
    started_timezone_col = "started_at_timezone" if "started_at_timezone" in work.columns else None
    finished_timezone_col = "finished_at_timezone" if "finished_at_timezone" in work.columns else started_timezone_col
    local_started = _local_datetime(
        work,
        "started_at",
        started_timezone_col,
        default_timezone=default_timezone,
    )
    if "finished_at" in work.columns:
        local_finished = _local_datetime(
            work,
            "finished_at",
            finished_timezone_col,
            default_timezone=default_timezone,
        )
    else:
        local_finished = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")

    sequence_week = (
        pd.to_numeric(work["n_week_sequence"], errors="coerce").astype("Int64")
        if "n_week_sequence" in work.columns
        else pd.Series(pd.NA, index=work.index, dtype="Int64")
    )
    weekday_number = local_started.dt.weekday.astype("Int64")
    relative_started = _relative_started_at(local_started, sequence_week, relative_anchor)
    relative_finished = _relative_finished_at(local_started, local_finished, relative_started)

    storyline_id = (
        work["storyline_id"].astype("string")
        if "storyline_id" in work.columns
        else work.index.astype("string")
    )
    row_type = work["type"].astype("string") if "type" in work.columns else pd.Series(pd.NA, index=work.index, dtype="string")
    leg_id = work["leg_id"].astype("string") if "leg_id" in work.columns else pd.Series(pd.NA, index=work.index, dtype="string")
    activity_id = (
        work["activity_id"].astype("string") if "activity_id" in work.columns else pd.Series(pd.NA, index=work.index, dtype="string")
    )
    leg_id = leg_id.mask(leg_id.isna() & row_type.eq("Track"), storyline_id)
    activity_id = activity_id.mask(activity_id.isna() & row_type.eq("Stay"), storyline_id)

    out = pd.DataFrame(
        {
            "storyline_id": storyline_id,
            "leg_id": leg_id,
            "activity_id": activity_id,
            "experiment_name": work["experiment_name"].astype("string") if "experiment_name" in work.columns else pd.NA,
            "user_id": work["user_id"].astype("string") if "user_id" in work.columns else pd.NA,
            "type": row_type,
            "phase": work["phase"].astype("string") if "phase" in work.columns else pd.NA,
            "phase_number": work["phase_number"] if "phase_number" in work.columns else pd.NA,
            "sequence_week_number": sequence_week,
            "weekday_number": weekday_number,
            "weekday_label": weekday_number.map(FRENCH_WEEKDAYS).astype("string"),
            "sequence_day_key": _sequence_key(sequence_week, weekday_number),
            "absolute_started_at": local_started,
            "absolute_finished_at": local_finished,
            "absolute_started_date": local_started.dt.date,
            "absolute_finished_date": local_finished.dt.date,
            "started_time": _format_time(local_started),
            "started_hour": local_started.dt.hour.astype("Int64"),
            "started_minute_of_day": (local_started.dt.hour * 60 + local_started.dt.minute).astype("Int64"),
            "started_timezone": work[started_timezone_col].astype("string") if started_timezone_col else default_timezone,
            "finished_time": _format_time(local_finished),
            "finished_hour": local_finished.dt.hour.astype("Int64"),
            "finished_minute_of_day": (local_finished.dt.hour * 60 + local_finished.dt.minute).astype("Int64"),
            "finished_timezone": work[finished_timezone_col].astype("string") if finished_timezone_col else default_timezone,
            "relative_started_at": relative_started,
            "relative_finished_at": relative_finished,
        }
    )

    for column in ("mode_niv1", "mode_niv2", "mode_mrmt", "purpose_niv1"):
        if column in work.columns:
            out[column] = work[column]
    for column in ("trip_id", "journey_id", "previous_leg_id", "next_leg_id"):
        if column in work.columns:
            out[column] = work[column].astype("string")

    out = _fill_columns_from_mapping(
        out,
        tables.get("map_track_trip_journey"),
        key="leg_id",
        columns=("trip_id", "journey_id"),
    )
    out = _fill_columns_from_mapping(
        out,
        tables.get("map_legs_staypoints"),
        key="activity_id",
        columns=("previous_leg_id", "next_leg_id"),
    )

    preferred_columns = [
        "storyline_id",
        "leg_id",
        "activity_id",
        "previous_leg_id",
        "next_leg_id",
        "trip_id",
        "journey_id",
    ]
    for column in preferred_columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[[*preferred_columns, *[column for column in out.columns if column not in preferred_columns]]]
