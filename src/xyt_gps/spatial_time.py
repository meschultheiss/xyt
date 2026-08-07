"""Reusable time-slice helpers for mobility tables."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import TimeSlice, default_time_slices


def _time_to_minutes(value: str) -> int:
    hour, minute = str(value).split(":", maxsplit=1)
    return int(hour) * 60 + int(minute)


def _minute_in_interval(minute: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end


def add_time_slices(
    table: pd.DataFrame,
    *,
    time_slices: Iterable[TimeSlice] | None = None,
    datetime_col: str = "started_at",
    output_col: str = "time_slice",
    timezone: str | None = "Europe/Zurich",
    fallback_label: str = "HC",
) -> pd.DataFrame:
    """Add a reusable daily time-slice label to a mobility table.

    Args:
        table: Input table with a datetime column.
        time_slices: Named intervals. Defaults to `HPM` 07:10-09:00 and
            `HPS` 17:30-20:00. Observations outside these intervals receive
            `fallback_label`, usually `HC`.
        datetime_col: Datetime column used to classify observations.
        output_col: Name of the created column.
        timezone: Local timezone used before extracting the hour. Set to
            `None` to keep UTC timestamps.
        fallback_label: Label assigned outside configured intervals.

    Returns:
        Copy of `table` with `output_col`.

    Raises:
        KeyError: If `datetime_col` is missing.
    """

    if datetime_col not in table.columns:
        raise KeyError(f"Table must contain a `{datetime_col}` column.")

    out = table.copy()
    slices = tuple(time_slices) if time_slices is not None else default_time_slices()
    intervals = [
        (time_slice.name, _time_to_minutes(time_slice.start), _time_to_minutes(time_slice.end))
        for time_slice in slices
    ]
    timestamps = pd.to_datetime(out[datetime_col], errors="coerce", utc=True, format="mixed")
    if timezone is not None:
        timestamps = timestamps.dt.tz_convert(timezone)
    minutes = timestamps.dt.hour.mul(60).add(timestamps.dt.minute)
    labels = pd.Series(fallback_label, index=out.index, dtype="object")
    labels.loc[timestamps.isna()] = pd.NA
    for name, start, end in intervals:
        mask = minutes.apply(lambda value: False if pd.isna(value) else _minute_in_interval(int(value), start, end))
        labels.loc[mask] = name
    out[output_col] = labels
    return out
