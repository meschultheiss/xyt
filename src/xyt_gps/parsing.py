"""Parsing helpers for structured GPS exports."""

from __future__ import annotations

import binascii
from collections.abc import Iterable

import pandas as pd
from shapely import wkb, wkt
from shapely.geometry.base import BaseGeometry

from .config import Phase


STORYLINE_DATE_COLUMNS = (
    "started_at",
    "finished_at",
    "started_on",
    "created_at",
    "updated_at",
    "started_at_in_timezone",
    "finished_at_in_timezone",
    "confirmed_at",
    "confirmed_at_in_timezone",
    "created_at_in_timezone",
    "updated_at_in_timezone",
)

TRIP_DATE_COLUMNS = ("started_at", "finished_at", "created_at")
JOURNEY_DATE_COLUMNS = ("started_at", "finished_at", "created_at")



def drop_nans_if_low_rate(
    df: pd.DataFrame,
    column: str,
    *,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Drop nulls only when the null rate is explicitly below the threshold."""

    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")
    nan_rate = float(df[column].isna().mean())
    if nan_rate > threshold:
        raise ValueError(
            f"Too many missing values in {column}: {nan_rate:.2%}, threshold={threshold:.2%}."
        )
    return df.dropna(subset=[column]).copy()


def parse_ewkb(value):
    """Parse an EWKB, WKB or WKT value into a shapely geometry.

    GPS exports usually provide geometry as EWKB hex strings. Parquet roundtrips
    can preserve the same geometry as raw WKB bytes. CSV roundtrips may expose
    WKT strings. All three forms are accepted.
    """

    if isinstance(value, BaseGeometry):
        return value
    if pd.isna(value):
        raise ValueError("EWKB value is missing. Validate or drop missing geometries first.")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return wkb.loads(bytes(value))
    text = str(value).strip()
    if text.upper().startswith(("POINT", "LINESTRING", "POLYGON", "MULTI", "GEOMETRYCOLLECTION")):
        return wkt.loads(text)
    try:
        return wkb.loads(binascii.unhexlify(text))
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid EWKB geometry value: {value!r}") from exc


def parse_date_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    utc: bool = True,
) -> pd.DataFrame:
    """Parse existing date columns and leave absent columns untouched."""

    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce", utc=utc, format="mixed")
    return out


def assign_phase(value, phases: Iterable[Phase], *, default: str = "Other") -> str:
    """Assign a date or timestamp to the configured experimental phase."""

    if pd.isna(value):
        return default
    date_value = pd.to_datetime(value).date()
    for phase in phases:
        if phase.start_date <= date_value <= phase.end_date:
            return phase.name
    return default


def assign_week_sequence(value, start, *, before_start_value=pd.NA):
    """Assign a timestamp to a 1-based experiment-relative week number.

    Week 1 starts on `start`. For example, if `start` is 2026-02-03, dates
    from 2026-02-03 to 2026-02-09 are week 1 and 2026-02-13 is week 2.
    """

    if start is None or pd.isna(value):
        return pd.NA
    value_date = pd.to_datetime(value, errors="coerce")
    start_date = pd.to_datetime(start, errors="coerce")
    if pd.isna(value_date) or pd.isna(start_date):
        return pd.NA
    if getattr(value_date, "tzinfo", None) is not None:
        value_date = value_date.tz_localize(None)
    if getattr(start_date, "tzinfo", None) is not None:
        start_date = start_date.tz_localize(None)
    delta_days = (value_date.normalize() - start_date.normalize()).days
    if delta_days < 0:
        return before_start_value
    return int(delta_days // 7 + 1)
