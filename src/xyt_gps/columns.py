"""Shared column-name helpers."""

from __future__ import annotations

import pandas as pd


PHASE_COLUMN = "phase"


def phase_column(df: pd.DataFrame) -> str | None:
    """Return ``phase`` when the standardized phase column is present."""

    return PHASE_COLUMN if PHASE_COLUMN in df.columns else None


def phase_series(
    df: pd.DataFrame,
    *,
    default_phase_name: str = "All",
) -> pd.Series:
    """Return phase values or a default analytical period."""

    column = phase_column(df)
    if column is None:
        return pd.Series(default_phase_name, index=df.index, dtype="object")
    return df[column].fillna(default_phase_name).astype(str)


def reject_legacy_phase_column(df: pd.DataFrame, *, table_name: str = "table") -> None:
    """Fail fast when a table still uses the deprecated ``Phase`` column."""

    if "Phase" in df.columns:
        raise ValueError(
            f"{table_name} contains deprecated column `Phase`; update upstream "
            "preparation to use `phase` before calling xyt_gps."
        )


def ensure_phase_column(
    df: pd.DataFrame,
    *,
    default_phase_name: str = "All",
) -> pd.DataFrame:
    """Return a copy with a standardized ``phase`` column."""

    reject_legacy_phase_column(df)
    out = df.copy()
    if PHASE_COLUMN not in out.columns:
        out[PHASE_COLUMN] = default_phase_name
    else:
        out[PHASE_COLUMN] = out[PHASE_COLUMN].fillna(default_phase_name).astype(str)
    return out
