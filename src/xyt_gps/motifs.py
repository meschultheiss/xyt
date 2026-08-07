"""Daily mobility motif helpers.

The functions in this module migrate the useful core of the historical
`GPStoGraph` workflow without exposing notebook-era graph objects as the main
API. A motif is represented as a daily directed transition structure between
visited places. Places can come from an existing `location_id` column, or be
derived from staypoint purpose and rounded coordinates.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd

from .models import MobilityDataset


def _as_staypoints(staypoints_or_dataset: MobilityDataset | pd.DataFrame) -> pd.DataFrame:
    if isinstance(staypoints_or_dataset, MobilityDataset):
        return pd.DataFrame(staypoints_or_dataset.staypoints).copy()
    return pd.DataFrame(staypoints_or_dataset).copy()


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Table must contain column(s): {missing}")


def _safe_text(value: Any, default: str = "unknown") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _derive_location_key(
    row: pd.Series,
    *,
    location_col: str | None,
    purpose_col: str | None,
    lon_col: str,
    lat_col: str,
    coordinate_precision: int,
) -> str:
    if location_col is not None and location_col in row.index and pd.notna(row[location_col]):
        return _safe_text(row[location_col])

    parts = []
    if purpose_col is not None and purpose_col in row.index:
        parts.append(_safe_text(row[purpose_col]))

    if lon_col in row.index and lat_col in row.index and pd.notna(row[lon_col]) and pd.notna(row[lat_col]):
        lon = round(float(row[lon_col]), coordinate_precision)
        lat = round(float(row[lat_col]), coordinate_precision)
        parts.append(f"{lon:.{coordinate_precision}f},{lat:.{coordinate_precision}f}")

    if not parts:
        raise KeyError(
            "Cannot derive a location key. Provide `location_col`, or keep "
            "`purpose_col` plus `lon_col`/`lat_col` in the staypoints table."
        )
    return "|".join(parts)


def _collapse_consecutive_duplicates(values: list[Hashable]) -> list[Hashable]:
    collapsed = []
    for value in values:
        if not collapsed or collapsed[-1] != value:
            collapsed.append(value)
    return collapsed


def _canonical_sequence(values: list[Hashable]) -> tuple[int, ...]:
    mapping = {}
    sequence = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        sequence.append(mapping[value])
    return tuple(sequence)


def _adjacency_flat(sequence: tuple[int, ...]) -> tuple[int, ...]:
    if not sequence:
        return tuple()
    n_locations = max(sequence) + 1
    matrix = [[0 for _ in range(n_locations)] for _ in range(n_locations)]
    for origin, destination in zip(sequence[:-1], sequence[1:]):
        matrix[origin][destination] = 1
    return tuple(value for row in matrix for value in row)


def _motif_signature(adjacency_flat: tuple[int, ...], n_locations: int) -> str:
    values = ",".join(str(value) for value in adjacency_flat)
    return f"n={n_locations};adj={values}"


def assign_mobility_motif_ids(
    motifs: pd.DataFrame,
    *,
    signature_col: str = "motif_signature",
    id_col: str = "motif_id",
    top_n: int = 9,
    other_motif_id: int = 99,
) -> pd.DataFrame:
    """Assign stable numeric ids to the most frequent motif signatures.

    Args:
        motifs: Motif table returned by `build_mobility_motifs()`.
        signature_col: Column containing the canonical motif signature.
        id_col: Name of the generated id column.
        top_n: Number of frequent motifs receiving ids from 1 to `top_n`.
        other_motif_id: Id used for less frequent motifs.

    Returns:
        Copy of `motifs` with a numeric `id_col`.

    Raises:
        KeyError: If `signature_col` is absent.
    """

    if signature_col not in motifs.columns:
        raise KeyError(f"motifs must contain {signature_col}.")
    out = motifs.copy()
    out[id_col] = other_motif_id
    top_signatures = out[signature_col].value_counts(dropna=False).index[:top_n]
    for index, signature in enumerate(top_signatures, start=1):
        out.loc[out[signature_col].eq(signature), id_col] = index
    return out


def build_mobility_motifs(
    staypoints_or_dataset: MobilityDataset | pd.DataFrame,
    *,
    user_col: str = "user_id",
    started_at_col: str = "started_at",
    finished_at_col: str = "finished_at",
    date_col: str | None = None,
    location_col: str | None = None,
    purpose_col: str | None = "purpose_niv1",
    lon_col: str = "lon",
    lat_col: str = "lat",
    coordinate_precision: int = 4,
    top_n_motifs: int = 9,
    other_motif_id: int = 99,
) -> pd.DataFrame:
    """Build daily mobility motifs from staypoints.

    The function works on `dataset.staypoints` or on a staypoint DataFrame.
    A motif is the daily sequence of places visited by one user after removing
    consecutive duplicate places. The sequence is relabelled in order of first
    appearance, then encoded as a flattened directed adjacency matrix. This
    keeps the old `motif_flat` idea while making the result easy to export and
    compare.

    Args:
        staypoints_or_dataset: `MobilityDataset` or staypoint table.
        user_col: User identifier column.
        started_at_col: Start timestamp column used for sorting.
        finished_at_col: Optional end timestamp column kept in motif nodes.
        date_col: Optional explicit date column. If omitted, the date is
            derived from `started_at_col`.
        location_col: Optional stable place identifier. If omitted, a place key
            is derived from purpose and rounded coordinates.
        purpose_col: Optional activity/purpose label used in derived place keys.
        lon_col: Longitude column used when deriving place keys.
        lat_col: Latitude column used when deriving place keys.
        coordinate_precision: Decimal precision for coordinate-derived keys.
        top_n_motifs: Number of frequent motif signatures assigned ids 1..N.
        other_motif_id: Id assigned to less frequent motifs.

    Returns:
        One row per user-day with the canonical sequence, adjacency signature,
        motif id and simple counts.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If no usable staypoint row remains.
    """

    staypoints = _as_staypoints(staypoints_or_dataset)
    _require_columns(staypoints, (user_col, started_at_col))
    if finished_at_col not in staypoints.columns:
        staypoints[finished_at_col] = pd.NaT

    nodes = staypoints.copy()
    nodes[started_at_col] = pd.to_datetime(nodes[started_at_col], errors="coerce", utc=True)
    nodes[finished_at_col] = pd.to_datetime(nodes[finished_at_col], errors="coerce", utc=True)
    nodes = nodes.dropna(subset=[user_col, started_at_col]).copy()
    if nodes.empty:
        raise ValueError("No usable staypoint row found for mobility motifs.")

    if date_col is not None and date_col in nodes.columns:
        nodes["date"] = pd.to_datetime(nodes[date_col], errors="coerce").dt.date
    else:
        nodes["date"] = nodes[started_at_col].dt.date

    nodes["location_key"] = nodes.apply(
        _derive_location_key,
        axis=1,
        location_col=location_col,
        purpose_col=purpose_col,
        lon_col=lon_col,
        lat_col=lat_col,
        coordinate_precision=coordinate_precision,
    )
    nodes = nodes.sort_values([user_col, "date", started_at_col]).reset_index(drop=True)
    nodes["user_id_day"] = (
        nodes[user_col].astype(str)
        + "_"
        + pd.to_datetime(nodes["date"]).dt.strftime("%Y%m%d")
    )

    rows = []
    for (user_id, date), group in nodes.groupby([user_col, "date"], sort=True):
        sequence = _collapse_consecutive_duplicates(group["location_key"].tolist())
        canonical_sequence = _canonical_sequence(sequence)
        edges = tuple(zip(canonical_sequence[:-1], canonical_sequence[1:]))
        adjacency = _adjacency_flat(canonical_sequence)
        n_locations = len(set(canonical_sequence))
        rows.append(
            {
                "user_id_day": f"{user_id}_{pd.to_datetime(date).strftime('%Y%m%d')}",
                "user_id": user_id,
                "date": date,
                "n_visits": len(group),
                "n_locations": n_locations,
                "n_transitions": max(0, len(canonical_sequence) - 1),
                "n_unique_transitions": len(set(edges)),
                "is_round_trip": bool(len(canonical_sequence) > 1 and canonical_sequence[0] == canonical_sequence[-1]),
                "start_location_key": sequence[0] if sequence else pd.NA,
                "end_location_key": sequence[-1] if sequence else pd.NA,
                "location_sequence": tuple(sequence),
                "canonical_sequence": canonical_sequence,
                "canonical_edges": edges,
                "adjacency_flat": adjacency,
                "motif_signature": _motif_signature(adjacency, n_locations),
            }
        )

    motifs = pd.DataFrame(rows)
    if motifs.empty:
        raise ValueError("No mobility motif could be built.")
    return assign_mobility_motif_ids(
        motifs,
        top_n=top_n_motifs,
        other_motif_id=other_motif_id,
    )


def summarize_mobility_motifs(
    motifs: pd.DataFrame,
    *,
    motif_id_col: str = "motif_id",
    signature_col: str = "motif_signature",
) -> pd.DataFrame:
    """Summarize motif frequencies and simple structural properties."""

    _require_columns(motifs, (motif_id_col, signature_col))
    total = len(motifs)
    if total == 0:
        return pd.DataFrame(
            columns=[
                motif_id_col,
                signature_col,
                "days_count",
                "days_share",
                "mean_locations",
                "mean_transitions",
                "round_trip_share",
            ]
        )
    summary = (
        motifs.groupby([motif_id_col, signature_col], dropna=False)
        .agg(
            days_count=("user_id_day", "count"),
            mean_locations=("n_locations", "mean"),
            mean_transitions=("n_transitions", "mean"),
            round_trip_share=("is_round_trip", "mean"),
        )
        .reset_index()
        .sort_values(["days_count", motif_id_col], ascending=[False, True])
    )
    summary["days_share"] = summary["days_count"] / total
    return summary[
        [
            motif_id_col,
            signature_col,
            "days_count",
            "days_share",
            "mean_locations",
            "mean_transitions",
            "round_trip_share",
        ]
    ]


def build_mobility_motif_sequences(
    motifs: pd.DataFrame,
    *,
    user_col: str = "user_id",
    date_col: str = "date",
    motif_id_col: str = "motif_id",
    n_days: int = 60,
    align_to_week: bool = True,
    fill_value: int = 0,
) -> pd.DataFrame:
    """Build a fixed-width daily motif sequence for each user.

    This is the package equivalent of the historical `motif_sequence()`
    helper. Each row is a user and each column is a relative day. Missing days
    are filled with `fill_value`. When `align_to_week=True`, the first observed
    motif is shifted so the first column corresponds to Monday.
    """

    _require_columns(motifs, (user_col, date_col, motif_id_col))
    if n_days <= 0:
        raise ValueError("n_days must be positive.")

    data = motifs[[user_col, date_col, motif_id_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[motif_id_col] = pd.to_numeric(data[motif_id_col], errors="coerce").fillna(fill_value).astype(int)
    data = data.dropna(subset=[user_col, date_col])

    columns = [f"day_{index:03d}" for index in range(1, n_days + 1)]
    rows = []
    for user_id, group in data.groupby(user_col, sort=True):
        series = (
            group.sort_values(date_col)
            .set_index(date_col)[motif_id_col]
            .resample("1D")
            .max()
            .fillna(fill_value)
            .astype(int)
        )
        pad_before = int(series.index[0].dayofweek) if align_to_week and len(series) else 0
        values = ([fill_value] * pad_before + series.tolist())[:n_days]
        if len(values) < n_days:
            values.extend([fill_value] * (n_days - len(values)))
        rows.append({"user_id": user_id, **dict(zip(columns, values))})

    return pd.DataFrame(rows, columns=["user_id", *columns])
