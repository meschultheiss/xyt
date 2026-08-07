"""Load project reference tables used by enrichment helpers.

The package knows the expected table schemas, but not the project values.
Reference values must be provided through explicit CSV paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


REFERENCE_TABLES: Mapping[str, str] = {
    "co2_factors": "co2_factors.csv",
    "occupancy_rates": "occupancy_rates.csv",
    "metabolic_equivalent_tasks": "metabolic_equivalent_tasks.csv",
}


def available_reference_tables() -> tuple[str, ...]:
    """Return the supported project reference table names."""

    return tuple(sorted(REFERENCE_TABLES))


def load_reference_table(name: str, path: str | Path | None = None) -> pd.DataFrame:
    """Load a reference table as a DataFrame.

    Parameters
    ----------
    name:
        Name returned by :func:`available_reference_tables`.
    path:
        Path to a project-specific CSV using the expected schema.
    """

    if name not in REFERENCE_TABLES:
        valid = ", ".join(available_reference_tables())
        raise ValueError(f"Unknown reference table {name!r}. Expected one of: {valid}.")
    if path is None:
        expected_filename = REFERENCE_TABLES[name]
        raise ValueError(
            f"Reference table {name!r} requires an explicit CSV path, "
            f"for example Notebooks/config/reference/{expected_filename}."
        )
    return pd.read_csv(path)


def load_co2_factors(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load CO2 factors keyed by MotionTag mode."""

    table = load_reference_table("co2_factors", path)
    _require_columns(
        table,
        {"mode", "co2_factor_g_per_km", "co2_direct_factor_g_per_km"},
        "co2_factors",
    )
    factors: dict[str, tuple[float, float]] = {}
    for row in table.itertuples(index=False):
        mode = str(getattr(row, "mode")).strip()
        if not mode:
            continue
        factors[mode] = (
            float(getattr(row, "co2_factor_g_per_km")),
            float(getattr(row, "co2_direct_factor_g_per_km")),
        )
    if not factors:
        raise ValueError("co2_factors reference table is empty.")
    return factors


def load_occupancy_by_distance_and_purpose(
    path: str | Path,
) -> dict[str, tuple[float, float, Mapping[str, float]]]:
    """Load occupancy rates indexed by distance class and journey purpose."""

    distance_rules, _purpose_rules = _load_occupancy_rules(path)
    return distance_rules


def load_other_purpose_occupancy(path: str | Path) -> dict[str, float]:
    """Load project-defined general occupancy rates keyed by journey purpose."""

    _distance_rules, purpose_rules = _load_occupancy_rules(path)
    return purpose_rules


def load_metabolic_equivalent_tasks(path: str | Path) -> dict[tuple[str, str], float]:
    """Load MET values keyed by `(mode_niv1, intensity)`."""

    table = load_reference_table("metabolic_equivalent_tasks", path)
    _require_columns(table, {"mode_niv1", "intensity", "mets"}, "metabolic_equivalent_tasks")
    values: dict[tuple[str, str], float] = {}
    for row in table.itertuples(index=False):
        mode = str(getattr(row, "mode_niv1")).strip()
        intensity = str(getattr(row, "intensity")).strip()
        if not mode or not intensity:
            continue
        values[(mode, intensity)] = float(getattr(row, "mets"))
    if not values:
        raise ValueError("metabolic_equivalent_tasks reference table is empty.")
    return values


def load_activity_speed_ranges(path: str | Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Load speed ranges keyed by `(mode_niv1, intensity)`.

    Bounds are expressed in km/h. Lower bounds are inclusive and upper bounds
    are exclusive. Missing upper bounds mean no upper limit.
    """

    table = load_reference_table("metabolic_equivalent_tasks", path)
    _require_columns(table, {"mode_niv1", "intensity", "mets"}, "metabolic_equivalent_tasks")
    if not {"min_speed_kmh", "max_speed_kmh"}.issubset(table.columns):
        return {}

    ranges: dict[tuple[str, str], tuple[float, float]] = {}
    for row in table.itertuples(index=False):
        mode = str(getattr(row, "mode_niv1")).strip()
        intensity = str(getattr(row, "intensity")).strip()
        min_speed_raw = getattr(row, "min_speed_kmh")
        max_speed_raw = getattr(row, "max_speed_kmh")
        if not mode or not intensity or (_is_blank(min_speed_raw) and _is_blank(max_speed_raw)):
            continue
        ranges[(mode, intensity)] = (
            _float_or_default(min_speed_raw, default=0.0),
            _float_or_default(max_speed_raw, default=float("inf")),
        )
    return ranges


def _load_occupancy_rules(
    path: str | Path,
) -> tuple[dict[str, tuple[float, float, Mapping[str, float]]], dict[str, float]]:
    table = load_reference_table("occupancy_rates", path)
    _require_columns(
        table,
        {"distance_class", "min_distance_m", "max_distance_m", "purpose", "occupancy_rate"},
        "occupancy_rates",
    )
    distance_rules: dict[str, tuple[float, float, dict[str, float]]] = {}
    purpose_rules: dict[str, float] = {}
    for row in table.itertuples(index=False):
        purpose = str(getattr(row, "purpose")).strip()
        rate = float(getattr(row, "occupancy_rate"))
        distance_class = getattr(row, "distance_class")
        min_distance_raw = getattr(row, "min_distance_m")
        max_distance_raw = getattr(row, "max_distance_m")
        has_distance_rule = any(
            not _is_blank(value)
            for value in (distance_class, min_distance_raw, max_distance_raw)
        )
        if has_distance_rule:
            label = str(distance_class).strip()
            min_distance = _float_or_default(min_distance_raw, default=0.0)
            max_distance = _float_or_default(max_distance_raw, default=float("inf"))
            if not label or not purpose:
                raise ValueError("occupancy_rates distance rows need distance_class and purpose.")
            if label not in distance_rules:
                distance_rules[label] = (min_distance, max_distance, {})
            existing_min, existing_max, coeffs = distance_rules[label]
            if existing_min != min_distance or existing_max != max_distance:
                raise ValueError(f"Inconsistent distance bounds for occupancy class {label!r}.")
            coeffs[purpose] = rate
        else:
            if not purpose:
                raise ValueError("occupancy_rates general purpose rows need purpose.")
            purpose_rules[purpose] = rate
    if not distance_rules:
        raise ValueError("occupancy_rates reference table has no distance rows.")
    return distance_rules, purpose_rules


def _is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _float_or_default(value: object, *, default: float) -> float:
    if _is_blank(value):
        return default
    return float(str(value).strip())


def _require_columns(table: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"{table_name} reference table is missing columns: {', '.join(missing)}.")


__all__ = [
    "REFERENCE_TABLES",
    "available_reference_tables",
    "load_activity_speed_ranges",
    "load_co2_factors",
    "load_metabolic_equivalent_tasks",
    "load_occupancy_by_distance_and_purpose",
    "load_other_purpose_occupancy",
    "load_reference_table",
]
