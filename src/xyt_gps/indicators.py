"""Mobility indicator helpers computed from structured mobility tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

import pandas as pd

from .columns import phase_column
from .config import ProjectConfig
from .demand import build_daily_demand_profile
from .models import IndicatorResult, MobilityDataset


PERSON_DAY_COLUMNS = [
    "user_id",
    "date",
    "phase",
    "mode",
    "distance_km",
    "travel_time_min",
    "trip_count",
    "leg_count",
]

PERSON_DAY_INDEX_COLUMNS = ["user_id", "date", "phase", "mode"]

ENRICHED_LEG_METRICS = {
    "co2_g": "co2_kg",
    "co2_direct_g": "co2_direct_kg",
    "calories_burned": "calories_kcal",
}

AIRPLANE_MODE_MARKERS = ("airplane", "mode::airplane", "avion")
AIRPLANE_MODE_COLUMNS = (
    "mode",
    "detected_mode",
    "modes",
    "main_mode",
    "mode_mrmt",
    "main_mode_mrmt",
    "mode_niv2",
    "main_mode_niv2",
)

DAY_BASIS_COLUMNS = {
    "days_in_range": "day_in_range",
    "tracked_days": "tracked_day",
    "active_days": "active_day",
}


def _infer_trip_mode_col(mode_col: str) -> str:
    if mode_col.startswith("main_"):
        return mode_col
    return f"main_{mode_col}"


def _date_series(df: pd.DataFrame, preferred_col: str, fallback_col: str = "started_at") -> pd.Series:
    if preferred_col in df.columns:
        return pd.to_datetime(df[preferred_col], errors="coerce").dt.normalize()
    if fallback_col in df.columns:
        return pd.to_datetime(df[fallback_col], errors="coerce").dt.normalize()
    raise KeyError(f"Table must contain {preferred_col} or {fallback_col}.")


def _duration_seconds(df: pd.DataFrame) -> pd.Series:
    if "duration" in df.columns:
        duration = df["duration"]
        if pd.api.types.is_timedelta64_dtype(duration):
            return duration.dt.total_seconds().fillna(0)
        numeric = pd.to_numeric(duration, errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(0)
        timedelta = pd.to_timedelta(duration, errors="coerce").dt.total_seconds()
        if timedelta.notna().any():
            return timedelta.fillna(0)
    missing = {"started_at", "finished_at"} - set(df.columns)
    if missing:
        raise KeyError("Table must contain duration or started_at/finished_at.")
    return (
        pd.to_datetime(df["finished_at"], errors="coerce", utc=True)
        - pd.to_datetime(df["started_at"], errors="coerce", utc=True)
    ).dt.total_seconds().fillna(0)


def _distance_meters(df: pd.DataFrame, distance_col: str | None = None) -> pd.Series:
    candidates = [distance_col, "length", "distance", "length_leg"]
    for column in candidates:
        if column and column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0)
    raise KeyError("Legs must contain a distance column: length, distance or length_leg.")


def _phase_records_from_config(config: ProjectConfig | None) -> list[dict]:
    if config is None:
        return []
    return [
        {
            "name": phase.name,
            "start": pd.to_datetime(phase.start),
            "end": pd.to_datetime(phase.end),
            "source_ids": [],
        }
        for phase in config.phases
    ]


def _phase_records_from_metadata(dataset: MobilityDataset) -> list[dict]:
    metadata = dataset.metadata or {}
    projects = []
    if "sources" in metadata:
        projects = [
            source.get("project", {})
            for source in metadata.get("sources", [])
            if isinstance(source, dict)
        ]
    elif "project" in metadata:
        projects = [metadata.get("project", {})]

    records = []
    for project in projects:
        source_ids = tuple(str(value) for value in project.get("source_ids", []) if value is not None)
        for phase in project.get("phases", []):
            records.append(
                {
                    "name": str(phase["name"]),
                    "start": pd.to_datetime(phase["start"], errors="coerce"),
                    "end": pd.to_datetime(phase["end"], errors="coerce"),
                    "source_ids": source_ids,
                }
            )
    return records


def _phase_records(dataset: MobilityDataset, config: ProjectConfig | None = None) -> list[dict]:
    configured = _phase_records_from_config(config)
    return configured if configured else _phase_records_from_metadata(dataset)


def _phase_values(dataset: MobilityDataset, phase_records: Iterable[dict] = ()) -> list[str]:
    values = [str(record["name"]) for record in phase_records if record.get("name")]
    for table in (dataset.legs, dataset.trips):
        column = phase_column(table)
        if column is not None:
            values.extend(table[column].dropna().astype(str).unique().tolist())
    return sorted(set(values))


def _phase_series(df: pd.DataFrame, *, default_phase_name: str) -> pd.Series:
    column = phase_column(df)
    if column is not None:
        return df[column].fillna(default_phase_name).astype(str)
    return pd.Series(default_phase_name, index=df.index, dtype="object")


def _mode_values(legs: pd.DataFrame, trips: pd.DataFrame, mode_col: str, trips_mode_col: str) -> list[str]:
    values = []
    if mode_col in legs.columns:
        values.extend(legs[mode_col].dropna().astype(str).unique().tolist())
    if trips_mode_col in trips.columns:
        values.extend(trips[trips_mode_col].dropna().astype(str).unique().tolist())
    return sorted(set(values))


def _excursion_mask(values: pd.Series) -> pd.Series:
    """Return true for rows flagged as excursions.

    The package writes excursion flags as 0/1 integers, but notebooks and
    imported data can also contain booleans or text values.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).ne(0)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y", "oui"})


def _without_excursions(table: pd.DataFrame, *, excursion_col: str = "excursion") -> pd.DataFrame:
    if excursion_col not in table.columns:
        return table.copy()
    return table.loc[~_excursion_mask(table[excursion_col])].copy()


def _airplane_mask(table: pd.DataFrame) -> pd.Series:
    if table.empty:
        return pd.Series(False, index=table.index)

    mask = pd.Series(False, index=table.index)
    for column in AIRPLANE_MODE_COLUMNS:
        if column not in table.columns:
            continue
        values = table[column].fillna("").astype(str).str.strip().str.lower()
        column_mask = pd.Series(False, index=table.index)
        for marker in AIRPLANE_MODE_MARKERS:
            column_mask = column_mask | values.str.contains(marker, regex=False)
        mask = mask | column_mask
    return mask


def _without_airplane(table: pd.DataFrame) -> pd.DataFrame:
    return table.loc[~_airplane_mask(table)].copy()


def _scoped_dataset(
    dataset: MobilityDataset,
    *,
    include_excursions: bool,
    include_airplane: bool,
) -> MobilityDataset:
    legs = dataset.legs
    trips = dataset.trips
    journeys = dataset.journeys
    if not include_excursions:
        legs = _without_excursions(legs)
        trips = _without_excursions(trips)
        journeys = _without_excursions(journeys)
    if not include_airplane:
        legs = _without_airplane(legs)
        trips = _without_airplane(trips)
        journeys = _without_airplane(journeys)
    return replace(
        dataset,
        legs=legs,
        trips=trips,
        journeys=journeys,
    )


def _has_excursion_flags(dataset: MobilityDataset) -> bool:
    return any(
        "excursion" in table.columns
        for table in (dataset.storyline, dataset.staypoints, dataset.legs, dataset.trips, dataset.journeys)
        if hasattr(table, "columns")
    )


def _phase_index(phase: str) -> int | None:
    if not str(phase).lower().startswith("phase"):
        return None
    suffix = str(phase)[5:]
    return int(suffix) if suffix.isdigit() else None


def _observed_user_phase_dates(person_day_observed: pd.DataFrame) -> dict[tuple, tuple[pd.Timestamp, pd.Timestamp]]:
    if person_day_observed.empty:
        return {}
    if "phase" not in person_day_observed.columns:
        return {}
    dates = person_day_observed.copy()
    dates["date"] = pd.to_datetime(dates["date"], errors="coerce")
    grouped = dates.dropna(subset=["date"]).groupby(["user_id", "phase"])["date"].agg(["min", "max"])
    return {
        key: (row["min"], row["max"])
        for key, row in grouped.iterrows()
    }


def _user_source_label(user_row: pd.Series) -> str | None:
    for column in ("experiment_name", "user_expe", "source_id"):
        source = user_row.get(column)
        if source is not None and not pd.isna(source):
            return str(source)
    user_id = user_row.get("user_id")
    if user_id is not None and "::" in str(user_id):
        return str(user_id).split("::", 1)[0]
    return None


def _configured_user_phase_range(
    user_row: pd.Series,
    phase: str,
    phase_records: Iterable[dict],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    matches = [record for record in phase_records if str(record.get("name")) == str(phase)]
    if not matches:
        return None

    source_id = _user_source_label(user_row)
    if source_id is not None:
        source_matches = [
            record
            for record in matches
            if source_id in tuple(str(value) for value in record.get("source_ids", ()))
        ]
        if len(source_matches) == 1:
            matches = source_matches

    if len(matches) != 1:
        return None

    start = pd.to_datetime(matches[0].get("start"), errors="coerce")
    end = pd.to_datetime(matches[0].get("end"), errors="coerce")
    if pd.isna(start) or pd.isna(end) or start > end:
        return None
    return start, end


def _user_phase_range(
    user_row: pd.Series,
    phase: str,
    observed_ranges: Mapping[tuple, tuple[pd.Timestamp, pd.Timestamp]],
    phase_records: Iterable[dict] = (),
) -> pd.DatetimeIndex:
    configured = _configured_user_phase_range(user_row, phase, phase_records)
    if configured is not None:
        start, end = configured
        return pd.date_range(start.normalize(), end.normalize(), freq="D")

    phase_number = _phase_index(phase)
    start = end = pd.NaT
    if phase_number is not None:
        start = pd.to_datetime(user_row.get(f"phase{phase_number}_start"), errors="coerce")
        end = pd.to_datetime(user_row.get(f"phase{phase_number}_end"), errors="coerce")

    if pd.isna(start) or pd.isna(end):
        observed = observed_ranges.get((user_row["user_id"], phase))
        if observed is not None:
            start, end = observed

    if pd.isna(start) or pd.isna(end) or start > end:
        return pd.DatetimeIndex([])
    return pd.date_range(start.normalize(), end.normalize(), freq="D")


def _calendar_rows(
    user_stats: pd.DataFrame,
    phases: Iterable[str],
    modes: Iterable[str],
    observed: pd.DataFrame,
    phase_records: Iterable[dict] = (),
) -> pd.DataFrame:
    if user_stats.empty or "user_id" not in user_stats.columns:
        return observed[["user_id", "date", "phase", "mode"]].drop_duplicates()

    phase_list = list(phases)
    mode_list = list(modes)
    if not phase_list or not mode_list:
        return observed[["user_id", "date", "phase", "mode"]].drop_duplicates()

    calendar_parts = []
    observed_ranges = _observed_user_phase_dates(observed)
    for _, user_row in user_stats.iterrows():
        for phase in phase_list:
            dates = _user_phase_range(user_row, phase, observed_ranges, phase_records)
            if dates.empty:
                continue
            calendar_parts.append(
                pd.MultiIndex.from_product(
                    [[user_row["user_id"]], dates.normalize(), [phase], mode_list],
                    names=["user_id", "date", "phase", "mode"],
                ).to_frame(index=False)
            )
    if not calendar_parts:
        return observed[["user_id", "date", "phase", "mode"]].drop_duplicates()
    return pd.concat(calendar_parts, ignore_index=True)


def _coverage_calendar_rows(
    coverage: pd.DataFrame,
    modes: Iterable[str],
    *,
    day_basis: str,
) -> pd.DataFrame:
    """Construire le calendrier indicateur depuis la table personne-jour."""

    if day_basis not in DAY_BASIS_COLUMNS:
        raise ValueError(
            f"day_basis must be one of {sorted(DAY_BASIS_COLUMNS)}, got {day_basis!r}."
        )
    flag_column = DAY_BASIS_COLUMNS[day_basis]
    required = {"user_id", "date", "phase", flag_column}
    missing = required - set(coverage.columns)
    if missing:
        raise KeyError(f"user_day_coverage missing required column(s): {sorted(missing)}")

    selected = coverage.loc[
        pd.to_numeric(coverage[flag_column], errors="coerce").fillna(0).gt(0),
        ["user_id", "date", "phase"],
    ].drop_duplicates()
    mode_list = list(modes)
    if selected.empty or not mode_list:
        return pd.DataFrame(columns=PERSON_DAY_INDEX_COLUMNS)
    selected["_key"] = 1
    mode_table = pd.DataFrame({"mode": mode_list, "_key": 1})
    return selected.merge(mode_table, on="_key", how="inner").drop(columns="_key")


def _prepare_leg_indicators(
    dataset: MobilityDataset,
    *,
    mode_col: str,
    distance_col: str | None,
    default_phase_name: str,
) -> pd.DataFrame:
    legs = dataset.legs.copy()
    required = {"user_id", mode_col}
    missing = required - set(legs.columns)
    if missing:
        raise KeyError(f"legs missing required column(s): {sorted(missing)}")

    travel_time_min = _duration_seconds(legs).div(60)
    out = pd.DataFrame(
        {
            "user_id": legs["user_id"],
            "date": _date_series(legs, "legs_date"),
            "phase": _phase_series(legs, default_phase_name=default_phase_name),
            "mode": legs[mode_col].fillna("Autres").astype(str),
            "distance_km": _distance_meters(legs, distance_col).div(1000),
            "travel_time_min": travel_time_min,
            "leg_count": 1.0,
        }
    )
    for source_col, target_col in ENRICHED_LEG_METRICS.items():
        if source_col in legs.columns:
            values = pd.to_numeric(legs[source_col], errors="coerce").fillna(0)
            if source_col.endswith("_g"):
                values = values.div(1000)
            out[target_col] = values
    if "mets" in legs.columns:
        mets = pd.to_numeric(legs["mets"], errors="coerce")
        out["mets_minutes"] = mets.mul(travel_time_min).fillna(0)

    value_cols = [column for column in out.columns if column not in PERSON_DAY_INDEX_COLUMNS]
    grouped = (
        out.groupby(PERSON_DAY_INDEX_COLUMNS, dropna=False)[value_cols]
        .sum()
        .reset_index()
    )
    if "mets_minutes" in grouped.columns:
        grouped["mets"] = grouped["mets_minutes"].div(
            grouped["travel_time_min"].where(grouped["travel_time_min"].gt(0))
        ).fillna(0)
        grouped = grouped.drop(columns=["mets_minutes"])
    return grouped


def _prepare_trip_indicators(
    dataset: MobilityDataset,
    *,
    trips_mode_col: str,
    default_phase_name: str,
) -> pd.DataFrame:
    trips = dataset.trips.copy()
    required = {"user_id", trips_mode_col}
    missing = required - set(trips.columns)
    if missing:
        raise KeyError(f"trips missing required column(s): {sorted(missing)}")

    trip_count = (
        pd.to_numeric(trips["trip_count"], errors="coerce").fillna(1)
        if "trip_count" in trips.columns
        else pd.Series(1, index=trips.index)
    )
    out = pd.DataFrame(
        {
            "user_id": trips["user_id"],
            "date": _date_series(trips, "trip_date"),
            "phase": _phase_series(trips, default_phase_name=default_phase_name),
            "mode": trips[trips_mode_col].fillna("Autres").astype(str),
            "trip_count": trip_count,
        }
    )
    return (
        out.groupby(["user_id", "date", "phase", "mode"], dropna=False)["trip_count"]
        .sum()
        .reset_index()
    )


def build_person_day_indicators(
    dataset: MobilityDataset,
    *,
    mode_col: str = "mode_niv1",
    trips_mode_col: str | None = None,
    distance_col: str | None = None,
    include_zero_days: bool = True,
    include_excursions: bool = False,
    include_airplane: bool = False,
    day_basis: str = "days_in_range",
    config: ProjectConfig | None = None,
    default_phase_name: str = "All",
) -> pd.DataFrame:
    """Build person-day mobility indicators by mode.

    The MVP indicators are distance, travel time and number of trips. Values
    are aggregated per user, date, analytical phase/period and mode. Distances
    are expressed in kilometers, durations in minutes. If the dataset has no
    `phase` column, all rows are assigned to `default_phase_name`.

    Args:
        dataset: Transformed mobility dataset.
        mode_col: Leg mode column, for example `mode_niv1`, `mode_niv2` or
            `mode_mrmt`.
        trips_mode_col: Trip mode column. If absent, it is inferred as
            `main_{mode_col}`.
        distance_col: Optional leg distance column override. By default the
            function tries `length`, `distance`, then `length_leg`.
        include_zero_days: When true, build a continuous user-day calendar
            from `user_stats` phase ranges and fill days without movement
            with zero.
        include_excursions: When false, exclude rows flagged with
            `excursion == 1` in legs and trips before computing indicators.
        include_airplane: When false, exclude legs and trips whose source or
            mapped mode columns identify airplane travel. Airplane is excluded
            by default because it can dominate distances and CO2 indicators.
        day_basis: Person-day denominator used for the calendar when
            `include_zero_days` is true: `days_in_range`, `tracked_days` or
            `active_days`.
        config: Optional project configuration. If omitted, phase metadata
            stored in `MobilityDataset` is used when available.
        default_phase_name: Analytical period name used when no phase split is
            present in the dataset.

    Returns:
        Long table with columns `user_id`, `date`, `phase`, `mode`,
        `distance_km`, `travel_time_min`, `trip_count` and `leg_count`.

    Raises:
        KeyError: If required columns are missing.
    """

    trips_mode_col = trips_mode_col or _infer_trip_mode_col(mode_col)
    dataset = _scoped_dataset(
        dataset,
        include_excursions=include_excursions,
        include_airplane=include_airplane,
    )
    leg_indicators = _prepare_leg_indicators(
        dataset,
        mode_col=mode_col,
        distance_col=distance_col,
        default_phase_name=default_phase_name,
    )
    trip_indicators = _prepare_trip_indicators(
        dataset,
        trips_mode_col=trips_mode_col,
        default_phase_name=default_phase_name,
    )
    observed = pd.merge(
        leg_indicators,
        trip_indicators,
        on=["user_id", "date", "phase", "mode"],
        how="outer",
    )
    for column in ("distance_km", "travel_time_min", "trip_count", "leg_count"):
        observed[column] = observed[column].fillna(0)
    optional_metric_cols = [
        column
        for column in observed.columns
        if column
        not in PERSON_DAY_INDEX_COLUMNS + ["distance_km", "travel_time_min", "trip_count", "leg_count"]
    ]
    for column in optional_metric_cols:
        observed[column] = observed[column].fillna(0)

    if include_zero_days:
        phase_records = _phase_records(dataset, config)
        phases = _phase_values(dataset, phase_records)
        if not phases:
            phases = [default_phase_name]
        modes = _mode_values(dataset.legs, dataset.trips, mode_col, trips_mode_col)
        if isinstance(dataset.user_day_coverage, pd.DataFrame) and not dataset.user_day_coverage.empty:
            calendar = _coverage_calendar_rows(
                dataset.user_day_coverage,
                modes,
                day_basis=day_basis,
            )
        else:
            calendar = _calendar_rows(dataset.user_stats, phases, modes, observed, phase_records)
        out = pd.merge(calendar, observed, on=["user_id", "date", "phase", "mode"], how="left")
        for column in ("distance_km", "travel_time_min", "trip_count", "leg_count", *optional_metric_cols):
            out[column] = out[column].fillna(0)
    else:
        out = observed

    ordered_columns = PERSON_DAY_COLUMNS + [
        column for column in optional_metric_cols if column not in PERSON_DAY_COLUMNS
    ]
    out = out[ordered_columns].sort_values(["user_id", "phase", "date", "mode"]).reset_index(drop=True)
    return out


def build_person_phase_indicators(
    person_day: pd.DataFrame,
    *,
    user_stats: pd.DataFrame | None = None,
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Average person-day indicators by user, phase and mode.

    Args:
        person_day: Table produced by `build_person_day_indicators`.
        user_stats: Optional user-level table used to attach weights.
        weight_col: User weight column. Missing weights are set to 1.

    Returns:
        Long table with mean daily distance, travel time and trip count by
        user, phase and mode. If `user_stats` is provided, the table also
        contains `weight_col`.
    """

    required = set(PERSON_DAY_COLUMNS)
    missing = required - set(person_day.columns)
    if missing:
        raise KeyError(f"person_day missing required column(s): {sorted(missing)}")
    value_columns = [
        column
        for column in person_day.columns
        if column not in {"user_id", "date", "phase", "mode"}
        and pd.api.types.is_numeric_dtype(person_day[column])
    ]
    out = (
        person_day.groupby(["user_id", "phase", "mode"], dropna=False)[value_columns]
        .mean()
        .reset_index()
    )
    if user_stats is None or "user_id" not in user_stats.columns or not weight_col:
        return out

    weight_table = user_stats[["user_id"]].drop_duplicates("user_id").copy()
    if weight_col in user_stats.columns:
        weight_table = user_stats[["user_id", weight_col]].drop_duplicates("user_id").copy()
    else:
        weight_table[weight_col] = 1.0
    out = out.merge(weight_table, on="user_id", how="left")
    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce").fillna(1.0)
    return out


def build_population_indicators(
    person_phase: pd.DataFrame,
    *,
    use_weights: bool = True,
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Average person-phase indicators at population level.

    Args:
        person_phase: Table produced by `build_person_phase_indicators`.
        use_weights: When true and `weight_col` is available, compute
            weighted means.
        weight_col: Weight column attached to `person_phase`.

    Returns:
        Long table with one row per phase and mode. `n_users` counts the
        number of users contributing to each phase-mode mean.
    """

    required = {"user_id", "phase", "mode", "distance_km", "travel_time_min", "trip_count"}
    missing = required - set(person_phase.columns)
    if missing:
        raise KeyError(f"person_phase missing required column(s): {sorted(missing)}")
    value_columns = [
        column
        for column in person_phase.columns
        if column not in {"user_id", "phase", "mode", weight_col}
        and pd.api.types.is_numeric_dtype(person_phase[column])
    ]
    rows = []
    for (phase, mode), group in person_phase.groupby(["phase", "mode"], dropna=False):
        weights_available = bool(weight_col and weight_col in group.columns)
        weights = (
            pd.to_numeric(group[weight_col], errors="coerce").fillna(1.0)
            if weights_available
            else pd.Series(1.0, index=group.index)
        )
        row = {
            "phase": phase,
            "mode": mode,
            "n_users": group["user_id"].nunique(),
        }
        if weights_available:
            row["weight_sum"] = float(weights.sum())
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.notna()
            if not valid.any():
                row[f"{column}_per_day"] = pd.NA
                continue
            valid_values = values.loc[valid]
            valid_weights = weights.loc[valid]
            if use_weights and weights_available and valid_weights.sum() > 0:
                row[f"{column}_per_day"] = float(valid_values.mul(valid_weights).sum() / valid_weights.sum())
            else:
                row[f"{column}_per_day"] = float(valid_values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _dataset_table_count(dataset: MobilityDataset, table_name: str) -> int:
    table = getattr(dataset, table_name)
    return int(len(table)) if hasattr(table, "__len__") else 0


def _occupancy_vehicle_mask(legs: pd.DataFrame) -> pd.Series:
    if legs.empty:
        return pd.Series(False, index=legs.index)
    if "mode_niv1" in legs.columns:
        return legs["mode_niv1"].fillna("").astype(str).eq("Voiture")
    if "mode" in legs.columns:
        modes = legs["mode"].fillna("").astype(str)
        return modes.str.contains("Car|Ecar|Taxi|Ridepooling|Carsharing", case=False, regex=True)
    return pd.Series(True, index=legs.index)


def _mean_occupancy_rate_by_source(legs: pd.DataFrame) -> dict[str, float]:
    required = {"occupancy_rate", "occupancy_source"}
    if legs.empty or not required.issubset(legs.columns):
        return {}
    vehicle_legs = legs.loc[_occupancy_vehicle_mask(legs)].copy()
    if vehicle_legs.empty:
        return {}
    vehicle_legs["occupancy_rate"] = pd.to_numeric(vehicle_legs["occupancy_rate"], errors="coerce")
    vehicle_legs = vehicle_legs.dropna(subset=["occupancy_rate", "occupancy_source"])
    if vehicle_legs.empty:
        return {}
    return {
        str(source): float(value)
        for source, value in vehicle_legs.groupby("occupancy_source", dropna=False)["occupancy_rate"].mean().items()
    }


def _nullable_bool(series: pd.Series) -> pd.Series:
    """Convert a flag column to booleans without pandas silent-downcast warnings."""

    return series.astype("boolean").fillna(False).astype(bool)


def _indicator_metadata(
    source_dataset: MobilityDataset,
    indicator_dataset: MobilityDataset,
    person_day: pd.DataFrame,
    population: pd.DataFrame,
    *,
    mode_col: str,
    trips_mode_col: str,
    include_zero_days: bool,
    include_excursions: bool,
    include_airplane: bool,
    day_basis: str,
    use_weights: bool,
    weight_col: str,
    default_phase_name: str,
) -> dict:
    user_day = person_day.copy()
    numeric_cols = [
        column
        for column in ("distance_km", "travel_time_min", "trip_count", "leg_count")
        if column in user_day.columns
    ]
    if numeric_cols:
        user_day = (
            user_day.groupby(["user_id", "date", "phase"], dropna=False)[numeric_cols]
            .sum()
            .reset_index()
        )
    else:
        user_day = user_day[["user_id", "date", "phase"]].drop_duplicates()

    user_ids = set()
    if "user_id" in indicator_dataset.user_stats.columns:
        user_ids.update(indicator_dataset.user_stats["user_id"].dropna().astype(str).tolist())
    if "user_id" in person_day.columns:
        user_ids.update(person_day["user_id"].dropna().astype(str).tolist())

    phases = (
        population["phase"].dropna().astype(str).drop_duplicates().tolist()
        if "phase" in population.columns and not population.empty
        else [default_phase_name]
    )
    metadata = {
        "mode_col": mode_col,
        "trips_mode_col": trips_mode_col,
        "include_zero_days": bool(include_zero_days),
        "day_basis": day_basis,
        "include_excursions": bool(include_excursions),
        "include_airplane": bool(include_airplane),
        "excursions_available": _has_excursion_flags(source_dataset),
        "use_weights": bool(use_weights),
        "weight_col": weight_col,
        "filtered_dataset": bool((source_dataset.metadata or {}).get("filtered_dataset", False)),
        "n_users": int(len(user_ids)),
        "source_user_count": int((source_dataset.metadata or {}).get("source_user_count", len(user_ids))),
        "n_person_days": int(len(user_day)),
        "n_legs": _dataset_table_count(indicator_dataset, "legs"),
        "n_trips": _dataset_table_count(indicator_dataset, "trips"),
        "n_journeys": _dataset_table_count(indicator_dataset, "journeys"),
        "phases": phases,
    }
    if "leg_count" in user_day.columns and not user_day.empty:
        metadata["mean_leg_count_per_user_day"] = float(user_day["leg_count"].mean())
    if "trip_count" in user_day.columns and not user_day.empty:
        metadata["mean_trip_count_per_user_day"] = float(user_day["trip_count"].mean())
    if {"distance_km", "trip_count", "leg_count"}.issubset(user_day.columns):
        zero_days = user_day.loc[
            user_day["distance_km"].le(0)
            & user_day["trip_count"].le(0)
            & user_day["leg_count"].le(0)
        ]
        metadata["zero_mobility_days_count"] = int(len(zero_days))
    if "occupancy_source" in indicator_dataset.legs.columns:
        metadata["occupancy_sources"] = sorted(
            indicator_dataset.legs["occupancy_source"].dropna().astype(str).unique().tolist()
        )
    mean_occupancy = _mean_occupancy_rate_by_source(indicator_dataset.legs)
    if mean_occupancy:
        metadata["mean_occupancy_rate_by_source"] = mean_occupancy
    if "occupancy_observed_available" in indicator_dataset.legs.columns:
        available = _nullable_bool(indicator_dataset.legs["occupancy_observed_available"])
        metadata["occupancy_observed_available"] = bool(available.any())
    if "occupancy" in indicator_dataset.legs.columns:
        observed = pd.to_numeric(indicator_dataset.legs["occupancy"], errors="coerce")
        metadata["occupancy_observed_non_null_share"] = float(observed.notna().mean()) if len(observed) else 0.0
    if "occupancy_declared" in indicator_dataset.legs.columns:
        vehicle_legs = indicator_dataset.legs.loc[_occupancy_vehicle_mask(indicator_dataset.legs)]
        declared = pd.to_numeric(vehicle_legs["occupancy_declared"], errors="coerce")
        metadata["car_legs_with_declared_occupancy_share"] = (
            float(declared.gt(0).fillna(False).mean()) if len(declared) else 0.0
        )
    if "filtered_user_count" in (source_dataset.metadata or {}):
        metadata["filtered_user_count"] = source_dataset.metadata["filtered_user_count"]
    if "bad_user" in source_dataset.user_stats.columns:
        bad_user = _nullable_bool(source_dataset.user_stats["bad_user"])
        metadata["bad_user_count"] = int(bad_user.sum())
    if "tracking_quality_ok" in source_dataset.user_stats.columns:
        tracking_quality_ok = _nullable_bool(source_dataset.user_stats["tracking_quality_ok"])
        metadata["tracking_quality_ok_count"] = int(tracking_quality_ok.sum())
    if "bad_signal_user" in source_dataset.user_stats.columns:
        bad_signal_user = _nullable_bool(source_dataset.user_stats["bad_signal_user"])
        metadata["bad_signal_user_count"] = int(bad_signal_user.sum())
    if not include_airplane:
        airplane_counts = {
            "legs": int(_airplane_mask(source_dataset.legs).sum()),
            "trips": int(_airplane_mask(source_dataset.trips).sum()),
            "journeys": int(_airplane_mask(source_dataset.journeys).sum()),
        }
        if any(airplane_counts.values()):
            metadata["airplane_rows_excluded"] = airplane_counts
    demand_profile = build_daily_demand_profile(
        indicator_dataset.legs,
        person_day,
        mode_col=mode_col,
        default_phase_name=default_phase_name,
    )
    if demand_profile:
        metadata["demand_profile_5min"] = demand_profile
    return metadata


def compute_mobility_indicators(
    dataset: MobilityDataset,
    *,
    mode_col: str = "mode_niv1",
    trips_mode_col: str | None = None,
    distance_col: str | None = None,
    include_zero_days: bool = True,
    include_excursions: bool = False,
    include_airplane: bool = False,
    day_basis: str = "days_in_range",
    use_weights: bool = True,
    weight_col: str = "weight",
    config: ProjectConfig | None = None,
    default_phase_name: str = "All",
) -> IndicatorResult:
    """Compute the first mobility indicator tables from a mobility dataset.

    Args:
        dataset: Transformed mobility dataset.
        mode_col: Leg mode column used as indicator granularity.
        trips_mode_col: Optional trip mode column override.
        distance_col: Optional leg distance column override.
        include_zero_days: Whether to materialize days from `day_basis`
            without movement as zero rows.
        include_excursions: Whether to include rows flagged as excursions in
            the indicator base.
        include_airplane: Whether to include airplane legs and trips in the
            indicator base. Defaults to false because airplane rows can
            dominate distance, CO2 and demand profiles.
        day_basis: Person-day denominator used when `include_zero_days` is
            true: `days_in_range`, `tracked_days` or `active_days`.
        use_weights: Whether to use `weight_col` from `dataset.user_stats`
            for population-level means.
        weight_col: User-level weight column. Missing weights are set to 1.
        config: Optional project configuration. If omitted, phase metadata
            stored in `MobilityDataset` is used when available.
        default_phase_name: Analytical period name used when no phase split is
            present in the dataset.

    Returns:
        `IndicatorResult` with `person_day`, `person_phase` and
        `population` tables.
    """

    trips_mode_col = trips_mode_col or _infer_trip_mode_col(mode_col)
    indicator_dataset = _scoped_dataset(
        dataset,
        include_excursions=include_excursions,
        include_airplane=include_airplane,
    )
    person_day = build_person_day_indicators(
        indicator_dataset,
        mode_col=mode_col,
        trips_mode_col=trips_mode_col,
        distance_col=distance_col,
        include_zero_days=include_zero_days,
        include_excursions=True,
        include_airplane=True,
        day_basis=day_basis,
        config=config,
        default_phase_name=default_phase_name,
    )
    person_phase = build_person_phase_indicators(
        person_day,
        user_stats=indicator_dataset.user_stats,
        weight_col=weight_col,
    )
    population = build_population_indicators(
        person_phase,
        use_weights=use_weights,
        weight_col=weight_col,
    )
    metadata = _indicator_metadata(
        dataset,
        indicator_dataset,
        person_day,
        population,
        mode_col=mode_col,
        trips_mode_col=trips_mode_col,
        include_zero_days=include_zero_days,
        include_excursions=include_excursions,
        include_airplane=include_airplane,
        day_basis=day_basis,
        use_weights=use_weights,
        weight_col=weight_col,
        default_phase_name=default_phase_name,
    )
    return IndicatorResult(
        person_day=person_day,
        person_phase=person_phase,
        population=population,
        mode_col=mode_col,
        metadata=metadata,
    )


def population_indicator_summary(indicators: IndicatorResult | pd.DataFrame) -> pd.DataFrame:
    """Return a compact population-level indicator summary.

    Args:
        indicators: `IndicatorResult` or a population table.

    Returns:
        Population indicator table sorted by phase and mode.
    """

    population = indicators.population if isinstance(indicators, IndicatorResult) else indicators
    return population.sort_values(["phase", "mode"]).reset_index(drop=True)
