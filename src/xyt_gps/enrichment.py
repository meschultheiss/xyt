"""Optional enrichment helpers for mobility analysis tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from .references import (
    load_activity_speed_ranges,
    load_co2_factors,
    load_metabolic_equivalent_tasks,
    load_occupancy_by_distance_and_purpose,
    load_other_purpose_occupancy,
)


@dataclass(frozen=True)
class CO2OccupancyConfig:
    """Parameters used to derive CO2 and occupancy metrics from legs.

    Factors are expressed in grams per kilometer. `co2_g` and
    `co2_direct_g` are then computed as leg-level totals. Occupancy is
    recomputed by default; set `prefer_observed_occupancy=True` to use a
    positive provider value from `occupancy_col` when available.
    """

    factors_g_per_km: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    occupancy_by_distance_and_purpose: Mapping[str, tuple[float, float, Mapping[str, float]]] = field(
        default_factory=dict
    )
    other_purpose_occupancy: Mapping[str, float] = field(default_factory=dict)
    car_modes: tuple[str, ...] = (
        "Mode::Carsharing",
        "Mode::Ecar",
        "Mode::Car",
        "Mode::Ridepooling",
        "Mode::CarPassenger",
    )
    default_purpose: str = "Tout motifs"
    default_occupancy: float = 1.0
    apply_occupancy_to_car_modes: bool = True
    prefer_observed_occupancy: bool = False
    occupancy_col: str = "occupancy"
    reference_label: str = "project reference files"

    @classmethod
    def from_reference_files(
        cls,
        *,
        co2_factors_path: str | Path,
        occupancy_rates_path: str | Path,
        **overrides: object,
    ) -> "CO2OccupancyConfig":
        """Build a CO2 config from explicit project CSV files."""

        values: dict[str, object] = {
            "factors_g_per_km": load_co2_factors(co2_factors_path),
            "occupancy_by_distance_and_purpose": load_occupancy_by_distance_and_purpose(
                occupancy_rates_path
            ),
            "other_purpose_occupancy": load_other_purpose_occupancy(occupancy_rates_path),
            "reference_label": "project reference files",
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class HealthConfig:
    """Parameters used to derive simple physical-activity metrics."""

    weight_kg: float = 70.0
    step_length_m: float = 0.75
    step_modes: tuple[str, ...] = ("Marche",)
    walk_speed_thresholds_kmh: tuple[float, float] = (3.0, 5.0)
    bike_speed_thresholds_kmh: tuple[float, float] = (12.0, 20.0)
    mets_by_mode_intensity: Mapping[tuple[str, str], float] = field(default_factory=dict)
    speed_ranges_by_mode_intensity: Mapping[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    default_mets: float = 1.0
    reference_label: str = "project reference files"

    @classmethod
    def from_reference_files(
        cls,
        *,
        metabolic_equivalent_tasks_path: str | Path,
        **overrides: object,
    ) -> "HealthConfig":
        """Build a health config from an explicit project MET CSV file."""

        values: dict[str, object] = {
            "mets_by_mode_intensity": load_metabolic_equivalent_tasks(
                metabolic_equivalent_tasks_path
            ),
            "speed_ranges_by_mode_intensity": load_activity_speed_ranges(
                metabolic_equivalent_tasks_path
            ),
            "reference_label": "project reference files",
        }
        values.update(overrides)
        return cls(**values)


def _numeric_series(df: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _distance_meters(df: pd.DataFrame, distance_col: str | None = None) -> pd.Series:
    for column in (distance_col, "length", "distance", "length_leg"):
        if column and column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    raise KeyError("legs must contain a distance column: length, distance or length_leg.")


def _duration_seconds(df: pd.DataFrame, duration_col: str = "duration") -> pd.Series:
    if duration_col in df.columns:
        duration = df[duration_col]
        if pd.api.types.is_timedelta64_dtype(duration):
            return duration.dt.total_seconds().fillna(0.0)
        numeric = pd.to_numeric(duration, errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(0.0)
        return pd.to_timedelta(duration, errors="coerce").dt.total_seconds().fillna(0.0)
    if {"started_at", "finished_at"}.issubset(df.columns):
        return (
            pd.to_datetime(df["finished_at"], errors="coerce", utc=True)
            - pd.to_datetime(df["started_at"], errors="coerce", utc=True)
        ).dt.total_seconds().fillna(0.0)
    raise KeyError("legs must contain duration or started_at/finished_at.")


def _distance_class(length_m: float, config: CO2OccupancyConfig) -> tuple[str | None, Mapping[str, float] | None]:
    for label, (min_dist, max_dist, coeffs) in config.occupancy_by_distance_and_purpose.items():
        if min_dist <= length_m < max_dist:
            return label, coeffs
    return None, None


def _occupancy_rate(length_m: float, purpose: str | None, config: CO2OccupancyConfig) -> tuple[float, str | None]:
    distance_class, coeffs = _distance_class(length_m, config)
    purpose_value = purpose or config.default_purpose
    if coeffs and purpose_value in coeffs:
        return float(coeffs[purpose_value]), distance_class
    if purpose_value in config.other_purpose_occupancy:
        return float(config.other_purpose_occupancy[purpose_value]), distance_class
    return float(config.default_occupancy), distance_class


def _attach_journey_purpose(
    legs: pd.DataFrame,
    journeys: pd.DataFrame | None,
    map_track_trip_journey: pd.DataFrame | None,
    *,
    purpose_col: str,
) -> pd.DataFrame:
    if "main_journey_purpose_mrmt" in legs.columns:
        return legs.copy()
    out = legs.copy()
    if journeys is None or map_track_trip_journey is None:
        out["main_journey_purpose_mrmt"] = pd.NA
        return out
    required_mapping = {"leg_id", "journey_id"}
    if not required_mapping.issubset(map_track_trip_journey.columns) or purpose_col not in journeys.columns:
        out["main_journey_purpose_mrmt"] = pd.NA
        return out
    purpose_by_leg = (
        map_track_trip_journey[["leg_id", "journey_id"]]
        .dropna(subset=["leg_id"])
        .drop_duplicates(subset=["leg_id"])
        .merge(
            journeys[["journey_id", purpose_col]].drop_duplicates(subset=["journey_id"]),
            on="journey_id",
            how="left",
        )
        .rename(columns={purpose_col: "main_journey_purpose_mrmt"})
    )
    return out.merge(purpose_by_leg[["leg_id", "main_journey_purpose_mrmt"]], on="leg_id", how="left")


def add_co2_occupancy_metrics(
    legs: pd.DataFrame,
    *,
    journeys: pd.DataFrame | None = None,
    map_track_trip_journey: pd.DataFrame | None = None,
    config: CO2OccupancyConfig | None = None,
    mode_col: str = "mode",
    distance_col: str | None = None,
    journey_purpose_col: str = "main_purpose_mrmt",
    prefer_observed_occupancy: bool | None = None,
    occupancy_col: str | None = None,
) -> pd.DataFrame:
    """Add occupancy and CO2 metrics to a leg table.

    The function is intentionally row-level: it does not aggregate. This keeps
    assumptions visible before mobility indicators are computed.

    By default, occupancy is recomputed from distance and purpose because
    provider columns are often sparse. Pass `prefer_observed_occupancy=True`
    to use a positive value from `occupancy_col` when available and fall back
    to the computed value otherwise.
    """

    if config is None:
        raise ValueError(
            "add_co2_occupancy_metrics requires an explicit CO2OccupancyConfig. "
            "Build it with CO2OccupancyConfig.from_reference_files(...)."
        )
    co2_config = config
    if not co2_config.factors_g_per_km or not co2_config.occupancy_by_distance_and_purpose:
        raise ValueError(
            "CO2OccupancyConfig is incomplete: provide co2_factors and occupancy_rates reference tables."
        )
    out = _attach_journey_purpose(
        legs,
        journeys,
        map_track_trip_journey,
        purpose_col=journey_purpose_col,
    )
    if mode_col not in out.columns:
        raise KeyError(f"legs must contain {mode_col}.")

    length_m = _distance_meters(out, distance_col)
    purpose = out["main_journey_purpose_mrmt"].fillna(co2_config.default_purpose).astype(str)
    modes = out[mode_col].fillna("Mode::Other").astype(str)

    computed_occupancy_rows = [
        _occupancy_rate(float(length), purpose_value, co2_config)
        if mode in co2_config.car_modes
        else (float(co2_config.default_occupancy), None)
        for length, purpose_value, mode in zip(length_m, purpose, modes, strict=False)
    ]
    computed_occupancy = pd.Series([value for value, _class in computed_occupancy_rows], index=out.index)
    observed_column = occupancy_col or co2_config.occupancy_col
    use_observed = (
        co2_config.prefer_observed_occupancy
        if prefer_observed_occupancy is None
        else bool(prefer_observed_occupancy)
    )
    observed_occupancy = (
        pd.to_numeric(out[observed_column], errors="coerce")
        if observed_column in out.columns
        else pd.Series(pd.NA, index=out.index, dtype="Float64")
    )
    observed_valid = observed_occupancy.notna() & observed_occupancy.gt(0)
    out["occupancy_rate_computed"] = computed_occupancy.astype(float)
    out["occupancy_declared"] = observed_occupancy.astype("Float64")
    out["occupancy_declared_available"] = observed_valid.astype(bool)
    out["occupancy_rate"] = out["occupancy_rate_computed"]
    out["occupancy_source"] = "computed"
    if use_observed:
        out.loc[observed_valid, "occupancy_rate"] = observed_occupancy.loc[observed_valid].astype(float)
        out.loc[observed_valid, "occupancy_source"] = "observed"
        out.loc[~observed_valid, "occupancy_source"] = "computed_fallback"
    out["occupancy_observed_available"] = bool(observed_column in out.columns)
    out["occupancy_distance_class"] = [_class for _value, _class in computed_occupancy_rows]
    out["co2_factor_g_per_km"] = [co2_config.factors_g_per_km.get(mode, (pd.NA, pd.NA))[0] for mode in modes]
    out["co2_direct_factor_g_per_km"] = [co2_config.factors_g_per_km.get(mode, (pd.NA, pd.NA))[1] for mode in modes]

    factor = pd.to_numeric(out["co2_factor_g_per_km"], errors="coerce")
    direct_factor = pd.to_numeric(out["co2_direct_factor_g_per_km"], errors="coerce")
    occupancy = pd.to_numeric(out["occupancy_rate"], errors="coerce").replace(0, pd.NA).fillna(1.0)
    divisor = occupancy.where(
        modes.isin(co2_config.car_modes) & co2_config.apply_occupancy_to_car_modes,
        1.0,
    )
    out["co2_g"] = length_m.div(1000).mul(factor).div(divisor)
    out["co2_direct_g"] = length_m.div(1000).mul(direct_factor).div(divisor)
    out["co2_assumption"] = (
        f"{co2_config.reference_label}; factors_g_per_km; car modes divided by occupancy"
        if co2_config.apply_occupancy_to_car_modes
        else f"{co2_config.reference_label}; factors_g_per_km; occupancy reported but not applied"
    )
    if use_observed:
        out["co2_assumption"] = out["co2_assumption"] + f"; occupancy source prefers `{observed_column}`"
    else:
        out["co2_assumption"] = out["co2_assumption"] + "; occupancy recomputed"
    return out


def _intensity(mode: str, speed_kmh: float, config: HealthConfig) -> str:
    configured_ranges = {
        intensity: speed_range
        for (range_mode, intensity), speed_range in config.speed_ranges_by_mode_intensity.items()
        if range_mode == mode
    }
    if configured_ranges:
        if pd.isna(speed_kmh):
            return "unknown"
        for intensity, (min_speed, max_speed) in configured_ranges.items():
            if min_speed <= speed_kmh < max_speed:
                return intensity
        return "unknown"

    if mode == "Marche":
        low, high = config.walk_speed_thresholds_kmh
    elif mode == "Vélo":
        low, high = config.bike_speed_thresholds_kmh
    else:
        return "nonactive"
    if pd.isna(speed_kmh):
        return "unknown"
    if speed_kmh < low:
        return "calme"
    if speed_kmh < high:
        return "moderee"
    return "intense"


def _intensity_speed_range(
    mode: str,
    intensity: str,
    config: HealthConfig,
) -> tuple[object, object]:
    if (mode, intensity) in config.speed_ranges_by_mode_intensity:
        min_speed, max_speed = config.speed_ranges_by_mode_intensity[(mode, intensity)]
        return min_speed, pd.NA if max_speed == float("inf") else max_speed
    if mode == "Marche":
        low, high = config.walk_speed_thresholds_kmh
    elif mode == "Vélo":
        low, high = config.bike_speed_thresholds_kmh
    else:
        return pd.NA, pd.NA
    if intensity == "calme":
        return 0.0, low
    if intensity == "moderee":
        return low, high
    if intensity == "intense":
        return high, pd.NA
    return pd.NA, pd.NA


def add_health_metrics(
    legs: pd.DataFrame,
    *,
    config: HealthConfig | None = None,
    mode_col: str = "mode_niv1",
    distance_col: str | None = None,
    duration_col: str = "duration",
) -> pd.DataFrame:
    """Add simple activity, intensity, MET and calorie metrics to legs."""

    if config is None:
        raise ValueError(
            "add_health_metrics requires an explicit HealthConfig. "
            "Build it with HealthConfig.from_reference_files(...)."
        )
    health_config = config
    if not health_config.mets_by_mode_intensity:
        raise ValueError(
            "HealthConfig is incomplete: provide a metabolic_equivalent_tasks reference table."
        )
    if mode_col not in legs.columns:
        raise KeyError(f"legs must contain {mode_col}.")
    out = legs.copy()
    length_m = _distance_meters(out, distance_col)
    duration_s = _duration_seconds(out, duration_col)
    duration_h = duration_s.div(3600)
    distance_km = length_m.div(1000)
    out["speed_kmh"] = distance_km.div(duration_h.where(duration_h.gt(0))).round(2)

    modes = out[mode_col].fillna("Autres").astype(str)
    out["step_count"] = (
        length_m.div(health_config.step_length_m)
        .where(modes.isin(health_config.step_modes), 0)
        .fillna(0)
        .round()
        .astype("Int64")
    )
    out["intensity"] = [
        _intensity(mode, speed, health_config)
        for mode, speed in zip(modes, out["speed_kmh"], strict=False)
    ]
    intensity_speed_ranges = [
        _intensity_speed_range(mode, intensity, health_config)
        for mode, intensity in zip(modes, out["intensity"], strict=False)
    ]
    out["intensity_min_speed_kmh"] = pd.Series(
        [min_speed for min_speed, _max_speed in intensity_speed_ranges],
        index=out.index,
        dtype="Float64",
    )
    out["intensity_max_speed_kmh"] = pd.Series(
        [max_speed for _min_speed, max_speed in intensity_speed_ranges],
        index=out.index,
        dtype="Float64",
    )
    out["mets"] = [
        health_config.mets_by_mode_intensity.get((mode, intensity), health_config.default_mets)
        for mode, intensity in zip(modes, out["intensity"], strict=False)
    ]
    out["calories_burned"] = (out["mets"].astype(float) * health_config.weight_kg * duration_h).fillna(0)
    out["health_assumption"] = (
        f"{health_config.reference_label}; weight_kg={health_config.weight_kg}; "
        f"step_length_m={health_config.step_length_m}"
    )
    return out


def build_leg_enrichment_tables(legs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return compact CO2 and health side tables keyed by `leg_id`."""

    tables = {}
    if "leg_id" not in legs.columns:
        return tables
    co2_columns = [
        column
        for column in [
            "leg_id",
            "main_journey_purpose_mrmt",
            "occupancy_rate_computed",
            "occupancy_declared",
            "occupancy_declared_available",
            "occupancy_rate",
            "occupancy_source",
            "occupancy_observed_available",
            "occupancy_distance_class",
            "co2_factor_g_per_km",
            "co2_direct_factor_g_per_km",
            "co2_g",
            "co2_direct_g",
            "co2_assumption",
        ]
        if column in legs.columns
    ]
    if len(co2_columns) > 1:
        tables["occupancy_co2"] = legs[co2_columns].copy()

    health_columns = [
        column
        for column in [
            "leg_id",
            "speed_kmh",
            "step_count",
            "intensity",
            "intensity_min_speed_kmh",
            "intensity_max_speed_kmh",
            "mets",
            "calories_burned",
            "health_assumption",
        ]
        if column in legs.columns
    ]
    if len(health_columns) > 1:
        tables["health"] = legs[health_columns].copy()
    return tables
