"""Prepare raw GPS tables before mobility object construction."""

from __future__ import annotations

from dataclasses import replace

import geopandas as gpd
import pandas as pd

from .config import ProjectConfig
from .columns import reject_legacy_phase_column
from .mappings import MobilityMappings, map_sequence, map_value
from .parsing import (
    JOURNEY_DATE_COLUMNS,
    STORYLINE_DATE_COLUMNS,
    TRIP_DATE_COLUMNS,
    assign_phase,
    assign_week_sequence,
    drop_nans_if_low_rate,
    parse_date_columns,
    parse_ewkb,
)
from .schema import GPS_SCHEMAS, validate_schema


def _assign_phase_column(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    reject_legacy_phase_column(df, table_name="GPS table")
    out = df.copy()
    if config.phases and "started_at" in out.columns:
        out["phase"] = out["started_at"].apply(lambda value: assign_phase(value, config.phases))
    return out


def _phase_number(name: object) -> int | pd.NA:
    digits = "".join(character for character in str(name) if character.isdigit())
    if not digits:
        return pd.NA
    return int(digits)


def _assign_analysis_phase_column(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    out = df.copy()
    if not config.analysis_phases or "started_at" not in out.columns:
        return out

    out["analysis_phase"] = out["started_at"].apply(
        lambda value: assign_phase(value, config.analysis_phases, default=pd.NA)
    )
    start_lookup = {phase.name: str(phase.start_date) for phase in config.analysis_phases}
    end_lookup = {phase.name: str(phase.end_date) for phase in config.analysis_phases}
    out["analysis_phase_number"] = out["analysis_phase"].map(_phase_number).astype("Int64")
    out["analysis_phase_start"] = out["analysis_phase"].map(start_lookup).astype("string")
    out["analysis_phase_end"] = out["analysis_phase"].map(end_lookup).astype("string")
    return out


def _assign_week_sequence_column(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    out = df.copy()
    start = config.experiment_start_date
    if start is None or "started_at" not in out.columns:
        return out
    out["n_week_sequence"] = out["started_at"].apply(
        lambda value: assign_week_sequence(value, start)
    ).astype("Int64")
    return out


def _move_column_to_front(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    return df[[column, *[name for name in df.columns if name != column]]]


def _drop_provider_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove residual provider `id` columns after explicit IDs are present."""

    if "id" not in df.columns:
        return df
    return df.drop(columns=["id"])


def apply_storyline_mappings(
    storyline: pd.DataFrame,
    mappings: MobilityMappings | None = None,
) -> pd.DataFrame:
    """Add purpose and mode aggregation columns to storyline rows."""

    mappings = mappings or MobilityMappings()
    out = storyline.copy()
    if "type" in out.columns and "purpose" in out.columns:
        mask = out["type"].eq("Stay")
        out.loc[mask, "purpose_niv1"] = out.loc[mask, "purpose"].apply(
            lambda value: map_value(
                value,
                mappings.storyline_purpose_niv1,
                default=mappings.default_purpose,
            )
        )
    if "type" in out.columns and "mode" in out.columns:
        mask = out["type"].eq("Track")
        out.loc[mask, "mode_niv1"] = out.loc[mask, "mode"].apply(
            lambda value: map_value(value, mappings.storyline_mode_niv1, default=mappings.default_mode)
        )
        out.loc[mask, "mode_niv2"] = out.loc[mask, "mode"].apply(
            lambda value: map_value(value, mappings.storyline_mode_niv2, default=mappings.default_mode)
        )
        out.loc[mask, "mode_mrmt"] = out.loc[mask, "mode"].apply(
            lambda value: map_value(value, mappings.storyline_mode_mrmt, default=mappings.default_mode)
        )
    return out


def apply_trip_journey_mappings(
    trips: pd.DataFrame,
    journeys: pd.DataFrame,
    mappings: MobilityMappings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add purpose and mode aggregation columns to trips and journeys."""

    mappings = mappings or MobilityMappings()
    trips_out = trips.copy()
    journeys_out = journeys.copy()
    if "purpose" in trips_out.columns:
        trips_out["purpose_mrmt"] = trips_out["purpose"].apply(
            lambda value: map_value(value, mappings.trip_purpose_mrmt, default=mappings.default_purpose)
        )
    if "main_mode" in trips_out.columns:
        trips_out["main_mode_niv1"] = trips_out["main_mode"].apply(
            lambda value: map_value(value, mappings.trip_mode_niv1, default=mappings.default_mode)
        )
        trips_out["main_mode_niv2"] = trips_out["main_mode"].apply(
            lambda value: map_value(value, mappings.trip_mode_niv2, default=mappings.default_mode)
        )
        trips_out["main_mode_mrmt"] = trips_out["main_mode"].apply(
            lambda value: map_value(value, mappings.trip_mode_mrmt, default=mappings.default_mode)
        )
    if "modes" in trips_out.columns:
        trips_out["modes_niv1"] = trips_out["modes"].apply(
            lambda value: map_sequence(value, mappings.trip_mode_niv1, default=mappings.default_mode)
        )
        trips_out["modes_niv2"] = trips_out["modes"].apply(
            lambda value: map_sequence(value, mappings.trip_mode_niv2, default=mappings.default_mode)
        )
        trips_out["modes_mrmt"] = trips_out["modes"].apply(
            lambda value: map_sequence(value, mappings.trip_mode_mrmt, default=mappings.default_mode)
        )

    if "purpose" in journeys_out.columns and "main_purpose" not in journeys_out.columns:
        journeys_out = journeys_out.rename(columns={"purpose": "main_purpose"})
    if "main_purpose" in journeys_out.columns:
        journeys_out["main_purpose_mrmt"] = journeys_out["main_purpose"].apply(
            lambda value: map_value(value, mappings.trip_purpose_mrmt, default=mappings.default_purpose)
        )
    if "main_mode" in journeys_out.columns:
        journeys_out["main_mode_niv1"] = journeys_out["main_mode"].apply(
            lambda value: map_value(value, mappings.trip_mode_niv1, default=mappings.default_mode)
        )
        journeys_out["main_mode_niv2"] = journeys_out["main_mode"].apply(
            lambda value: map_value(value, mappings.trip_mode_niv2, default=mappings.default_mode)
        )
        journeys_out["main_mode_mrmt"] = journeys_out["main_mode"].apply(
            lambda value: map_value(value, mappings.trip_mode_mrmt, default=mappings.default_mode)
        )
    if "modes" in journeys_out.columns:
        journeys_out["modes_niv1"] = journeys_out["modes"].apply(
            lambda value: map_sequence(value, mappings.trip_mode_niv1, default=mappings.default_mode)
        )
        journeys_out["modes_niv2"] = journeys_out["modes"].apply(
            lambda value: map_sequence(value, mappings.trip_mode_niv2, default=mappings.default_mode)
        )
        journeys_out["modes_mrmt"] = journeys_out["modes"].apply(
            lambda value: map_sequence(value, mappings.trip_mode_mrmt, default=mappings.default_mode)
        )
    return trips_out, journeys_out


def prepare_storyline(
    storyline: pd.DataFrame,
    config: ProjectConfig,
    *,
    drop_nan_threshold: float = 0.01,
) -> gpd.GeoDataFrame:
    """Validate, parse geometry/dates, assign phases and map modes/purposes.

    `drop_nan_threshold` controls both validation and row dropping for
    `geometry`, `started_at` and `finished_at`. Keep the default strict for
    production data. Use a higher explicit threshold only for deliberately
    altered or geofenced demo data where null geometries are expected.
    """

    storyline_schema = GPS_SCHEMAS["storyline"]
    storyline_schema = replace(
        storyline_schema,
        max_null_rate={
            **storyline_schema.max_null_rate,
            "geometry": drop_nan_threshold,
            "started_at": drop_nan_threshold,
            "finished_at": drop_nan_threshold,
        },
    )
    report = validate_schema(storyline, storyline_schema)
    if not report.ok:
        raise ValueError(report.to_frame().to_string(index=False))

    out = _drop_provider_id_column(storyline.copy())
    for column in ("geometry", "started_at", "finished_at"):
        out = drop_nans_if_low_rate(out, column, threshold=drop_nan_threshold)
    out["geometry"] = out["geometry"].apply(parse_ewkb)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=config.target_crs)
    out = parse_date_columns(out, STORYLINE_DATE_COLUMNS)
    out = _assign_phase_column(out, config)
    out = _assign_analysis_phase_column(out, config)
    out = _assign_week_sequence_column(out, config)
    out = apply_storyline_mappings(out, config.mappings)
    return out


def prepare_trips(trips: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Validate and parse raw trips."""

    report = validate_schema(trips, GPS_SCHEMAS["trips"])
    if not report.ok:
        raise ValueError(report.to_frame().to_string(index=False))
    out = _drop_provider_id_column(trips.copy())
    for column in ("duration", "distance", "tracks_count"):
        if column in out.columns:
            out[column] = out[column].astype("Int64")
    out["trip_count"] = 1
    out = parse_date_columns(out, TRIP_DATE_COLUMNS)
    out = _assign_phase_column(out, config)
    out = _assign_analysis_phase_column(out, config)
    out = _assign_week_sequence_column(out, config)
    if "started_at" in out.columns:
        out["trip_date"] = out["started_at"].dt.date
    return _move_column_to_front(out, "trip_id")


def prepare_journeys(journeys: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Validate and parse raw journeys."""

    report = validate_schema(journeys, GPS_SCHEMAS["journeys"])
    if not report.ok:
        raise ValueError(report.to_frame().to_string(index=False))
    out = journeys.copy()
    out = out.rename(columns={"purpose": "main_purpose"})
    out = _drop_provider_id_column(out)
    for column in ("duration", "distance", "tracks_count_without_walks", "trips_count", "tracks_count"):
        if column in out.columns:
            out[column] = out[column].astype("Int64")
    out["journey_count"] = 1
    out = parse_date_columns(out, JOURNEY_DATE_COLUMNS)
    out = _assign_phase_column(out, config)
    out = _assign_analysis_phase_column(out, config)
    out = _assign_week_sequence_column(out, config)
    if "started_at" in out.columns:
        out["journey_date"] = out["started_at"].dt.date
    return _move_column_to_front(out, "journey_id")
