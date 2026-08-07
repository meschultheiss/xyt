"""Orchestration from structured GPS exports to mobility tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd

from .config import ProjectConfig
from .mobility_tables import add_length_quantile_flags, add_user_id_day, split_storyline
from .models import MobilityDataset, RawGpsData, SchemaValidationResult
from .parsing import assign_phase, drop_nans_if_low_rate, parse_date_columns, parse_ewkb
from .prepare_tables import (
    apply_storyline_mappings,
    apply_trip_journey_mappings,
    prepare_journeys,
    prepare_storyline,
    prepare_trips,
)
from .quality import resample_missing_stays
from .relations import (
    add_excursion_flags_to_trips_journeys,
    add_journey_to_trips,
    add_trip_destination_activity,
    build_legs_staypoints_map,
    build_track_trip_journey_map,
)
from .schema import validate_gps_raw
from .spatial import (
    add_signal_quality_flags as add_signal_quality_flags_table,
    add_signal_quality_to_user_stats,
    clean_leg_geometries as clean_leg_geometries_table,
)
from .user_stats import add_excursion_stats_to_user_stats, build_user_stats
from .quality_tracking import build_user_day_coverage

if TYPE_CHECKING:
    from .io import RawSampleConfig


logger = logging.getLogger(__name__)


def _row_count(table: pd.DataFrame | None) -> int:
    return 0 if table is None else len(table)


def _source_values(table: pd.DataFrame) -> list[str]:
    """Return stable source labels from current source columns."""

    for column in ("experiment_name", "user_expe", "source_id"):
        if column in table.columns:
            return sorted(table[column].dropna().astype(str).unique().tolist())
    return []


def _project_config_metadata(config: ProjectConfig, *, source_ids: list[str] | None = None) -> dict:
    """Return the project metadata needed for provenance and indicators."""

    return {
        "experiment_name": config.experiment_name,
        "motiontag_project_name": config.motiontag_project_name,
        "period": config.period,
        "target_crs": config.target_crs,
        "operations_crs": config.operations_crs,
        "timezone": config.timezone,
        "source_ids": source_ids or [],
        "phases": [
            {
                "name": phase.name,
                "start": str(phase.start_date),
                "end": str(phase.end_date),
            }
            for phase in config.phases
        ],
    }


def _transform_options_metadata(
    *,
    resample_missing_days: bool,
    clean_leg_geometries: bool,
    add_length_outlier_flags: bool,
    add_signal_quality_flags: bool,
) -> dict:
    """Return explicit preparation switches used to produce a dataset."""

    return {
        "resample_missing_days": resample_missing_days,
        "clean_leg_geometries": clean_leg_geometries,
        "add_length_outlier_flags": add_length_outlier_flags,
        "add_signal_quality_flags": add_signal_quality_flags,
    }


def prepare_mobility_dataset(
    raw: RawGpsData,
    config: ProjectConfig,
    *,
    sociodemo: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    weight_col: str = "weight",
    default_weight: float = 1.0,
    resample_missing_days: bool = False,
    clean_leg_geometries: bool = True,
    add_length_outlier_flags: bool = True,
    add_signal_quality_flags: bool = True,
    validation: dict[str, SchemaValidationResult] | None = None,
) -> MobilityDataset:
    """Run the transparent preparation workflow.

    The function orchestrates validation, storyline parsing, trip and journey
    preparation, mappings, split into staypoints and legs, GPS quality flags,
    user tracking stats and relation tables. Each step is also exposed as a
    smaller public function so notebooks can inspect intermediate states.

    Args:
        raw: Raw GPS tables loaded with `load_gps_export` or
            `load_gps_sources`.
        config: Project configuration controlling mappings, phases,
            thresholds and CRS.
        sociodemo: Optional user-level side table with a `user_id` column.
        weights: Optional user-level weighting table with `user_id` and
            `weight_col`.
        weight_col: Name of the weight column to merge/create.
        default_weight: Weight used when no weighting table is provided, or
            when a user has no matching weight.
        resample_missing_days: When true, insert transparent
            `Resampled_stay` rows for missing user-days.
        clean_leg_geometries: When true, convert continuous `MultiLineString`
            legs to `LineString` and drop discontinuous geometries.
        add_length_outlier_flags: When true, add per-mode length quantile flags
            to legs.
        add_signal_quality_flags: When true, compute GPS signal-loss metrics
            on legs and user-level signal-quality flags.
        validation: Optional validation reports. If absent, reports attached
            to `raw` are reused or recomputed.

    Returns:
        A `MobilityDataset` containing parsed storyline, staypoints, legs,
        trips, journeys, user stats and mapping tables.

    Raises:
        ValueError: If raw schema validation contains blocking errors.
    """

    logger.info(
        "prepare_mobility_dataset: start storyline=%s trips=%s journeys=%s user_statistics=%s",
        _row_count(raw.storyline),
        _row_count(raw.trips),
        _row_count(raw.journeys),
        _row_count(raw.user_statistics),
    )
    reports = validation or raw.validation or validate_gps_raw(raw)
    errors = [issue for report in reports.values() for issue in report.issues if issue.severity == "error"]
    if errors:
        error_frame = pd.concat([report.to_frame() for report in reports.values()], ignore_index=True)
        raise ValueError(error_frame.to_string(index=False))
    logger.info("prepare_mobility_dataset: raw schema validated")

    storyline = prepare_storyline(raw.storyline, config)
    logger.info("prepare_mobility_dataset: storyline prepared rows=%s", len(storyline))
    source_ids = _source_values(storyline)
    if resample_missing_days:
        storyline, _missing = resample_missing_stays(storyline, config)
        logger.info("prepare_mobility_dataset: missing-day resampling added_rows=%s", len(_missing))
    trips = prepare_trips(raw.trips, config)
    journeys = prepare_journeys(raw.journeys, config)
    trips, journeys = apply_trip_journey_mappings(trips, journeys, config.mappings)
    logger.info(
        "prepare_mobility_dataset: trips and journeys prepared trips=%s journeys=%s",
        len(trips),
        len(journeys),
    )
    staypoints, legs = split_storyline(storyline)
    logger.info(
        "prepare_mobility_dataset: storyline split staypoints=%s legs=%s",
        len(staypoints),
        len(legs),
    )
    if clean_leg_geometries:
        legs = clean_leg_geometries_table(legs)
        logger.info("prepare_mobility_dataset: leg geometries cleaned legs=%s", len(legs))
    legs = add_user_id_day(legs)
    if add_length_outlier_flags:
        legs = add_length_quantile_flags(
            legs,
            quantiles=config.spatial_quality_thresholds.outlier_quantiles_by_mode,
        )
        logger.info("prepare_mobility_dataset: length outlier flags added")
    if add_signal_quality_flags:
        legs = add_signal_quality_flags_table(legs, config)
        logger.info("prepare_mobility_dataset: signal quality flags added")
    map_track_trip_journey = build_track_trip_journey_map(
        legs,
        trips,
        journeys,
        tolerance=config.matching_thresholds.leg_trip_journey_tolerance,
    )
    map_legs_staypoints = build_legs_staypoints_map(
        legs,
        staypoints,
        tolerance=config.matching_thresholds.leg_trip_journey_tolerance,
    )
    logger.info(
        "prepare_mobility_dataset: relation maps built track_trip_journey=%s legs_staypoints=%s",
        len(map_track_trip_journey),
        len(map_legs_staypoints),
    )
    trips = add_journey_to_trips(trips, journeys, map_track_trip_journey)
    trips = add_trip_destination_activity(trips, map_track_trip_journey, map_legs_staypoints)
    trips, journeys = add_excursion_flags_to_trips_journeys(
        trips,
        journeys,
        legs,
        map_track_trip_journey,
    )
    user_day_coverage = build_user_day_coverage(storyline, config)
    user_stats = build_user_stats(
        storyline,
        config,
        user_statistics=raw.user_statistics,
        sociodemo=sociodemo,
        weights=weights,
        weight_col=weight_col,
        default_weight=default_weight,
        user_day_coverage=user_day_coverage,
    )
    user_stats = add_excursion_stats_to_user_stats(user_stats, legs)
    logger.info("prepare_mobility_dataset: user stats built users=%s", len(user_stats))
    user_stats = add_signal_quality_to_user_stats(
        user_stats,
        legs,
        signal_quality_computed=add_signal_quality_flags,
    )
    if "tracking_quality_ok" in user_stats.columns:
        tracking_bad = ~user_stats["tracking_quality_ok"].astype("boolean").fillna(False).astype(bool)
    else:
        tracking_bad = pd.Series(False, index=user_stats.index)
    if "bad_signal_user" in user_stats.columns:
        bad_signal = user_stats["bad_signal_user"].astype("boolean").fillna(False).astype(bool)
    else:
        bad_signal = pd.Series(False, index=user_stats.index)
    user_stats["bad_user"] = tracking_bad | bad_signal
    metadata = {
        "project": _project_config_metadata(config, source_ids=source_ids),
        "transform_options": _transform_options_metadata(
            resample_missing_days=resample_missing_days,
            clean_leg_geometries=clean_leg_geometries,
            add_length_outlier_flags=add_length_outlier_flags,
            add_signal_quality_flags=add_signal_quality_flags,
        ),
        "computed_steps": {
            "signal_quality": add_signal_quality_flags,
            "length_outlier_flags": add_length_outlier_flags,
            "clean_leg_geometries": clean_leg_geometries,
            "missing_day_resampling": resample_missing_days,
        },
    }
    logger.info("prepare_mobility_dataset: complete")
    return MobilityDataset(
        storyline=storyline,
        staypoints=staypoints,
        legs=legs,
        trips=trips,
        journeys=journeys,
        user_stats=user_stats,
        map_track_trip_journey=map_track_trip_journey,
        map_legs_staypoints=map_legs_staypoints,
        user_day_coverage=user_day_coverage,
        validation=reports,
        metadata=metadata,
    )


def _concat_frames(frames: list, *, geometry: str | None = None, crs=None):
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if geometry and geometry in combined.columns:
        return gpd.GeoDataFrame(combined, geometry=geometry, crs=crs)
    return combined


def concat_mobility_datasets(datasets: Iterable[MobilityDataset]) -> MobilityDataset:
    """Concatenate transformed mobility datasets from several sources.

    Args:
        datasets: Already transformed datasets, usually one per project or
            period.

    Returns:
        One `MobilityDataset` with concatenated tables and namespaced
        validation reports.

    Raises:
        ValueError: If no dataset is provided.
    """

    dataset_list = list(datasets)
    if not dataset_list:
        raise ValueError("At least one MobilityDataset is required.")

    validation: dict[str, SchemaValidationResult] = {}
    for dataset in dataset_list:
        source_values = _source_values(dataset.storyline)
        prefix = source_values[0] if len(source_values) == 1 else f"dataset_{len(validation)}"
        for table, report in dataset.validation.items():
            validation[f"{prefix}.{table}"] = report

    crs = getattr(dataset_list[0].storyline, "crs", None)
    return MobilityDataset(
        storyline=_concat_frames([dataset.storyline for dataset in dataset_list], geometry="geometry", crs=crs),
        staypoints=_concat_frames([dataset.staypoints for dataset in dataset_list], geometry="geometry", crs=crs),
        legs=_concat_frames([dataset.legs for dataset in dataset_list], geometry="geometry", crs=crs),
        trips=_concat_frames([dataset.trips for dataset in dataset_list]),
        journeys=_concat_frames([dataset.journeys for dataset in dataset_list]),
        user_stats=_concat_frames([dataset.user_stats for dataset in dataset_list]),
        map_track_trip_journey=_concat_frames([dataset.map_track_trip_journey for dataset in dataset_list]),
        map_legs_staypoints=_concat_frames([dataset.map_legs_staypoints for dataset in dataset_list]),
        user_day_coverage=_concat_frames([dataset.user_day_coverage for dataset in dataset_list]),
        validation=validation,
        metadata={
            "sources": [dataset.metadata for dataset in dataset_list],
            "computed_steps": {
                "signal_quality": all(
                    dataset.metadata.get("computed_steps", {}).get("signal_quality", False)
                    for dataset in dataset_list
                ),
                "length_outlier_flags": all(
                    dataset.metadata.get("computed_steps", {}).get("length_outlier_flags", False)
                    for dataset in dataset_list
                ),
            },
        },
    )


def _select_sociodemo_for_source(
    sociodemo_by_source,
    *,
    config: ProjectConfig,
    source_id: str,
    namespace_ids: bool,
) -> pd.DataFrame | None:
    if sociodemo_by_source is None:
        return None
    if isinstance(sociodemo_by_source, pd.DataFrame):
        sociodemo = sociodemo_by_source.copy()
    else:
        sociodemo = None
        for key in (source_id, config.experiment_name, config.motiontag_project_name):
            if key in sociodemo_by_source:
                sociodemo = sociodemo_by_source[key]
                break
        if sociodemo is None:
            return None
        sociodemo = sociodemo.copy()

    if "user_id" in sociodemo.columns and namespace_ids:
        if "raw_user_id" not in sociodemo.columns:
            sociodemo["raw_user_id"] = sociodemo["user_id"]
        prefix = f"{source_id}::"
        sociodemo["user_id"] = sociodemo["user_id"].apply(
            lambda value: value if pd.isna(value) or str(value).startswith(prefix) else f"{prefix}{value}"
        )
    return sociodemo


def prepare_mobility_datasets(
    configs: Iterable[ProjectConfig],
    *,
    sociodemo_by_source: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
    weights_by_source: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
    weight_col: str = "weight",
    default_weight: float = 1.0,
    sample: RawSampleConfig | None = None,
    resample_missing_days: bool = False,
    clean_leg_geometries: bool = True,
    add_length_outlier_flags: bool = True,
    add_signal_quality_flags: bool = True,
    validate: bool = True,
    must_exist: bool = True,
    namespace_ids: bool = True,
) -> MobilityDataset:
    """Load, transform and concatenate several GPS sources or periods.

    Args:
        configs: Project configurations, one per project or period.
        sociodemo_by_source: Optional sociodemographic table or mapping from
            source id/project name to a sociodemographic table.
        weights_by_source: Optional weighting table or mapping from source
            id/project name to a user-level weighting table.
        weight_col: Name of the weight column to merge/create.
        default_weight: Weight used when no weighting table is provided, or
            when a user has no matching weight.
        sample: Optional raw sampling strategy applied to each source.
        resample_missing_days: Whether to add transparent missing-day stays.
        clean_leg_geometries: Whether to normalize/drop problematic leg
            geometries before length and quality calculations.
        add_length_outlier_flags: Whether to add per-mode length outlier flags.
        add_signal_quality_flags: Whether to compute GPS signal-loss flags.
        validate: Whether to validate raw exports before transformation.
        must_exist: Whether expected raw CSV files must exist.
        namespace_ids: Whether to prefix ids by source id before
            concatenation.

    Returns:
        A concatenated `MobilityDataset`.
    """

    from .io import load_gps_source, source_id_for_config

    datasets = []
    for config in configs:
        source_id = source_id_for_config(config)
        logger.info("prepare_mobility_datasets: loading source=%s", source_id)
        raw = load_gps_source(
            config,
            source_id=source_id,
            sample=sample,
            validate=validate,
            must_exist=must_exist,
            namespace_ids=namespace_ids,
        )
        sociodemo = _select_sociodemo_for_source(
            sociodemo_by_source,
            config=config,
            source_id=source_id,
            namespace_ids=namespace_ids,
        )
        weights = _select_sociodemo_for_source(
            weights_by_source,
            config=config,
            source_id=source_id,
            namespace_ids=namespace_ids,
        )
        datasets.append(
            prepare_mobility_dataset(
                raw,
                config,
                sociodemo=sociodemo,
                weights=weights,
                weight_col=weight_col,
                default_weight=default_weight,
                resample_missing_days=resample_missing_days,
                clean_leg_geometries=clean_leg_geometries,
                add_length_outlier_flags=add_length_outlier_flags,
                add_signal_quality_flags=add_signal_quality_flags,
            )
        )
        logger.info("prepare_mobility_datasets: source prepared=%s", source_id)
    logger.info("prepare_mobility_datasets: concatenating sources=%s", len(datasets))
    return concat_mobility_datasets(datasets)
