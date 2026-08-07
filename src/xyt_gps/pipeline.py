"""Recommended end-to-end entry point for the mobility pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from .config import ProjectConfig
from .indicators import compute_mobility_indicators
from .io import RawSampleConfig, load_gps_export
from .models import IndicatorResult, MobilityDataset, RawGpsData
from .transform import prepare_mobility_dataset


@dataclass(frozen=True)
class MobilityPipelineResult:
    """Named result returned by `run_mobility_pipeline`.

    Attributes:
        raw: Raw GPS tables after loading or validation.
        dataset: Structured mobility tables.
        indicators: Optional mobility indicators. `None` when
            `compute_indicators=False`.
    """

    raw: RawGpsData
    dataset: MobilityDataset
    indicators: IndicatorResult | None

    def __iter__(self) -> Iterator:
        """Allow `raw, dataset, indicators = run_mobility_pipeline(...)`."""

        yield self.raw
        yield self.dataset
        yield self.indicators


def run_mobility_pipeline(
    config: ProjectConfig,
    *,
    raw: RawGpsData | None = None,
    sample: RawSampleConfig | None = None,
    sociodemo: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    weight_col: str = "weight",
    default_weight: float = 1.0,
    validate: bool = True,
    must_exist: bool = True,
    resample_missing_days: bool = False,
    clean_leg_geometries: bool = True,
    add_length_outlier_flags: bool = True,
    add_signal_quality_flags: bool = True,
    compute_indicators: bool = True,
    mode_col: str = "mode_niv1",
    trips_mode_col: str | None = None,
    distance_col: str | None = None,
    include_zero_days: bool = True,
    include_excursions: bool = False,
    include_airplane: bool = False,
    use_weights: bool = True,
    default_phase_name: str = "All",
) -> MobilityPipelineResult:
    """Run the recommended single-source GPS-to-indicators workflow.

    This is the simplest entry point for analysts discovering the package:
    load raw GPS tables, prepare mobility tables and optionally compute the
    generic indicators. Pass `raw` when tables have already been loaded in a
    notebook; otherwise the function calls `load_gps_export(config)`.

    Args:
        config: Project configuration.
        raw: Optional preloaded raw GPS tables. When omitted, files are loaded
            from `config` with `load_gps_export`.
        sample: Optional sampling strategy used only when `raw` is omitted.
        sociodemo: Optional user-level sociodemographic table.
        weights: Optional user-level weighting table.
        weight_col: Name of the user weight column.
        default_weight: Default user weight when no weight is available.
        validate: Whether to validate raw tables during loading.
        must_exist: Whether expected files must exist when loading from disk.
        resample_missing_days: Add explicit `Resampled_stay` rows for missing
            tracked days.
        clean_leg_geometries: Normalize continuous leg geometries.
        add_length_outlier_flags: Add per-mode length outlier flags.
        add_signal_quality_flags: Compute GPS signal-loss quality columns.
        compute_indicators: Whether to compute `IndicatorResult`.
        mode_col: Leg mode column used for indicators.
        trips_mode_col: Optional trip mode column override.
        distance_col: Optional leg distance column override.
        include_zero_days: Include tracked days without movement in daily means.
        include_excursions: Include legs/trips flagged as excursions. Defaults
            to false because excursions are usually analysed separately.
        include_airplane: Include airplane legs/trips in indicators. Defaults
            to false because airplane rows can dominate distances and CO2.
        use_weights: Use user weights in population indicators.
        default_phase_name: Period label when no phase split exists.

    Returns:
        `MobilityPipelineResult` containing raw tables, mobility tables and
        optional indicators.
    """

    raw_data = raw if raw is not None else load_gps_export(
        config,
        sample=sample,
        validate=validate,
        must_exist=must_exist,
    )
    dataset = prepare_mobility_dataset(
        raw_data,
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
    indicators = None
    if compute_indicators:
        indicators = compute_mobility_indicators(
            dataset,
            mode_col=mode_col,
            trips_mode_col=trips_mode_col,
            distance_col=distance_col,
            include_zero_days=include_zero_days,
            include_excursions=include_excursions,
            include_airplane=include_airplane,
            use_weights=use_weights,
            weight_col=weight_col,
            config=config,
            default_phase_name=default_phase_name,
        )
    return MobilityPipelineResult(raw=raw_data, dataset=dataset, indicators=indicators)


__all__ = ["MobilityPipelineResult", "run_mobility_pipeline"]
