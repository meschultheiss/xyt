"""Configuration objects for project-specific GPS transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from .mappings import MobilityMappings


@dataclass(frozen=True)
class Phase:
    """Named experimental phase used to tag and filter mobility records.

    Args:
        name: Stable phase name used in output columns and reports.
        start: Inclusive phase start date.
        end: Inclusive phase end date.
    """

    name: str
    start: str | pd.Timestamp
    end: str | pd.Timestamp

    @property
    def start_date(self):
        return pd.to_datetime(self.start).date()

    @property
    def end_date(self):
        return pd.to_datetime(self.end).date()


@dataclass(frozen=True)
class TimeSlice:
    """Named daily time interval used for reusable temporal aggregation.

    Args:
        name: Stable output label, for example `HPM`, `HC` or `HPS`.
        start: Inclusive start time formatted as `HH:MM`.
        end: Exclusive end time formatted as `HH:MM`.
    """

    name: str
    start: str
    end: str


def default_time_slices() -> tuple[TimeSlice, ...]:
    """Return default mobility-dashboard time slices.

    `HC` is intentionally not defined as an interval: it is the fallback label
    for observations outside the configured peak periods.
    """

    return (
        TimeSlice("HPM", "07:10", "09:00"),
        TimeSlice("HPS", "17:30", "20:00"),
    )


@dataclass(frozen=True)
class TrackingThresholds:
    """Minimum tracking duration expected for analysis.

    Args:
        min_days_by_phase: Minimum number of tracked days required by phase
            name. Missing phases default to one day in the filtering helpers.
        min_total_tracked_days: Minimum active tracking days across the full
            observation period.
        round_to_full_weeks: If true, phase durations are rounded down to
            complete weeks after applying the minimum-day threshold.
    """

    min_days_by_phase: Mapping[str, int] = field(default_factory=dict)
    min_total_tracked_days: int = 7
    round_to_full_weeks: bool = True


@dataclass(frozen=True)
class SignalLossThreshold:
    """Mode-specific signal-loss thresholds.

    `max_gap_m` is the absolute largest distance allowed between two
    consecutive points of a leg geometry, in meters. `max_relative_gap`
    is the same gap divided by total leg length.
    """

    modes: tuple[str, ...]
    max_gap_m: float
    max_relative_gap: float


def default_signal_loss_thresholds() -> dict[int, dict[str, SignalLossThreshold]]:
    """Return the default signal-loss thresholds.

    These values reproduce the current spatial-quality convention. Some
    absolute thresholds are intentionally very low in that notebook,
    making the relative signal-loss threshold the effective criterion.
    They should remain configurable by project.
    """

    road_modes = (
        "Mode::Car",
        "Mode::Motorbike",
        "Mode::Bus",
        "Mode::Tram",
        "Mode::Subway",
        "Mode::KickScooter",
        "Mode::Other",
        "Mode::TaxiUber",
        "Mode::Carsharing",
        "Mode::Ecar",
    )
    rail_modes = ("Mode::Train", "Mode::RegionalTrain", "Mode::LightRail")
    cycling_modes = ("Mode::Bicycle", "Mode::Ebicycle")
    walking_modes = ("Mode::Walk",)
    plane_boat_modes = ("Mode::Boat", "Mode::Airplane")
    return {
        1: {
            "road": SignalLossThreshold(road_modes, max_gap_m=1, max_relative_gap=0.60),
            "rail": SignalLossThreshold(rail_modes, max_gap_m=2000, max_relative_gap=0.70),
            "cycling": SignalLossThreshold(cycling_modes, max_gap_m=1, max_relative_gap=0.65),
            "walking": SignalLossThreshold(walking_modes, max_gap_m=1, max_relative_gap=0.95),
            "plane_boat": SignalLossThreshold(
                plane_boat_modes,
                max_gap_m=100_000_000,
                max_relative_gap=1.00,
            ),
        },
        2: {
            "road": SignalLossThreshold(road_modes, max_gap_m=1, max_relative_gap=0.45),
            "rail": SignalLossThreshold(rail_modes, max_gap_m=1500, max_relative_gap=0.55),
            "cycling": SignalLossThreshold(cycling_modes, max_gap_m=1, max_relative_gap=0.50),
            "walking": SignalLossThreshold(walking_modes, max_gap_m=1, max_relative_gap=0.75),
            "plane_boat": SignalLossThreshold(
                plane_boat_modes,
                max_gap_m=100_000_000,
                max_relative_gap=1.00,
            ),
        },
    }


@dataclass(frozen=True)
class SpatialQualityThresholds:
    """Spatial quality thresholds that may vary by project.

    Args:
        max_consecutive_point_distance_m: Reserved threshold for future
            point-jump filters.
        max_relative_signal_loss: Reserved global threshold for future simple
            signal-loss filters.
        outlier_quantiles_by_mode: Quantiles used to flag unusually long legs
            within each mode.
        signal_loss_thresholds_by_level: Mode-specific thresholds used to
            create `low_quality_legs_*` flags.
        bad_signal_user_quantile: Quantile used to identify users with very
            high average signal loss.
        signal_loss_mode_column: Column used to match mode-specific signal
            thresholds.
    """

    max_consecutive_point_distance_m: float | None = None
    max_relative_signal_loss: float | None = None
    outlier_quantiles_by_mode: tuple[float, ...] = (0.98, 0.99)
    signal_loss_thresholds_by_level: Mapping[int, Mapping[str, SignalLossThreshold]] = field(
        default_factory=default_signal_loss_thresholds
    )
    bad_signal_user_quantile: float = 0.995
    signal_loss_mode_column: str = "mode"


@dataclass(frozen=True)
class MatchingThresholds:
    """Temporal matching and future map-matching parameters.

    Args:
        leg_trip_journey_tolerance: Maximum temporal tolerance used when
            matching legs, trips and journeys.
        osrm_max_points_per_chunk: Reserved chunk size for future OSRM
            requests.
    """

    leg_trip_journey_tolerance: str = "5s"
    osrm_max_points_per_chunk: int = 99


@dataclass(frozen=True)
class ProjectConfig:
    """Project parameters that should remain explicit in the workflow.

    `ProjectConfig` centralizes import paths, project names, periods,
    coordinate systems, phases, thresholds and mappings. The goal is to avoid
    hiding methodological choices inside large transformation functions.

    Args:
        experiment_name: Optional analytical experiment name, for example
            `declic-prefig` or `declic-ziplo`. Leave empty for generic GPS
            processing without experiment grouping.
        motiontag_project_name: Optional provider project name used in export
            file names. Required only when loading files by inferred provider
            names.
        period: Optional export period string used in provider file names.
            Required only when loading files by inferred provider names.
        raw_data_dir: Directory containing raw GPS CSV exports.
        export_dir: Optional output directory.
        target_crs: CRS attached to parsed GPS geometries.
        operations_crs: Metric CRS used for distance calculations.
        csv_sep: CSV separator used by GPS exports.
        timezone: Local timezone label.
        phases: Optional analytical phases used for phase assignment and
            tracking filters. Leave empty for analyses without phase split.
        analysis_phases: Optional adjusted phases used for analytical filters
            when official phase dates include transition weeks or partial
            weeks. Leave empty to use only official `phases`.
        tracking_thresholds: User-level tracking-quality thresholds.
        spatial_quality_thresholds: Leg and user GPS-quality thresholds.
        matching_thresholds: Temporal matching thresholds.
        mappings: Mode and purpose mappings.
        time_slices: Daily time slices used for reusable temporal
            aggregations. Values outside the configured intervals are labelled
            `HC` by default.
        start_expe: Optional experiment start date used to build the
            1-based `n_week_sequence` column. If omitted, the first phase
            start date is used when phases are configured.
        end_expe: Optional experiment end date. If omitted, the last phase
            end date is used when phases are configured.
        excursion_area_path: Optional geospatial file path used by project
            notebooks to tag excursion legs or trips.
        reference_year: Optional year used to derive age from sociodemographic
            data.
        storyline_prefix_candidates: Accepted storyline file prefixes.
    """

    experiment_name: str | None = None
    motiontag_project_name: str | None = None
    period: str | None = None
    raw_data_dir: str | Path = "."
    export_dir: str | Path | None = None
    target_crs: str = "EPSG:4326"
    operations_crs: str = "EPSG:2056"
    csv_sep: str = ";"
    timezone: str = "Europe/Zurich"
    phases: tuple[Phase, ...] = ()
    tracking_thresholds: TrackingThresholds = field(default_factory=TrackingThresholds)
    spatial_quality_thresholds: SpatialQualityThresholds = field(
        default_factory=SpatialQualityThresholds
    )
    matching_thresholds: MatchingThresholds = field(default_factory=MatchingThresholds)
    mappings: MobilityMappings = field(default_factory=MobilityMappings)
    time_slices: tuple[TimeSlice, ...] = field(default_factory=default_time_slices)
    analysis_phases: tuple[Phase, ...] = ()
    start_expe: str | pd.Timestamp | None = None
    end_expe: str | pd.Timestamp | None = None
    excursion_area_path: str | Path | None = None
    reference_year: int | None = None
    storyline_prefix_candidates: tuple[str, ...] = (
        "StorylineWithTripId",
        "StorylineWithUserAnnotations",
        "StorylineExport",
    )

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_data_dir)

    @property
    def export_path(self) -> Path | None:
        return Path(self.export_dir) if self.export_dir is not None else None

    @property
    def experiment_start_date(self):
        """Return the experiment start date used for relative week numbering."""

        if self.start_expe is not None:
            return pd.to_datetime(self.start_expe).date()
        if self.phases:
            return min(phase.start_date for phase in self.phases)
        return None

    @property
    def experiment_end_date(self):
        """Return the configured experiment end date when available."""

        if self.end_expe is not None:
            return pd.to_datetime(self.end_expe).date()
        if self.phases:
            return max(phase.end_date for phase in self.phases)
        return None
