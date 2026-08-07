"""Small data containers returned by xyt_gps functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    """Single schema-validation issue.

    Attributes:
        table: Logical table name, for example `storyline` or `trips`.
        severity: Issue severity. Current values are `error` and `warning`.
        message: Human-readable explanation.
        column: Optional column concerned by the issue.
    """

    table: str
    severity: str
    message: str
    column: str | None = None


@dataclass
class SchemaValidationResult:
    """Schema validation result for one raw table.

    Attributes:
        table: Logical table name.
        n_rows: Number of rows checked.
        n_columns: Number of columns checked.
        columns: Observed column names.
        missing_required: Required columns absent from the table.
        missing_recommended: Recommended columns absent from the table.
        unexpected: Observed columns not described by the schema.
        null_rates: Null-rate by observed column.
        issues: Structured validation issues.
    """

    table: str
    n_rows: int
    n_columns: int
    columns: list[str]
    missing_required: list[str] = field(default_factory=list)
    missing_recommended: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    null_rates: dict[str, float] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return true when no blocking validation error was found."""

        return not any(issue.severity == "error" for issue in self.issues)

    def to_frame(self) -> pd.DataFrame:
        """Return validation issues as a tabular report."""

        return pd.DataFrame(
            [
                {
                    "table": issue.table,
                    "severity": issue.severity,
                    "column": issue.column,
                    "message": issue.message,
                }
                for issue in self.issues
            ]
        )


@dataclass
class RawGpsData:
    """Raw GPS export tables before transformation.

    Attributes:
        storyline: Raw provider storyline table containing the GPS events.
        trips: Raw `Trips` table.
        journeys: Raw `Journeys` table.
        user_statistics: Optional raw `UserStatistics` table.
        validation: Optional validation reports produced during loading.
        paths: Source path metadata, usually a `GpsExportPaths` object.
    """

    storyline: pd.DataFrame
    trips: pd.DataFrame
    journeys: pd.DataFrame
    user_statistics: pd.DataFrame | None = None
    validation: dict[str, SchemaValidationResult] = field(default_factory=dict)
    paths: Any | None = None


@dataclass
class MobilityDataset:
    """Structured mobility tables produced by the preparation workflow.

    Attributes:
        storyline: Parsed storyline with harmonized dates, geometry and
            project mappings.
        staypoints: Stay and resampled-stay rows.
        legs: Track rows enriched with quality and signal-loss fields.
        trips: Prepared trips enriched with journey references.
        journeys: Prepared journeys.
        user_stats: User-level quality, tracking and optional side-table
            information.
        user_day_coverage: Canonical person-day calendar distinguishing days
            in the individual range, tracked days and movement days.
        map_track_trip_journey: Mapping table between legs, trips and
            journeys.
        map_legs_staypoints: Mapping table between staypoints and adjacent
            legs.
        validation: Raw schema-validation reports.
        metadata: Lightweight provenance for the transformation: project,
            phases, options and computed quality steps.
    """

    storyline: Any
    staypoints: Any
    legs: Any
    trips: pd.DataFrame
    journeys: pd.DataFrame
    user_stats: pd.DataFrame
    map_track_trip_journey: pd.DataFrame
    map_legs_staypoints: pd.DataFrame
    user_day_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation: dict[str, SchemaValidationResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndicatorResult:
    """Mobility indicator tables computed from a `MobilityDataset`.

    Attributes:
        person_day: Long table with one row per user, day, phase and mode.
        person_phase: Mean daily indicators by user, phase and mode.
        population: Population-level means by phase and mode.
        mode_col: Mode column used to build the tables.
        metadata: Calculation options and small identity-card statistics used
            to interpret the indicators.
    """

    person_day: pd.DataFrame
    person_phase: pd.DataFrame
    population: pd.DataFrame
    mode_col: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return indicator tables keyed by stable export names."""

        return {
            "person_day": self.person_day,
            "person_phase": self.person_phase,
            "population": self.population,
        }
