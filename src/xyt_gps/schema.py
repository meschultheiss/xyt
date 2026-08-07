"""Schema validation for raw GPS mobility exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from .models import RawGpsData, SchemaValidationResult, ValidationIssue


@dataclass(frozen=True)
class SchemaSpec:
    """Column contract for one raw GPS input table."""

    table: str
    required: tuple[str, ...]
    recommended: tuple[str, ...] = ()
    nullable_checked: tuple[str, ...] = ()
    max_null_rate: Mapping[str, float] = field(default_factory=dict)


GPS_SCHEMAS: dict[str, SchemaSpec] = {
    "storyline": SchemaSpec(
        table="storyline",
        required=(
            "storyline_id",
            "user_id",
            "type",
            "started_at",
            "finished_at",
            "started_on",
            "geometry",
        ),
        recommended=(
            "created_at",
            "updated_at",
            "started_at_in_timezone",
            "finished_at_in_timezone",
            "confirmed_at_in_timezone",
            "created_at_in_timezone",
            "updated_at_in_timezone",
            "purpose",
            "mode",
            "detected_mode",
            "length",
            "trip_id",
        ),
        nullable_checked=("geometry", "started_at", "finished_at"),
        max_null_rate={"geometry": 0.01, "started_at": 0.01, "finished_at": 0.01},
    ),
    "trips": SchemaSpec(
        table="trips",
        required=("trip_id", "user_id", "started_at", "finished_at"),
        recommended=(
            "duration",
            "distance",
            "tracks_count",
            "created_at",
            "purpose",
            "main_mode",
            "modes",
        ),
        nullable_checked=("started_at", "finished_at"),
        max_null_rate={"started_at": 0.01, "finished_at": 0.01},
    ),
    "journeys": SchemaSpec(
        table="journeys",
        required=("journey_id", "user_id", "started_at", "finished_at"),
        recommended=(
            "duration",
            "distance",
            "tracks_count_without_walks",
            "trips_count",
            "tracks_count",
            "created_at",
            "purpose",
            "main_mode",
            "modes",
        ),
        nullable_checked=("started_at", "finished_at"),
        max_null_rate={"started_at": 0.01, "finished_at": 0.01},
    ),
    "user_statistics": SchemaSpec(
        table="user_statistics",
        required=("user_id",),
        recommended=(
            "first_tracked_at",
            "latest_tracked_at",
            "days_in_range_count",
            "tracked_days_count",
            "active_days_count",
            "zero_movement_tracked_days_count",
            "untracked_days_count",
            "inactive_days_count",
        ),
    ),
}


def expected_gps_schema() -> pd.DataFrame:
    """Return the expected GPS import schema as an inspectable table.

    The current schema describes the structured GPS multi-table contract used
    by the package: storyline, trips, journeys and user statistics. It is
    intentionally generic at the public API level because the landing step may
    adapt source-specific files and column names before data loading.
    """

    rows = []
    for table, spec in GPS_SCHEMAS.items():
        for column in spec.required:
            rows.append({"table": table, "column": column, "status": "required"})
        for column in spec.recommended:
            rows.append({"table": table, "column": column, "status": "recommended"})
    return pd.DataFrame(rows)


def check_raw_import_columns(
    storyline: pd.DataFrame,
    user_statistics: pd.DataFrame | None = None,
    *,
    trips: pd.DataFrame | None = None,
    journeys: pd.DataFrame | None = None,
    include_recommended: bool = True,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """Check the raw column structure before building `RawGpsData`.

    This is a lightweight, notebook-friendly check. It focuses on the column
    names expected at the raw import stage. It does not parse dates,
    geometries or modes.

    Args:
        storyline: Raw storyline table to check.
        user_statistics: Optional raw user statistics table to check.
        trips: Optional raw trips table to check.
        journeys: Optional raw journeys table to check.
        include_recommended: Include non-blocking recommended columns in the
            report.
        raise_on_error: Raise a `ValueError` when a required column is missing.

    Returns:
        A DataFrame with one row per expected column. Important columns are:
        `table`, `expected_column`, `present`, `status` and `message`.
    """

    rows: list[dict[str, object]] = []
    table_specs = [("storyline", storyline, GPS_SCHEMAS["storyline"])]
    if trips is not None:
        table_specs.append(("trips", trips, GPS_SCHEMAS["trips"]))
    if journeys is not None:
        table_specs.append(("journeys", journeys, GPS_SCHEMAS["journeys"]))
    if user_statistics is not None:
        table_specs.append(("user_statistics", user_statistics, GPS_SCHEMAS["user_statistics"]))

    for table_name, frame, spec in table_specs:
        observed_columns = set(frame.columns)
        expected_columns = list(spec.required)
        if include_recommended:
            expected_columns.extend(spec.recommended)

        for column in expected_columns:
            required = column in spec.required
            observed_column = column if column in observed_columns else None

            present = observed_column is not None
            if present and observed_column == column:
                status = "ok"
                message = "Column found."
            elif present:
                status = "unexpected_alias"
                message = f"`{observed_column}` is not accepted as an alias for `{column}`."
            elif required:
                status = "missing_required"
                message = "Required column is missing or has a different name."
            else:
                status = "missing_recommended"
                message = "Recommended column is missing; related enrichments may be skipped."

            rows.append(
                {
                    "table": table_name,
                    "expected_column": column,
                    "observed_column": observed_column,
                    "required": required,
                    "present": present,
                    "status": status,
                    "message": message,
                }
            )

    report = pd.DataFrame(rows)
    errors = report.loc[report["status"].eq("missing_required")]
    if raise_on_error and not errors.empty:
        raise ValueError(
            "Raw import column check failed:\n"
            + errors[["table", "expected_column", "message"]].to_string(index=False)
        )
    return report


def validate_schema(
    df: pd.DataFrame,
    spec: SchemaSpec,
    *,
    allow_extra_columns: bool = True,
) -> SchemaValidationResult:
    """Validate column presence and basic null rates without changing data."""

    columns = list(df.columns)
    column_set = set(columns)
    required = set(spec.required)
    recommended = set(spec.recommended)
    expected = required | recommended
    missing_required = sorted(required - column_set)
    missing_recommended = sorted(recommended - column_set)
    unexpected = sorted(column_set - expected)
    issues: list[ValidationIssue] = []

    for column in missing_required:
        issues.append(
            ValidationIssue(
                table=spec.table,
                severity="error",
                column=column,
                message="Required raw column is missing.",
            )
        )

    for column in missing_recommended:
        issues.append(
            ValidationIssue(
                table=spec.table,
                severity="warning",
                column=column,
                message="Recommended raw column is missing; downstream functions may skip related enrichments.",
            )
        )

    if unexpected and not allow_extra_columns:
        for column in unexpected:
            issues.append(
                ValidationIssue(
                    table=spec.table,
                    severity="warning",
                    column=column,
                    message="Unexpected raw column.",
                )
            )

    null_rates: dict[str, float] = {}
    for column in spec.nullable_checked:
        if column not in df.columns:
            continue
        rate = float(df[column].isna().mean())
        null_rates[column] = rate
        max_rate = spec.max_null_rate.get(column)
        if max_rate is not None and rate > max_rate:
            issues.append(
                ValidationIssue(
                    table=spec.table,
                    severity="error",
                    column=column,
                    message=f"Null rate {rate:.2%} exceeds accepted threshold {max_rate:.2%}.",
                )
            )

    return SchemaValidationResult(
        table=spec.table,
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=columns,
        missing_required=missing_required,
        missing_recommended=missing_recommended,
        unexpected=unexpected,
        null_rates=null_rates,
        issues=issues,
    )


def validate_gps_raw(raw: RawGpsData) -> dict[str, SchemaValidationResult]:
    """Validate all raw GPS tables that are present."""

    reports = {
        "storyline": validate_schema(raw.storyline, GPS_SCHEMAS["storyline"]),
        "trips": validate_schema(raw.trips, GPS_SCHEMAS["trips"]),
        "journeys": validate_schema(raw.journeys, GPS_SCHEMAS["journeys"]),
    }
    if raw.user_statistics is not None:
        reports["user_statistics"] = validate_schema(
            raw.user_statistics,
            GPS_SCHEMAS["user_statistics"],
        )
    return reports
