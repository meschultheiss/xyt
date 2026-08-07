"""Export helpers for structured mobility tables.

The export layer is kept separate from transformation code so formats such as
CSV, Parquet or Excel remain optional package concerns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import geopandas as gpd
import pandas as pd

from .models import IndicatorResult, MobilityDataset
from .quality import build_tracking_quality_report
from .spatial_analytics import build_spatial_analytics_tables, write_spatial_analytics_tables
from .table_dictionary import build_attribute_dictionary, build_table_descriptions
from .temporal import build_relative_time_index


MOBILITY_DATASET_TABLES = (
    "storyline",
    "staypoints",
    "legs",
    "trips",
    "journeys",
    "user_stats",
    "user_day_coverage",
    "map_track_trip_journey",
    "map_legs_staypoints",
)

TECHNICAL_EXPORT_COLUMNS = (
    "Phase",
    "phase_reference_column",
    "match_column",
    "xyt_source_id",
)

TABLE_PRIMARY_ID_COLUMNS = {
    "storyline": ("storyline_id", "leg_id", "activity_id"),
    "staypoints": ("activity_id",),
    "legs": ("leg_id",),
    "trips": ("trip_id",),
    "journeys": ("journey_id",),
    "user_stats": ("user_id",),
    "user_day_coverage": ("user_id", "date"),
    "user_statistics": ("user_statistics_id", "user_id"),
    "public_transport_legs": ("public_transport_leg_id", "leg_id"),
    "map_track_trip_journey": ("leg_id", "trip_id", "journey_id"),
    "map_legs_staypoints": ("previous_leg_id", "activity_id", "next_leg_id"),
    "relative_time_index": (
        "storyline_id",
        "leg_id",
        "activity_id",
        "previous_leg_id",
        "next_leg_id",
        "trip_id",
        "journey_id",
    ),
}


def mobility_dataset_tables(dataset: MobilityDataset) -> dict[str, pd.DataFrame]:
    """Return the named tables contained in a `MobilityDataset`.

    Args:
        dataset: Transformed mobility dataset.

    Returns:
        Dictionary keyed by stable table names. The order follows the
        transformation workflow and is reused by export helpers.
    """

    return {name: getattr(dataset, name) for name in MOBILITY_DATASET_TABLES}


def _ensure_supported_formats(formats: Iterable[str]) -> tuple[str, ...]:
    supported = {"csv", "geojson", "parquet", "xlsx", "pickle"}
    aliases = {"pkl": "pickle"}
    normalized = tuple(aliases.get(format_name.lower(), format_name.lower()) for format_name in formats)
    unsupported = sorted(set(normalized) - supported)
    if unsupported:
        raise ValueError(f"Unsupported export format(s): {unsupported}.")
    return normalized


def _prepare_csv_table(table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(table).copy()
    if isinstance(table, gpd.GeoDataFrame) and table.geometry.name in out.columns:
        geometry_col = table.geometry.name
        out[geometry_col] = table.geometry.to_wkt()
        out[f"{geometry_col}_crs"] = str(table.crs) if table.crs is not None else pd.NA
    return out


def prepare_public_export_table(table: pd.DataFrame, *, table_name: str | None = None) -> pd.DataFrame:
    """Return a public-facing table with technical workflow columns removed.

    This is a final safety and presentation pass, not the primary place where
    workflow columns should be created. Current pipeline tables should already
    use `phase` and avoid provider/workflow columns. This helper removes
    residual construction columns and moves identifiers to the front. It fails
    if a deprecated `Phase` column is still present, because that means the
    upstream preparation is not aligned with the current package contract.
    """

    out = table.copy()
    if "Phase" in out.columns:
        raise ValueError(
            f"Cannot export {table_name or 'table'}: deprecated column `Phase` is still present. "
            "Update upstream preparation to use `phase`."
        )
    if table_name == "public_transport_legs" and "track_id" in out.columns:
        if "leg_id" not in out.columns:
            out = out.rename(columns={"track_id": "leg_id"})
        else:
            out = out.drop(columns=["track_id"])

    primary_ids = TABLE_PRIMARY_ID_COLUMNS.get(str(table_name), ())
    if "id" in out.columns and any(column in out.columns for column in primary_ids):
        out = out.drop(columns=["id"])

    out = out.drop(columns=list(TECHNICAL_EXPORT_COLUMNS), errors="ignore")

    leading = []
    for column in (*primary_ids, "experiment_name", "user_id", "phase", "n_week_sequence"):
        if column in out.columns and column not in leading:
            leading.append(column)
    return out[[*leading, *[column for column in out.columns if column not in leading]]]


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    _prepare_csv_table(table).to_csv(path, index=False)


def _write_geojson(table: pd.DataFrame, path: Path) -> bool:
    if not isinstance(table, gpd.GeoDataFrame) or "geometry" not in table.columns:
        return False
    gdf = table.copy()
    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    path.write_text(gdf.to_json(drop_id=True, default=str), encoding="utf-8")
    return True


def _write_parquet(table: pd.DataFrame, path: Path) -> None:
    try:
        if isinstance(table, gpd.GeoDataFrame):
            table.to_parquet(path, index=False)
        else:
            table.to_parquet(path, index=False)
    except ImportError as error:
        raise ImportError(
            "Parquet export requires the optional export dependencies. "
            "Install them with `python -m pip install -r requirements-export.txt`."
        ) from error


def _write_pickle(table: pd.DataFrame, path: Path) -> None:
    table.to_pickle(path)


def _validation_report_to_frame(dataset: MobilityDataset) -> pd.DataFrame:
    frames = []
    for table, report in dataset.validation.items():
        frame = report.to_frame()
        if frame.empty:
            frame = pd.DataFrame(
                [
                    {
                        "table": table,
                        "severity": "ok",
                        "column": None,
                        "message": "No validation issue.",
                    }
                ]
            )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["table", "severity", "column", "message"])
    return pd.concat(frames, ignore_index=True)


def _quality_report_tables(
    dataset: MobilityDataset,
    *,
    selection_table: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    reports = {
        "quality_tracking_report": build_tracking_quality_report(dataset.user_stats),
    }
    if selection_table is not None:
        reports["quality_user_selection"] = selection_table
    return reports


def _write_xlsx(
    tables: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, table in tables.items():
                _prepare_csv_table(table).to_excel(writer, sheet_name=name[:31], index=False)
    except ImportError as error:
        raise ImportError(
            "Excel export requires the optional export dependencies. "
            "Install them with `python -m pip install -r requirements-export.txt`."
        ) from error


def write_mobility_dataset(
    dataset: MobilityDataset,
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("csv", "geojson"),
    extra_tables: Mapping[str, pd.DataFrame] | None = None,
    include_validation: bool = True,
    include_quality_reports: bool = True,
    selection_table: pd.DataFrame | None = None,
    overwrite: bool = True,
    clean_public_columns: bool = True,
    separate_format_dirs: bool = False,
) -> pd.DataFrame:
    """Write intermediate `MobilityDataset` tables to disk.

    The function writes the inspectable states of the preparation workflow:
    storyline, legs, staypoints, trips, journeys, user stats, mapping tables
    and optional validation and quality reports. It returns a manifest so the
    caller can see exactly what was exported.

    Args:
        dataset: Transformed mobility dataset.
        output_dir: Destination directory.
        formats: Export formats. Supported values are `csv`, `geojson`,
            `parquet`, `pickle`/`pkl` and `xlsx`. `geojson` is written only
            for geospatial tables.
        extra_tables: Optional additional tables to export with the same
            formats, for example `user_presence` or `participation_grid`.
            Table names must be stable file-safe identifiers.
        include_validation: Whether to export raw-schema validation issues.
        include_quality_reports: Whether to export tracking-quality summary
            and optional user selection table.
        selection_table: Optional table produced by
            `build_user_selection_table`.
        overwrite: If false, fail when a target file already exists.
        clean_public_columns: When true, remove internal workflow aliases from
            exported tables and keep normalized public column names.
        separate_format_dirs: When true, write each non-Excel format in a
            subdirectory named after the format (`csv/`, `parquet/`, `pkl/`).

    Returns:
        A manifest with one row per written file: table, format, path and
        number of rows.

    Raises:
        FileExistsError: If `overwrite` is false and a file already exists.
        ValueError: If an unsupported format is requested.
        ImportError: If optional dependencies for Parquet or Excel are
            missing.
    """

    export_formats = _ensure_supported_formats(formats)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tables = mobility_dataset_tables(dataset)
    if extra_tables:
        tables = {**tables, **{str(name): table for name, table in extra_tables.items()}}
    if include_validation:
        tables = {**tables, "validation_issues": _validation_report_to_frame(dataset)}
    if include_quality_reports:
        tables = {**tables, **_quality_report_tables(dataset, selection_table=selection_table)}
    if clean_public_columns:
        tables = {
            table_name: prepare_public_export_table(table, table_name=table_name)
            for table_name, table in tables.items()
        }

    manifest_rows = []

    if "xlsx" in export_formats:
        xlsx_path = output_path / "mobility_dataset.xlsx"
        if xlsx_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {xlsx_path}")
        _write_xlsx(tables, xlsx_path)
        manifest_rows.append(
            {
                "table": "all_tables",
                "format": "xlsx",
                "path": str(xlsx_path),
                "rows": sum(len(table) for table in tables.values()),
            }
        )

    for table_name, table in tables.items():
        for format_name in export_formats:
            if format_name == "xlsx":
                continue
            suffix = "pkl" if format_name == "pickle" else format_name
            format_dir = output_path / suffix if separate_format_dirs else output_path
            format_dir.mkdir(parents=True, exist_ok=True)
            table_path = format_dir / f"{table_name}.{suffix}"
            if table_path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {table_path}")

            written = True
            if format_name == "csv":
                _write_csv(table, table_path)
            elif format_name == "geojson":
                written = _write_geojson(table, table_path)
            elif format_name == "parquet":
                _write_parquet(table, table_path)
            elif format_name == "pickle":
                _write_pickle(table, table_path)

            if written:
                manifest_rows.append(
                    {
                        "table": table_name,
                        "format": format_name,
                        "path": str(table_path),
                        "rows": len(table),
                    }
                )

    manifest = pd.DataFrame(manifest_rows, columns=["table", "format", "path", "rows"])
    manifest_path = output_path / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def export_mobility_tables(*args, **kwargs) -> pd.DataFrame:
    """Alias for `write_mobility_dataset`.

    `write_mobility_dataset` is the preferred name because it says that the
    full structured dataset is written, including reports and mapping tables.
    """

    return write_mobility_dataset(*args, **kwargs)


def write_indicator_result(
    indicators: IndicatorResult,
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("parquet", "csv"),
    overwrite: bool = True,
    separate_format_dirs: bool = False,
) -> pd.DataFrame:
    """Write mobility indicator tables to disk.

    Args:
        indicators: Result returned by `compute_mobility_indicators`.
        output_dir: Destination directory.
        formats: Export formats. Supported values are `csv`, `parquet` and
            `pickle`/`pkl`.
        overwrite: If false, fail when a target file already exists.
        separate_format_dirs: When true, write each format in a subdirectory
            named after the format (`csv/`, `parquet/`, `pkl/`).

    Returns:
        Manifest with one row per written file.
    """

    export_formats = _ensure_supported_formats(formats)
    supported = {"csv", "parquet", "pickle"}
    unsupported = sorted(set(export_formats) - supported)
    if unsupported:
        raise ValueError(f"Unsupported indicator export format(s): {unsupported}.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for table_name, table in indicators.tables().items():
        for format_name in export_formats:
            suffix = "pkl" if format_name == "pickle" else format_name
            format_dir = output_path / suffix if separate_format_dirs else output_path
            format_dir.mkdir(parents=True, exist_ok=True)
            table_path = format_dir / f"indicators_{table_name}.{suffix}"
            if table_path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {table_path}")
            if format_name == "csv":
                table.to_csv(table_path, index=False)
            elif format_name == "parquet":
                _write_parquet(table, table_path)
            elif format_name == "pickle":
                _write_pickle(table, table_path)
            manifest_rows.append(
                {
                    "table": f"indicators_{table_name}",
                    "format": format_name,
                    "path": str(table_path),
                    "rows": len(table),
                }
            )

    manifest = pd.DataFrame(manifest_rows, columns=["table", "format", "path", "rows"])
    manifest_path = output_path / "indicator_export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _merge_extra_tables(
    *,
    dataset: MobilityDataset,
    extra_tables: Mapping[str, pd.DataFrame] | None,
    include_relative_time_index: bool,
    relative_anchor: str,
    default_timezone: str,
) -> dict[str, pd.DataFrame] | None:
    tables: dict[str, pd.DataFrame] = {}
    if extra_tables:
        tables.update({str(name): table for name, table in extra_tables.items()})

    if include_relative_time_index:
        if "relative_time_index" in tables:
            raise ValueError(
                "`extra_tables` already contains `relative_time_index`. "
                "Pass `include_relative_time_index=False` to export your own table."
            )
        tables["relative_time_index"] = build_relative_time_index(
            mobility_dataset_tables(dataset),
            relative_anchor=relative_anchor,
            default_timezone=default_timezone,
        )

    return tables or None


def _clean_spatial_formats(
    formats: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Return cautious defaults for spatial analytics exports.

    `leg_points_h3` can be very large. When the clean export asks for CSV and
    Parquet, keep the detailed point table in Parquet by default and write CSV
    only for aggregated spatial tables.
    """

    normalized = _ensure_supported_formats(formats)
    spatial_formats = tuple(
        format_name
        for format_name in normalized
        if format_name in {"csv", "parquet", "pickle"}
    )
    leg_point_formats = tuple(
        format_name for format_name in spatial_formats if format_name != "csv"
    )
    if not leg_point_formats:
        leg_point_formats = spatial_formats
    return {
        "__default__": spatial_formats,
        "leg_points_h3": leg_point_formats,
    }


def _tag_manifest(manifest: pd.DataFrame, section: str) -> pd.DataFrame:
    out = manifest.copy()
    out.insert(0, "section", section)
    return out


def _write_table_dictionary(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    formats: Iterable[str],
    overwrite: bool,
    separate_format_dirs: bool,
) -> pd.DataFrame:
    metadata_formats = tuple(
        format_name
        for format_name in _ensure_supported_formats(formats)
        if format_name in {"csv", "parquet", "pickle"}
    ) or ("csv",)
    metadata_tables = {
        "table_descriptions": build_table_descriptions(tables),
        "attribute_dictionary": build_attribute_dictionary(tables),
    }
    manifest_rows = []
    for table_name, table in metadata_tables.items():
        for format_name in metadata_formats:
            suffix = "pkl" if format_name == "pickle" else format_name
            format_dir = output_dir / suffix if separate_format_dirs else output_dir
            format_dir.mkdir(parents=True, exist_ok=True)
            table_path = format_dir / f"{table_name}.{suffix}"
            if table_path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {table_path}")
            if format_name == "csv":
                table.to_csv(table_path, index=False)
            elif format_name == "parquet":
                _write_parquet(table, table_path)
            elif format_name == "pickle":
                _write_pickle(table, table_path)
            manifest_rows.append(
                {
                    "table": table_name,
                    "format": format_name,
                    "path": str(table_path),
                    "rows": len(table),
                }
            )
    return pd.DataFrame(manifest_rows, columns=["table", "format", "path", "rows"])


def export_clean_dataset(
    dataset: MobilityDataset,
    indicators: IndicatorResult | None,
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("parquet", "csv"),
    include_spatial_analytics: bool = False,
    include_relative_time_index: bool = True,
    include_table_dictionary: bool = True,
    relative_anchor: str = "2000-01-03",
    default_timezone: str = "Europe/Zurich",
    extra_tables: Mapping[str, pd.DataFrame] | None = None,
    selection_table: pd.DataFrame | None = None,
    overwrite: bool = True,
    separate_format_dirs: bool = False,
    spatial_analytics_options: Mapping[str, Any] | None = None,
    spatial_analytics_formats: Iterable[str] | Mapping[str, Iterable[str]] | None = None,
    manifest_name: str = "clean_export_manifest.json",
) -> pd.DataFrame:
    """Write the recommended clean export set and return a unified manifest.

    This is the public, analyst-facing export function. It combines structured
    mobility tables, optional indicator tables, a relative time index and
    optional H3 spatial analytics in one readable manifest. Fine-grained export
    helpers remain available when a notebook needs to expose each construction
    step separately.

    Args:
        dataset: Structured mobility dataset returned by
            `prepare_mobility_dataset`.
        indicators: Optional indicator result returned by
            `compute_mobility_indicators`. Pass `None` to export only the
            cleaned mobility dataset.
        output_dir: Destination directory.
        formats: Formats for mobility and indicator tables. Supported values
            are `parquet`, `csv`, `pickle`/`pkl`, `geojson` and `xlsx`.
            The clean default is `("parquet", "csv")`.
        include_spatial_analytics: Whether to build and export dashboard-ready
            H3 tables. Disabled by default because it can be expensive.
        include_relative_time_index: Whether to add `relative_time_index` to
            the exported clean tables.
        include_table_dictionary: Whether to export `table_descriptions` and
            `attribute_dictionary` metadata for the clean tables.
        relative_anchor: Date used as week 1 day 1 in the relative time index.
        default_timezone: Fallback timezone for relative dates when row-level
            timezone columns are absent.
        extra_tables: Additional project tables to export alongside the clean
            dataset, for example questionnaire, CO2 occupancy or health tables.
        selection_table: Optional table returned by
            `build_user_selection_table`; when provided it is exported with
            the quality reports.
        overwrite: If false, fail when a target file already exists.
        separate_format_dirs: When true, write each format in a subdirectory.
        spatial_analytics_options: Optional keyword arguments passed to
            `build_spatial_analytics_tables`.
        spatial_analytics_formats: Optional formats for spatial tables. When
            omitted, the function reuses `formats` but avoids CSV for the
            detailed `leg_points_h3` table.
        manifest_name: Name of the unified JSON manifest written in
            `output_dir`.

    Returns:
        Manifest with one row per written file. Columns are `section`, `table`,
        `format`, `path` and `rows`.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / manifest_name
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {manifest_path}")

    export_formats = _ensure_supported_formats(formats)
    merged_extra_tables = _merge_extra_tables(
        dataset=dataset,
        extra_tables=extra_tables,
        include_relative_time_index=include_relative_time_index,
        relative_anchor=relative_anchor,
        default_timezone=default_timezone,
    )

    manifests = [
        _tag_manifest(
            write_mobility_dataset(
                dataset,
                output_path,
                formats=export_formats,
                extra_tables=merged_extra_tables,
                selection_table=selection_table,
                overwrite=overwrite,
                clean_public_columns=True,
                separate_format_dirs=separate_format_dirs,
            ),
            "mobility_dataset",
        )
    ]

    if include_table_dictionary:
        dictionary_source_tables = {
            table_name: prepare_public_export_table(table, table_name=table_name)
            for table_name, table in {
                **mobility_dataset_tables(dataset),
                **dict(merged_extra_tables or {}),
            }.items()
        }
        manifests.append(
            _tag_manifest(
                _write_table_dictionary(
                    dictionary_source_tables,
                    output_path,
                    formats=export_formats,
                    overwrite=overwrite,
                    separate_format_dirs=separate_format_dirs,
                ),
                "metadata",
            )
        )

    if indicators is not None:
        indicator_formats = tuple(
            format_name
            for format_name in export_formats
            if format_name in {"csv", "parquet", "pickle"}
        )
        if not indicator_formats:
            raise ValueError(
                "`export_clean_dataset` cannot export indicators with the requested formats. "
                "Use one of: csv, parquet, pickle."
            )
        manifests.append(
            _tag_manifest(
                write_indicator_result(
                    indicators,
                    output_path,
                    formats=indicator_formats,
                    overwrite=overwrite,
                    separate_format_dirs=separate_format_dirs,
                ),
                "indicators",
            )
        )

    if include_spatial_analytics:
        spatial_tables = build_spatial_analytics_tables(
            dataset,
            **dict(spatial_analytics_options or {}),
        )
        spatial_formats = (
            spatial_analytics_formats
            if spatial_analytics_formats is not None
            else _clean_spatial_formats(export_formats)
        )
        manifests.append(
            _tag_manifest(
                write_spatial_analytics_tables(
                    spatial_tables,
                    output_path,
                    formats=spatial_formats,
                    overwrite=overwrite,
                    manifest_name="spatial_analytics_manifest.json",
                    separate_format_dirs=separate_format_dirs,
                ),
                "spatial_analytics",
            )
        )

    manifest = pd.concat(manifests, ignore_index=True)
    manifest = manifest[["section", "table", "format", "path", "rows"]]
    manifest_path.write_text(
        json.dumps(manifest.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
