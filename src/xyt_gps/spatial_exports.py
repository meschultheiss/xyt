"""Dashboard-oriented spatial table and DuckDB exports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from .config import ProjectConfig, TimeSlice, default_time_slices
from .models import MobilityDataset
from .spatial_h3 import (
    DEFAULT_H3_COUNT_DIMENSION_SETS,
    DEFAULT_H3_COUNT_METRICS,
    aggregate_h3_frequencies,
    build_h3_count_matrix,
    legs_to_h3_points,
)
from .spatial_time import add_time_slices


def _normalise_formats(formats: Iterable[str]) -> tuple[str, ...]:
    supported = {"parquet", "csv", "pickle"}
    aliases = {"pkl": "pickle"}
    normalized = tuple(aliases.get(format_name.lower(), format_name.lower()) for format_name in formats)
    unsupported = sorted(set(normalized) - supported)
    if unsupported:
        raise ValueError(f"Unsupported spatial analytics export format(s): {unsupported}.")
    return normalized


def _csv_ready(table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(table).copy()
    if isinstance(table, gpd.GeoDataFrame) and table.geometry.name in out.columns:
        geometry_col = table.geometry.name
        out[geometry_col] = table.geometry.to_wkt()
        out[f"{geometry_col}_crs"] = str(table.crs) if table.crs is not None else pd.NA
    return out


def _write_table(table: pd.DataFrame, path: Path, format_name: str) -> None:
    if format_name == "csv":
        _csv_ready(table).to_csv(path, index=False)
    elif format_name == "parquet":
        try:
            table.to_parquet(path, index=False)
        except ImportError as error:
            raise ImportError(
                "Parquet export requires optional export dependencies. "
                "Install them with `python -m pip install -r requirements-export.txt`."
            ) from error
    elif format_name == "pickle":
        table.to_pickle(path)


def _legs_from_dataset_or_table(dataset_or_legs: MobilityDataset | pd.DataFrame) -> pd.DataFrame:
    if isinstance(dataset_or_legs, MobilityDataset):
        legs = dataset_or_legs.legs.copy()
        trips = dataset_or_legs.trips
        if "trip_id" not in legs.columns or "trip_id" not in trips.columns:
            return legs

        trip_cols = [
            column
            for column in (
                "trip_id",
                "purpose",
                "purpose_mrmt",
                "main_mode_niv1",
                "main_mode_niv2",
                "main_mode_mrmt",
            )
            if column in trips.columns
        ]
        if len(trip_cols) <= 1:
            return legs

        dimensions = trips[trip_cols].drop_duplicates("trip_id").copy()
        rename = {
            column: f"trip_{column}"
            for column in dimensions.columns
            if column != "trip_id" and column in legs.columns
        }
        dimensions = dimensions.rename(columns=rename)
        out = legs.merge(dimensions, on="trip_id", how="left")
        if "purpose_mrmt" not in out.columns and "trip_purpose_mrmt" in out.columns:
            out["purpose_mrmt"] = out["trip_purpose_mrmt"]
        if "trip_purpose" not in out.columns and "purpose" in dimensions.columns:
            out["trip_purpose"] = out["purpose"]
        return out
    return dataset_or_legs


def build_spatial_analytics_tables(
    dataset_or_legs: MobilityDataset | pd.DataFrame,
    *,
    h3_resolution: int | Iterable[int] = 9,
    config: ProjectConfig | None = None,
    frequency_group_cols: Iterable[str] | None = None,
    count_dimension_sets: Iterable[Iterable[str]] = DEFAULT_H3_COUNT_DIMENSION_SETS,
    count_metrics: Iterable[str] = DEFAULT_H3_COUNT_METRICS,
    include_count_matrix: bool = True,
    time_slices: Iterable[TimeSlice] | None = None,
    datetime_col: str = "started_at",
    time_slice_col: str = "time_slice",
    timezone: str | None = None,
    sample_distance_m: float | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    chunk_size: int = 250,
) -> dict[str, pd.DataFrame]:
    """Build H3 spatial analytics tables without writing files.

    Args:
        dataset_or_legs: A `MobilityDataset` or a leg table.
        h3_resolution: H3 resolution or list of resolutions used for point
            indexing.
        config: Optional project configuration. When provided, its timezone
            and time slices are used unless explicit values are passed.
        frequency_group_cols: Grouping columns for `aggregate_h3_frequencies`.
            Defaults to H3 cell only.
        count_dimension_sets: Dimension sets used for the wide count matrix.
        count_metrics: Count metrics included in the wide count matrix.
        include_count_matrix: Whether to export `h3_count_matrix`.
        time_slices: Optional reusable daily intervals. Defaults to
            `config.time_slices` or package defaults.
        datetime_col: Datetime column used to assign time slices.
        time_slice_col: Name of the time-slice column.
        timezone: Local timezone used for time slices. Defaults to
            `config.timezone` or `Europe/Zurich`.
        sample_distance_m: Optional regular sampling distance in metres before
            H3 indexing.
        parallel: If true, H3 point indexing is split by leg chunks and
            processed with a thread pool.
        max_workers: Optional number of worker threads when `parallel=True`.
        chunk_size: Number of legs per H3 indexing chunk.

    Returns:
        Dictionary with `leg_points_h3`, `h3_frequency` and, when requested,
        `h3_count_matrix`.
    """

    legs = _legs_from_dataset_or_table(dataset_or_legs)
    slice_definitions = tuple(
        time_slices
        if time_slices is not None
        else (config.time_slices if config is not None else default_time_slices())
    )
    timezone_value = timezone if timezone is not None else (config.timezone if config is not None else "Europe/Zurich")
    if datetime_col in legs.columns:
        legs = add_time_slices(
            legs,
            time_slices=slice_definitions,
            datetime_col=datetime_col,
            output_col=time_slice_col,
            timezone=timezone_value,
        )
    h3_points = legs_to_h3_points(
        legs,
        h3_resolution=h3_resolution,
        sample_distance_m=sample_distance_m,
        parallel=parallel,
        max_workers=max_workers,
        chunk_size=chunk_size,
    )
    h3_frequency = aggregate_h3_frequencies(
        h3_points,
        group_cols=frequency_group_cols,
    )

    tables: dict[str, pd.DataFrame] = {
        "leg_points_h3": h3_points,
        "h3_frequency": h3_frequency,
    }
    if include_count_matrix:
        tables["h3_count_matrix"] = build_h3_count_matrix(
            h3_points,
            dimension_sets=count_dimension_sets,
            metrics=count_metrics,
        )
    return tables


def _formats_for_table(
    table_name: str,
    formats: Iterable[str] | Mapping[str, Iterable[str]],
) -> tuple[str, ...]:
    if isinstance(formats, Mapping):
        table_formats = formats.get(table_name, formats.get("__default__", ()))
        return _normalise_formats(table_formats)
    return _normalise_formats(formats)


def write_spatial_analytics_tables(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    formats: Iterable[str] | Mapping[str, Iterable[str]] = ("parquet", "csv"),
    overwrite: bool = True,
    manifest_name: str = "spatial_analytics_manifest.json",
    separate_format_dirs: bool = False,
) -> pd.DataFrame:
    """Write precomputed spatial analytics tables to disk.

    Use this when tables have already been built with
    `build_spatial_analytics_tables()` and need to be inspected before export.

    Args:
        tables: Mapping of table names to DataFrames.
        output_dir: Destination directory.
        formats: Output formats: `parquet`, `csv`, `pickle`/`pkl`.
            Defaults to `parquet` and `csv`. Pass a mapping to configure
            formats per table, for example
            `{"__default__": ("parquet", "csv"), "leg_points_h3": ("parquet",)}`.
        overwrite: If false, fail when a target file already exists.
        manifest_name: JSON manifest file name.
        separate_format_dirs: When true, write each format in a subdirectory
            named after the format (`csv/`, `parquet/`, `pkl/`).

    Returns:
        Manifest with one row per written file.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for table_name, table in tables.items():
        export_formats = _formats_for_table(table_name, formats)
        for format_name in export_formats:
            suffix = "pkl" if format_name == "pickle" else format_name
            format_dir = output_path / suffix if separate_format_dirs else output_path
            format_dir.mkdir(parents=True, exist_ok=True)
            table_path = format_dir / f"{table_name}.{suffix}"
            if table_path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {table_path}")
            _write_table(table, table_path, format_name)
            manifest_rows.append(
                {
                    "table": table_name,
                    "format": format_name,
                    "path": str(table_path),
                    "rows": len(table),
                }
            )

    manifest = pd.DataFrame(manifest_rows, columns=["table", "format", "path", "rows"])
    manifest_path = output_path / manifest_name
    manifest_path.write_text(
        manifest.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def write_spatial_analytics_exports(
    dataset_or_legs: MobilityDataset | pd.DataFrame,
    output_dir: str | Path,
    *,
    h3_resolution: int | Iterable[int] = 9,
    config: ProjectConfig | None = None,
    frequency_group_cols: Iterable[str] | None = None,
    count_dimension_sets: Iterable[Iterable[str]] = DEFAULT_H3_COUNT_DIMENSION_SETS,
    count_metrics: Iterable[str] = DEFAULT_H3_COUNT_METRICS,
    include_count_matrix: bool = True,
    time_slices: Iterable[TimeSlice] | None = None,
    datetime_col: str = "started_at",
    time_slice_col: str = "time_slice",
    timezone: str | None = None,
    sample_distance_m: float | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    chunk_size: int = 250,
    formats: Iterable[str] = ("parquet", "csv"),
    overwrite: bool = True,
) -> pd.DataFrame:
    """Write H3 point, frequency and count-matrix tables for dashboards.

    Args:
        dataset_or_legs: A `MobilityDataset` or a leg table.
        output_dir: Destination directory, for example
            `Data/Output/2-transformed-data/spatial-analytics`.
        h3_resolution: H3 resolution or list of resolutions used for point
            indexing.
        config: Optional project configuration. When provided, its timezone
            and time slices are used unless explicit values are passed.
        frequency_group_cols: Grouping columns for `aggregate_h3_frequencies`.
            Defaults to H3 cell only.
        count_dimension_sets: Dimension sets used for the wide count matrix.
        count_metrics: Count metrics included in the wide count matrix.
        include_count_matrix: Whether to export `h3_count_matrix`.
        time_slices: Optional reusable daily intervals. Defaults to
            `config.time_slices` or package defaults.
        datetime_col: Datetime column used to assign time slices.
        time_slice_col: Name of the time-slice column.
        timezone: Local timezone used for time slices. Defaults to
            `config.timezone` or `Europe/Zurich`.
        sample_distance_m: Optional regular sampling distance in metres before
            H3 indexing.
        parallel: If true, H3 point indexing is split by leg chunks and
            processed with a thread pool.
        max_workers: Optional number of worker threads when `parallel=True`.
        chunk_size: Number of legs per H3 indexing chunk.
        formats: Output formats: `parquet`, `csv`, `pickle`/`pkl`.
            Defaults to `parquet` and `csv`.
        overwrite: If false, fail when a target file already exists.

    Returns:
        Manifest with one row per written file.
    """

    tables = build_spatial_analytics_tables(
        dataset_or_legs,
        h3_resolution=h3_resolution,
        config=config,
        frequency_group_cols=frequency_group_cols,
        count_dimension_sets=count_dimension_sets,
        count_metrics=count_metrics,
        include_count_matrix=include_count_matrix,
        time_slices=time_slices,
        datetime_col=datetime_col,
        time_slice_col=time_slice_col,
        timezone=timezone,
        sample_distance_m=sample_distance_m,
        parallel=parallel,
        max_workers=max_workers,
        chunk_size=chunk_size,
    )
    return write_spatial_analytics_tables(
        tables,
        output_dir,
        formats=formats,
        overwrite=overwrite,
    )


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _prepare_duckdb_table(table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(table).copy()
    if isinstance(table, gpd.GeoDataFrame) and table.geometry.name in table.columns:
        geometry_col = table.geometry.name
        geometry = table.geometry
        out[f"{geometry_col}_wkb"] = geometry.to_wkb()
        out[f"{geometry_col}_crs"] = str(table.crs) if table.crs is not None else None
        if all(isinstance(value, Point) or value is None for value in geometry):
            if "lon" not in out.columns:
                out["lon"] = geometry.x
            if "lat" not in out.columns:
                out["lat"] = geometry.y
        out = out.drop(columns=[geometry_col])
    for column in out.columns:
        series = out[column]
        if series.dtype != "object":
            continue
        values = series.dropna()
        if values.empty:
            continue
        has_boolean = values.map(lambda value: isinstance(value, bool)).any()
        if not has_boolean:
            continue
        normalized = values.map(lambda value: int(value) if isinstance(value, bool) else value)
        if normalized.isin([0, 1]).all():
            out[column] = series.map(
                {
                    True: True,
                    False: False,
                    1: True,
                    0: False,
                },
                na_action="ignore",
            ).astype("boolean")
        else:
            out[column] = series.astype("string")
    return out


def _load_duckdb_spatial(conn, *, require_spatial_extension: bool) -> tuple[bool, str | None]:
    try:
        conn.execute("INSTALL spatial;")
        conn.execute("LOAD spatial;")
        return True, None
    except Exception as install_error:
        try:
            conn.execute("LOAD spatial;")
            return True, None
        except Exception as load_error:
            if require_spatial_extension:
                raise RuntimeError(
                    "DuckDB spatial extension could not be loaded. "
                    "Install/load the extension manually or use GeoParquet/H3 "
                    "tables without spatial SQL functions."
                ) from load_error
            return False, f"{type(load_error).__name__}: {load_error} (install: {install_error})"


def write_duckdb_spatial_database(
    tables: Mapping[str, pd.DataFrame] | MobilityDataset,
    database_path: str | Path,
    *,
    overwrite: bool = True,
    load_spatial: bool = True,
    require_spatial_extension: bool = False,
) -> pd.DataFrame:
    """Write mobility tables to a local DuckDB database.

    GeoDataFrame geometries are stored as WKB columns so that the database can
    be queried even without the spatial extension. When the extension is
    available, companion views ending with `_spatial` are created with a DuckDB
    geometry column.

    Args:
        tables: Mapping of table names to DataFrames, or a `MobilityDataset`.
        database_path: Destination `.duckdb` file.
        overwrite: If true, replace an existing database file.
        load_spatial: Try to install and load DuckDB's spatial extension.
        require_spatial_extension: If true, fail when the spatial extension
            cannot be loaded. Keep false for offline or lightweight use.

    Returns:
        Manifest with table names, row counts and spatial-extension status.

    Raises:
        ImportError: If the optional `duckdb` dependency is missing.
        FileExistsError: If the database exists and `overwrite` is false.
    """

    try:
        import duckdb
    except ImportError as error:
        raise ImportError(
            "DuckDB exports require the optional analytics dependencies. "
            "Install them with `python -m pip install -r requirements-analytics.txt`."
        ) from error

    if isinstance(tables, MobilityDataset):
        from .export import mobility_dataset_tables

        table_mapping = mobility_dataset_tables(tables)
    else:
        table_mapping = dict(tables)

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        path.unlink()

    conn = duckdb.connect(str(path))
    spatial_loaded = False
    spatial_error = None
    if load_spatial:
        spatial_loaded, spatial_error = _load_duckdb_spatial(
            conn,
            require_spatial_extension=require_spatial_extension,
        )

    manifest_rows = []
    try:
        for table_name, table in table_mapping.items():
            prepared = _prepare_duckdb_table(table)
            relation_name = f"_xyt_{table_name}"
            conn.register(relation_name, prepared)
            quoted_table = _quote_identifier(table_name)
            conn.execute(f"CREATE TABLE {quoted_table} AS SELECT * FROM {_quote_identifier(relation_name)};")
            conn.unregister(relation_name)

            spatial_view_created = False
            geometry_wkb_cols = [column for column in prepared.columns if column.endswith("_wkb")]
            if spatial_loaded and geometry_wkb_cols:
                geometry_wkb_col = geometry_wkb_cols[0]
                view_name = f"{table_name}_spatial"
                try:
                    conn.execute(
                        "CREATE VIEW "
                        f"{_quote_identifier(view_name)} AS SELECT *, "
                        f"ST_GeomFromWKB({_quote_identifier(geometry_wkb_col)}) AS geom "
                        f"FROM {quoted_table};"
                    )
                    spatial_view_created = True
                except Exception:
                    spatial_view_created = False

            manifest_rows.append(
                {
                    "table": table_name,
                    "rows": len(table),
                    "columns": len(prepared.columns),
                    "database_path": str(path),
                    "spatial_extension_loaded": spatial_loaded,
                    "spatial_view_created": spatial_view_created,
                    "spatial_error": spatial_error,
                }
            )
    finally:
        conn.close()

    return pd.DataFrame(
        manifest_rows,
        columns=[
            "table",
            "rows",
            "columns",
            "database_path",
            "spatial_extension_loaded",
            "spatial_view_created",
            "spatial_error",
        ],
    )
