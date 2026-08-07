"""Input helpers for structured GPS exports and project side tables."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .models import RawGpsData
from .schema import validate_gps_raw


@dataclass(frozen=True)
class GpsExportPaths:
    """Resolved file paths for one structured GPS export period.

    Attributes:
        storyline: Path to the storyline CSV.
        trips: Path to the trips CSV.
        journeys: Path to the journeys CSV.
        user_statistics: Optional path to the user statistics CSV.
    """

    storyline: Path
    trips: Path
    journeys: Path
    user_statistics: Path | None = None


@dataclass(frozen=True)
class RawSampleConfig:
    """Sampling options for loading large raw GPS exports.

    Use `RawSampleConfig.by_users(n)` when checking a large export: it keeps
    all rows for a small number of selected users and preserves the relation
    between storyline, trips, journeys and user statistics. Use
    `RawSampleConfig.random_rows(n)` only for quick schema inspection.

    Attributes:
        mode: Sampling strategy, either `users` or `rows`.
        n: Number of users or rows to keep.
        random_state: Optional random seed.
        chunksize: CSV chunk size for user-based sampling.
        user_id_column: User identifier column in storyline, trips and
            journeys.
        user_statistics_id_column: User identifier column in user statistics.
        storyline_type_column: Column used to identify track rows.
        track_type_value: Value used for track rows in storyline.
    """

    mode: str
    n: int
    random_state: int | None = 42
    chunksize: int = 100_000
    user_id_column: str = "user_id"
    user_statistics_id_column: str = "user_id"
    storyline_type_column: str = "type"
    track_type_value: str = "Track"

    def __post_init__(self):
        if self.mode not in {"users", "rows"}:
            raise ValueError("RawSampleConfig.mode must be 'users' or 'rows'.")
        if self.n <= 0:
            raise ValueError("RawSampleConfig.n must be greater than 0.")
        if self.chunksize <= 0:
            raise ValueError("RawSampleConfig.chunksize must be greater than 0.")

    @classmethod
    def by_users(
        cls,
        n: int,
        *,
        random_state: int | None = 42,
        chunksize: int = 100_000,
        user_id_column: str = "user_id",
        user_statistics_id_column: str = "user_id",
    ) -> "RawSampleConfig":
        """Keep all raw rows for `n` randomly selected users."""

        return cls(
            mode="users",
            n=n,
            random_state=random_state,
            chunksize=chunksize,
            user_id_column=user_id_column,
            user_statistics_id_column=user_statistics_id_column,
        )

    @classmethod
    def random_rows(
        cls,
        n: int,
        *,
        random_state: int | None = 42,
        chunksize: int = 100_000,
        user_id_column: str = "user_id",
        user_statistics_id_column: str = "user_id",
    ) -> "RawSampleConfig":
        """Keep `n` random rows per table; storyline rows prefer Track rows."""

        return cls(
            mode="rows",
            n=n,
            random_state=random_state,
            chunksize=chunksize,
            user_id_column=user_id_column,
            user_statistics_id_column=user_statistics_id_column,
        )


SOURCE_METADATA_COLUMNS = (
    "experiment_name",
    "motiontag_project_name",
    "period",
)


def _motiontag_file(raw_path: Path, prefix: str, project_name: str, period: str) -> Path:
    return raw_path / f"{prefix}.{project_name}.{period}.csv"


def _require_motiontag_naming(config: ProjectConfig) -> tuple[str, str]:
    missing = []
    if not config.motiontag_project_name:
        missing.append("motiontag_project_name")
    if not config.period:
        missing.append("period")
    if missing:
        raise ValueError(
            "Cannot infer provider file names without "
            + ", ".join(missing)
            + ". Provide these ProjectConfig fields, or load raw tables manually into RawGpsData."
        )
    return str(config.motiontag_project_name), str(config.period)


def infer_gps_export_paths(config: ProjectConfig, *, must_exist: bool = True) -> GpsExportPaths:
    """Infer expected raw file paths from project parameters.

    Args:
        config: Project configuration containing `raw_data_dir`,
            `motiontag_project_name`, `period` and accepted storyline
            prefixes.
        must_exist: When true, raise an error if any expected file is absent.

    Returns:
        A `GpsExportPaths` object with resolved CSV paths.

    Raises:
        FileNotFoundError: If `must_exist` is true and at least one expected
            CSV file is missing.
    """

    raw_path = config.raw_path
    motiontag_project_name, period = _require_motiontag_naming(config)
    storyline_path = None
    for prefix in config.storyline_prefix_candidates:
        candidate = _motiontag_file(raw_path, prefix, motiontag_project_name, period)
        if candidate.exists():
            storyline_path = candidate
            break
        if storyline_path is None:
            storyline_path = candidate

    if storyline_path is None:
        raise ValueError("ProjectConfig.storyline_prefix_candidates must contain at least one prefix.")
    paths = GpsExportPaths(
        storyline=storyline_path,
        trips=_motiontag_file(raw_path, "Trips", motiontag_project_name, period),
        journeys=_motiontag_file(raw_path, "Journeys", motiontag_project_name, period),
        user_statistics=_motiontag_file(
            raw_path,
            "UserStatistics",
            motiontag_project_name,
            period,
        ),
    )

    if must_exist:
        missing = [
            str(path)
            for path in (paths.storyline, paths.trips, paths.journeys, paths.user_statistics)
            if path is not None and not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing raw GPS file(s): " + ", ".join(missing))

    return paths


def source_id_for_config(config: ProjectConfig) -> str:
    """Build a stable source id for multi-project imports.

    The source id is used to namespace identifiers when several projects or
    periods are loaded in one pass.
    """

    source_name = config.experiment_name or config.motiontag_project_name or "source"
    return f"{source_name}__{config.period}" if config.period else str(source_name)


def _namespace_value(value, source_id: str):
    if pd.isna(value):
        return value
    text = str(value)
    if text.startswith(f"{source_id}::"):
        return text
    return f"{source_id}::{text}"


def _add_source_metadata(
    df: pd.DataFrame | None,
    config: ProjectConfig,
    *,
    source_id: str,
    id_columns: tuple[str, ...],
    namespace_ids: bool,
) -> pd.DataFrame | None:
    if df is None:
        return None

    out = df.copy()
    if config.experiment_name:
        out["experiment_name"] = config.experiment_name
    if config.motiontag_project_name:
        out["motiontag_project_name"] = config.motiontag_project_name
    if config.period:
        out["period"] = config.period

    if namespace_ids:
        for column in id_columns:
            if column in out.columns:
                raw_column = f"raw_{column}"
                if raw_column not in out.columns:
                    out[raw_column] = out[column]
                out[column] = out[column].apply(lambda value: _namespace_value(value, source_id))
    return out


def _sample_frame(df: pd.DataFrame | None, sample: RawSampleConfig) -> pd.DataFrame | None:
    if df is None:
        return None
    if df.empty or len(df) <= sample.n:
        return df.copy().reset_index(drop=True)
    return df.sample(n=sample.n, random_state=sample.random_state).sort_index().reset_index(drop=True)


def _sample_storyline_rows(storyline: pd.DataFrame, sample: RawSampleConfig) -> pd.DataFrame:
    if sample.storyline_type_column not in storyline.columns:
        return _sample_frame(storyline, sample)
    tracks = storyline[storyline[sample.storyline_type_column].eq(sample.track_type_value)]
    if tracks.empty:
        return _sample_frame(storyline, sample)
    return _sample_frame(tracks, sample)


def _filter_by_values(
    df: pd.DataFrame | None,
    column: str,
    values,
) -> pd.DataFrame | None:
    if df is None:
        return None
    if column not in df.columns:
        return df.copy().reset_index(drop=True)
    return df[df[column].isin(values)].copy().reset_index(drop=True)


def _read_csv_header(path: Path, sep: str) -> pd.Index:
    return pd.read_csv(path, sep=sep, nrows=0).columns


def _selected_users_from_storyline(
    path: Path,
    *,
    sep: str,
    sample: RawSampleConfig,
) -> set:
    try:
        chunks = pd.read_csv(
            path,
            sep=sep,
            usecols=[sample.user_id_column],
            chunksize=sample.chunksize,
        )
        seen_users = set()
        user_values = []
        for chunk in chunks:
            for value in chunk[sample.user_id_column].dropna().unique():
                if value not in seen_users:
                    seen_users.add(value)
                    user_values.append(value)
        users = pd.Series(user_values, dtype="object")
    except ValueError as error:
        columns = _read_csv_header(path, sep)
        if sample.user_id_column not in columns:
            raise KeyError(f"Column not found in storyline: {sample.user_id_column}") from error
        raise

    if users.empty:
        raise ValueError("Cannot sample by users: no user id found in storyline.")
    selected = users.sample(
        n=min(sample.n, len(users)),
        random_state=sample.random_state,
    )
    return set(selected)


def _read_csv_filtered_by_values(
    path: Path,
    *,
    sep: str,
    column: str,
    values: set,
    chunksize: int,
) -> pd.DataFrame:
    parts = []
    try:
        chunks = pd.read_csv(path, sep=sep, chunksize=chunksize)
        for chunk in chunks:
            if column not in chunk.columns:
                raise KeyError(f"Column not found in {path.name}: {column}")
            selected = chunk[chunk[column].isin(values)]
            if not selected.empty:
                parts.append(selected)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if parts:
        return pd.concat(parts, ignore_index=True, sort=False).reset_index(drop=True)
    return pd.read_csv(path, sep=sep, nrows=0)


def _read_user_statistics_filtered_by_users(
    path: Path | None,
    *,
    sep: str,
    sample: RawSampleConfig,
    selected_users: set,
) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None

    columns = _read_csv_header(path, sep)
    if sample.user_statistics_id_column not in columns:
        raise KeyError(f"Column not found in {path.name}: {sample.user_statistics_id_column}")
    column = sample.user_statistics_id_column

    return _read_csv_filtered_by_values(
        path,
        sep=sep,
        column=column,
        values=selected_users,
        chunksize=sample.chunksize,
    )


def _sample_by_users(raw: RawGpsData, sample: RawSampleConfig) -> RawGpsData:
    if sample.user_id_column not in raw.storyline.columns:
        raise KeyError(f"Column not found in storyline: {sample.user_id_column}")

    users = raw.storyline[sample.user_id_column].dropna().drop_duplicates()
    if users.empty:
        raise ValueError("Cannot sample by users: no user id found in storyline.")
    selected = users.sample(
        n=min(sample.n, len(users)),
        random_state=sample.random_state,
    )
    selected_values = set(selected)

    storyline = _filter_by_values(raw.storyline, sample.user_id_column, selected_values)
    trips = _filter_by_values(raw.trips, sample.user_id_column, selected_values)
    journeys = _filter_by_values(raw.journeys, sample.user_id_column, selected_values)

    user_statistics = raw.user_statistics
    if user_statistics is not None:
        if sample.user_statistics_id_column not in user_statistics.columns:
            raise KeyError(f"Column not found in user_statistics: {sample.user_statistics_id_column}")
        user_statistics = _filter_by_values(
            user_statistics,
            sample.user_statistics_id_column,
            selected_values,
        )

    return RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=user_statistics,
        paths=raw.paths,
    )


def _sample_by_rows(raw: RawGpsData, sample: RawSampleConfig) -> RawGpsData:
    return RawGpsData(
        storyline=_sample_storyline_rows(raw.storyline, sample),
        trips=_sample_frame(raw.trips, sample),
        journeys=_sample_frame(raw.journeys, sample),
        user_statistics=_sample_frame(raw.user_statistics, sample),
        paths=raw.paths,
    )


def sample_raw_gps_data(
    raw: RawGpsData,
    sample: RawSampleConfig | None,
) -> RawGpsData:
    """Apply a raw-data sample before validation or transformation.

    Args:
        raw: Raw GPS tables already loaded in memory.
        sample: Sampling configuration. If `None`, the raw object is returned
            unchanged.

    Returns:
        A new `RawGpsData` object containing sampled tables.

    Raises:
        KeyError: If user-based sampling is requested and the user id column
            is missing.
        ValueError: If the sampling mode is unsupported.
    """

    if sample is None:
        return raw
    if sample.mode == "users":
        return _sample_by_users(raw, sample)
    if sample.mode == "rows":
        return _sample_by_rows(raw, sample)
    raise ValueError("Unsupported sample mode.")


def _load_gps_export_by_user_sample(
    paths: GpsExportPaths,
    config: ProjectConfig,
    sample: RawSampleConfig,
) -> RawGpsData:
    selected_users = _selected_users_from_storyline(
        paths.storyline,
        sep=config.csv_sep,
        sample=sample,
    )
    storyline = _read_csv_filtered_by_values(
        paths.storyline,
        sep=config.csv_sep,
        column=sample.user_id_column,
        values=selected_users,
        chunksize=sample.chunksize,
    )
    trips = _read_csv_filtered_by_values(
        paths.trips,
        sep=config.csv_sep,
        column=sample.user_id_column,
        values=selected_users,
        chunksize=sample.chunksize,
    )
    journeys = _read_csv_filtered_by_values(
        paths.journeys,
        sep=config.csv_sep,
        column=sample.user_id_column,
        values=selected_users,
        chunksize=sample.chunksize,
    )
    user_statistics = _read_user_statistics_filtered_by_users(
        paths.user_statistics,
        sep=config.csv_sep,
        sample=sample,
        selected_users=selected_users,
    )
    return RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=user_statistics,
        paths=paths,
    )


def _with_validation(raw: RawGpsData, *, validate: bool) -> RawGpsData:
    validation = validate_gps_raw(raw) if validate else {}
    return RawGpsData(
        storyline=raw.storyline,
        trips=raw.trips,
        journeys=raw.journeys,
        user_statistics=raw.user_statistics,
        validation=validation,
        paths=raw.paths,
    )


def load_gps_export(
    config: ProjectConfig,
    *,
    sample: RawSampleConfig | None = None,
    validate: bool = True,
    must_exist: bool = True,
) -> RawGpsData:
    """Load raw GPS CSV exports without transforming their schema.

    Args:
        config: Project configuration used to infer file paths and CSV
            separator.
        sample: Optional sampling strategy. `RawSampleConfig.by_users` reads
            CSV files by chunks and keeps all records for selected users.
        validate: When true, attach schema-validation reports to the returned
            object.
        must_exist: When true, fail if expected CSV files are missing.

    Returns:
        Raw storyline, trips, journeys and optional user-statistics tables.

    Raises:
        FileNotFoundError: If required files are missing.
        KeyError: If user-based sampling references a missing user column.
    """

    paths = infer_gps_export_paths(config, must_exist=must_exist)
    if sample is not None and sample.mode == "users":
        raw = _load_gps_export_by_user_sample(paths, config, sample)
        return _with_validation(raw, validate=validate)

    storyline = pd.read_csv(paths.storyline, sep=config.csv_sep)
    trips = pd.read_csv(paths.trips, sep=config.csv_sep)
    journeys = pd.read_csv(paths.journeys, sep=config.csv_sep)

    user_statistics = None
    if paths.user_statistics is not None and paths.user_statistics.exists():
        user_statistics = pd.read_csv(paths.user_statistics, sep=config.csv_sep)

    raw = RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=user_statistics,
        paths=paths,
    )
    sampled = sample_raw_gps_data(raw, sample)
    return _with_validation(sampled, validate=validate)


def load_gps_source(
    config: ProjectConfig,
    *,
    source_id: str | None = None,
    sample: RawSampleConfig | None = None,
    validate: bool = True,
    must_exist: bool = True,
    namespace_ids: bool = True,
) -> RawGpsData:
    """Load one GPS source and add project metadata columns.

    Args:
        config: Project configuration for one project or period.
        source_id: Optional source identifier. If absent, it is built from
            the configuration.
        sample: Optional raw sampling strategy.
        validate: Whether to attach validation reports.
        must_exist: Whether expected files must exist.
        namespace_ids: When true, prefix ids with `source_id` to avoid
            collisions in multi-source imports.

    Returns:
        Raw GPS tables with project metadata and optional `raw_*` identifier
        columns. `source_id` is used for optional identifier namespacing but is
        not added as a data column.
    """

    source_id = source_id or source_id_for_config(config)
    raw = load_gps_export(config, sample=sample, validate=False, must_exist=must_exist)
    storyline = _add_source_metadata(
        raw.storyline,
        config,
        source_id=source_id,
        id_columns=("storyline_id", "user_id", "trip_id"),
        namespace_ids=namespace_ids,
    )
    trips = _add_source_metadata(
        raw.trips,
        config,
        source_id=source_id,
        id_columns=("trip_id", "user_id"),
        namespace_ids=namespace_ids,
    )
    journeys = _add_source_metadata(
        raw.journeys,
        config,
        source_id=source_id,
        id_columns=("journey_id", "user_id"),
        namespace_ids=namespace_ids,
    )
    user_statistics = _add_source_metadata(
        raw.user_statistics,
        config,
        source_id=source_id,
        id_columns=("user_id",),
        namespace_ids=namespace_ids,
    )
    source_raw = RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=user_statistics,
        paths=raw.paths,
    )
    return _with_validation(source_raw, validate=validate)


def concat_raw_gps_data(
    raws: Iterable[RawGpsData],
    *,
    validate: bool = True,
) -> RawGpsData:
    """Concatenate several already-loaded raw GPS datasets.

    Args:
        raws: Raw datasets to concatenate. They should already contain source
            metadata if they come from different projects or periods.
        validate: Whether to validate the concatenated raw tables.

    Returns:
        One `RawGpsData` object containing all rows.

    Raises:
        ValueError: If `raws` is empty.
    """

    raw_list = list(raws)
    if not raw_list:
        raise ValueError("At least one RawGpsData object is required.")

    user_statistics_parts = [
        raw.user_statistics for raw in raw_list if raw.user_statistics is not None
    ]
    raw = RawGpsData(
        storyline=pd.concat([raw.storyline for raw in raw_list], ignore_index=True, sort=False),
        trips=pd.concat([raw.trips for raw in raw_list], ignore_index=True, sort=False),
        journeys=pd.concat([raw.journeys for raw in raw_list], ignore_index=True, sort=False),
        user_statistics=(
            pd.concat(user_statistics_parts, ignore_index=True, sort=False)
            if user_statistics_parts
            else None
        ),
        paths=[raw.paths for raw in raw_list],
    )
    return _with_validation(raw, validate=validate)


def load_gps_sources(
    configs: Iterable[ProjectConfig],
    *,
    sample: RawSampleConfig | None = None,
    validate: bool = True,
    must_exist: bool = True,
    namespace_ids: bool = True,
) -> RawGpsData:
    """Load and concatenate raw GPS exports from several sources.

    Args:
        configs: Project configurations, one per project or period.
        sample: Optional sampling strategy applied to each source.
        validate: Whether to attach validation reports.
        must_exist: Whether expected files must exist.
        namespace_ids: Whether to prefix identifiers by source id.

    Returns:
        A concatenated `RawGpsData` object ready for transformation.
    """

    raws = [
        load_gps_source(
            config,
            sample=sample,
            validate=validate,
            must_exist=must_exist,
            namespace_ids=namespace_ids,
        )
        for config in configs
    ]
    return concat_raw_gps_data(raws, validate=validate)


def load_sociodemo(path: str | Path, *, user_id_column: str = "Id") -> pd.DataFrame:
    """Load a sociodemographic side table and standardize the user id column.

    Args:
        path: CSV or Excel file path.
        user_id_column: Column to rename to `user_id`.

    Returns:
        A pandas table that can be passed to `prepare_mobility_dataset`.
    """

    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if user_id_column in df.columns and user_id_column != "user_id":
        df = df.rename(columns={user_id_column: "user_id"})
    return df
