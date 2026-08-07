"""Demo sample-data helpers.

The default sample is a personal GPS storyline explicitly authorized for
this package demo. It is loaded only when requested.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import warnings

import pandas as pd

from .models import RawGpsData
from .schema import validate_gps_raw


SAMPLE_GPS_RELATIVE_PATH = Path(
    "package-xyt-gps/examples/sample_data/sample_data_marced_motiontag.pkl"
)

_MODE_TO_TRIP_MODE = {
    "Mode::Airplane": "Airplane",
    "Mode::Bicycle": "Bicycle",
    "Mode::Bikesharing": "Bikesharing",
    "Mode::Boat": "Boat",
    "Mode::Bus": "Bus",
    "Mode::Car": "Car",
    "Mode::Carsharing": "Carsharing",
    "Mode::Ebicycle": "E-Bike",
    "Mode::Ferry": "Ferry",
    "Mode::Hiking": "Hiking",
    "Mode::LightRail": "LightRail",
    "Mode::RegionalTrain": "Regional Train",
    "Mode::Subway": "Subway",
    "Mode::Train": "Train",
    "Mode::Tram": "Tram",
    "Mode::Walk": "Walking",
}


def find_sample_gps_path(start: str | Path | None = None) -> Path:
    """Find the authorized demo pickle from the current repo layout."""

    candidates: list[Path] = []
    if start is not None:
        start_path = Path(start).resolve()
        if start_path.is_file():
            candidates.append(start_path)
        else:
            candidates.append(start_path / SAMPLE_GPS_RELATIVE_PATH)
            candidates.append(start_path / SAMPLE_GPS_RELATIVE_PATH.name)

    cwd = Path.cwd().resolve()
    candidates.extend(parent / SAMPLE_GPS_RELATIVE_PATH for parent in (cwd, *cwd.parents))

    module_path = Path(__file__).resolve()
    candidates.extend(
        parent / SAMPLE_GPS_RELATIVE_PATH for parent in (module_path.parent, *module_path.parents)
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Demo sample not found. Provide an explicit path to "
        "sample_data_marced_motiontag.pkl."
    )


def _read_sample_storyline(path: Path) -> pd.DataFrame:
    with path.open("rb") as file:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r"numpy\.core.*is deprecated.*",
            )
            obj = pickle.load(file)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {path}, got {type(obj)!r}.")
    return obj.copy()


def _duration_seconds(started_at: pd.Series, finished_at: pd.Series) -> pd.Series:
    duration = pd.to_datetime(finished_at, utc=True) - pd.to_datetime(started_at, utc=True)
    return duration.dt.total_seconds().round().astype("Int64")


def _trip_mode_from_storyline(value) -> str:
    if pd.isna(value):
        return "Other"
    return _MODE_TO_TRIP_MODE.get(str(value), "Other")


def _build_demo_trips(storyline: pd.DataFrame) -> pd.DataFrame:
    tracks = storyline[storyline["type"].eq("Track")].copy()
    tracks["_storyline_index"] = tracks.index
    tracks = tracks.sort_values(["user_id", "started_at", "finished_at"]).reset_index(drop=True)
    if tracks.empty:
        raise ValueError("The sample storyline does not contain any Track row.")

    trip_ids = [f"sample_trip_{index:06d}" for index in range(len(tracks))]
    storyline.loc[tracks["_storyline_index"], "trip_id"] = trip_ids

    trips = pd.DataFrame(
        {
            "trip_id": trip_ids,
            "user_id": tracks["user_id"].to_numpy(),
            "started_at": tracks["started_at"].to_numpy(),
            "finished_at": tracks["finished_at"].to_numpy(),
            "duration": _duration_seconds(tracks["started_at"], tracks["finished_at"]).to_numpy(),
            "distance": tracks["length"].fillna(0).round().astype("Int64").to_numpy(),
            "tracks_count": 1,
            "purpose": "Other",
            "main_mode": tracks["mode"].apply(_trip_mode_from_storyline).to_numpy(),
            "modes": tracks["mode"].apply(_trip_mode_from_storyline).to_numpy(),
        }
    )
    return storyline, trips


def _build_demo_journeys(trips: pd.DataFrame) -> pd.DataFrame:
    out = trips.copy()
    out["_started_at"] = pd.to_datetime(out["started_at"], utc=True)
    out["_finished_at"] = pd.to_datetime(out["finished_at"], utc=True)
    out["_date"] = out["_started_at"].dt.date

    rows = []
    for index, ((user_id, date), group) in enumerate(out.groupby(["user_id", "_date"]), start=1):
        mode = group["main_mode"].mode()
        rows.append(
            {
                "journey_id": f"sample_journey_{index:06d}",
                "user_id": user_id,
                "started_at": group["_started_at"].min().isoformat(),
                "finished_at": group["_finished_at"].max().isoformat(),
                "duration": int(group["duration"].fillna(0).sum()),
                "distance": int(group["distance"].fillna(0).sum()),
                "tracks_count_without_walks": int(group["main_mode"].ne("Walking").sum()),
                "trips_count": int(len(group)),
                "tracks_count": int(group["tracks_count"].fillna(0).sum()),
                "purpose": "Other",
                "main_mode": mode.iloc[0] if not mode.empty else "Other",
                "modes": " + ".join(group["main_mode"].dropna().astype(str).unique()),
            }
        )
    return pd.DataFrame(rows)


def _build_demo_user_statistics(storyline: pd.DataFrame) -> pd.DataFrame:
    out = storyline.copy()
    out["_started_at"] = pd.to_datetime(out["started_at"], utc=True)
    rows = []
    for user_id, group in out.groupby("user_id"):
        dates = group["_started_at"].dt.date.drop_duplicates().sort_values()
        first = pd.Timestamp(dates.iloc[0])
        latest = pd.Timestamp(dates.iloc[-1])
        movement_dates = group.loc[
            group.get("type", pd.Series("Track", index=group.index))
            .astype(str)
            .str.lower()
            .str.endswith("track"),
            "_started_at",
        ].dt.date.nunique()
        tracked_days = int(len(dates))
        active_days = int(movement_dates)
        rows.append(
            {
                "user_id": user_id,
                "first_tracked_at": first.isoformat(),
                "latest_tracked_at": latest.isoformat(),
                "days_in_range_count": int((latest - first).days + 1),
                "tracked_days_count": tracked_days,
                "active_days_count": active_days,
                "zero_movement_tracked_days_count": tracked_days - active_days,
                "untracked_days_count": int((latest - first).days + 1 - tracked_days),
                "inactive_days_count": int((latest - first).days + 1 - tracked_days),
            }
        )
    return pd.DataFrame(rows)


def load_sample_gps(
    path: str | Path | None = None,
    *,
    user_id: str = "sample_user",
    max_rows: int | None = None,
    validate: bool = True,
) -> RawGpsData:
    """Load the authorized personal demo storyline as a structured GPS raw dataset.

    The source pickle contains a storyline table only. The function pseudonymizes
    `user_id` by default and derives minimal `Trips`, `Journeys` and
    `UserStatistics` tables so that tutorial transformations can run without a
    full raw GPS export.
    """

    source_path = Path(path) if path is not None else find_sample_gps_path()
    storyline = _read_sample_storyline(source_path)
    if max_rows is not None:
        storyline = storyline.head(max_rows).copy()

    if "user_id" in storyline.columns:
        storyline["user_id"] = user_id
    if "storyline_id" not in storyline.columns:
        if "id" in storyline.columns:
            storyline["storyline_id"] = storyline["id"]
        else:
            storyline["storyline_id"] = [f"sample_storyline_{index:06d}" for index in range(len(storyline))]
    storyline = storyline.drop(columns=["id"], errors="ignore")

    storyline, trips = _build_demo_trips(storyline)
    journeys = _build_demo_journeys(trips)
    user_statistics = _build_demo_user_statistics(storyline)

    raw_without_validation = RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=user_statistics,
        paths={"sample_storyline": source_path},
    )
    validation = validate_gps_raw(raw_without_validation) if validate else {}
    return RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=user_statistics,
        validation=validation,
        paths={"sample_storyline": source_path},
    )
