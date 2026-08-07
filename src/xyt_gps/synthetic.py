"""Synthetic GPS testset generation from the authorized sample.

The generator is intentionally explicit and parameter-driven. With one
authorized source user, the package cannot learn a defensible population model;
it can, however, bootstrap realistic structured GPS days and inject controlled
tracking anomalies for testing the transformation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from shapely import affinity, wkb
from shapely.geometry.base import BaseGeometry

from .config import Phase
from .models import RawGpsData
from .sample_data import load_sample_gps
from .schema import validate_gps_raw


_STORYLINE_DATE_COLUMNS = (
    "started_at",
    "finished_at",
    "confirmed_at",
    "created_at",
    "updated_at",
    "started_at_in_timezone",
    "finished_at_in_timezone",
    "confirmed_at_in_timezone",
    "created_at_in_timezone",
    "updated_at_in_timezone",
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

_DEFAULT_TRACKING_PROFILES = {
    "complete": {"weight": 0.25, "active_day_probability": 0.92},
    "regular": {"weight": 0.35, "active_day_probability": 0.76},
    "incomplete": {"weight": 0.20, "active_day_probability": 0.52},
    "sparse": {"weight": 0.10, "active_day_probability": 0.30},
    "dropout": {"weight": 0.10, "active_day_probability": 0.78},
}


@dataclass(frozen=True)
class SyntheticExperiment:
    """Experiment window used by the synthetic Declic generator.

    Args:
        experiment_name: Stable experiment identifier added to all generated
            tables, for example `declic-mobility-vague1`.
        phases: Optional experiment phases. Empty phases are allowed for a
            generic GPS testset without analytical phase split.
        motiontag_project_name: Optional provider project name. Defaults to
            `experiment_name`.
    """

    experiment_name: str
    phases: tuple[Phase, ...] = ()
    motiontag_project_name: str | None = None

    @property
    def source_id(self) -> str:
        return self.experiment_name

    @property
    def provider_name(self) -> str:
        return self.motiontag_project_name or self.experiment_name


@dataclass(frozen=True)
class SyntheticAnomalyRates:
    """Controlled anomaly rates injected into synthetic raw tables.

    Args:
        missing_geometry_rate: Share of storyline rows with a missing geometry.
            Keep this below the raw schema tolerance when the generated dataset
            must pass `prepare_mobility_dataset()` without manual cleaning.
        unconfirmed_rate: Share of rows where confirmation timestamps are
            cleared.
        mode_mismatch_rate: Share of track rows where `detected_mode` differs
            from the final `mode`.
        extreme_length_rate: Share of track rows with inflated length values.
    """

    missing_geometry_rate: float = 0.002
    unconfirmed_rate: float = 0.20
    mode_mismatch_rate: float = 0.06
    extreme_length_rate: float = 0.01


@dataclass
class SyntheticGpsDataset:
    """Generated structured GPS testset and companion construction tables.

    Attributes:
        raw: Raw GPS-like tables ready for validation or transformation.
        user_presence: User-level construction table with experiment and phase
            windows.
        generation_manifest: Summary of generated rows, users and active days.
    """

    raw: RawGpsData
    user_presence: pd.DataFrame
    generation_manifest: pd.DataFrame = field(default_factory=pd.DataFrame)

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return all generated tables keyed by export name."""

        tables = {
            "storyline": self.raw.storyline,
            "trips": self.raw.trips,
            "journeys": self.raw.journeys,
            "user_statistics": self.raw.user_statistics
            if self.raw.user_statistics is not None
            else pd.DataFrame(),
            "user_presence": self.user_presence,
            "generation_manifest": self.generation_manifest,
        }
        return tables


def default_declic_synthetic_experiments() -> tuple[SyntheticExperiment, ...]:
    """Return the five Declic experiment windows used for synthetic tests."""

    return (
        SyntheticExperiment(
            "declic-mobility-prefig",
            phases=(
                Phase("Phase1", "2025-04-12", "2025-05-02"),
                Phase("Phase2", "2025-05-05", "2025-06-01"),
                Phase("Phase3", "2025-06-09", "2025-06-29"),
            ),
        ),
        SyntheticExperiment(
            "declic-mobility-vague1",
            phases=(
                Phase("Phase1", "2026-02-23", "2026-03-08"),
                Phase("Phase2", "2026-03-09", "2026-04-05"),
                Phase("Phase3", "2026-04-06", "2026-04-19"),
            ),
        ),
        SyntheticExperiment(
            "declic-mobility-vague2",
            phases=(
                Phase("Phase1", "2026-03-30", "2026-04-12"),
                Phase("Phase2", "2026-04-13", "2026-05-10"),
                Phase("Phase3", "2026-05-11", "2026-05-24"),
            ),
        ),
        SyntheticExperiment(
            "declic-mobility-vague3",
            phases=(
                Phase("Phase1", "2026-04-27", "2026-05-17"),
                Phase("Phase2", "2026-05-18", "2026-06-14"),
                Phase("Phase3", "2026-06-15", "2026-06-28"),
            ),
        ),
        SyntheticExperiment(
            "declic-mobility-ziplo",
            phases=(
                Phase("Phase1", "2026-03-02", "2026-03-15"),
                Phase("Phase2", "2026-03-16", "2026-06-28"),
                Phase("Phase3", "2026-06-29", "2026-07-12"),
            ),
        ),
    )


def _phase_dates(experiment: SyntheticExperiment) -> pd.DataFrame:
    if not experiment.phases:
        raise ValueError(
            f"Synthetic experiment `{experiment.experiment_name}` has no phase window. "
            "Provide phases or use a generic date-window generator."
        )
    rows = []
    for index, phase in enumerate(experiment.phases, start=1):
        for date in pd.date_range(phase.start_date, phase.end_date, freq="D"):
            rows.append(
                {
                    "date": date.date(),
                    "phase": phase.name,
                    "phase_number": index,
                    "phase_start": str(phase.start_date),
                    "phase_end": str(phase.end_date),
                }
            )
    return pd.DataFrame(rows)


def _prepare_sample_storyline(sample_path: str | Path | None) -> pd.DataFrame:
    raw = load_sample_gps(path=sample_path, user_id="template_user", validate=False)
    storyline = raw.storyline.copy()
    storyline["_started_at"] = pd.to_datetime(storyline["started_at"], errors="coerce", utc=True)
    storyline["_finished_at"] = pd.to_datetime(storyline["finished_at"], errors="coerce", utc=True)
    storyline = storyline.dropna(subset=["_started_at", "_finished_at", "geometry"]).copy()
    storyline["_template_date"] = storyline["_started_at"].dt.date
    storyline["_template_started_midnight"] = storyline["_started_at"].dt.normalize()
    storyline["_geometry_object"] = storyline["geometry"].apply(_parse_geometry)
    return storyline


def _parse_geometry(value) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        return value
    return wkb.loads(bytes.fromhex(str(value)))


def _geometry_to_wkb_hex(geometry: BaseGeometry | None) -> str | pd.NA:
    if geometry is None or pd.isna(geometry):
        return pd.NA
    return geometry.wkb_hex


def _trip_mode_from_storyline(value) -> str:
    if pd.isna(value):
        return "Other"
    return _MODE_TO_TRIP_MODE.get(str(value), "Other")


def _choose_profile(rng: np.random.Generator, profiles: dict[str, dict[str, float]]) -> str:
    names = list(profiles)
    weights = np.array([profiles[name]["weight"] for name in names], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(names, p=weights))


def _select_active_dates(
    phase_dates: pd.DataFrame,
    *,
    profile: str,
    profiles: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    dates = phase_dates.copy().reset_index(drop=True)
    probability = float(profiles[profile]["active_day_probability"])
    probabilities = np.full(len(dates), probability)

    if profile == "dropout" and len(dates) > 8:
        dropout_index = int(rng.integers(low=max(2, len(dates) // 3), high=len(dates)))
        probabilities[dropout_index:] = np.minimum(probabilities[dropout_index:], 0.18)
    elif profile == "sparse":
        probabilities = np.where(dates["phase_number"].eq(2), probabilities * 0.9, probabilities)

    active = rng.random(len(dates)) < probabilities
    if not active.any():
        active[int(rng.integers(0, len(active)))] = True
    return dates.loc[active].copy()


def _sample_day_template(
    templates: dict[object, pd.DataFrame],
    template_dates: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    template_date = rng.choice(template_dates)
    return templates[template_date].copy()


def _shift_storyline_day(
    template_day: pd.DataFrame,
    *,
    target_date: object,
    user_id: str,
    experiment: SyntheticExperiment,
    phase_row: pd.Series,
    geometry_transform: dict[str, float],
) -> pd.DataFrame:
    out = template_day.copy()
    target_midnight = pd.Timestamp(target_date, tz="UTC")
    old_started = out["_started_at"]
    new_started = target_midnight + (old_started - out["_template_started_midnight"])

    for column in _STORYLINE_DATE_COLUMNS:
        if column not in out.columns:
            continue
        parsed = pd.to_datetime(out[column], errors="coerce", utc=True, format="mixed")
        shifted = new_started + (parsed - old_started)
        out[column] = shifted.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    out["started_on"] = str(target_date)
    out["user_id"] = user_id
    out["experiment_name"] = experiment.experiment_name
    out["motiontag_project_name"] = experiment.provider_name
    out["phase"] = phase_row["phase"]
    out["phase_number"] = phase_row["phase_number"]
    out["phase_start"] = phase_row["phase_start"]
    out["phase_end"] = phase_row["phase_end"]
    out["geometry"] = out["_geometry_object"].apply(
        lambda geom: _geometry_to_wkb_hex(_transform_geometry(geom, geometry_transform))
    )
    return out.drop(
        columns=[
            "_started_at",
            "_finished_at",
            "_template_date",
            "_template_started_midnight",
            "_geometry_object",
        ],
        errors="ignore",
    )


def _transform_geometry(geometry: BaseGeometry, params: dict[str, float]) -> BaseGeometry:
    shifted = affinity.translate(geometry, xoff=params["xoff"], yoff=params["yoff"])
    scaled = affinity.scale(shifted, xfact=params["scale"], yfact=params["scale"], origin="centroid")
    return affinity.rotate(scaled, params["rotation"], origin="centroid")


def _apply_storyline_ids(storyline: pd.DataFrame) -> pd.DataFrame:
    out = storyline.sort_values(["experiment_name", "user_id", "started_at", "finished_at"]).reset_index(drop=True)
    out["storyline_id"] = [f"syn_sl_{index:09d}" for index in range(len(out))]
    track_mask = out["type"].eq("Track")
    trip_ids = [f"syn_trip_{index:09d}" for index in range(int(track_mask.sum()))]
    out.loc[track_mask, "trip_id"] = trip_ids
    out.loc[~track_mask, "trip_id"] = pd.NA
    return out


def _inject_anomalies(
    storyline: pd.DataFrame,
    *,
    anomaly_rates: SyntheticAnomalyRates,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = storyline.copy()
    n_rows = len(out)
    if n_rows == 0:
        return out

    missing_geometry_count = int(n_rows * anomaly_rates.missing_geometry_rate)
    if missing_geometry_count > 0:
        missing_index = rng.choice(out.index.to_numpy(), size=missing_geometry_count, replace=False)
        out.loc[missing_index, "geometry"] = pd.NA

    confirmation_cols = [column for column in ("confirmed_at", "confirmed_at_in_timezone") if column in out.columns]
    if confirmation_cols and anomaly_rates.unconfirmed_rate > 0:
        unconfirmed_count = int(n_rows * anomaly_rates.unconfirmed_rate)
        if unconfirmed_count > 0:
            unconfirmed_index = rng.choice(out.index.to_numpy(), size=unconfirmed_count, replace=False)
            out.loc[unconfirmed_index, confirmation_cols] = pd.NA

    track_index = out.index[out["type"].eq("Track")].to_numpy()
    if len(track_index) > 0 and anomaly_rates.mode_mismatch_rate > 0 and "detected_mode" in out.columns:
        mismatch_count = int(len(track_index) * anomaly_rates.mode_mismatch_rate)
        if mismatch_count > 0:
            mismatch_index = rng.choice(track_index, size=mismatch_count, replace=False)
            alternative_modes = np.array(["Mode::Walk", "Mode::Bicycle", "Mode::Car", "Mode::Train"])
            out.loc[mismatch_index, "detected_mode"] = rng.choice(alternative_modes, size=mismatch_count)

    if len(track_index) > 0 and anomaly_rates.extreme_length_rate > 0 and "length" in out.columns:
        extreme_count = int(len(track_index) * anomaly_rates.extreme_length_rate)
        if extreme_count > 0:
            extreme_index = rng.choice(track_index, size=extreme_count, replace=False)
            factors = rng.uniform(4.0, 12.0, size=extreme_count)
            out.loc[extreme_index, "length"] = pd.to_numeric(
                out.loc[extreme_index, "length"],
                errors="coerce",
            ).fillna(0) * factors
    return out


def _build_synthetic_trips(storyline: pd.DataFrame) -> pd.DataFrame:
    tracks = storyline[storyline["type"].eq("Track")].copy()
    started = pd.to_datetime(tracks["started_at"], errors="coerce", utc=True)
    finished = pd.to_datetime(tracks["finished_at"], errors="coerce", utc=True)
    out = pd.DataFrame(
        {
            "trip_id": tracks["trip_id"].to_numpy(),
            "user_id": tracks["user_id"].to_numpy(),
            "started_at": tracks["started_at"].to_numpy(),
            "finished_at": tracks["finished_at"].to_numpy(),
            "duration": (finished - started).dt.total_seconds().round().astype("Int64").to_numpy(),
            "distance": pd.to_numeric(tracks["length"], errors="coerce").fillna(0).round().astype("Int64").to_numpy(),
            "tracks_count": 1,
            "purpose": "Other",
            "main_mode": tracks["mode"].apply(_trip_mode_from_storyline).to_numpy(),
            "modes": tracks["mode"].apply(_trip_mode_from_storyline).to_numpy(),
            "created_at": tracks.get("created_at", tracks["started_at"]).to_numpy(),
        }
    )
    for column in (
        "experiment_name",
        "motiontag_project_name",
        "phase",
        "phase_number",
        "phase_start",
        "phase_end",
    ):
        if column in tracks.columns:
            out[column] = tracks[column].to_numpy()
    return out


def _build_synthetic_journeys(trips: pd.DataFrame) -> pd.DataFrame:
    if trips.empty:
        return pd.DataFrame(columns=("journey_id", "user_id", "started_at", "finished_at"))

    out = trips.copy()
    out["_started_at"] = pd.to_datetime(out["started_at"], errors="coerce", utc=True)
    out["_finished_at"] = pd.to_datetime(out["finished_at"], errors="coerce", utc=True)
    out["_date"] = out["_started_at"].dt.date

    group_cols = ["experiment_name", "user_id", "_date"]
    rows = []
    for index, (group_key, group) in enumerate(out.groupby(group_cols, dropna=False), start=1):
        experiment_name, user_id, _date = group_key
        mode = group["main_mode"].mode()
        first_row = group.sort_values("_started_at").iloc[0]
        rows.append(
            {
                "journey_id": f"syn_journey_{index:09d}",
                "user_id": user_id,
                "started_at": group["_started_at"].min().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "finished_at": group["_finished_at"].max().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "duration": int(group["duration"].fillna(0).sum()),
                "distance": int(group["distance"].fillna(0).sum()),
                "tracks_count_without_walks": int(group["main_mode"].ne("Walking").sum()),
                "trips_count": int(len(group)),
                "tracks_count": int(group["tracks_count"].fillna(0).sum()),
                "created_at": group["_started_at"].min().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "purpose": "Other",
                "main_mode": mode.iloc[0] if not mode.empty else "Other",
                "modes": " + ".join(group["main_mode"].dropna().astype(str).unique()),
                "experiment_name": experiment_name,
                "motiontag_project_name": first_row.get("motiontag_project_name"),
                "phase": first_row.get("phase"),
                "phase_number": first_row.get("phase_number"),
                "phase_start": first_row.get("phase_start"),
                "phase_end": first_row.get("phase_end"),
            }
        )
    return pd.DataFrame(rows)


def _build_synthetic_user_statistics(storyline: pd.DataFrame, user_profiles: pd.DataFrame) -> pd.DataFrame:
    temp = storyline.copy()
    temp["_started_at"] = pd.to_datetime(temp["started_at"], errors="coerce", utc=True)
    rows = []
    for user_id, group in temp.groupby("user_id", dropna=False):
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
    stats = pd.DataFrame(rows)
    profile_cols = [
        "user_id",
        "experiment_name",
        "motiontag_project_name",
        "synthetic_tracking_profile",
    ]
    return pd.merge(
        stats,
        user_profiles[profile_cols].drop_duplicates(subset=["user_id"]),
        on="user_id",
        how="left",
    )


def _build_user_presence(
    experiments: Iterable[SyntheticExperiment],
    user_profiles: pd.DataFrame,
    storyline: pd.DataFrame,
    trips: pd.DataFrame,
    user_statistics: pd.DataFrame,
) -> pd.DataFrame:
    active = storyline.copy()
    active["_date"] = pd.to_datetime(active["started_at"], errors="coerce", utc=True).dt.date
    active_days = (
        active.drop_duplicates(subset=["user_id", "phase", "_date"])
        .groupby(["user_id", "phase"], dropna=False)["_date"]
        .count()
        .rename("active_days")
        .reset_index()
    )
    active_by_user_phase = active_days.pivot_table(
        index="user_id",
        columns="phase",
        values="active_days",
        fill_value=0,
        aggfunc="sum",
    )

    rows = []
    for experiment in experiments:
        users = user_profiles.loc[user_profiles["experiment_name"].eq(experiment.experiment_name)].copy()
        for _, user in users.iterrows():
            row = {
                "experiment_name": experiment.experiment_name,
                "motiontag_project_name": experiment.provider_name,
                "user_id": user["user_id"],
                "generated_user": True,
                "synthetic_tracking_profile": user["synthetic_tracking_profile"],
                "in_storyline": user["user_id"] in set(storyline["user_id"]),
                "in_trips": user["user_id"] in set(trips["user_id"]),
                "in_user_statistics": user["user_id"] in set(user_statistics["user_id"]),
                "in_questionnaire": False,
            }
            for index, phase in enumerate(experiment.phases, start=1):
                row[f"phase_{index}_start"] = str(phase.start_date)
                row[f"phase_{index}_end"] = str(phase.end_date)
                row[f"phase_{index}_active_days"] = int(
                    active_by_user_phase.loc[user["user_id"], phase.name]
                    if user["user_id"] in active_by_user_phase.index
                    and phase.name in active_by_user_phase.columns
                    else 0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _build_generation_manifest(dataset: SyntheticGpsDataset) -> pd.DataFrame:
    raw = dataset.raw
    rows = []
    for experiment_name, users in dataset.user_presence.groupby("experiment_name", dropna=False):
        user_ids = set(users["user_id"])
        rows.append(
            {
                "experiment_name": experiment_name,
                "users": len(user_ids),
                "storyline_rows": int(raw.storyline["user_id"].isin(user_ids).sum()),
                "trip_rows": int(raw.trips["user_id"].isin(user_ids).sum()),
                "journey_rows": int(raw.journeys["user_id"].isin(user_ids).sum()),
                "mean_active_days": float(
                    users[[col for col in users.columns if col.endswith("_active_days")]]
                    .sum(axis=1)
                    .mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def generate_synthetic_declic_gps(
    *,
    sample_path: str | Path | None = None,
    experiments: Iterable[SyntheticExperiment] | None = None,
    users_per_experiment: int = 50,
    random_state: int | None = 42,
    anomaly_rates: SyntheticAnomalyRates | None = None,
    validate: bool = True,
) -> SyntheticGpsDataset:
    """Generate a Declic-like synthetic GPS dataset from the sample.

    The method uses bootstrap resampling of observed sample days, shifts them
    to the requested experiment phases, perturbs geometries and injects
    controlled anomalies. It is suited for tests, tutorials and pipeline
    validation. It is not a trained behavioral model.

    Args:
        sample_path: Optional path to the authorized sample pickle. If omitted,
            the package searches the repository layout.
        experiments: Experiment definitions. Defaults to prefiguration, waves
            1-3 and ZIPLO.
        users_per_experiment: Number of synthetic users generated for each
            experiment.
        random_state: Seed used for reproducible generation.
        anomaly_rates: Rates for missing geometry, unconfirmed rows, mode
            mismatch and extreme lengths.
        validate: If true, attach raw schema validation reports to the output.

    Returns:
        A `SyntheticGpsDataset` containing raw GPS-like tables,
        `user_presence` and a generation manifest.
    """

    if users_per_experiment < 1:
        raise ValueError("users_per_experiment must be at least 1.")

    rng = np.random.default_rng(random_state)
    experiment_list = tuple(experiments or default_declic_synthetic_experiments())
    anomaly_rates = anomaly_rates or SyntheticAnomalyRates()

    sample_storyline = _prepare_sample_storyline(sample_path)
    templates = {
        date: group.copy()
        for date, group in sample_storyline.groupby("_template_date", sort=False)
    }
    template_dates = np.array(list(templates.keys()), dtype=object)

    storyline_parts = []
    profile_rows = []
    for experiment in experiment_list:
        dates = _phase_dates(experiment)
        for user_index in range(1, users_per_experiment + 1):
            user_id = f"syn_{experiment.experiment_name.replace('-', '_')}_{user_index:03d}"
            profile = _choose_profile(rng, _DEFAULT_TRACKING_PROFILES)
            geometry_transform = {
                "xoff": float(rng.normal(0, 0.055)),
                "yoff": float(rng.normal(0, 0.035)),
                "scale": float(rng.uniform(0.88, 1.12)),
                "rotation": float(rng.uniform(-8.0, 8.0)),
            }
            active_dates = _select_active_dates(
                dates,
                profile=profile,
                profiles=_DEFAULT_TRACKING_PROFILES,
                rng=rng,
            )
            profile_rows.append(
                {
                    "experiment_name": experiment.experiment_name,
                    "motiontag_project_name": experiment.provider_name,
                    "user_id": user_id,
                    "synthetic_tracking_profile": profile,
                    "synthetic_active_days": len(active_dates),
                }
            )

            for _, phase_row in active_dates.iterrows():
                day = _sample_day_template(templates, template_dates, rng)
                storyline_parts.append(
                    _shift_storyline_day(
                        day,
                        target_date=phase_row["date"],
                        user_id=user_id,
                        experiment=experiment,
                        phase_row=phase_row,
                        geometry_transform=geometry_transform,
                    )
                )

    storyline = pd.concat(storyline_parts, ignore_index=True, sort=False)
    storyline = _apply_storyline_ids(storyline)
    storyline = _inject_anomalies(storyline, anomaly_rates=anomaly_rates, rng=rng)
    trips = _build_synthetic_trips(storyline)
    journeys = _build_synthetic_journeys(trips)
    user_profiles = pd.DataFrame(profile_rows)
    user_statistics = _build_synthetic_user_statistics(storyline, user_profiles)
    user_presence = _build_user_presence(
        experiment_list,
        user_profiles,
        storyline,
        trips,
        user_statistics,
    )

    raw_without_validation = RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=user_statistics,
        paths={"synthetic_source": "authorized_sample_bootstrap"},
    )
    validation = validate_gps_raw(raw_without_validation) if validate else {}
    raw = RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=user_statistics,
        validation=validation,
        paths={"synthetic_source": "authorized_sample_bootstrap"},
    )

    dataset = SyntheticGpsDataset(raw=raw, user_presence=user_presence)
    dataset.generation_manifest = _build_generation_manifest(dataset)
    return dataset


def write_synthetic_gps_dataset(
    dataset: SyntheticGpsDataset,
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("parquet",),
    overwrite: bool = True,
) -> pd.DataFrame:
    """Write a generated synthetic GPS dataset as landing-ready files.

    Args:
        dataset: Synthetic dataset produced by
            `generate_synthetic_declic_gps()`.
        output_dir: Destination directory.
        formats: Formats for large event tables. Supported values are `csv`
            and `parquet`. Defaults to Parquet to avoid very large CSV files.
            Construction tables are always written as CSV.
        overwrite: If false, fail when a target file already exists.

    Returns:
        A manifest with one row per written file.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    normalized_formats = tuple(format_name.lower() for format_name in formats)
    unsupported = sorted(set(normalized_formats) - {"csv", "parquet"})
    if unsupported:
        raise ValueError(f"Unsupported synthetic export format(s): {unsupported}.")

    event_tables = {
        "storyline": dataset.raw.storyline,
        "trips": dataset.raw.trips,
        "journeys": dataset.raw.journeys,
    }
    construction_tables = {
        "user_statistics": dataset.raw.user_statistics
        if dataset.raw.user_statistics is not None
        else pd.DataFrame(),
        "user_presence": dataset.user_presence,
        "generation_manifest": dataset.generation_manifest,
    }
    manifest_rows = []

    for table_name, table in event_tables.items():
        for format_name in normalized_formats:
            path = output_path / f"{table_name}.{format_name}"
            if path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {path}")
            if format_name == "csv":
                table.to_csv(path, index=False)
            elif format_name == "parquet":
                table.to_parquet(path, index=False)
            manifest_rows.append(
                {"table": table_name, "format": format_name, "path": str(path), "rows": len(table)}
            )

    for table_name, table in construction_tables.items():
        path = output_path / f"{table_name}.csv"
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        table.to_csv(path, index=False)
        manifest_rows.append(
            {"table": table_name, "format": "csv", "path": str(path), "rows": len(table)}
        )

    readme_path = output_path / "README.md"
    if readme_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {readme_path}")
    readme_path.write_text(_synthetic_readme(dataset), encoding="utf-8")
    manifest_rows.append(
        {"table": "readme", "format": "md", "path": str(readme_path), "rows": 1}
    )

    manifest = pd.DataFrame(manifest_rows, columns=["table", "format", "path", "rows"])
    manifest.to_csv(output_path / "export_manifest.csv", index=False)
    return manifest


def _synthetic_readme(dataset: SyntheticGpsDataset) -> str:
    experiments = ", ".join(dataset.user_presence["experiment_name"].drop_duplicates().astype(str))
    users = dataset.user_presence["user_id"].nunique()
    rows = len(dataset.raw.storyline)
    return (
        "# Synthetic GPS landing dataset\n\n"
        "This dataset was generated from the authorized xyt_gps sample using "
        "day-level bootstrap resampling, date shifting, geometry perturbation "
        "and controlled tracking anomalies.\n\n"
        f"- Experiments: {experiments}\n"
        f"- Synthetic users: {users}\n"
        f"- Storyline rows: {rows}\n\n"
        "The dataset is intended for package tests, tutorials and workflow "
        "validation. It is not a trained population model and should not be "
        "used for substantive mobility analysis.\n"
    )
