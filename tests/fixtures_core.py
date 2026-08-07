"""Small GPS fixtures shared by focused unit tests."""

from __future__ import annotations

import pandas as pd
from shapely.geometry import LineString, Point

from xyt_gps import Phase, ProjectConfig, RawGpsData, TrackingThresholds


def minimal_project_config() -> ProjectConfig:
    return ProjectConfig(
        experiment_name="test",
        motiontag_project_name="Test",
        period="2025-01-01--2025-01-02",
        raw_data_dir=".",
        phases=(Phase("Phase1", "2025-01-01", "2025-01-01"),),
        tracking_thresholds=TrackingThresholds(
            min_days_by_phase={"Phase1": 1},
            min_total_tracked_days=1,
            round_to_full_weeks=False,
        ),
    )


def minimal_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    track = LineString([(6.1, 46.2), (6.2, 46.25)])
    stay = Point(6.1, 46.2)
    storyline = pd.DataFrame(
        [
            {
                "storyline_id": "stay-1",
                "user_id": "u1",
                "type": "Stay",
                "started_at": "2025-01-01T08:00:00Z",
                "finished_at": "2025-01-01T09:00:00Z",
                "started_on": "2025-01-01",
                "started_at_in_timezone": "2025-01-01T09:00:00+01:00",
                "finished_at_in_timezone": "2025-01-01T10:00:00+01:00",
                "geometry": stay.wkb_hex,
                "purpose": "home",
                "mode": None,
                "detected_mode": None,
                "length": None,
            },
            {
                "storyline_id": "leg-1",
                "user_id": "u1",
                "type": "Track",
                "started_at": "2025-01-01T09:00:00Z",
                "finished_at": "2025-01-01T09:30:00Z",
                "started_on": "2025-01-01",
                "started_at_in_timezone": "2025-01-01T10:00:00+01:00",
                "finished_at_in_timezone": "2025-01-01T10:30:00+01:00",
                "geometry": track.wkb_hex,
                "purpose": None,
                "mode": "Mode::Walk",
                "detected_mode": "Mode::Walk",
                "length": 1200,
                "trip_id": "trip-1",
            },
        ]
    )
    trips = pd.DataFrame(
        [
            {
                "trip_id": "trip-1",
                "user_id": "u1",
                "started_at": "2025-01-01T09:00:00Z",
                "finished_at": "2025-01-01T09:30:00Z",
                "duration": 1800,
                "distance": 1200,
                "tracks_count": 1,
                "purpose": "At home",
                "main_mode": "Walking",
                "modes": "Walking",
            }
        ]
    )
    journeys = pd.DataFrame(
        [
            {
                "journey_id": "journey-1",
                "user_id": "u1",
                "started_at": "2025-01-01T08:55:00Z",
                "finished_at": "2025-01-01T09:45:00Z",
                "duration": 3000,
                "distance": 1200,
                "tracks_count_without_walks": 0,
                "trips_count": 1,
                "tracks_count": 1,
                "purpose": "At home",
                "main_mode": "Walking",
                "modes": "Walking",
            }
        ]
    )
    return storyline, trips, journeys


def minimal_raw_gps_data() -> RawGpsData:
    storyline, trips, journeys = minimal_raw_tables()
    return RawGpsData(storyline=storyline, trips=trips, journeys=journeys)
