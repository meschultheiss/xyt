"""Focused tests for relation tables."""

from __future__ import annotations

import pandas as pd

from xyt_gps import build_track_trip_journey_map


def test_build_track_trip_journey_map_matches_without_existing_trip_id():
    legs = pd.DataFrame(
        [
            {
                "leg_id": "leg-1",
                "user_id": "u1",
                "started_at": pd.Timestamp("2025-01-01T09:10:00Z"),
                "finished_at": pd.Timestamp("2025-01-01T09:20:00Z"),
            }
        ]
    )
    trips = pd.DataFrame(
        [
            {
                "trip_id": "trip-1",
                "user_id": "u1",
                "started_at": pd.Timestamp("2025-01-01T09:00:00Z"),
                "finished_at": pd.Timestamp("2025-01-01T09:30:00Z"),
            }
        ]
    )
    journeys = pd.DataFrame(
        [
            {
                "journey_id": "journey-1",
                "user_id": "u1",
                "started_at": pd.Timestamp("2025-01-01T08:55:00Z"),
                "finished_at": pd.Timestamp("2025-01-01T09:45:00Z"),
            }
        ]
    )

    mapping = build_track_trip_journey_map(legs, trips, journeys, tolerance="0s")

    assert mapping.to_dict("records") == [
        {"leg_id": "leg-1", "trip_id": "trip-1", "journey_id": "journey-1"}
    ]


def test_build_track_trip_journey_map_keeps_unmatched_legs_visible():
    legs = pd.DataFrame(
        [
            {
                "leg_id": "leg-1",
                "user_id": "u1",
                "started_at": pd.Timestamp("2025-01-01T12:00:00Z"),
                "finished_at": pd.Timestamp("2025-01-01T12:10:00Z"),
            }
        ]
    )
    trips = pd.DataFrame(
        [
            {
                "trip_id": "trip-1",
                "user_id": "u1",
                "started_at": pd.Timestamp("2025-01-01T09:00:00Z"),
                "finished_at": pd.Timestamp("2025-01-01T09:30:00Z"),
            }
        ]
    )
    journeys = pd.DataFrame(
        [
            {
                "journey_id": "journey-1",
                "user_id": "u1",
                "started_at": pd.Timestamp("2025-01-01T08:55:00Z"),
                "finished_at": pd.Timestamp("2025-01-01T09:45:00Z"),
            }
        ]
    )

    mapping = build_track_trip_journey_map(legs, trips, journeys)

    assert mapping.loc[0, "leg_id"] == "leg-1"
    assert pd.isna(mapping.loc[0, "trip_id"])
    assert pd.isna(mapping.loc[0, "journey_id"])
