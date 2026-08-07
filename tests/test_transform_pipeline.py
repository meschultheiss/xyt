"""Focused tests for the main mobility preparation pipeline."""

from __future__ import annotations

import logging

from fixtures_core import minimal_project_config, minimal_raw_gps_data
from xyt_gps import prepare_mobility_dataset


def test_prepare_mobility_dataset_builds_linked_tables_and_metadata():
    dataset = prepare_mobility_dataset(minimal_raw_gps_data(), minimal_project_config())

    assert len(dataset.storyline) == 2
    assert len(dataset.staypoints) == 1
    assert len(dataset.legs) == 1
    assert dataset.map_track_trip_journey.loc[0, "leg_id"] == "leg-1"
    assert dataset.map_track_trip_journey.loc[0, "trip_id"] == "trip-1"
    assert dataset.map_track_trip_journey.loc[0, "journey_id"] == "journey-1"
    assert dataset.metadata["project"]["experiment_name"] == "test"
    assert dataset.metadata["computed_steps"]["signal_quality"] is True


def test_prepare_mobility_dataset_logs_main_steps(caplog):
    with caplog.at_level(logging.INFO, logger="xyt_gps.transform"):
        prepare_mobility_dataset(minimal_raw_gps_data(), minimal_project_config())

    messages = [record.getMessage() for record in caplog.records]

    assert any("raw schema validated" in message for message in messages)
    assert any("storyline split" in message for message in messages)
    assert any("relation maps built" in message for message in messages)
    assert any("complete" in message for message in messages)
