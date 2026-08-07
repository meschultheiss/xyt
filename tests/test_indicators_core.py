"""Focused tests for mobility indicators."""

from __future__ import annotations

from fixtures_core import minimal_project_config, minimal_raw_gps_data
from xyt_gps import compute_mobility_indicators, prepare_mobility_dataset


def test_compute_mobility_indicators_returns_expected_core_tables():
    dataset = prepare_mobility_dataset(minimal_raw_gps_data(), minimal_project_config())

    result = compute_mobility_indicators(dataset, mode_col="mode_niv1")

    assert set(result.tables()) == {"person_day", "person_phase", "population"}
    assert result.metadata["n_users"] == 1
    assert result.metadata["include_zero_days"] is True


def test_compute_mobility_indicators_uses_zero_day_calendar():
    dataset = prepare_mobility_dataset(minimal_raw_gps_data(), minimal_project_config())

    result = compute_mobility_indicators(dataset, mode_col="mode_niv1")
    walk_population = result.population.loc[result.population["mode"].eq("Marche")].iloc[0]

    assert walk_population["distance_km_per_day"] == 1.2
    assert walk_population["travel_time_min_per_day"] == 30
    assert walk_population["trip_count_per_day"] == 1
