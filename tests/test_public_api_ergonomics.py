"""Tests for the compact public API promoted before PyPI publication."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import xyt_gps as xyt
from fixtures_core import minimal_project_config, minimal_raw_gps_data


def _minimal_dataset():
    return xyt.prepare_mobility_dataset(
        minimal_raw_gps_data(),
        minimal_project_config(),
        resample_missing_days=False,
    )


def test_recommended_public_api_symbols_are_importable():
    recommended_symbols = {
        "ProjectConfig",
        "Phase",
        "TimeSlice",
        "RawGpsData",
        "load_gps_export",
        "check_raw_import_columns",
        "prepare_mobility_dataset",
        "mobility_dataset_tables",
        "build_user_selection_table",
        "build_origin_destination_zone_correspondence",
        "filter_mobility_dataset_by_users",
        "compute_mobility_indicators",
        "build_relative_time_index",
        "export_clean_dataset",
        "plot_gps_traces",
        "plot_participation_heatmap",
        "plot_indicator_bars",
        "plot_mobility_motif_graphs",
        "available_reference_tables",
        "load_reference_table",
    }

    missing = [name for name in sorted(recommended_symbols) if not hasattr(xyt, name)]

    assert missing == []


def test_reference_tables_are_public_and_can_be_overridden(tmp_path):
    assert xyt.available_reference_tables() == (
        "co2_factors",
        "metabolic_equivalent_tasks",
        "occupancy_rates",
    )
    with pytest.raises(ValueError, match="explicit CSV path"):
        xyt.load_reference_table("co2_factors")
    with pytest.raises(ValueError, match="explicit CO2OccupancyConfig"):
        xyt.add_co2_occupancy_metrics(pd.DataFrame())

    legs = pd.DataFrame(
        [
            {
                "leg_id": "leg-1",
                "user_id": "u1",
                "mode": "Mode::Car",
                "mode_niv1": "Marche",
                "length": 1000,
                "duration": 600,
            }
        ]
    )

    custom_co2_path = tmp_path / "co2_factors.csv"
    custom_co2_path.write_text(
        "\n".join(
            [
                "mode,co2_factor_g_per_km,co2_direct_factor_g_per_km",
                "Mode::Car,100.0,50.0",
            ]
        ),
        encoding="utf-8",
    )
    custom_occupancy_path = tmp_path / "occupancy_rates.csv"
    custom_occupancy_path.write_text(
        "\n".join(
            [
                "distance_class,min_distance_m,max_distance_m,purpose,occupancy_rate",
                "all,0.0,,Tout motifs,2.0",
            ]
        ),
        encoding="utf-8",
    )
    co2_reference = xyt.load_reference_table("co2_factors", custom_co2_path)
    custom_co2_config = xyt.CO2OccupancyConfig.from_reference_files(
        co2_factors_path=custom_co2_path,
        occupancy_rates_path=custom_occupancy_path,
    )
    custom_co2 = xyt.add_co2_occupancy_metrics(legs, config=custom_co2_config)

    assert "Mode::Car" in set(co2_reference["mode"])
    assert custom_co2.loc[0, "co2_factor_g_per_km"] == 100.0
    assert custom_co2.loc[0, "co2_g"] == 50.0

    custom_mets_path = tmp_path / "metabolic_equivalent_tasks.csv"
    custom_mets_path.write_text(
        "\n".join(
            [
                "mode_niv1,intensity,mets,min_speed_kmh,max_speed_kmh",
                "Marche,moderee,9.0,3.0,7.0",
            ]
        ),
        encoding="utf-8",
    )
    met_reference = xyt.load_reference_table("metabolic_equivalent_tasks", custom_mets_path)
    custom_health_config = xyt.HealthConfig.from_reference_files(
        metabolic_equivalent_tasks_path=custom_mets_path,
    )
    custom_health = xyt.add_health_metrics(legs, config=custom_health_config)

    assert {"mode_niv1", "intensity", "mets"}.issubset(met_reference.columns)
    assert custom_health.loc[0, "intensity"] == "moderee"
    assert custom_health.loc[0, "intensity_min_speed_kmh"] == 3.0
    assert custom_health.loc[0, "intensity_max_speed_kmh"] == 7.0
    assert custom_health.loc[0, "mets"] == 9.0


def test_internal_helpers_remain_accessible_but_leave_star_import_surface():
    internal_helpers = {
        "assign_phase",
        "drop_nans_if_low_rate",
        "export_mobility_tables",
        "parse_date_columns",
        "parse_ewkb",
        "validate_schema",
    }

    missing = [name for name in sorted(internal_helpers) if not hasattr(xyt, name)]
    still_exported = [name for name in sorted(internal_helpers) if name in xyt.__all__]

    assert missing == []
    assert still_exported == []


def test_explicit_notebook_construction_path_remains_supported(tmp_path):
    raw = minimal_raw_gps_data()
    config = minimal_project_config()

    validation = xyt.validate_gps_raw(raw)
    storyline = xyt.prepare_storyline(raw.storyline, config)
    trips = xyt.prepare_trips(raw.trips, config)
    journeys = xyt.prepare_journeys(raw.journeys, config)
    trips, journeys = xyt.apply_trip_journey_mappings(trips, journeys, config.mappings)
    staypoints, legs = xyt.split_storyline(storyline)
    legs = xyt.add_user_id_day(legs)
    legs = xyt.add_signal_quality_flags(legs, config)
    map_track_trip_journey = xyt.build_track_trip_journey_map(legs, trips, journeys)
    map_legs_staypoints = xyt.build_legs_staypoints_map(legs, staypoints)
    trips = xyt.add_journey_to_trips(trips, journeys, map_track_trip_journey)
    trips = xyt.add_trip_destination_activity(trips, map_track_trip_journey, map_legs_staypoints)
    user_stats = xyt.build_user_stats(storyline, config)
    user_stats = xyt.add_signal_quality_to_user_stats(user_stats, legs)

    dataset = xyt.MobilityDataset(
        storyline=storyline,
        staypoints=staypoints,
        legs=legs,
        trips=trips,
        journeys=journeys,
        user_stats=user_stats,
        map_track_trip_journey=map_track_trip_journey,
        map_legs_staypoints=map_legs_staypoints,
        validation=validation,
        metadata={"source": "explicit_notebook_path_test"},
    )

    manifest = xyt.write_mobility_dataset(
        dataset,
        tmp_path / "explicit",
        formats=("csv",),
        include_validation=True,
        include_quality_reports=True,
    )

    assert len(dataset.legs) == 1
    assert dataset.map_track_trip_journey.loc[0, "trip_id"] == "trip-1"
    assert set(xyt.mobility_dataset_tables(dataset)) <= set(manifest["table"])
    assert "quality_tracking_report" in set(manifest["table"])
    assert (tmp_path / "explicit" / "legs.csv").exists()


def test_export_clean_dataset_writes_manifest_and_expected_tables(tmp_path):
    dataset = _minimal_dataset()
    selection_table = xyt.build_user_selection_table(dataset.user_stats)
    indicators = xyt.compute_mobility_indicators(dataset, mode_col="mode_niv1")

    manifest = xyt.export_clean_dataset(
        dataset,
        indicators,
        tmp_path / "clean",
        formats=("csv", "pkl"),
        selection_table=selection_table,
        extra_tables={
            "questionnaire_reference": pd.DataFrame(
                {"user_id": ["u1"], "questionnaire_status": ["synthetic"]}
            )
        },
    )

    exported = set(zip(manifest["section"], manifest["table"], manifest["format"]))

    assert ("mobility_dataset", "legs", "csv") in exported
    assert ("mobility_dataset", "relative_time_index", "csv") in exported
    assert ("mobility_dataset", "quality_user_selection", "csv") in exported
    assert ("mobility_dataset", "questionnaire_reference", "csv") in exported
    assert ("indicators", "indicators_population", "csv") in exported
    assert ("indicators", "indicators_person_day", "pickle") in exported
    assert ("metadata", "table_descriptions", "csv") in exported
    assert ("metadata", "attribute_dictionary", "csv") in exported
    assert (tmp_path / "clean" / "clean_export_manifest.json").exists()
    assert (tmp_path / "clean" / "relative_time_index.csv").exists()
    assert (tmp_path / "clean" / "indicators_population.csv").exists()
    assert (tmp_path / "clean" / "attribute_dictionary.csv").exists()

    manifest_json = json.loads((tmp_path / "clean" / "clean_export_manifest.json").read_text())
    assert len(manifest_json) == len(manifest)
    assert {row["section"] for row in manifest_json} == {"mobility_dataset", "indicators", "metadata"}


def test_export_clean_dataset_accepts_missing_indicators(tmp_path):
    manifest = xyt.export_clean_dataset(
        _minimal_dataset(),
        None,
        tmp_path / "clean",
        formats=("csv",),
    )

    assert set(manifest["section"]) == {"mobility_dataset", "metadata"}
    assert "relative_time_index" in set(manifest["table"])
    assert "attribute_dictionary" in set(manifest["table"])
    assert (tmp_path / "clean" / "clean_export_manifest.json").exists()


def test_export_clean_dataset_rejects_relative_time_index_collision(tmp_path):
    with pytest.raises(ValueError, match="relative_time_index"):
        xyt.export_clean_dataset(
            _minimal_dataset(),
            None,
            tmp_path / "clean",
            formats=("csv",),
            extra_tables={"relative_time_index": pd.DataFrame({"user_id": ["u1"]})},
        )


def test_export_clean_dataset_rejects_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported export format"):
        xyt.export_clean_dataset(
            _minimal_dataset(),
            None,
            tmp_path / "clean",
            formats=("feather",),
        )


def test_export_clean_dataset_supports_separate_format_dirs(tmp_path):
    manifest = xyt.export_clean_dataset(
        _minimal_dataset(),
        None,
        tmp_path / "clean",
        formats=("csv", "pkl"),
        separate_format_dirs=True,
    )

    assert ("mobility_dataset", "legs", "csv") in set(
        zip(manifest["section"], manifest["table"], manifest["format"])
    )
    assert (tmp_path / "clean" / "csv" / "legs.csv").exists()
    assert (tmp_path / "clean" / "pkl" / "legs.pkl").exists()
    assert (tmp_path / "clean" / "csv" / "attribute_dictionary.csv").exists()


def test_export_clean_dataset_can_include_spatial_analytics(tmp_path):
    manifest = xyt.export_clean_dataset(
        _minimal_dataset(),
        None,
        tmp_path / "clean",
        formats=("parquet", "csv"),
        include_spatial_analytics=True,
        spatial_analytics_options={
            "h3_resolution": 8,
            "include_count_matrix": False,
        },
    )

    exported = set(zip(manifest["section"], manifest["table"], manifest["format"]))

    assert ("spatial_analytics", "leg_points_h3", "parquet") in exported
    assert ("spatial_analytics", "leg_points_h3", "csv") not in exported
    assert ("spatial_analytics", "h3_frequency", "csv") in exported
    assert (tmp_path / "clean" / "leg_points_h3.parquet").exists()
    assert not (tmp_path / "clean" / "leg_points_h3.csv").exists()
