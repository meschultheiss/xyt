import pandas as pd
import pickle
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from xyt_gps import (
    CO2OccupancyConfig,
    HealthConfig,
    Phase,
    IndicatorResult,
    MatchingThresholds,
    MobilityPipelineResult,
    MobilityMappings,
    ProjectConfig,
    RawSampleConfig,
    RawGpsData,
    SpatialQualityThresholds,
    TrackingThresholds,
    add_journey_origin_destination_from_trips,
    add_leg_origin_destination_zones,
    add_excursion_flags,
    add_excursion_flags_to_trips_journeys,
    add_excursion_stats_to_user_stats,
    add_co2_occupancy_metrics,
    add_health_metrics,
    build_attribute_dictionary,
    build_daily_tracking_presence,
    build_daily_demand_profile,
    build_mobility_motif_sequences,
    build_mobility_motifs,
    build_mode_detection_precision,
    build_tracking_gap_report,
    build_table_descriptions,
    build_user_confirmation_rates,
    build_user_day_coverage,
    build_user_selection_table,
    build_weekly_participation_grid,
    calculate_tracking_periods,
    calculate_user_tracking_stats,
    add_signal_loss_metrics,
    add_signal_quality_flags,
    add_spatial_zone_labels,
    add_trip_origin_destination_from_legs,
    add_user_id_day,
    add_signal_quality_to_user_stats,
    build_tracking_quality_report,
    build_track_trip_journey_map,
    build_origin_destination_zone_correspondence,
    build_user_stats,
    check_raw_import_columns,
    compute_mobility_indicators,
    expected_gps_schema,
    find_sample_gps_path,
    filter_mobility_dataset_by_users,
    flag_tracking_quality,
    get_extreme_legs_by_mode,
    generate_synthetic_declic_gps,
    load_sample_gps,
    load_gps_export,
    load_gps_sources,
    mobility_dataset_tables,
    mode_purpose_mapping,
    plot_gps_on_map,
    plot_indicator_bars,
    plot_mobility_motif_graphs,
    plot_participation_heatmap,
    prepare_journeys,
    prepare_storyline,
    prepare_trips,
    run_mobility_pipeline,
    sample_raw_gps_data,
    select_analysis_users,
    select_valid_tracking_users,
    source_id_for_config,
    summarize_leg_lengths_by_mode,
    summarize_mobility_motifs,
    summarize_phase_tracking,
    summarize_phase_window_calendar_balance,
    summarize_phase_window_weekday_balance,
    summarize_participation_grid,
    summarize_weekday_balance,
    split_storyline,
    classify_leg_relation_to_area,
    prepare_mobility_dataset,
    validate_gps_raw,
    write_mobility_dataset,
)
from xyt_gps.parsing import parse_ewkb


def _co2_test_config() -> CO2OccupancyConfig:
    return CO2OccupancyConfig(
        factors_g_per_km={
            "Mode::Walk": (0.0, 0.0),
            "Mode::Car": (298.3, 180.8),
        },
        occupancy_by_distance_and_purpose={
            "0-5km": (0.0, 5000.0, {"Tout motifs": 1.81, "Loisir": 2.08}),
            "5-10km": (5000.0, 10000.0, {"Tout motifs": 1.78, "Loisir": 2.16}),
            "10-20km": (10000.0, 20000.0, {"Tout motifs": 1.80, "Loisir": 2.15}),
        },
        other_purpose_occupancy={
            "Travail": 1.19,
            "Formation": 1.78,
            "Achat et services": 1.92,
            "Accompagnement": 2.18,
        },
        reference_label="test reference files",
    )


def _health_test_config() -> HealthConfig:
    return HealthConfig(
        mets_by_mode_intensity={
            ("Marche", "calme"): 2.8,
            ("Marche", "moderee"): 3.5,
            ("Marche", "intense"): 5.0,
            ("Vélo", "calme"): 4.0,
            ("Vélo", "moderee"): 6.8,
            ("Vélo", "intense"): 10.0,
            ("Voiture", "nonactive"): 1.2,
            ("TP", "nonactive"): 1.8,
        },
        reference_label="test reference files",
    )


def test_parse_ewkb_accepts_hex_and_bytes_geometry():
    geometry = Point(6.1, 46.2)

    assert parse_ewkb(geometry.wkb_hex).equals_exact(geometry, tolerance=0)
    assert parse_ewkb(geometry.wkb).equals_exact(geometry, tolerance=0)
    assert parse_ewkb(geometry.wkt).equals_exact(geometry, tolerance=0)


def test_top_level_public_api_remains_importable():
    import xyt_gps as xyt

    expected_names = [
        "prepare_mobility_dataset",
        "prepare_mobility_datasets",
        "run_mobility_pipeline",
        "MobilityPipelineResult",
        "prepare_storyline",
        "prepare_trips",
        "prepare_journeys",
        "build_weekly_participation_grid",
        "build_tracking_quality_report",
        "build_attribute_dictionary",
        "build_table_descriptions",
        "build_user_selection_table",
        "compute_mobility_indicators",
        "build_daily_demand_profile",
        "write_mobility_dataset",
        "build_spatial_analytics_tables",
        "write_duckdb_spatial_database",
        "plot_gps_on_map",
        "plot_gps_traces",
        "plot_mobility_motif_graphs",
        "plot_participation_heatmap",
    ]

    for name in expected_names:
        assert hasattr(xyt, name), name
        assert name in xyt.__all__


def _config():
    return ProjectConfig(
        experiment_name="test",
        motiontag_project_name="Test",
        period="2025-01-01--2025-01-02",
        raw_data_dir=".",
        phases=(Phase("Phase1", "2025-01-01", "2025-01-31"),),
    )


def _raw_tables():
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


def _write_motiontag_export(base_path, config):
    storyline, trips, journeys = _raw_tables()
    user_statistics = pd.DataFrame([{"user_id": "u1", "active_days_count": 1}])
    base_path.mkdir(parents=True, exist_ok=True)
    project = config.motiontag_project_name
    period = config.period
    storyline.to_csv(base_path / f"StorylineWithTripId.{project}.{period}.csv", sep=";", index=False)
    trips.to_csv(base_path / f"Trips.{project}.{period}.csv", sep=";", index=False)
    journeys.to_csv(base_path / f"Journeys.{project}.{period}.csv", sep=";", index=False)
    user_statistics.to_csv(base_path / f"UserStatistics.{project}.{period}.csv", sep=";", index=False)


def _raw_tables_for_two_users():
    parts = []
    trip_parts = []
    journey_parts = []
    for user_id in ("u1", "u2"):
        storyline, trips, journeys = _raw_tables()
        storyline = storyline.copy()
        trips = trips.copy()
        journeys = journeys.copy()
        storyline["user_id"] = user_id
        storyline["storyline_id"] = storyline["storyline_id"].map(lambda value: f"{user_id}-{value}")
        storyline["trip_id"] = storyline["trip_id"].map(
            lambda value: f"{user_id}-{value}" if pd.notna(value) else value
        )
        trips["user_id"] = user_id
        trips["trip_id"] = trips["trip_id"].map(lambda value: f"{user_id}-{value}")
        journeys["user_id"] = user_id
        journeys["journey_id"] = journeys["journey_id"].map(lambda value: f"{user_id}-{value}")
        parts.append(storyline)
        trip_parts.append(trips)
        journey_parts.append(journeys)
    return (
        pd.concat(parts, ignore_index=True),
        pd.concat(trip_parts, ignore_index=True),
        pd.concat(journey_parts, ignore_index=True),
    )


def test_validate_gps_raw_accepts_minimal_tables():
    storyline, trips, journeys = _raw_tables()
    raw = RawGpsData(storyline=storyline, trips=trips, journeys=journeys)
    reports = validate_gps_raw(raw)
    assert reports["storyline"].ok
    assert reports["trips"].ok
    assert reports["journeys"].ok


def test_public_schema_and_mapping_helpers_are_available():
    gps_schema = expected_gps_schema()
    mapping = mode_purpose_mapping()

    assert {"table", "column", "status"}.issubset(gps_schema.columns)
    assert isinstance(mapping, MobilityMappings)


def test_synthetic_declic_generation_returns_valid_raw_tables():
    try:
        sample_path = find_sample_gps_path()
    except FileNotFoundError:
        pytest.skip("Authorized local sample data is not available.")

    synthetic = generate_synthetic_declic_gps(
        sample_path=sample_path,
        users_per_experiment=2,
        random_state=123,
    )

    raw = synthetic.raw
    reports = validate_gps_raw(raw)
    dataset = prepare_mobility_dataset(raw, ProjectConfig(), resample_missing_days=False)

    assert synthetic.user_presence["experiment_name"].nunique() == 5
    assert synthetic.user_presence["user_id"].nunique() == 10
    assert set(reports) == {"storyline", "trips", "journeys", "user_statistics"}
    assert all(report.ok for report in reports.values())
    assert raw.storyline["phase"].notna().any()
    assert raw.storyline["experiment_name"].nunique() == 5
    assert len(dataset.legs) > 0
    assert dataset.user_stats["user_id"].nunique() == 10


def test_validate_gps_raw_accepts_user_statistics_user_id_alias():
    storyline, trips, journeys = _raw_tables()
    raw = RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=pd.DataFrame([{"user_id": "u1", "active_days_count": 1}]),
    )

    reports = validate_gps_raw(raw)

    assert reports["user_statistics"].ok


def test_check_raw_import_columns_reports_motion_tables_and_user_statistics_structure():
    storyline, trips, journeys = _raw_tables()
    user_statistics = pd.DataFrame([{"user_id": "u1", "active_days_count": 1}])

    report = check_raw_import_columns(
        storyline,
        user_statistics,
        trips=trips,
        journeys=journeys,
    )

    storyline_required = report.loc[
        report["table"].eq("storyline") & report["required"]
    ]
    trips_required = report.loc[report["table"].eq("trips") & report["required"]]
    journeys_required = report.loc[report["table"].eq("journeys") & report["required"]]
    user_id_row = report.loc[
        report["table"].eq("user_statistics")
        & report["expected_column"].eq("user_id")
    ].iloc[0]

    assert storyline_required["status"].eq("ok").all()
    assert trips_required["status"].eq("ok").all()
    assert journeys_required["status"].eq("ok").all()
    assert user_id_row["status"] == "ok"
    assert user_id_row["observed_column"] == "user_id"


def test_check_raw_import_columns_can_raise_on_missing_required_column():
    storyline, _trips, _journeys = _raw_tables()
    broken_storyline = storyline.drop(columns=["geometry"])

    with pytest.raises(ValueError, match="geometry"):
        check_raw_import_columns(broken_storyline, raise_on_error=True)


def test_prepare_storyline_splits_stays_and_legs():
    storyline, trips, journeys = _raw_tables()
    config = _config()
    parsed = prepare_storyline(storyline, config)
    staypoints, legs = split_storyline(parsed)
    legs = add_user_id_day(legs)
    assert len(staypoints) == 1
    assert len(legs) == 1
    assert staypoints.loc[0, "purpose_niv1"] == "Domicile"
    assert staypoints.loc[0, "lon"] == pytest.approx(6.1)
    assert staypoints.loc[0, "lat"] == pytest.approx(46.2)
    assert legs.loc[0, "mode_niv1"] == "Marche"
    assert legs.loc[0, "user_id_day"] == "u1_20250101"
    assert legs.loc[0, "duration"] == 1800
    assert legs.loc[0, "track_count"] == 1


def test_split_storyline_rounds_fractional_second_leg_duration():
    storyline, _trips, _journeys = _raw_tables()
    storyline.loc[storyline["type"].eq("Track"), "finished_at"] = "2025-01-01T09:30:00.200Z"

    parsed = prepare_storyline(storyline, _config())
    _staypoints, legs = split_storyline(parsed)

    assert legs.loc[0, "duration"] == 1800


def test_project_config_can_override_mode_and_purpose_mappings():
    storyline, trips, journeys = _raw_tables()
    config = ProjectConfig(
        experiment_name="test",
        motiontag_project_name="Test",
        period="2025-01-01--2025-01-02",
        raw_data_dir=".",
        phases=(Phase("Phase1", "2025-01-01", "2025-01-31"),),
        mappings=MobilityMappings(
            storyline_purpose_niv1={"Chez soi": ("home",)},
            storyline_mode_niv1={"Mobilite active": ("Mode::Walk",)},
            trip_mode_niv1={"Mobilite active": ("Walking",)},
        ),
    )
    parsed = prepare_storyline(storyline, config)
    staypoints, legs = split_storyline(parsed)

    assert staypoints.loc[0, "purpose_niv1"] == "Chez soi"
    assert legs.loc[0, "mode_niv1"] == "Mobilite active"


def test_project_config_exposes_thresholds_from_delivery_context():
    config = _config()
    assert config.tracking_thresholds.min_total_tracked_days == 7
    assert config.spatial_quality_thresholds.outlier_quantiles_by_mode == (0.98, 0.99)
    assert config.matching_thresholds.osrm_max_points_per_chunk == 99
    assert isinstance(config.spatial_quality_thresholds, SpatialQualityThresholds)
    assert isinstance(config.matching_thresholds, MatchingThresholds)


def test_track_trip_journey_map_uses_tolerance():
    legs = pd.DataFrame(
        [
            {
                "leg_id": "leg-1",
                "trip_id": "trip-1",
                "user_id": "u1",
                "started_at": pd.Timestamp("2025-01-01T09:00:00Z"),
                "finished_at": pd.Timestamp("2025-01-01T09:30:00Z"),
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
                "started_at": pd.Timestamp("2025-01-01T09:15:30Z"),
                "finished_at": pd.Timestamp("2025-01-01T09:16:00Z"),
            }
        ]
    )

    without_tolerance = build_track_trip_journey_map(legs, trips, journeys, tolerance="0s")
    with_tolerance = build_track_trip_journey_map(legs, trips, journeys, tolerance="1min")

    assert pd.isna(without_tolerance.loc[0, "journey_id"])
    assert with_tolerance.loc[0, "journey_id"] == "journey-1"


def test_track_trip_journey_map_does_not_keep_missing_trip_ids():
    legs = pd.DataFrame(
        [
            {
                "leg_id": "leg-1",
                "trip_id": "trip-missing",
                "user_id": "u1",
                "started_at": pd.Timestamp("2025-01-01T09:00:00Z"),
                "finished_at": pd.Timestamp("2025-01-01T09:30:00Z"),
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


def test_excursion_flags_are_propagated_to_trips_and_journeys():
    legs = pd.DataFrame(
        [
            {"leg_id": "leg-1", "excursion": True},
            {"leg_id": "leg-2", "excursion": False},
        ]
    )
    trips = pd.DataFrame([{"trip_id": "trip-1"}, {"trip_id": "trip-2"}])
    journeys = pd.DataFrame([{"journey_id": "journey-1"}, {"journey_id": "journey-2"}])
    mapping = pd.DataFrame(
        [
            {"leg_id": "leg-1", "trip_id": "trip-1", "journey_id": "journey-1"},
            {"leg_id": "leg-2", "trip_id": "trip-2", "journey_id": "journey-2"},
        ]
    )

    trips_out, journeys_out = add_excursion_flags_to_trips_journeys(
        trips,
        journeys,
        legs,
        mapping,
    )

    assert trips_out.set_index("trip_id").loc["trip-1", "excursion"] == 1
    assert trips_out.set_index("trip_id").loc["trip-2", "excursion"] == 0
    assert journeys_out.set_index("journey_id").loc["journey-1", "excursion"] == 1
    assert journeys_out.set_index("journey_id").loc["journey-2", "excursion"] == 0


def test_excursion_stats_are_added_to_user_stats_from_legs():
    user_stats = pd.DataFrame({"user_id": ["u1", "u2"]})
    legs = pd.DataFrame(
        [
            {"user_id": "u1", "excursion": 1},
            {"user_id": "u1", "excursion": 0},
            {"user_id": "u2", "excursion": 0},
        ]
    )

    out = add_excursion_stats_to_user_stats(user_stats, legs)

    u1 = out.set_index("user_id").loc["u1"]
    u2 = out.set_index("user_id").loc["u2"]
    assert u1["excursion_legs_count"] == 1
    assert u1["total_legs_count"] == 2
    assert u1["excursion_leg_ratio"] == 0.5
    assert u2["excursion_legs_count"] == 0


def test_prepare_trips_and_journeys_use_explicit_ids():
    storyline, trips, journeys = _raw_tables()
    config = _config()
    prepared_trips = prepare_trips(trips, config)
    prepared_journeys = prepare_journeys(journeys, config)
    assert "trip_id" in prepared_trips.columns
    assert "journey_id" in prepared_journeys.columns


def test_prepare_trips_and_journeys_accept_explicit_landing_ids():
    _storyline, trips, journeys = _raw_tables()
    config = _config()
    trips = trips.assign(id=trips["trip_id"])
    journeys = journeys.assign(id=journeys["journey_id"])

    prepared_trips = prepare_trips(trips, config)
    prepared_journeys = prepare_journeys(journeys, config)

    assert prepared_trips.columns.tolist().count("trip_id") == 1
    assert prepared_journeys.columns.tolist().count("journey_id") == 1
    assert "id" not in prepared_trips.columns
    assert "id" not in prepared_journeys.columns


def test_load_gps_sources_concatenates_and_namespaces_ids(tmp_path):
    config_a = ProjectConfig(
        experiment_name="declic",
        motiontag_project_name="Declic",
        period="2025-01-01--2025-01-02",
        raw_data_dir=tmp_path / "declic",
    )
    config_b = ProjectConfig(
        experiment_name="declic-ziplo",
        motiontag_project_name="Declic-ziplo",
        period="2025-02-01--2025-02-02",
        raw_data_dir=tmp_path / "declic_ziplo",
    )
    _write_motiontag_export(config_a.raw_path, config_a)
    _write_motiontag_export(config_b.raw_path, config_b)

    raw = load_gps_sources([config_a, config_b])
    source_a = source_id_for_config(config_a)
    source_b = source_id_for_config(config_b)

    assert len(raw.storyline) == 4
    assert "xyt_source_id" not in raw.storyline.columns
    assert raw.storyline["storyline_id"].str.startswith((source_a, source_b)).all()
    assert raw.storyline["user_id"].str.startswith((source_a, source_b)).all()
    assert "raw_user_id" in raw.storyline.columns


def test_sample_raw_gps_data_by_users_keeps_all_rows_for_selected_users():
    storyline, trips, journeys = _raw_tables_for_two_users()
    raw = RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=pd.DataFrame(
            [
                {"user_id": "u1", "active_days_count": 1},
                {"user_id": "u2", "active_days_count": 1},
            ]
        ),
    )

    sampled = sample_raw_gps_data(raw, RawSampleConfig.by_users(1, random_state=1))

    selected_users = set(sampled.storyline["user_id"])
    assert len(selected_users) == 1
    assert set(sampled.trips["user_id"]) == selected_users
    assert set(sampled.journeys["user_id"]) == selected_users
    assert set(sampled.user_statistics["user_id"]) == selected_users
    assert len(sampled.storyline) == 2


def test_load_gps_export_by_users_filters_csvs_with_chunks(tmp_path):
    config = ProjectConfig(
        experiment_name="declic",
        motiontag_project_name="Declic",
        period="2025-01-01--2025-01-02",
        raw_data_dir=tmp_path,
    )
    storyline, trips, journeys = _raw_tables_for_two_users()
    user_statistics = pd.DataFrame(
        [
            {"user_id": "u1", "active_days_count": 1},
            {"user_id": "u2", "active_days_count": 1},
        ]
    )
    project = config.motiontag_project_name
    period = config.period
    storyline.to_csv(tmp_path / f"StorylineWithTripId.{project}.{period}.csv", sep=";", index=False)
    trips.to_csv(tmp_path / f"Trips.{project}.{period}.csv", sep=";", index=False)
    journeys.to_csv(tmp_path / f"Journeys.{project}.{period}.csv", sep=";", index=False)
    user_statistics.to_csv(tmp_path / f"UserStatistics.{project}.{period}.csv", sep=";", index=False)

    raw = load_gps_export(
        config,
        sample=RawSampleConfig.by_users(1, random_state=1, chunksize=1),
    )

    selected_users = set(raw.storyline["user_id"])
    assert len(selected_users) == 1
    assert set(raw.trips["user_id"]) == selected_users
    assert set(raw.journeys["user_id"]) == selected_users
    assert set(raw.user_statistics["user_id"]) == selected_users
    assert len(raw.storyline) == 2


def test_sample_raw_gps_data_random_rows_prefers_track_rows():
    storyline, trips, journeys = _raw_tables_for_two_users()
    raw = RawGpsData(
        storyline=storyline,
        trips=trips,
        journeys=journeys,
        user_statistics=pd.DataFrame(
            [
                {"user_id": "u1", "active_days_count": 1},
                {"user_id": "u2", "active_days_count": 1},
            ]
        ),
    )

    sampled = sample_raw_gps_data(raw, RawSampleConfig.random_rows(1, random_state=1))

    assert len(sampled.storyline) == 1
    assert sampled.storyline.loc[0, "type"] == "Track"
    assert len(sampled.trips) == 1
    assert len(sampled.journeys) == 1
    assert len(sampled.user_statistics) == 1


def test_build_user_stats_flags_tracking_quality():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "started_at": pd.Timestamp(f"2025-01-{day:02d}T08:00:00Z"), "type": "Track"}
            for day in range(1, 8)
        ]
        + [
            {"user_id": "u2", "started_at": pd.Timestamp("2025-01-01T08:00:00Z"), "type": "Track"},
            {"user_id": "u2", "started_at": pd.Timestamp("2025-01-03T08:00:00Z"), "type": "Track"},
        ]
    )
    config = ProjectConfig(
        experiment_name="test",
        motiontag_project_name="Test",
        period="2025-01-01--2025-01-07",
        raw_data_dir=".",
        phases=(Phase("baseline", "2025-01-01", "2025-01-07"),),
        tracking_thresholds=TrackingThresholds(
            min_days_by_phase={"baseline": 7},
            min_total_tracked_days=7,
            round_to_full_weeks=False,
        ),
    )

    stats = build_user_stats(storyline, config)
    valid_users = set(select_valid_tracking_users(stats))
    report = build_tracking_quality_report(stats)

    u1 = stats.loc[stats["user_id"].eq("u1")].iloc[0]
    u2 = stats.loc[stats["user_id"].eq("u2")].iloc[0]
    assert bool(u1["tracking_quality_ok"])
    assert not bool(u2["tracking_quality_ok"])
    assert u2["inactive_days_count"] == 1
    assert u2["tracking_quality_reason"] == "insufficient_total_tracked_days"
    assert valid_users == {"u1"}
    assert report.loc[0, "users_count"] == 2
    assert report.loc[0, "tracking_quality_ok_count"] == 1


def test_tracking_quality_selection_only_uses_configured_phases():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "started_at": pd.Timestamp(f"2025-01-{day:02d}T08:00:00Z"), "type": "Track"}
            for day in range(1, 5)
        ]
    )
    config = ProjectConfig(
        experiment_name="test",
        motiontag_project_name="Test",
        period="2025-01-01--2025-01-04",
        raw_data_dir=".",
        phases=(
            Phase("Phase1", "2025-01-01", "2025-01-02"),
            Phase("Phase2", "2025-01-03", "2025-01-04"),
        ),
        tracking_thresholds=TrackingThresholds(
            min_days_by_phase={"Phase1": 2, "Phase2": 2},
            min_total_tracked_days=4,
            round_to_full_weeks=False,
        ),
    )

    stats = build_user_stats(storyline, config)
    selection = build_user_selection_table(
        stats,
        require_tracking_quality=True,
        exclude_bad_signal_users=False,
    )

    assert "phase1_tracked_days_ok" in stats.columns
    assert "phase2_tracked_days_ok" in stats.columns
    assert "phase3_tracked_days_ok" not in stats.columns
    assert stats.loc[0, "tracking_quality_ok"]
    assert selection.loc[0, "analysis_user_ok"]
    assert selection.loc[0, "analysis_user_reason"] == "ok"


def test_build_user_stats_tracks_resampled_days_and_default_weight():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "started_at": pd.Timestamp("2025-01-01T08:00:00Z"), "type": "Stay"},
            {"user_id": "u1", "started_at": pd.Timestamp("2025-01-02T08:00:00Z"), "type": "Resampled_stay"},
            {"user_id": "u1", "started_at": pd.Timestamp("2025-01-03T08:00:00Z"), "type": "Track"},
        ]
    )

    stats = build_user_stats(storyline, _config())

    row = stats.iloc[0]
    assert row["tracked_days_count"] == 2
    assert row["active_days_count"] == 1
    assert row["zero_movement_tracked_days_count"] == 1
    assert row["untracked_days_count"] == 1
    assert row["resampled_days_count"] == 1
    assert row["observed_tracking_days_count"] == 2
    assert row["observed_active_days_count"] == 1
    assert row["weight"] == 1.0


def test_user_day_coverage_week_sequence_uses_experiment_start():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "started_at": "2025-01-10T08:00:00Z", "type": "Track"},
            {"user_id": "u1", "started_at": "2025-01-11T08:00:00Z", "type": "Track"},
        ]
    )
    config = ProjectConfig(
        experiment_name="test",
        start_expe="2025-01-01",
        phases=(
            Phase("Phase1", "2025-01-01", "2025-01-10"),
            Phase("Phase2", "2025-01-11", "2025-01-20"),
        ),
    )

    coverage = build_user_day_coverage(storyline, config)

    assert coverage["n_week_sequence"].tolist() == [2, 2]
    assert coverage["phase_week_number"].tolist() == [2, 1]


def test_build_user_stats_drops_residual_provider_id_when_user_id_exists():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "started_at": pd.Timestamp("2025-01-01T08:00:00Z"), "type": "Stay"},
            {"user_id": "u2", "started_at": pd.Timestamp("2025-01-01T08:00:00Z"), "type": "Track"},
        ]
    )
    user_statistics = pd.DataFrame(
        [
            {"id": "u1", "user_id": "u1", "operating_system": "ios"},
            {"id": "u2", "user_id": "u2", "operating_system": "android"},
        ]
    )

    stats = build_user_stats(storyline, _config(), user_statistics=user_statistics)

    assert stats.columns.tolist().count("user_id") == 1
    assert "id" not in stats.columns
    assert set(stats["operating_system"]) == {"ios", "android"}
    assert len(stats) == 2


def test_build_user_stats_keeps_users_without_tracking():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "started_at": pd.Timestamp("2025-01-01T08:00:00Z"), "type": "Track"},
        ]
    )
    user_statistics = pd.DataFrame(
        [
            {"user_id": "u1", "operating_system": "ios"},
            {"user_id": "u2", "operating_system": "android"},
        ]
    )

    stats = build_user_stats(storyline, _config(), user_statistics=user_statistics)
    untracked = stats.loc[stats["user_id"].eq("u2")].iloc[0]

    assert set(stats["user_id"]) == {"u1", "u2"}
    assert untracked["active_days_count"] == 0
    assert untracked["n_days_phase_1"] == 0
    assert not bool(untracked["tracking_quality_ok"])


def test_weekly_participation_grid_counts_observed_days_before_resampling():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "type": "Track", "started_at": "2025-01-01T08:00:00Z"},
            {"user_id": "u1", "type": "Stay", "started_at": "2025-01-02T08:00:00Z"},
            {"user_id": "u1", "type": "Resampled_stay", "started_at": "2025-01-03T08:00:00Z"},
            {"user_id": "u1", "type": "Track", "started_at": "2025-01-08T08:00:00Z"},
        ]
    )
    config = ProjectConfig(
        experiment_name="participation-test",
        motiontag_project_name="Participation",
        period="2025-01-01--2025-01-14",
        raw_data_dir=".",
        phases=(Phase("Phase1", "2025-01-01", "2025-01-14"),),
    )

    daily = build_daily_tracking_presence(storyline, config)
    grid = build_weekly_participation_grid(storyline, config, user_ids=["u1", "u2"])
    summary = summarize_participation_grid(grid, good_week_min_days=2)

    u1_week_1 = grid.loc[
        grid["user_id"].eq("u1") & grid["phase_week_number"].eq(1)
    ].iloc[0]
    u1_week_2 = grid.loc[
        grid["user_id"].eq("u1") & grid["phase_week_number"].eq(2)
    ].iloc[0]
    u2_week_1 = grid.loc[
        grid["user_id"].eq("u2") & grid["phase_week_number"].eq(1)
    ].iloc[0]

    assert len(daily) == 3
    assert u1_week_1["tracked_days_count"] == 2
    assert u1_week_1["active_days_count"] == 1
    assert u1_week_1["participation_score"] == 2
    assert u1_week_2["tracked_days_count"] == 1
    assert u1_week_2["active_days_count"] == 1
    assert u2_week_1["tracked_days_count"] == 0
    assert u2_week_1["active_days_count"] == 0
    assert grid["tracked_days_count"].between(0, 7).all()
    assert grid["active_days_count"].between(0, 7).all()
    assert summary.loc[summary["scope"].eq("all"), "good_or_complete_weeks_count"].item() == 1


def test_summarize_weekday_balance_reports_calendar_and_person_days():
    coverage = pd.DataFrame(
        [
            {
                "experiment_name": "test",
                "phase": "Phase1",
                "user_id": user_id,
                "date": date,
                "day_in_range": 1,
                "tracked_day": tracked_day,
                "active_day": active_day,
            }
            for user_id, tracked_day, active_day in [("u1", 1, 1), ("u2", 0, 0)]
            for date in pd.date_range("2025-01-06", "2025-01-12", freq="D")
        ]
        + [
            {
                "experiment_name": "test",
                "phase": "Phase2",
                "user_id": "u1",
                "date": date,
                "day_in_range": 1,
                "tracked_day": 1,
                "active_day": 1,
            }
            for date in pd.date_range("2025-01-13", "2025-01-17", freq="D")
        ]
    )

    summary = summarize_weekday_balance(coverage, day_basis="day_in_range")
    phase1 = summary.loc[summary["phase"].eq("Phase1")].iloc[0]
    phase2 = summary.loc[summary["phase"].eq("Phase2")].iloc[0]
    tracked = summarize_weekday_balance(coverage, day_basis="tracked_day")
    tracked_phase1 = tracked.loc[tracked["phase"].eq("Phase1")].iloc[0]

    assert phase1["calendar_weekday_balanced"]
    assert phase1["calendar_all_weekdays_present"]
    assert phase1["calendar_weekday_spread"] == 0
    assert phase1["person_days_count"] == 14
    assert phase1["person_days_lundi"] == 2
    assert phase1["person_weekday_balanced"]
    assert not phase2["calendar_weekday_balanced"]
    assert not phase2["calendar_all_weekdays_present"]
    assert not phase2["person_weekday_balanced"]
    assert phase2["calendar_weekday_spread"] == 1
    assert phase2["calendar_samedi"] == 0
    assert tracked_phase1["person_days_count"] == 7
    assert tracked_phase1["person_days_lundi"] == 1
    assert tracked_phase1["person_weekday_balanced"]

    phase_windows = pd.DataFrame(
        [
            {
                "window_type": "analysis",
                "experiment_name": "test",
                "phase": "Phase1",
                "phase_start": "2025-01-06",
                "phase_end": "2025-01-12",
            },
            {
                "window_type": "analysis",
                "experiment_name": "test",
                "phase": "Phase2",
                "phase_start": "2025-01-13",
                "phase_end": "2025-01-17",
            },
        ]
    )
    compact = summarize_phase_window_weekday_balance(
        coverage,
        phase_windows,
        day_basis="day_in_range",
    )
    compact_phase1 = compact.loc[compact["phase"].eq("Phase1")].iloc[0]
    compact_phase2 = compact.loc[compact["phase"].eq("Phase2")].iloc[0]

    assert compact_phase1["n_lun"] == 2
    assert compact_phase1["n_dim"] == 2
    assert compact_phase1["weekday_balanced"]
    assert compact_phase2["n_sam"] == 0
    assert not compact_phase2["weekday_balanced"]


def test_summarize_phase_window_calendar_balance_flags_partial_weeks():
    phase_windows = pd.DataFrame(
        [
            {
                "window_type": "official",
                "experiment_name": "test",
                "phase": "Phase2",
                "phase_start": "2026-03-09",
                "phase_end": "2026-04-08",
            },
            {
                "window_type": "analysis",
                "experiment_name": "test",
                "phase": "Phase2",
                "phase_start": "2026-03-09",
                "phase_end": "2026-04-05",
            },
        ]
    )

    summary = summarize_phase_window_calendar_balance(phase_windows)
    official = summary.loc[summary["window_type"].eq("official")].iloc[0]
    analysis = summary.loc[summary["window_type"].eq("analysis")].iloc[0]

    assert official["n_lun"] == 5
    assert official["n_jeu"] == 4
    assert official["weekday_spread"] == 1
    assert not official["weekday_balanced"]
    assert analysis["n_lun"] == 4
    assert analysis["n_dim"] == 4
    assert analysis["weekday_spread"] == 0
    assert analysis["weekday_balanced"]
    assert analysis["full_monday_sunday_weeks"]


def test_user_day_coverage_keeps_official_and_analysis_phases_separate():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "type": "Track", "started_at": "2026-04-06T08:00:00Z"},
            {"user_id": "u1", "type": "Track", "started_at": "2026-04-13T08:00:00Z"},
        ]
    )
    config = ProjectConfig(
        experiment_name="phase-window-test",
        raw_data_dir=".",
        start_expe="2026-03-30",
        phases=(
            Phase("Phase1", "2026-03-30", "2026-04-05"),
            Phase("Phase2", "2026-04-06", "2026-04-19"),
        ),
        analysis_phases=(
            Phase("Phase1", "2026-03-30", "2026-04-05"),
            Phase("Phase2", "2026-04-13", "2026-04-19"),
        ),
    )

    coverage = build_user_day_coverage(storyline, config)
    transition_day = coverage.loc[coverage["date"].eq(pd.Timestamp("2026-04-06"))].iloc[0]
    analysis_day = coverage.loc[coverage["date"].eq(pd.Timestamp("2026-04-13"))].iloc[0]

    assert transition_day["phase"] == "Phase2"
    assert pd.isna(transition_day["analysis_phase"])
    assert transition_day["n_week_sequence"] == 2
    assert analysis_day["phase"] == "Phase2"
    assert analysis_day["analysis_phase"] == "Phase2"
    assert analysis_day["analysis_phase_start"] == "2026-04-13"
    assert analysis_day["n_week_sequence"] == 3


def test_weekly_participation_grid_handles_empty_observed_phase_days():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "type": "Track", "started_at": "2024-12-20T08:00:00Z"},
        ]
    )
    config = ProjectConfig(
        experiment_name="participation-test",
        raw_data_dir=".",
        phases=(Phase("Phase1", "2025-01-01", "2025-01-14"),),
    )

    grid = build_weekly_participation_grid(storyline, config, user_ids=["u1", "u2"])

    assert len(grid) == 4
    assert set(grid["user_id"]) == {"u1", "u2"}
    assert set(grid["phase_week_number"]) == {1, 2}
    assert grid["active_days_count"].eq(0).all()


def test_tracking_gap_report_and_phase_summary_are_explicit():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "type": "Track", "started_at": "2025-01-01T08:00:00Z"},
            {"user_id": "u1", "type": "Track", "started_at": "2025-01-02T08:00:00Z"},
            {"user_id": "u1", "type": "Track", "started_at": "2025-01-05T08:00:00Z"},
            {"user_id": "u2", "type": "Track", "started_at": "2025-01-03T08:00:00Z"},
        ]
    )
    config = ProjectConfig(
        phases=(Phase("Phase1", "2025-01-01", "2025-01-07"),),
        tracking_thresholds=TrackingThresholds(
            min_days_by_phase={"Phase1": 1},
            min_total_tracked_days=1,
            round_to_full_weeks=False,
        ),
    )

    gap_report = build_tracking_gap_report(storyline, config, user_ids=["u1", "u2"])
    user_stats = flag_tracking_quality(calculate_tracking_periods(calculate_user_tracking_stats(storyline), config), config)
    phase_summary = summarize_phase_tracking(user_stats, config, day_thresholds=(1, 3))

    u1 = gap_report.loc[gap_report["user_id"].eq("u1")].iloc[0]
    assert u1["active_days_count"] == 3
    assert u1["missing_days_count"] == 4
    assert u1["max_consecutive_days"] == 2
    assert u1["max_gap_days"] == 2
    assert phase_summary.loc[0, "users_ge_1_days"] == 2
    assert phase_summary.loc[0, "users_ge_3_days"] == 1


def test_leg_length_reports_confirmation_rates_and_mode_precision():
    storyline, _trips, _journeys = _raw_tables()
    storyline["confirmed_at"] = [pd.NaT, "2025-01-01T09:31:00Z"]
    parsed = prepare_storyline(storyline, _config())
    staypoints, legs = split_storyline(parsed)
    legs = add_user_id_day(legs)

    length_summary = summarize_leg_lengths_by_mode(legs, mode_col="mode_niv1")
    extreme_legs = get_extreme_legs_by_mode(legs, mode_col="mode_niv1")
    confirmations = build_user_confirmation_rates(parsed)
    precision = build_mode_detection_precision(parsed, group_cols=("mode", "mode_niv1"))

    assert length_summary.loc[0, "legs_count"] == 1
    assert extreme_legs.loc[0, "distance_km"] == 1.2
    assert confirmations.loc[0, "confirmed_tracks_count"] == 1
    assert confirmations.loc[0, "confirmed_stays_count"] == 0
    assert precision.loc[precision["grouping"].eq("mode"), "precision"].iloc[0] == 1.0


def test_build_user_selection_table_combines_tracking_and_signal_quality():
    user_stats = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "tracking_quality_ok": True,
                "tracking_quality_reason": "ok",
                "bad_signal_user": 0,
                "low_quality_legs_share": 0.05,
            },
            {
                "user_id": "u2",
                "tracking_quality_ok": False,
                "tracking_quality_reason": "insufficient_total_tracked_days",
                "bad_signal_user": 0,
                "low_quality_legs_share": 0.02,
            },
            {
                "user_id": "u3",
                "tracking_quality_ok": True,
                "tracking_quality_reason": "ok",
                "bad_signal_user": 1,
                "low_quality_legs_share": 0.10,
            },
            {
                "user_id": "u4",
                "tracking_quality_ok": True,
                "tracking_quality_reason": "ok",
                "bad_signal_user": 0,
                "low_quality_legs_share": 0.80,
            },
        ]
    )

    selection_table = build_user_selection_table(
        user_stats,
        require_tracking_quality=True,
        max_low_quality_legs_share=0.25,
    )
    selected = set(select_analysis_users(selection_table))

    assert selected == {"u1"}
    assert selection_table.loc[
        selection_table["user_id"].eq("u2"), "analysis_user_reason"
    ].item() == "insufficient_total_tracked_days"
    assert selection_table.loc[
        selection_table["user_id"].eq("u3"), "analysis_user_reason"
    ].item() == "bad_signal_user"
    assert selection_table.loc[
        selection_table["user_id"].eq("u4"), "analysis_user_reason"
    ].item() == "too_many_low_quality_legs"

    signal_only_selection = build_user_selection_table(
        user_stats,
        require_tracking_quality=False,
        max_low_quality_legs_share=0.25,
    )
    signal_only_selected = set(select_analysis_users(signal_only_selection))

    assert signal_only_selected == {"u1", "u2"}
    assert signal_only_selection.loc[
        signal_only_selection["user_id"].eq("u2"), "analysis_user_reason"
    ].item() == "ok"


def test_add_signal_loss_metrics_uses_metric_crs():
    config = _config()
    legs = gpd.GeoDataFrame(
        [
            {
                "leg_id": "leg-1",
                "user_id": "u1",
                "mode": "Mode::Walk",
                "geometry": LineString([(2_600_000, 1_100_000), (2_600_003, 1_100_000), (2_600_004, 1_100_000)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:2056",
    )

    out = add_signal_loss_metrics(legs, config)

    assert out.loc[0, "max_signal_loss_m"] == 3
    assert out.loc[0, "length_leg"] == 4
    assert out.loc[0, "relative_signal_loss"] == 0.75


def test_add_excursion_flags_supports_circle_reference():
    traces = gpd.GeoDataFrame(
        {
            "leg_id": ["near", "far"],
            "geometry": [
                LineString([(6.0, 46.0), (6.001, 46.001)]),
                LineString([(7.0, 47.0), (7.001, 47.001)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    flagged = add_excursion_flags(
        traces,
        center=(6.0, 46.0),
        radius_m=1000,
        operations_crs="EPSG:2056",
    )

    assert flagged.set_index("leg_id").loc["near", "excursion"] == 0
    assert flagged.set_index("leg_id").loc["far", "excursion"] == 1


def test_spatial_zone_labels_and_origin_destination_propagation():
    zones = gpd.GeoDataFrame(
        {
            "zone": ["A", "B"],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    staypoints = gpd.GeoDataFrame(
        {"activity_id": ["a1", "a2"], "geometry": [Point(1, 1), Point(3, 1)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    legs = gpd.GeoDataFrame(
        {
            "leg_id": ["l1", "l2"],
            "geometry": [
                LineString([(1, 1), (3, 1)]),
                LineString([(3, 1), (5, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    mapping = pd.DataFrame(
        [
            {"leg_id": "l1", "trip_id": "t1", "journey_id": "j1"},
            {"leg_id": "l2", "trip_id": "t1", "journey_id": "j1"},
        ]
    )
    trips = pd.DataFrame({"trip_id": ["t1"]})
    journeys = pd.DataFrame({"journey_id": ["j1"]})

    labelled_stays = add_spatial_zone_labels(
        staypoints,
        zones,
        zone_label_col="zone",
        output_col="activity_zone",
    )
    labelled_legs = add_leg_origin_destination_zones(
        legs,
        zones,
        zone_label_col="zone",
        fill_value="Outside",
    )
    relation = classify_leg_relation_to_area(
        legs,
        area=zones.loc[zones["zone"].eq("A"), "geometry"].iloc[0],
        code_col="area_relation_code",
    )
    trips_out = add_trip_origin_destination_from_legs(trips, labelled_legs, mapping)
    journeys_out = add_journey_origin_destination_from_trips(journeys, trips_out, mapping)

    assert labelled_stays.set_index("activity_id").loc["a1", "activity_zone"] == "A"
    assert labelled_stays.set_index("activity_id").loc["a2", "activity_zone"] == "B"
    assert labelled_legs.set_index("leg_id").loc["l1", "origin_zone"] == "A"
    assert labelled_legs.set_index("leg_id").loc["l1", "destination_zone"] == "B"
    assert labelled_legs.set_index("leg_id").loc["l2", "destination_zone"] == "Outside"
    assert relation.set_index("leg_id").loc["l1", "area_relation"] == "exchange"
    assert relation.set_index("leg_id").loc["l1", "area_relation_code"] == 0.5
    assert trips_out.loc[0, "trip_origin_zone"] == "A"
    assert trips_out.loc[0, "trip_destination_zone"] == "Outside"
    assert journeys_out.loc[0, "journey_origin_zone"] == "A"
    assert journeys_out.loc[0, "journey_destination_zone"] == "Outside"


def test_origin_destination_zone_correspondence_supports_multiple_layers_and_trips():
    zones = gpd.GeoDataFrame(
        {
            "zone_code": ["A", "B"],
            "zone_name": ["Alpha", "Beta"],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    communes = gpd.GeoDataFrame(
        {
            "NUMERO_COM": ["100", "200"],
            "COMMUNE": ["Commune A", "Commune B"],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    legs = gpd.GeoDataFrame(
        {
            "leg_id": ["l1", "l2"],
            "user_id": ["u1", "u1"],
            "experiment_name": ["exp", "exp"],
            "geometry": [
                LineString([(1, 1), (3, 1)]),
                LineString([(3, 1), (5, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    mapping = pd.DataFrame(
        [
            {"leg_id": "l1", "trip_id": "t1", "journey_id": "j1"},
            {"leg_id": "l2", "trip_id": "t1", "journey_id": "j1"},
        ]
    )

    correspondence = build_origin_destination_zone_correspondence(
        legs,
        {
            "girec": {"zones": zones, "zone_id_col": "zone_code", "zone_name_col": "zone_name"},
            "commune": {"zones": communes, "zone_id_col": "NUMERO_COM", "zone_name_col": "COMMUNE"},
        },
        map_track_trip_journey=mapping,
    )

    leg_rows = correspondence.loc[correspondence["object_type"].eq("leg")].set_index("leg_id")
    trip_rows = correspondence.loc[correspondence["object_type"].eq("trip")].set_index("trip_id")
    assert leg_rows.loc["l1", "girec_zone_o"] == "A"
    assert leg_rows.loc["l1", "girec_zone_d"] == "B"
    assert leg_rows.loc["l1", "commune_zone_o"] == "100"
    assert leg_rows.loc["l2", "commune_zone_d"] == "Outside"
    assert trip_rows.loc["t1", "girec_zone_o"] == "A"
    assert trip_rows.loc["t1", "girec_zone_d"] == "Outside"
    assert trip_rows.loc["t1", "commune_zone_o"] == "100"
    assert trip_rows.loc["t1", "commune_zone_d"] == "Outside"


def test_origin_destination_zone_correspondence_generates_zone_ids_with_warning():
    zones_without_id = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    legs = gpd.GeoDataFrame(
        {"leg_id": ["l1"], "geometry": [LineString([(1, 1), (1.5, 1.5)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.warns(UserWarning) as warnings:
        correspondence = build_origin_destination_zone_correspondence(
            legs,
            {"custom": zones_without_id},
        )

    warning_messages = [str(warning.message) for warning in warnings]
    assert any("zone id column" in message for message in warning_messages)
    assert any("zone name column" in message for message in warning_messages)
    leg_row = correspondence.loc[correspondence["object_type"].eq("leg")].iloc[0]
    assert leg_row["custom_zone_o"] == "custom_000001"
    assert leg_row["custom_zone_d"] == "custom_000001"


def test_add_signal_quality_flags_by_mode_and_user():
    config = _config()
    legs = gpd.GeoDataFrame(
        [
            {
                "leg_id": "good",
                "user_id": "u1",
                "mode": "Mode::Walk",
                "geometry": LineString([(2_600_000, 1_100_000), (2_600_001, 1_100_000), (2_600_002, 1_100_000)]),
            },
            {
                "leg_id": "bad",
                "user_id": "u2",
                "mode": "Mode::Walk",
                "geometry": LineString([(2_600_000, 1_100_000), (2_600_003, 1_100_000), (2_600_003.1, 1_100_000)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:2056",
    )

    out = add_signal_quality_flags(legs, config)
    user_stats = add_signal_quality_to_user_stats(pd.DataFrame({"user_id": ["u1", "u2"]}), out)

    assert out.loc[out["leg_id"].eq("good"), "low_quality_legs_1"].item() == 0
    assert out.loc[out["leg_id"].eq("bad"), "low_quality_legs_1"].item() == 1
    assert out.loc[out["leg_id"].eq("bad"), "low_quality_legs_2"].item() == 1
    assert out.loc[out["leg_id"].eq("bad"), "signal_quality_reason"].item() == "walking_signal_loss_level_1"
    assert user_stats.loc[user_stats["user_id"].eq("u2"), "low_quality_legs_count"].item() == 1
    assert user_stats.loc[user_stats["user_id"].eq("u2"), "bad_signal_user"].item() == 1


def test_prepare_mobility_dataset_adds_signal_quality_columns():
    storyline, trips, journeys = _raw_tables()
    config = _config()
    raw = RawGpsData(storyline=storyline, trips=trips, journeys=journeys)

    dataset = prepare_mobility_dataset(raw, config)

    assert "relative_signal_loss" in dataset.legs.columns
    assert "low_quality_legs_1" in dataset.legs.columns
    assert "bad_signal_user" in dataset.legs.columns
    assert "low_quality_legs_count" in dataset.user_stats.columns


def test_filter_mobility_dataset_by_users_keeps_mapping_tables_consistent():
    storyline, trips, journeys = _raw_tables_for_two_users()
    config = _config()
    raw = RawGpsData(storyline=storyline, trips=trips, journeys=journeys)

    dataset = prepare_mobility_dataset(raw, config)
    filtered = filter_mobility_dataset_by_users(dataset, ["u1"])

    assert set(filtered.storyline["user_id"]) == {"u1"}
    assert set(filtered.legs["user_id"]) == {"u1"}
    assert set(filtered.trips["user_id"]) == {"u1"}
    assert set(filtered.journeys["user_id"]) == {"u1"}
    assert set(filtered.user_stats["user_id"]) == {"u1"}
    assert set(filtered.map_track_trip_journey["leg_id"]) == set(filtered.legs["leg_id"])
    assert set(filtered.map_legs_staypoints["activity_id"]) == set(filtered.staypoints["activity_id"])
    assert filtered.metadata["filtered_dataset"] is True
    assert filtered.metadata["filtered_user_count"] == 1
    assert filtered.metadata["source_user_count"] == 2


def test_write_mobility_dataset_exports_tables_and_reports(tmp_path):
    storyline, trips, journeys = _raw_tables()
    config = _config()
    raw = RawGpsData(storyline=storyline, trips=trips, journeys=journeys)
    dataset = prepare_mobility_dataset(raw, config)
    selection_table = build_user_selection_table(dataset.user_stats)

    manifest = write_mobility_dataset(
        dataset,
        tmp_path / "exports",
        formats=("csv", "geojson", "pkl"),
        extra_tables={"user_presence": pd.DataFrame({"user_id": ["u1"], "source": ["test"]})},
        selection_table=selection_table,
    )
    table_names = set(manifest["table"])

    assert set(mobility_dataset_tables(dataset)) <= table_names
    assert "validation_issues" in table_names
    assert "user_presence" in table_names
    assert "quality_tracking_report" in table_names
    assert "quality_user_selection" in table_names
    assert (tmp_path / "exports" / "legs.csv").exists()
    assert (tmp_path / "exports" / "legs.geojson").exists()
    assert (tmp_path / "exports" / "legs.pkl").exists()
    assert (tmp_path / "exports" / "user_stats.csv").exists()
    assert (tmp_path / "exports" / "user_presence.csv").exists()
    assert (tmp_path / "exports" / "export_manifest.json").exists()


def test_build_attribute_dictionary_documents_tables_columns_and_na_counts():
    tables = {
        "legs": gpd.GeoDataFrame(
            {
                "leg_id": ["l1", "l2"],
                "mode": ["Mode::Walk", None],
                "length": [120.0, None],
                "geometry": [LineString([(6.1, 46.2), (6.2, 46.3)]), None],
            },
            geometry="geometry",
            crs="EPSG:4326",
        ),
        "map_track_trip_journey": pd.DataFrame(
            {"leg_id": ["l1"], "trip_id": ["t1"], "journey_id": [pd.NA]}
        ),
    }

    table_descriptions = build_table_descriptions(tables)
    attribute_dictionary = build_attribute_dictionary(tables)

    assert set(table_descriptions["table"]) == {"legs", "map_track_trip_journey"}
    assert table_descriptions.loc[table_descriptions["table"].eq("legs"), "row_count"].iat[0] == 2
    length_row = attribute_dictionary[
        attribute_dictionary["table"].eq("legs") & attribute_dictionary["column"].eq("length")
    ].iloc[0]
    geometry_row = attribute_dictionary[
        attribute_dictionary["table"].eq("legs") & attribute_dictionary["column"].eq("geometry")
    ].iloc[0]
    assert length_row["na_count"] == 1
    assert length_row["dtype"] == "float64"
    assert "Longueur" in length_row["description"]
    assert geometry_row["dtype"] == "geometry"
    assert geometry_row["geometry_crs"] == "EPSG:4326"


def test_prepare_mobility_dataset_can_skip_optional_wp1_enrichments():
    storyline, trips, journeys = _raw_tables()
    config = _config()
    raw = RawGpsData(storyline=storyline, trips=trips, journeys=journeys)

    dataset = prepare_mobility_dataset(
        raw,
        config,
        add_length_outlier_flags=False,
        add_signal_quality_flags=False,
    )

    assert len(dataset.legs) == 1
    assert "user_id_day" in dataset.legs.columns
    assert "relative_signal_loss" not in dataset.legs.columns
    assert "extreme98_length_mode" not in dataset.legs.columns
    assert not bool(dataset.user_stats.loc[0, "signal_quality_computed"])
    assert pd.isna(dataset.user_stats.loc[0, "bad_signal_user"])
    assert pd.isna(dataset.user_stats.loc[0, "low_quality_legs_count"])


def test_user_selection_requires_computed_signal_quality_when_signal_filter_is_enabled():
    user_stats = pd.DataFrame(
        {
            "user_id": ["u1"],
            "tracking_quality_ok": [True],
            "tracking_quality_reason": ["ok"],
            "signal_quality_computed": [False],
            "bad_signal_user": [pd.NA],
        }
    )

    with pytest.raises(ValueError, match="Signal-quality filtering was requested"):
        build_user_selection_table(user_stats, exclude_bad_signal_users=True)

    selection_table = build_user_selection_table(user_stats, exclude_bad_signal_users=False)

    assert selection_table.loc[0, "analysis_user_ok"]


def test_compute_mobility_indicators_builds_person_day_and_population_tables():
    storyline, trips, journeys = _raw_tables()
    config = ProjectConfig(
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
    dataset = prepare_mobility_dataset(
        RawGpsData(storyline=storyline, trips=trips, journeys=journeys),
        config,
    )

    result = compute_mobility_indicators(dataset, mode_col="mode_niv1")
    person_day = result.person_day
    population = result.population

    walk_day = person_day.loc[person_day["mode"].eq("Marche")].iloc[0]
    walk_pop = population.loc[population["mode"].eq("Marche")].iloc[0]

    assert set(result.tables()) == {"person_day", "person_phase", "population"}
    assert walk_day["distance_km"] == 1.2
    assert walk_day["travel_time_min"] == 30
    assert walk_day["trip_count"] == 1
    assert walk_day["leg_count"] == 1
    assert walk_pop["distance_km_per_day"] == 1.2
    assert walk_pop["travel_time_min_per_day"] == 30
    assert walk_pop["trip_count_per_day"] == 1
    assert walk_pop["leg_count_per_day"] == 1
    assert result.metadata["n_users"] == 1
    assert result.metadata["include_zero_days"]
    assert not result.metadata["include_excursions"]
    demand_profile = result.metadata["demand_profile_5min"]
    phase_profile = demand_profile["phases"][0]
    all_modes = phase_profile["series"][0]
    assert demand_profile["bin_minutes"] == 5
    assert demand_profile["heatmap_bin_minutes"] == 60
    assert phase_profile["phase"] == "Phase1"
    assert all_modes["mode"] == "Tous modes"
    assert max(all_modes["values"]) == 1.0
    assert demand_profile["heatmaps"][0]["label"] == "Jours de semaine"
    direct_profile = build_daily_demand_profile(dataset.legs, person_day, mode_col="mode_niv1")
    assert direct_profile["phases"][0]["series"][0]["mode"] == "Tous modes"
    assert direct_profile["heatmaps"]


def test_compute_mobility_indicators_alias_is_available():
    storyline, trips, journeys = _raw_tables()
    dataset = prepare_mobility_dataset(
        RawGpsData(storyline=storyline, trips=trips, journeys=journeys),
        _config(),
    )

    result = compute_mobility_indicators(dataset, mode_col="mode_niv1")

    assert isinstance(result, IndicatorResult)
    assert not result.population.empty


def test_co2_enrichment_prefers_computed_occupancy_by_default():
    legs = pd.DataFrame(
        [
            {
                "leg_id": "leg-1",
                "user_id": "u1",
                "mode": "Mode::Car",
                "length": 10_000,
                "duration": 1200,
                "occupancy": 3.0,
            }
        ]
    )

    computed = add_co2_occupancy_metrics(legs, config=_co2_test_config())
    observed = add_co2_occupancy_metrics(
        legs,
        config=_co2_test_config(),
        prefer_observed_occupancy=True,
    )

    assert computed.loc[0, "occupancy_source"] == "computed"
    assert computed.loc[0, "occupancy_rate"] != 3.0
    assert observed.loc[0, "occupancy_source"] == "observed"
    assert observed.loc[0, "occupancy_rate"] == 3.0
    assert observed.loc[0, "co2_g"] < computed.loc[0, "co2_g"]


def test_enrichment_metrics_are_available_in_wp2_indicators():
    storyline, trips, journeys = _raw_tables()
    config = ProjectConfig(
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
    dataset = prepare_mobility_dataset(
        RawGpsData(storyline=storyline, trips=trips, journeys=journeys),
        config,
    )

    legs = add_co2_occupancy_metrics(
        dataset.legs,
        journeys=dataset.journeys,
        map_track_trip_journey=dataset.map_track_trip_journey,
        config=_co2_test_config(),
    )
    legs = add_health_metrics(legs, config=_health_test_config())
    dataset.legs = legs

    result = compute_mobility_indicators(dataset, mode_col="mode_niv1")

    assert {"co2_kg", "co2_direct_kg", "calories_kcal", "mets"}.issubset(result.person_day.columns)
    assert {"co2_kg_per_day", "calories_kcal_per_day", "mets_per_day"}.issubset(result.population.columns)
    walk_day = result.person_day.loc[result.person_day["mode"].eq("Marche")].iloc[0]
    assert walk_day["co2_kg"] == 0
    assert walk_day["calories_kcal"] > 0


def test_compute_mobility_indicators_uses_configured_phase_calendar_for_zero_days():
    storyline, trips, journeys = _raw_tables()
    config = ProjectConfig(
        experiment_name="test",
        motiontag_project_name="Test",
        period="2025-01-01--2025-01-03",
        raw_data_dir=".",
        phases=(Phase("baseline", "2025-01-01", "2025-01-03"),),
        tracking_thresholds=TrackingThresholds(
            min_days_by_phase={"baseline": 1},
            min_total_tracked_days=1,
            round_to_full_weeks=False,
        ),
    )
    dataset = prepare_mobility_dataset(
        RawGpsData(storyline=storyline, trips=trips, journeys=journeys),
        config,
    )

    result = compute_mobility_indicators(dataset, mode_col="mode_niv1")
    marche = result.person_day[result.person_day["mode"].eq("Marche")]

    assert set(marche["phase"]) == {"baseline"}
    assert set(pd.to_datetime(marche["date"]).dt.date.astype(str)) == {"2025-01-01"}
    assert result.metadata["day_basis"] == "days_in_range"
    assert marche["trip_count"].sum() == 1


def test_prepare_mobility_dataset_supports_generic_config_without_experiment_or_phases():
    storyline, trips, journeys = _raw_tables()
    config = ProjectConfig()

    dataset = prepare_mobility_dataset(
        RawGpsData(storyline=storyline, trips=trips, journeys=journeys),
        config,
        resample_missing_days=True,
    )

    assert "Phase" not in dataset.storyline.columns
    assert "Phase" not in dataset.legs.columns
    assert dataset.metadata["project"]["experiment_name"] is None
    assert dataset.metadata["project"]["phases"] == []


def test_prepare_mobility_dataset_keeps_storyline_ids_for_resampled_rows():
    storyline, trips, journeys = _raw_tables()
    extra_stay = storyline.iloc[[0]].copy()
    extra_stay["storyline_id"] = "stay-3"
    extra_stay["started_at"] = "2025-01-03T08:00:00Z"
    extra_stay["finished_at"] = "2025-01-03T09:00:00Z"
    extra_stay["started_on"] = "2025-01-03"
    extra_stay["started_at_in_timezone"] = "2025-01-03T09:00:00+01:00"
    extra_stay["finished_at_in_timezone"] = "2025-01-03T10:00:00+01:00"
    storyline = pd.concat([storyline, extra_stay], ignore_index=True)

    dataset = prepare_mobility_dataset(
        RawGpsData(storyline=storyline, trips=trips, journeys=journeys),
        _config(),
        resample_missing_days=True,
    )
    resampled = dataset.storyline.loc[dataset.storyline["type"].eq("Resampled_stay")]

    assert len(resampled) == 1
    assert resampled["storyline_id"].notna().all()
    assert "id" not in resampled.columns
    assert dataset.storyline["storyline_id"].isna().sum() == 0


def test_prepare_mobility_dataset_merges_weights_when_provided():
    storyline, trips, journeys = _raw_tables()
    weights = pd.DataFrame({"user_id": ["u1"], "weight": [2.5]})

    dataset = prepare_mobility_dataset(
        RawGpsData(storyline=storyline, trips=trips, journeys=journeys),
        ProjectConfig(),
        weights=weights,
    )

    assert dataset.user_stats.loc[0, "weight"] == 2.5


def test_compute_mobility_indicators_uses_default_period_when_no_phase_is_present():
    storyline, trips, journeys = _raw_tables()
    dataset = prepare_mobility_dataset(
        RawGpsData(storyline=storyline, trips=trips, journeys=journeys),
        ProjectConfig(),
    )

    result = compute_mobility_indicators(dataset, include_zero_days=True)

    assert set(result.person_day["phase"]) == {"All"}
    assert set(result.person_phase["phase"]) == {"All"}
    assert set(result.population["phase"]) == {"All"}


def test_compute_mobility_indicators_can_exclude_excursions():
    storyline, trips, journeys = _raw_tables()
    dataset = prepare_mobility_dataset(
        RawGpsData(storyline=storyline, trips=trips, journeys=journeys),
        ProjectConfig(),
    )
    dataset.legs["excursion"] = 1
    dataset.trips["excursion"] = 1

    result = compute_mobility_indicators(
        dataset,
        include_zero_days=False,
        include_excursions=False,
    )

    assert result.person_day.empty
    assert result.metadata["include_excursions"] is False
    assert result.metadata["excursions_available"] is True


def test_compute_mobility_indicators_excludes_airplane_by_default():
    storyline, trips, journeys = _raw_tables()
    dataset = prepare_mobility_dataset(
        RawGpsData(storyline=storyline, trips=trips, journeys=journeys),
        ProjectConfig(),
    )
    airplane_leg = dataset.legs.iloc[[0]].copy()
    airplane_leg["leg_id"] = "leg-airplane"
    airplane_leg["storyline_id"] = "leg-airplane"
    airplane_leg["mode"] = "Mode::Airplane"
    airplane_leg["mode_niv1"] = "Autres"
    airplane_leg["mode_mrmt"] = "Airplane"
    airplane_leg["length"] = 100_000
    airplane_trip = dataset.trips.iloc[[0]].copy()
    airplane_trip["trip_id"] = "trip-airplane"
    airplane_trip["main_mode"] = "Airplane"
    airplane_trip["main_mode_niv1"] = "Autres"
    airplane_trip["main_mode_mrmt"] = "Airplane"
    dataset.legs = pd.concat([dataset.legs, airplane_leg], ignore_index=True)
    dataset.trips = pd.concat([dataset.trips, airplane_trip], ignore_index=True)

    default_result = compute_mobility_indicators(
        dataset,
        mode_col="mode_mrmt",
        trips_mode_col="main_mode_mrmt",
        include_zero_days=False,
    )
    with_airplane = compute_mobility_indicators(
        dataset,
        mode_col="mode_mrmt",
        trips_mode_col="main_mode_mrmt",
        include_zero_days=False,
        include_airplane=True,
    )

    assert "Airplane" not in set(default_result.person_day["mode"])
    assert default_result.metadata["include_airplane"] is False
    assert default_result.metadata["airplane_rows_excluded"]["legs"] == 1
    assert default_result.metadata["airplane_rows_excluded"]["trips"] == 1
    assert "Airplane" in set(with_airplane.person_day["mode"])
    assert with_airplane.metadata["include_airplane"] is True


def test_weekly_participation_grid_uses_observed_period_without_configured_phases():
    storyline = pd.DataFrame(
        [
            {"user_id": "u1", "type": "Track", "started_at": "2025-01-01T08:00:00Z"},
            {"user_id": "u1", "type": "Track", "started_at": "2025-01-02T08:00:00Z"},
            {"user_id": "u1", "type": "Track", "started_at": "2025-01-08T08:00:00Z"},
            {"user_id": "u2", "type": "Track", "started_at": "2025-01-09T08:00:00Z"},
        ]
    )

    grid = build_weekly_participation_grid(
        storyline,
        ProjectConfig(),
        user_ids=["u1", "u2"],
    )

    u1_week_1 = grid.loc[grid["user_id"].eq("u1") & grid["protocol_week_number"].eq(1)].iloc[0]
    u1_week_2 = grid.loc[grid["user_id"].eq("u1") & grid["protocol_week_number"].eq(2)].iloc[0]
    u2_week_1 = grid.loc[grid["user_id"].eq("u2") & grid["protocol_week_number"].eq(1)].iloc[0]

    assert "experiment_name" not in grid.columns
    assert set(grid["phase"]) == {"All"}
    assert u1_week_1["active_days_count"] == 2
    assert u1_week_2["active_days_count"] == 1
    assert u2_week_1["active_days_count"] == 0


def test_plot_participation_heatmap_returns_github_style_html():
    participation_grid = pd.DataFrame(
        [
            {"user_id": "u1", "protocol_week_number": 1, "active_days_count": 0, "phase": "Phase1"},
            {"user_id": "u1", "protocol_week_number": 2, "active_days_count": 7, "phase": "Phase2"},
            {"user_id": "u2", "protocol_week_number": 1, "active_days_count": 3, "phase": "Phase1"},
            {"user_id": "u2", "protocol_week_number": 2, "active_days_count": 5, "phase": "Phase2"},
        ]
    )

    html = plot_participation_heatmap(
        participation_grid,
        title="Test participation",
        experiment_name="test-experiment",
        cell_width=16,
        cell_height=8,
        user_label_colors={"u2": "#e9413f"},
    )
    html_text = html.data if hasattr(html, "data") else html

    assert "Test participation" in html_text
    assert "test-experiment" in html_text
    assert "width: 16px" in html_text
    assert "height: 8px" in html_text
    assert "u1" in html_text
    assert "u2" in html_text
    assert "color:#e9413f" in html_text
    assert "#216e39" in html_text
    assert "xyt-participation-cell" in html_text
    assert "xyt-participation-phase-separator" in html_text
    assert "Changement de phase" in html_text


def test_plot_participation_heatmap_can_export_png(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    participation_grid = pd.DataFrame(
        [
            {"user_id": "u1", "protocol_week_number": 1, "active_days_count": 0, "phase": "Phase1"},
            {"user_id": "u1", "protocol_week_number": 2, "active_days_count": 7, "phase": "Phase2"},
            {"user_id": "u2", "protocol_week_number": 1, "active_days_count": 3, "phase": "Phase1"},
            {"user_id": "u2", "protocol_week_number": 2, "active_days_count": 5, "phase": "Phase2"},
        ]
    )
    png_path = tmp_path / "participation.png"

    plot_participation_heatmap(
        participation_grid,
        title="PNG participation",
        experiment_name="test-experiment",
        cell_width=16,
        cell_height=8,
        save_png_path=png_path,
    )

    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_plot_indicator_bars_returns_html_from_indicator_result():
    indicators = IndicatorResult(
        person_day=pd.DataFrame(),
        person_phase=pd.DataFrame(),
        population=pd.DataFrame(
            [
                {
                    "phase": "All",
                    "mode": "Marche",
                    "distance_km_per_day": 4.5,
                    "travel_time_min_per_day": 42.0,
                    "trip_count_per_day": 3.0,
                },
                {
                    "phase": "All",
                    "mode": "Vélo",
                    "distance_km_per_day": 18.25,
                    "travel_time_min_per_day": 35.0,
                    "trip_count_per_day": 1.0,
                },
            ]
        ),
        mode_col="mode",
        metadata={
            "n_users": 2,
            "source_user_count": 3,
            "mean_leg_count_per_user_day": 3.24,
            "include_zero_days": True,
            "filtered_dataset": True,
            "occupancy_sources": ["computed", "observed"],
            "mean_occupancy_rate_by_source": {"computed": 1.35, "observed": 1.7},
            "demand_profile_5min": {
                "bin_minutes": 5,
                "heatmap_bin_minutes": 60,
                "unit": "personnes moyennes en déplacement",
                "time_labels": ["00:00", "00:05", "00:10"],
                "phases": [
                    {
                        "phase": "Phase1",
                        "n_days": 1,
                        "series": [
                            {"mode": "Tous modes", "values": [0, 2, 1]},
                            {"mode": "Marche", "values": [0, 1, 0]},
                            {"mode": "Vélo", "values": [0, 1, 1]},
                        ],
                    }
                ],
                "heatmaps": [
                    {
                        "key": "weekday",
                        "label": "Jours de semaine",
                        "time_labels": ["00:00", "01:00", "02:00"],
                        "modes": [
                            {
                                "mode": "Tous modes",
                                "phases": [
                                    {
                                        "phase": "Phase1",
                                        "rows": [
                                            {"label": "Lundi", "values": [0, 2, 1]},
                                            {"label": "Mardi", "values": [1, 0, 0]},
                                        ],
                                    }
                                ],
                            },
                            {
                                "mode": "Marche",
                                "phases": [
                                    {
                                        "phase": "Phase1",
                                        "rows": [
                                            {"label": "Lundi", "values": [0, 1, 0]},
                                            {"label": "Mardi", "values": [1, 0, 0]},
                                        ],
                                    }
                                ],
                            },
                        ],
                    },
                    {
                        "key": "motif",
                        "label": "Motifs agrégés",
                        "source_col": "purpose_mrmt",
                        "time_labels": ["00:00", "01:00", "02:00"],
                        "modes": [
                            {
                                "mode": "Tous modes",
                                "phases": [
                                    {
                                        "phase": "Phase1",
                                        "rows": [
                                            {"label": "Domicile", "values": [0, 2, 0]},
                                            {"label": "Travail", "values": [0, 0, 1]},
                                        ],
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
        },
    )

    html = plot_indicator_bars(indicators, title="Indicateurs test")
    html_text = html.data if hasattr(html, "data") else html

    assert '<meta charset="utf-8">' in html_text
    assert "Indicateurs test" in html_text
    assert "Carte d'identité des données" in html_text
    assert "Distance par jour" in html_text
    assert "Déplacements par jour" in html_text
    assert "Trips par jour" not in html_text
    assert "Marche" in html_text
    assert "Vélo" in html_text
    assert "Tous modes" in html_text
    assert "#e84b8a" in html_text
    assert "18.2" in html_text
    assert "18.25" not in html_text
    assert "Utilisateurs" in html_text
    assert "Utilisateurs retenus / inscrits" in html_text
    assert "2/3" in html_text
    assert "Étapes moyennes" in html_text
    assert "Périmètre de calcul" in html_text
    assert "Sous-échantillon filtré" in html_text
    assert "Taux moyen d&#x27;occupation" in html_text
    assert "observé: 1.7" in html_text
    assert "Part occupation renseignée" not in html_text
    assert "xyt-indicator-bar" in html_text
    assert "Barre de référence" in html_text
    assert "catégorie « Autres »" in html_text
    assert "Demande journalière par heure" in html_text
    assert "Courbe de demande Phase1" in html_text
    assert "Configurations temporelles des fréquentations" in html_text
    assert "xyt-demand-heatmap-mode-select" in html_text
    assert "Motifs agrégés" in html_text
    assert "qualité des traces" in html_text
    highlighted = plot_indicator_bars(
        indicators,
        title="Indicateurs test",
        facet_colors={"All": "#e9413f"},
    )
    highlighted_text = highlighted.data if hasattr(highlighted, "data") else highlighted
    assert "color:#e9413f" in highlighted_text


def test_plot_indicator_bars_keeps_mode_order_stable_between_phases():
    indicators = pd.DataFrame(
        [
            {"phase": "Phase1", "mode": "Voiture", "co2_kg_per_day": 8.6},
            {"phase": "Phase1", "mode": "TP", "co2_kg_per_day": 0.1},
            {"phase": "Phase1", "mode": "Vélo", "co2_kg_per_day": 0.0},
            {"phase": "Phase1", "mode": "Autres", "co2_kg_per_day": 0.0},
            {"phase": "Phase1", "mode": "Marche", "co2_kg_per_day": 0.0},
            {"phase": "Phase2", "mode": "Voiture", "co2_kg_per_day": 7.5},
            {"phase": "Phase2", "mode": "Autres", "co2_kg_per_day": 3.7},
            {"phase": "Phase2", "mode": "TP", "co2_kg_per_day": 0.2},
            {"phase": "Phase2", "mode": "Vélo", "co2_kg_per_day": 0.0},
            {"phase": "Phase2", "mode": "Marche", "co2_kg_per_day": 0.0},
        ]
    )

    html = plot_indicator_bars(
        indicators,
        metrics=["co2_kg_per_day"],
        include_all_modes=False,
        show_identity_card=False,
    )
    html_text = html.data if hasattr(html, "data") else html
    phase2_html = html_text[html_text.index(">Phase2<") :]

    assert phase2_html.index(">Voiture<") < phase2_html.index(">TP<")
    assert phase2_html.index(">TP<") < phase2_html.index(">Vélo<")
    assert phase2_html.index(">Vélo<") < phase2_html.index(">Marche<")
    assert phase2_html.index(">Marche<") < phase2_html.index(">Autres<")


def test_plot_indicator_bars_uses_separate_scale_for_all_modes():
    indicators = pd.DataFrame(
        [
            {"phase": "Phase1", "mode": "Voiture", "distance_km_per_day": 10.0},
            {"phase": "Phase1", "mode": "TP", "distance_km_per_day": 2.0},
        ]
    )

    html = plot_indicator_bars(
        indicators,
        metrics=["distance_km_per_day"],
        include_all_modes=True,
        show_identity_card=False,
    )
    html_text = html.data if hasattr(html, "data") else html

    assert html_text.count("width:100.00%") >= 2
    assert "par mode" in html_text


def test_build_mobility_motifs_uses_canonical_daily_sequences():
    staypoints = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "started_at": "2025-01-01T08:00:00Z",
                "finished_at": "2025-01-01T09:00:00Z",
                "purpose_niv1": "Domicile",
                "lon": 6.10,
                "lat": 46.20,
            },
            {
                "user_id": "u1",
                "started_at": "2025-01-01T09:05:00Z",
                "finished_at": "2025-01-01T09:30:00Z",
                "purpose_niv1": "Domicile",
                "lon": 6.10,
                "lat": 46.20,
            },
            {
                "user_id": "u1",
                "started_at": "2025-01-01T10:00:00Z",
                "finished_at": "2025-01-01T17:00:00Z",
                "purpose_niv1": "Travail",
                "lon": 6.20,
                "lat": 46.25,
            },
            {
                "user_id": "u1",
                "started_at": "2025-01-01T18:00:00Z",
                "finished_at": "2025-01-01T23:00:00Z",
                "purpose_niv1": "Domicile",
                "lon": 6.10,
                "lat": 46.20,
            },
            {
                "user_id": "u2",
                "started_at": "2025-01-01T07:00:00Z",
                "finished_at": "2025-01-01T08:00:00Z",
                "purpose_niv1": "Domicile",
                "lon": 7.10,
                "lat": 47.20,
            },
            {
                "user_id": "u2",
                "started_at": "2025-01-01T09:00:00Z",
                "finished_at": "2025-01-01T16:00:00Z",
                "purpose_niv1": "Travail",
                "lon": 7.20,
                "lat": 47.25,
            },
            {
                "user_id": "u2",
                "started_at": "2025-01-01T17:00:00Z",
                "finished_at": "2025-01-01T23:00:00Z",
                "purpose_niv1": "Domicile",
                "lon": 7.10,
                "lat": 47.20,
            },
            {
                "user_id": "u1",
                "started_at": "2025-01-02T08:00:00Z",
                "finished_at": "2025-01-02T09:00:00Z",
                "purpose_niv1": "Domicile",
                "lon": 6.10,
                "lat": 46.20,
            },
            {
                "user_id": "u1",
                "started_at": "2025-01-02T10:00:00Z",
                "finished_at": "2025-01-02T11:00:00Z",
                "purpose_niv1": "Achats",
                "lon": 6.30,
                "lat": 46.30,
            },
        ]
    )

    motifs = build_mobility_motifs(staypoints)
    summary = summarize_mobility_motifs(motifs)
    sequences = build_mobility_motif_sequences(motifs, n_days=3, align_to_week=False)

    u1_day1 = motifs.loc[motifs["user_id_day"].eq("u1_20250101")].iloc[0]
    u2_day1 = motifs.loc[motifs["user_id_day"].eq("u2_20250101")].iloc[0]
    u1_sequence = sequences.loc[sequences["user_id"].eq("u1")].iloc[0]

    assert u1_day1["n_visits"] == 4
    assert u1_day1["canonical_sequence"] == (0, 1, 0)
    assert u1_day1["n_transitions"] == 2
    assert u1_day1["is_round_trip"]
    assert u1_day1["motif_signature"] == u2_day1["motif_signature"]
    assert u1_day1["motif_id"] == u2_day1["motif_id"] == 1
    assert summary.loc[summary["motif_id"].eq(1), "days_count"].item() == 2
    assert u1_sequence["day_001"] == 1
    assert u1_sequence["day_002"] == 2
    assert u1_sequence["day_003"] == 0

    motif_graphs = plot_mobility_motif_graphs(motifs, top_n=2)
    motif_graphs_html = motif_graphs.data if hasattr(motif_graphs, "data") else motif_graphs
    assert "Graphes des motifs journaliers" in motif_graphs_html
    assert "Motif 1" in motif_graphs_html
    assert "<svg" in motif_graphs_html


def test_load_sample_gps_derives_minimal_tables(tmp_path):
    storyline, trips, journeys = _raw_tables()
    sample_path = tmp_path / "sample.pkl"
    with sample_path.open("wb") as file:
        pickle.dump(storyline, file)

    raw = load_sample_gps(sample_path, user_id="demo_user")

    assert raw.validation["storyline"].ok
    assert raw.validation["trips"].ok
    assert raw.validation["journeys"].ok
    assert raw.storyline["user_id"].eq("demo_user").all()
    assert raw.storyline.loc[raw.storyline["type"].eq("Track"), "trip_id"].notna().all()
    assert len(raw.trips) == 1
    assert len(raw.journeys) == 1
    assert raw.user_statistics is not None
    assert raw.user_statistics.loc[0, "user_id"] == "demo_user"
