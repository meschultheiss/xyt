import sys
from pathlib import Path

import pandas as pd
import pytest


PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
NOTEBOOK_UTILS = Path(__file__).resolve().parents[2] / "Notebooks" / "utils"
if not NOTEBOOK_UTILS.exists():
    pytest.skip(
        "Projet-specific notebook utility tests require Notebooks/utils.",
        allow_module_level=True,
    )
sys.path.insert(0, str(PACKAGE_SRC))
sys.path.insert(0, str(NOTEBOOK_UTILS))

from xyt_gps import build_relative_time_index  # noqa: E402
from thematic_utils import (  # noqa: E402
    compare_track_event_summaries_by_experiment,
    filter_cleaned_payload_by_experiments,
    filter_thematic_dataset,
    summarize_track_events_by_experiment,
)


def test_build_relative_time_index_aligns_shifted_waves_on_sequence_week():
    tables = {
        "storyline": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-vague2"],
                "storyline_id": ["leg-v1", "leg-v2"],
                "type": ["Track", "Track"],
                "user_id": ["user-a", "user-b"],
                "phase": ["Phase2", "Phase2"],
                "phase_number": [2, 2],
                "n_week_sequence": [4, 4],
                "started_at": ["2026-03-25T06:30:00Z", "2026-05-06T06:30:00Z"],
                "finished_at": ["2026-03-25T06:50:00Z", "2026-05-06T06:45:00Z"],
                "started_at_timezone": ["Europe/Zurich", "Europe/Zurich"],
                "finished_at_timezone": ["Europe/Zurich", "Europe/Zurich"],
            }
        ),
        "map_track_trip_journey": pd.DataFrame(
            {
                "leg_id": ["leg-v1", "leg-v2"],
                "trip_id": ["trip-v1", "trip-v2"],
                "journey_id": ["journey-v1", "journey-v2"],
            }
        ),
    }

    index = build_relative_time_index(tables)

    assert len(index) == len(tables["storyline"])
    assert index["sequence_week_number"].tolist() == [4, 4]
    assert index["weekday_label"].tolist() == ["mercredi", "mercredi"]
    assert index["relative_started_at"].dt.date.astype(str).nunique() == 1
    assert index["absolute_started_date"].astype(str).tolist() == ["2026-03-25", "2026-05-06"]
    assert index["leg_id"].tolist() == ["leg-v1", "leg-v2"]
    assert index["trip_id"].tolist() == ["trip-v1", "trip-v2"]


def test_build_relative_time_index_keeps_rows_without_sequence_week():
    tables = {
        "storyline": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1"],
                "storyline_id": ["activity-other"],
                "type": ["Stay"],
                "user_id": ["user-a"],
                "phase": ["Other"],
                "n_week_sequence": [pd.NA],
                "started_at": ["2026-02-22T19:41:40Z"],
                "finished_at": ["2026-02-23T15:09:10Z"],
                "started_at_timezone": ["Europe/Zurich"],
                "finished_at_timezone": ["Europe/Zurich"],
            }
        ),
        "map_legs_staypoints": pd.DataFrame(
            {
                "activity_id": ["activity-other"],
                "previous_leg_id": ["leg-before"],
                "next_leg_id": ["leg-after"],
            }
        ),
    }

    index = build_relative_time_index(tables)

    assert len(index) == 1
    assert pd.isna(index.loc[0, "sequence_week_number"])
    assert pd.isna(index.loc[0, "relative_started_at"])
    assert index.loc[0, "weekday_label"] == "dimanche"
    assert index.loc[0, "activity_id"] == "activity-other"
    assert index.loc[0, "previous_leg_id"] == "leg-before"
    assert index.loc[0, "next_leg_id"] == "leg-after"


def test_build_relative_time_index_fills_missing_existing_mapping_columns():
    tables = {
        "storyline": pd.DataFrame(
            {
                "storyline_id": ["leg-a", "leg-b"],
                "type": ["Track", "Track"],
                "trip_id": [pd.NA, "trip-kept"],
                "started_at": ["2026-03-23 07:00:00+00:00", "2026-03-23 08:00:00+00:00"],
                "finished_at": ["2026-03-23 07:10:00+00:00", "2026-03-23 08:10:00+00:00"],
                "n_week_sequence": [1, 1],
            }
        ),
        "map_track_trip_journey": pd.DataFrame(
            {
                "leg_id": ["leg-a", "leg-b"],
                "trip_id": ["trip-filled", "trip-from-map"],
                "journey_id": ["journey-a", "journey-b"],
            }
        ),
    }

    index = build_relative_time_index(tables)

    assert index["trip_id"].tolist() == ["trip-filled", "trip-kept"]
    assert index["journey_id"].tolist() == ["journey-a", "journey-b"]


def test_build_relative_time_index_warns_when_anchor_is_not_monday():
    tables = {
        "storyline": pd.DataFrame(
            {
                "storyline_id": ["leg-a"],
                "type": ["Track"],
                "started_at": ["2026-03-23 07:00:00+00:00"],
                "finished_at": ["2026-03-23 07:10:00+00:00"],
                "n_week_sequence": [1],
            }
        )
    }

    with pytest.warns(UserWarning, match="relative_anchor"):
        build_relative_time_index(tables, relative_anchor="2000-01-04")


def test_summarize_and_compare_track_events_by_experiment():
    raw_storyline = pd.DataFrame(
        {
            "experiment_name": ["vague1", "vague1", "vague1", "ziplo"],
            "type": ["Track", "Track", "Stay", "Track"],
            "storyline_id": ["leg-a", "leg-b", "stay-a", "leg-z"],
            "trip_id": ["trip-a", "trip-a", "trip-a", "trip-z"],
            "user_id": ["u1", "u2", "u1", "u3"],
            "length": [1000, 2500, 0, 5000],
            "started_on": ["2026-03-01", "2026-03-02", "2026-03-02", "2026-04-01"],
        }
    )
    clean_legs = pd.DataFrame(
        {
            "experiment_name": ["vague1", "ziplo"],
            "leg_id": ["leg-a", "leg-z"],
            "trip_id": ["trip-a", "trip-z"],
            "user_id": ["u1", "u3"],
            "length": [900, 5000],
            "started_at": ["2026-03-01T07:00:00Z", "2026-04-01T08:00:00Z"],
        }
    )

    raw_summary = summarize_track_events_by_experiment(
        raw_storyline,
        experiment_names=["vague1", "ziplo"],
        event_id_column="storyline_id",
    )
    clean_summary = summarize_track_events_by_experiment(
        clean_legs,
        experiment_names=["vague1", "ziplo"],
        type_filter=None,
        event_id_column="leg_id",
    )
    comparison = compare_track_event_summaries_by_experiment(
        raw_summary,
        clean_summary,
        before_label="landing",
        after_label="clean",
    )

    vague1_raw = raw_summary.set_index("experiment_name").loc["vague1"]
    assert vague1_raw["leg_count"] == 2
    assert vague1_raw["trip_count"] == 1
    assert vague1_raw["user_count"] == 2
    assert vague1_raw["distance_km"] == 3.5
    assert vague1_raw["unique_days"] == 2

    vague1_comparison = comparison.set_index("experiment_name").loc["vague1"]
    assert vague1_comparison["delta_leg_count"] == -1
    assert vague1_comparison["retained_leg_count_pct"] == 50.0
    assert vague1_comparison["retained_distance_km_pct"] == pytest.approx(25.71)


def test_filter_cleaned_payload_by_experiments_keeps_relation_tables_by_ids():
    loaded = {
        "storyline": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-ziplo"],
                "user_id": ["u1", "u2"],
                "storyline_id": ["leg-a", "leg-b"],
                "leg_id": ["leg-a", "leg-b"],
                "trip_id": ["trip-a", "trip-b"],
                "phase": ["Phase1", "Phase1"],
            }
        ),
        "staypoints": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-ziplo"],
                "user_id": ["u1", "u2"],
                "activity_id": ["act-a", "act-b"],
                "phase": ["Phase1", "Phase1"],
            }
        ),
        "legs": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-ziplo"],
                "user_id": ["u1", "u2"],
                "leg_id": ["leg-a", "leg-b"],
                "trip_id": ["trip-a", "trip-b"],
                "phase": ["Phase1", "Phase1"],
            }
        ),
        "trips": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-ziplo"],
                "user_id": ["u1", "u2"],
                "trip_id": ["trip-a", "trip-b"],
                "journey_id": ["journey-a", "journey-b"],
                "phase": ["Phase1", "Phase1"],
            }
        ),
        "journeys": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-ziplo"],
                "user_id": ["u1", "u2"],
                "journey_id": ["journey-a", "journey-b"],
                "phase": ["Phase1", "Phase1"],
            }
        ),
        "user_stats": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-ziplo"],
                "user_id": ["u1", "u2"],
            }
        ),
        "user_day_coverage": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-ziplo"],
                "user_id": ["u1", "u2"],
                "phase": ["Phase1", "Phase1"],
            }
        ),
        "map_track_trip_journey": pd.DataFrame(
            {
                "experiment_name": ["declic-all", "declic-all"],
                "leg_id": ["leg-a", "leg-b"],
                "trip_id": ["trip-a", "trip-b"],
                "journey_id": ["journey-a", "journey-b"],
            }
        ),
        "map_legs_staypoints": pd.DataFrame(
            {
                "experiment_name": ["declic-all", "declic-all"],
                "activity_id": ["act-a", "act-b"],
            }
        ),
        "public_transport_legs": pd.DataFrame(),
        "user_expe": pd.DataFrame(),
        "occupancy_co2": pd.DataFrame(
            {"experiment_name": ["declic-all", "declic-all"], "leg_id": ["leg-a", "leg-b"]}
        ),
        "health": pd.DataFrame(
            {"experiment_name": ["declic-all", "declic-all"], "leg_id": ["leg-a", "leg-b"]}
        ),
        "od_zone_correspondence": pd.DataFrame(
            {
                "experiment_name": ["declic-all", "declic-all", "declic-all", "declic-all"],
                "object_type": ["leg", "leg", "trip", "trip"],
                "object_id": ["leg-a", "leg-b", "trip-a", "trip-b"],
                "leg_id": ["leg-a", "leg-b", pd.NA, pd.NA],
                "trip_id": ["trip-a", "trip-b", "trip-a", "trip-b"],
                "journey_id": ["journey-a", "journey-b", "journey-a", "journey-b"],
                "modus_zone_o": ["1", "2", "1", "2"],
                "modus_zone_d": ["3", "4", "3", "4"],
            }
        ),
        "relative_time_index": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-ziplo"],
                "user_id": ["u1", "u2"],
                "leg_id": ["leg-a", "leg-b"],
                "phase": ["Phase1", "Phase1"],
            }
        ),
        "questionnaires": {
            "declic-all": {
                "questionnaire_declic": pd.DataFrame(
                    {
                        "experiment_name": ["declic-mobility-vague1", "declic-mobility-ziplo"],
                        "user_id": ["u1", "u2"],
                    }
                )
            }
        },
    }

    filtered = filter_cleaned_payload_by_experiments(
        loaded,
        ["declic-mobility-vague1"],
        group_name="declic-mobilite-2025-2026",
    )

    assert filtered["selected_experiments"] == ["declic-mobility-vague1"]
    assert filtered["legs"]["leg_id"].tolist() == ["leg-a"]
    assert filtered["health"]["leg_id"].tolist() == ["leg-a"]
    assert filtered["health"]["experiment_name"].tolist() == ["declic-mobility-vague1"]
    assert filtered["map_track_trip_journey"]["leg_id"].tolist() == ["leg-a"]
    assert filtered["map_legs_staypoints"]["activity_id"].tolist() == ["act-a"]
    assert filtered["od_zone_correspondence"]["object_id"].tolist() == ["leg-a", "trip-a"]
    assert filtered["od_zone_correspondence"]["experiment_name"].tolist() == [
        "declic-mobility-vague1",
        "declic-mobility-vague1",
    ]
    assert set(filtered["questionnaires"]) == {"declic-mobility-vague1"}
    assert filtered["questionnaires"]["declic-mobility-vague1"]["questionnaire_declic"][
        "user_id"
    ].tolist() == ["u1"]


def test_filter_thematic_dataset_filters_od_zone_correspondence_by_leg_and_trip_ids():
    loaded = {
        "storyline": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-vague1"],
                "user_id": ["u1", "u1"],
                "storyline_id": ["leg-a", "leg-b"],
                "leg_id": ["leg-a", "leg-b"],
                "trip_id": ["trip-a", "trip-b"],
                "journey_id": ["journey-a", "journey-b"],
                "phase": ["Phase1", "Phase2"],
            }
        ),
        "staypoints": pd.DataFrame(),
        "legs": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-vague1"],
                "user_id": ["u1", "u1"],
                "leg_id": ["leg-a", "leg-b"],
                "trip_id": ["trip-a", "trip-b"],
                "journey_id": ["journey-a", "journey-b"],
                "phase": ["Phase1", "Phase2"],
            }
        ),
        "trips": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-vague1"],
                "user_id": ["u1", "u1"],
                "trip_id": ["trip-a", "trip-b"],
                "journey_id": ["journey-a", "journey-b"],
                "phase": ["Phase1", "Phase2"],
            }
        ),
        "journeys": pd.DataFrame(),
        "user_stats": pd.DataFrame({"experiment_name": ["declic-mobility-vague1"], "user_id": ["u1"]}),
        "user_day_coverage": pd.DataFrame(
            {"experiment_name": ["declic-mobility-vague1"], "user_id": ["u1"], "phase": ["Phase1"]}
        ),
        "map_track_trip_journey": pd.DataFrame(),
        "map_legs_staypoints": pd.DataFrame(),
        "public_transport_legs": pd.DataFrame(),
        "user_expe": pd.DataFrame(),
        "occupancy_co2": pd.DataFrame(),
        "health": pd.DataFrame(),
        "od_zone_correspondence": pd.DataFrame(
            {
                "experiment_name": ["declic-mobility-vague1", "declic-mobility-vague1", "declic-mobility-vague1", "declic-mobility-vague1"],
                "object_type": ["leg", "leg", "trip", "trip"],
                "object_id": ["leg-a", "leg-b", "trip-a", "trip-b"],
                "leg_id": ["leg-a", "leg-b", pd.NA, pd.NA],
                "trip_id": ["trip-a", "trip-b", "trip-a", "trip-b"],
                "journey_id": ["journey-a", "journey-b", "journey-a", "journey-b"],
            }
        ),
        "relative_time_index": pd.DataFrame(),
        "questionnaires": {},
    }

    filtered = filter_thematic_dataset(
        loaded,
        experiment_names=["declic-mobility-vague1"],
        phases=["Phase1"],
    )

    assert filtered["od_zone_correspondence"]["object_id"].tolist() == ["leg-a", "trip-a"]
