import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PACKAGE_READY_UTILS_DIR = (
    Path(__file__).resolve().parents[2] / "Notebooks" / "utils"
)
if not PACKAGE_READY_UTILS_DIR.exists():
    pytest.skip(
        "Projet-specific notebook utility tests require Notebooks/utils.",
        allow_module_level=True,
    )
sys.path.insert(0, str(PACKAGE_READY_UTILS_DIR))

from data_landing import (  # noqa: E402
    build_landed_tables,
    build_user_presence_table,
    load_config_bundle,
    load_experiment_data,
    scan_landing_alias_columns,
    write_landing_output,
)


def test_load_config_bundle_supports_shared_and_per_experiment_files(tmp_path):
    config_dir = tmp_path / "config"
    experiments_dir = config_dir / "experiments"
    experiments_dir.mkdir(parents=True)
    raw_dir = tmp_path / "raw" / "ziplo"
    questionnaire_path = tmp_path / "questionnaires" / "ziplo.csv"
    user_mapping_path = tmp_path / "user_mapping.csv"

    (config_dir / "shared.json").write_text(
        """
{
  "user-mapping-table": {"path": "../user_mapping.csv"},
  "landing-column-contract": {
    "tables": {
      "storyline": {"must_have": ["user_id"], "nice_to_have": ["geometry"]}
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    (experiments_dir / "declic-mobility-ziplo.json").write_text(
        """
{
  "experiment_name": "declic-mobility-ziplo",
  "storyline_name": "declic-mobility-ziplo",
  "motiontag_project_name": null,
  "dump_date_range": "2026-02-23--2026-05-20",
  "paths": {
    "gps_dump_dir": "../raw/ziplo",
    "storyline": "../raw/ziplo/storyline.csv",
    "trips": "../raw/ziplo/trips.csv",
    "journeys": "../raw/ziplo/journeys.csv",
    "user_statistics": "../raw/ziplo/user_statistics.csv",
    "public_transport_legs": "../raw/ziplo/public_transport_legs.csv",
    "questionnaire": "../questionnaires/ziplo.csv"
  },
  "phase_1": ["2026-03-02", "2026-03-09"],
  "phase_2": ["2026-03-10", "2026-07-12"],
  "phase_3": ["2026-07-13", "2026-07-26"]
}
""".strip(),
        encoding="utf-8",
    )
    bundle = load_config_bundle(config_dir)

    assert bundle["user-mapping-table"]["path"] == str(user_mapping_path.resolve())
    assert bundle["landing-column-contract"]["tables"]["storyline"]["must_have"] == ["user_id"]
    assert [item["experiment_name"] for item in bundle["config-experiment"]] == ["declic-mobility-ziplo"]
    experiment = bundle["config-experiment"][0]
    assert experiment["paths"]["gps_dump_dir"] == str(raw_dir.resolve())
    assert experiment["paths"]["questionnaire"] == str(questionnaire_path.resolve())
    assert experiment["paths"]["storyline"] == str((raw_dir / "storyline.csv").resolve())
    assert experiment["paths"]["public_transport_legs"] == str((raw_dir / "public_transport_legs.csv").resolve())
    assert experiment["phase_1"] == ("2026-03-02", "2026-03-09")


def test_load_experiment_data_concatenates_ordered_csv_parts(tmp_path):
    storyline_part_1 = tmp_path / "storyline-part1.csv"
    storyline_part_2 = tmp_path / "storyline-part2.csv"
    trips_path = tmp_path / "trips.csv"
    journeys_path = tmp_path / "journeys.csv"
    user_statistics_path = tmp_path / "user-statistics.csv"

    pd.DataFrame({"id": ["storyline-1"], "user_id": ["user-1"]}).to_csv(
        storyline_part_1,
        sep=";",
        index=False,
    )
    pd.DataFrame({"id": ["storyline-2"], "user_id": ["user-2"]}).to_csv(
        storyline_part_2,
        sep=";",
        index=False,
    )
    for path in (trips_path, journeys_path, user_statistics_path):
        pd.DataFrame({"id": ["row-1"], "user_id": ["user-1"]}).to_csv(
            path,
            sep=";",
            index=False,
        )

    payload = load_experiment_data(
        {
            "experiment_name": "declic-all",
            "paths": {
                "storyline": [str(storyline_part_1), str(storyline_part_2)],
                "trips": str(trips_path),
                "journeys": str(journeys_path),
                "user_statistics": str(user_statistics_path),
            },
        }
    )

    assert payload["paths"]["storyline"] == [storyline_part_1, storyline_part_2]
    assert payload["storyline"]["id"].tolist() == ["storyline-1", "storyline-2"]


def test_load_experiment_data_loads_questionnaires_by_component(tmp_path):
    source_paths = {
        table_name: tmp_path / f"{table_name}.csv"
        for table_name in ("storyline", "trips", "journeys", "user_statistics")
    }
    for path in source_paths.values():
        pd.DataFrame({"id": ["row-1"], "user_id": ["user-1"]}).to_csv(
            path,
            sep=";",
            index=False,
        )

    declic_questionnaire_path = tmp_path / "questionnaire-declic.xlsx"
    ziplo_questionnaire_path = tmp_path / "questionnaire-ziplo.xlsx"
    pd.DataFrame(
        {"invite_code": ["invite-a", "invite-b"], "question_declic": [1, 2]}
    ).to_excel(declic_questionnaire_path, index=False)
    pd.DataFrame(
        {"invite_code": ["invite-z"], "question_ziplo": [3]}
    ).to_excel(ziplo_questionnaire_path, index=False)

    payload = load_experiment_data(
        {
            "experiment_name": "declic-all",
            "paths": {name: str(path) for name, path in source_paths.items()},
            "component_experiments": [
                {
                    "experiment_name": "declic-mobility-vague1",
                    "paths": {"questionnaire": str(declic_questionnaire_path)},
                },
                {
                    "experiment_name": "declic-mobility-vague2",
                    "paths": {"questionnaire": str(declic_questionnaire_path)},
                },
                {
                    "experiment_name": "declic-mobility-ziplo",
                    "paths": {"questionnaire": str(ziplo_questionnaire_path)},
                },
            ],
        }
    )

    assert set(payload["questionnaires"]) == {
        "declic-mobility-vague1",
        "declic-mobility-vague2",
        "declic-mobility-ziplo",
    }
    assert payload["questionnaires"]["declic-mobility-vague1"] is payload[
        "questionnaires"
    ]["declic-mobility-vague2"]
    assert payload["questionnaires"]["declic-mobility-ziplo"][
        "invite_code"
    ].tolist() == ["invite-z"]
    assert payload["questionnaire_paths"] == {
        "declic-mobility-vague1": declic_questionnaire_path,
        "declic-mobility-vague2": declic_questionnaire_path,
        "declic-mobility-ziplo": ziplo_questionnaire_path,
    }


def test_composite_landing_keeps_declic_and_ziplo_questionnaires_separate():
    component_names = [
        "declic-mobility-vague1",
        "declic-mobility-vague2",
        "declic-mobility-ziplo",
    ]
    users = ["user-a", "user-b", "user-z"]
    invite_codes = ["invite-a", "invite-b", "invite-z"]
    components = [
        {
            "experiment_name": experiment_name,
            "phase_1": ("2026-01-01", "2026-01-07"),
            "phase_2": (None, None),
            "phase_3": (None, None),
            "start_expe": "2026-01-01",
            "end_expe": "2026-01-07",
        }
        for experiment_name in component_names
    ]
    source_config = {
        "experiment_name": "declic-all",
        "component_experiments": components,
    }
    storyline = pd.DataFrame(
        [
            {
                "id": f"storyline-{user_id}",
                "user_id": user_id,
                "type": "Track",
                "started_at": "2026-01-03T08:00:00Z",
                "finished_at": "2026-01-03T08:30:00Z",
                "started_on": "2026-01-03",
                "geometry": "LINESTRING (0 0, 1 1)",
            }
            for user_id in users
        ]
    )
    trips = pd.DataFrame(
        {
            "id": [f"trip-{user_id}" for user_id in users],
            "user_id": users,
            "started_at": ["2026-01-03T08:00:00Z"] * len(users),
            "finished_at": ["2026-01-03T08:30:00Z"] * len(users),
        }
    )
    journeys = trips.rename(columns={"id": "journey_id"}).copy()
    user_statistics = pd.DataFrame(
        {"id": users, "active_days_count": [7, 7, 7]}
    )
    declic_questionnaire = pd.DataFrame(
        {
            "Invite Code": ["invite-a", "invite-b", "not-in-project"],
            "question_declic": [1, 2, 99],
        }
    )
    ziplo_questionnaire = pd.DataFrame(
        {"Invite Code": ["invite-z"], "question_ziplo": [3]}
    )
    experiment_data = {
        "declic-all": {
            "config": source_config,
            "paths": {"public_transport_legs": None},
            "storyline": storyline,
            "trips": trips,
            "journeys": journeys,
            "user_statistics": user_statistics,
            "public_transport_legs": None,
            "questionnaire": None,
            "questionnaires": {
                "declic-mobility-vague1": declic_questionnaire,
                "declic-mobility-vague2": declic_questionnaire,
                "declic-mobility-ziplo": ziplo_questionnaire,
            },
        }
    }
    mapped_users = pd.DataFrame(
        {
            "experiment_name": component_names,
            "user_id": users,
            "project_raw": ["Vague 1", "Vague 2", "ZIPLO"],
            "invite_code": invite_codes,
        }
    )

    user_expe = build_user_presence_table(experiment_data, mapped_users)
    landed_tables, questionnaire_rules, *_ = build_landed_tables(
        experiment_data,
        mapped_users,
        user_expe,
    )

    assert user_expe["in_questionnaire"].all()
    assert len(landed_tables["questionnaire_declic"]) == 2
    assert set(landed_tables["questionnaire_declic"]["experiment_name"]) == {
        "declic-mobility-vague1",
        "declic-mobility-vague2",
    }
    assert landed_tables["questionnaire_declic"]["invite_code"].tolist() == [
        "invite-a",
        "invite-b",
    ]
    assert len(landed_tables["questionnaire_declic_ziplo"]) == 1
    assert landed_tables["questionnaire_declic_ziplo"].iloc[0][
        "invite_code"
    ] == "invite-z"
    assert {
        questionnaire_rules[experiment_name]["questionnaire_column"]
        for experiment_name in component_names
    } == {"invite_code"}


def test_composite_landing_assigns_experiment_and_phase_from_user_mapping():
    source_config = {
        "experiment_name": "declic-all",
        "component_experiments": [
            {
                "experiment_name": "experiment-a",
                "phase_1": ("2026-01-01", "2026-01-07"),
                "phase_2": ("2026-01-08", "2026-01-14"),
                "phase_3": (None, None),
                "start_expe": "2026-01-01",
                "end_expe": "2026-01-14",
            },
            {
                "experiment_name": "experiment-b",
                "phase_1": ("2026-02-01", "2026-02-07"),
                "phase_2": ("2026-02-08", "2026-02-14"),
                "phase_3": (None, None),
                "start_expe": "2026-02-01",
                "end_expe": "2026-02-14",
            },
        ],
    }
    storyline = pd.DataFrame(
        [
            {
                "id": "storyline-a",
                "user_id": "user-a",
                "type": "Track",
                "started_at": "2026-01-03T08:00:00Z",
                "finished_at": "2026-01-03T08:30:00Z",
                "started_on": "2026-01-03",
                "geometry": "LINESTRING (0 0, 1 1)",
            },
            {
                "id": "storyline-b",
                "user_id": "user-b",
                "type": "Track",
                "started_at": "2026-02-10T08:00:00Z",
                "finished_at": "2026-02-10T08:30:00Z",
                "started_on": "2026-02-10",
                "geometry": "LINESTRING (0 0, 1 1)",
            },
            {
                "id": "storyline-unassigned",
                "user_id": "user-unassigned",
                "type": "Track",
                "started_at": "2026-02-10T09:00:00Z",
                "finished_at": "2026-02-10T09:30:00Z",
                "started_on": "2026-02-10",
                "geometry": "LINESTRING (0 0, 1 1)",
            },
        ]
    )
    trips = pd.DataFrame(
        {
            "id": ["trip-a", "trip-b", "trip-unassigned"],
            "user_id": ["user-a", "user-b", "user-unassigned"],
            "started_at": [
                "2026-01-03T08:00:00Z",
                "2026-02-10T08:00:00Z",
                "2026-02-10T09:00:00Z",
            ],
            "finished_at": [
                "2026-01-03T08:30:00Z",
                "2026-02-10T08:30:00Z",
                "2026-02-10T09:30:00Z",
            ],
        }
    )
    journeys = trips.rename(columns={"id": "journey_id"}).copy()
    user_statistics = pd.DataFrame(
        {
            "id": ["user-a", "user-b", "user-unassigned"],
            "tracking_days": [7, 7, 1],
        }
    )
    experiment_data = {
        "declic-all": {
            "config": source_config,
            "paths": {"public_transport_legs": None},
            "storyline": storyline,
            "trips": trips,
            "journeys": journeys,
            "user_statistics": user_statistics,
            "public_transport_legs": None,
            "questionnaire": None,
        }
    }
    mapped_users = pd.DataFrame(
        {
            "experiment_name": ["experiment-a", "experiment-b"],
            "user_id": ["user-a", "user-b"],
            "project_raw": ["Project A", "Project B"],
            "invite_code": ["invite-a", "invite-b"],
        }
    )

    with pytest.warns(UserWarning, match="experiment_name='unassigned'"):
        user_expe = build_user_presence_table(experiment_data, mapped_users)
    landed_tables, *_ = build_landed_tables(
        experiment_data,
        mapped_users,
        user_expe,
    )

    storyline_by_user = landed_tables["storyline"].set_index("user_id")
    assert storyline_by_user.loc["user-a", "experiment_name"] == "experiment-a"
    assert storyline_by_user.loc["user-a", "phase"] == "Phase1"
    assert storyline_by_user.loc["user-b", "experiment_name"] == "experiment-b"
    assert storyline_by_user.loc["user-b", "phase"] == "Phase2"
    assert storyline_by_user.loc["user-unassigned", "experiment_name"] == "unassigned"
    assert pd.isna(storyline_by_user.loc["user-unassigned", "phase"])

    user_expe_by_user = user_expe.set_index("user_id")
    assert user_expe_by_user.loc["user-a", "phase_1_start"] == "2026-01-01"
    assert user_expe_by_user.loc["user-a", "phase_1_end"] == "2026-01-07"
    assert user_expe_by_user.loc["user-b", "phase_2_start"] == "2026-02-08"
    assert user_expe_by_user.loc["user-b", "phase_2_end"] == "2026-02-14"
    assert user_expe_by_user.loc["user-unassigned", "experiment_name"] == "unassigned"
    assert pd.isna(user_expe_by_user.loc["user-unassigned", "start_expe"])


def test_prefig_user_presence_uses_source_users_without_2026_allocation():
    config = {
        "experiment_name": "declic-mobility-prefig",
        "motiontag_project_name": "Declic-mobilite",
        "phase_1": ("2025-04-12", "2025-05-02"),
        "phase_2": ("2025-05-03", "2025-06-02"),
        "phase_3": ("2025-06-04", "2025-06-28"),
        "start_expe": "2025-04-12",
        "end_expe": "2025-06-28",
    }
    experiment_data = {
        "declic-mobility-prefig": {
            "config": config,
            "storyline": pd.DataFrame({"user_id": ["user-a"]}),
            "trips": pd.DataFrame({"user_id": ["user-a"]}),
            "journeys": pd.DataFrame({"user_id": ["user-a"]}),
            "user_statistics": pd.DataFrame({"id": ["user-a", "user-b"]}),
            "questionnaire": pd.DataFrame({"id": ["user-a"]}),
        }
    }
    empty_allocation = pd.DataFrame(
        columns=["experiment_name", "user_id", "project_raw", "invite_code"]
    )

    user_expe = build_user_presence_table(experiment_data, empty_allocation)

    assert set(user_expe["user_id"]) == {"user-a", "user-b"}
    assert set(user_expe["experiment_name"]) == {"declic-mobility-prefig"}
    assert user_expe.set_index("user_id").loc["user-a", "in_questionnaire"]
    assert not user_expe.set_index("user_id").loc["user-b", "in_storyline"]
    assert set(user_expe["phase_1_start"]) == {"2025-04-12"}
    assert set(user_expe["phase_3_end"]) == {"2025-06-28"}


def test_landing_contract_keeps_phase_dates_in_user_presence_and_not_dump_range_in_tables(tmp_path):
    config = {
        "experiment_name": "declic-mobility-vague1",
        "dump_date_range": "2026-01-01--2026-12-31",
        "phase_1": ("2026-03-01", "2026-03-07"),
        "phase_2": ("2026-03-08", "2026-03-21"),
        "phase_3": ("2026-03-22", "2026-03-28"),
        "paths": {"excursion_area": str(tmp_path / "zones" / "vague1.geojson")},
    }
    prefig_config = {
        "experiment_name": "declic-mobility-prefig",
        "dump_date_range": "2025-01-01--2025-12-31",
        "phase_1": ("2025-03-01", "2025-03-07"),
        "phase_2": ("2025-03-08", "2025-03-21"),
        "phase_3": ("2025-03-22", "2025-03-28"),
    }
    ziplo_config = {
        "experiment_name": "declic-mobility-ziplo",
        "dump_date_range": "2026-01-01--2026-12-31",
        "phase_1": ("2026-04-01", "2026-04-07"),
        "phase_2": ("2026-04-08", "2026-04-21"),
        "phase_3": ("2026-04-22", "2026-04-28"),
    }
    experiment_data = {
        "declic-mobility-vague1": {
            "config": config,
            "paths": {"public_transport_legs": None},
            "storyline": pd.DataFrame(
                [
                    {
                        "id": "leg-1",
                        "user_id": "u1",
                        "type": "Track",
                        "started_at": "2026-03-02T08:00:00Z",
                        "finished_at": "2026-03-02T08:30:00Z",
                        "started_on": "2026-03-02",
                        "geometry": "LINESTRING (0 0, 1 1)",
                    }
                ]
            ),
            "trips": pd.DataFrame(
                [
                    {
                        "id": "trip-1",
                        "user_id": "u1",
                        "started_at": "2026-03-02T08:00:00Z",
                        "finished_at": "2026-03-02T08:30:00Z",
                    }
                ]
            ),
            "journeys": pd.DataFrame(
                [
                    {
                        "id": "journey-1",
                        "user_id": "u1",
                        "started_at": "2026-03-02T08:00:00Z",
                        "finished_at": "2026-03-02T08:30:00Z",
                    }
                ]
            ),
            "user_statistics": pd.DataFrame(
                [{"id": "u1", "email": "sensitive@example.org", "active_days_count": 1}]
            ),
            "questionnaire": pd.DataFrame(
                [{"RegistrationCode": "abc", "question_vague": "yes"}]
            ),
        },
        "declic-mobility-prefig": {
            "config": prefig_config,
            "paths": {"public_transport_legs": None},
            "storyline": pd.DataFrame(
                [
                    {
                        "id": "leg-0",
                        "user_id": "u0",
                        "type": "Track",
                        "started_at": "2025-03-02T08:00:00Z",
                        "finished_at": "2025-03-02T08:30:00Z",
                        "started_on": "2025-03-02",
                        "geometry": "LINESTRING (0 0, 1 1)",
                    }
                ]
            ),
            "trips": pd.DataFrame(
                [
                    {
                        "id": "trip-0",
                        "user_id": "u0",
                        "started_at": "2025-03-02T08:00:00Z",
                        "finished_at": "2025-03-02T08:30:00Z",
                    }
                ]
            ),
            "journeys": pd.DataFrame(
                [
                    {
                        "id": "journey-0",
                        "user_id": "u0",
                        "started_at": "2025-03-02T08:00:00Z",
                        "finished_at": "2025-03-02T08:30:00Z",
                    }
                ]
            ),
            "user_statistics": pd.DataFrame(
                [{"id": "u0", "email": "prefig@example.org", "active_days_count": 1}]
            ),
            "questionnaire": pd.DataFrame(
                [{"user_id": "u0", "question_prefig": "yes"}]
            ),
        },
        "declic-mobility-ziplo": {
            "config": ziplo_config,
            "paths": {"public_transport_legs": None},
            "storyline": pd.DataFrame(
                [
                    {
                        "id": "leg-2",
                        "user_id": "u2",
                        "type": "Track",
                        "started_at": "2026-04-02T08:00:00Z",
                        "finished_at": "2026-04-02T08:30:00Z",
                        "started_on": "2026-04-02",
                        "geometry": "LINESTRING (0 0, 1 1)",
                    }
                ]
            ),
            "trips": pd.DataFrame(
                [
                    {
                        "id": "trip-2",
                        "user_id": "u2",
                        "started_at": "2026-04-02T08:00:00Z",
                        "finished_at": "2026-04-02T08:30:00Z",
                    }
                ]
            ),
            "journeys": pd.DataFrame(
                [
                    {
                        "id": "journey-2",
                        "user_id": "u2",
                        "started_at": "2026-04-02T08:00:00Z",
                        "finished_at": "2026-04-02T08:30:00Z",
                    }
                ]
            ),
            "user_statistics": pd.DataFrame(
                [{"id": "u2", "email": "ziplo@example.org", "active_days_count": 1}]
            ),
            "questionnaire": pd.DataFrame(
                [{"RegistrationCode": "def", "question_ziplo": "yes"}]
            ),
        }
    }
    user_allocation = pd.DataFrame(
        [
            {
                "experiment_name": "declic-mobility-vague1",
                "user_id": "u1",
                "project_raw": "Exp A",
                "invite_code": "abc",
            },
            {
                "experiment_name": "declic-mobility-prefig",
                "user_id": "u0",
                "project_raw": "Exp Prefig",
                "invite_code": "prefig",
            },
            {
                "experiment_name": "declic-mobility-ziplo",
                "user_id": "u2",
                "project_raw": "Exp ZIPLO",
                "invite_code": "def",
            }
        ]
    )

    user_presence = build_user_presence_table(experiment_data, user_allocation)
    (
        landed_tables,
        _questionnaire_rules,
        _unmatched,
        sensitive_report,
        column_contract_report,
    ) = build_landed_tables(
        experiment_data,
        user_allocation,
        user_presence,
    )

    vague_presence = user_presence.loc[user_presence["experiment_name"].eq("declic-mobility-vague1")].iloc[0]
    assert vague_presence["phase_1_start"] == "2026-03-01"
    assert vague_presence["phase_3_end"] == "2026-03-28"
    assert "dump_date_range" not in landed_tables["storyline"].columns
    assert "dump_date_range" not in landed_tables["trips"].columns
    assert "dump_date_range" not in landed_tables["user_statistics"].columns
    assert "email" not in landed_tables["user_statistics"].columns
    assert {"storyline_id", "user_id"}.issubset(landed_tables["storyline"].columns)
    assert {"trip_id", "user_id"}.issubset(landed_tables["trips"].columns)
    assert {"journey_id", "user_id"}.issubset(landed_tables["journeys"].columns)
    assert {"user_statistics_id", "user_id"}.issubset(landed_tables["user_statistics"].columns)
    assert "id" not in landed_tables["storyline"].columns
    assert "id" not in landed_tables["trips"].columns
    assert "id" not in landed_tables["journeys"].columns
    assert "id" not in landed_tables["user_statistics"].columns
    missing_must_have = column_contract_report.loc[column_contract_report["severity"].eq("error")]
    assert missing_must_have.empty
    assert "phase" in landed_tables["storyline"].columns
    assert "Phase" not in landed_tables["storyline"].columns
    assert set(landed_tables["storyline"]["phase"]) == {"Phase1"}
    assert "phase" not in landed_tables["user_statistics"].columns
    assert "Phase" not in landed_tables["user_statistics"].columns
    assert "phase_number" not in landed_tables["user_statistics"].columns
    assert "phase_reference_column" not in landed_tables["user_statistics"].columns
    assert set(sensitive_report["table_name"]) == {"user_statistics"}
    assert set(sensitive_report["column"]) == {"email"}
    assert "questionnaires" not in landed_tables
    assert set(landed_tables["questionnaire_declic_prefig"]["experiment_name"]) == {"declic-mobility-prefig"}
    assert set(landed_tables["questionnaire_declic"]["experiment_name"]) == {"declic-mobility-vague1"}
    assert set(landed_tables["questionnaire_declic_ziplo"]["experiment_name"]) == {"declic-mobility-ziplo"}
    alias_report = scan_landing_alias_columns(landed_tables, {})
    assert alias_report.empty

    manifest, _summary = write_landing_output(
        landed_tables,
        experiment_data,
        [config, prefig_config, ziplo_config],
        _questionnaire_rules,
        tmp_path,
        public_transport_unmatched_rows=_unmatched,
        sensitive_columns_report=sensitive_report,
        column_contract_report=column_contract_report,
        formats=("parquet", "csv"),
    )

    assert manifest["exported_files"]["user_statistics"]["files"] == ["user_statistics.csv"]
    assert manifest["sensitive_columns_report"]["rows"] == 1
    assert manifest["column_contract_report"]["missing_must_have"] == 0
    assert manifest["column_alias_report"]["different_value_aliases"] == 0
    assert manifest["exported_files"]["questionnaire_declic_prefig"]["files"] == ["questionnaire_declic_prefig.csv"]
    assert manifest["exported_files"]["questionnaire_declic"]["files"] == ["questionnaire_declic.csv"]
    assert manifest["exported_files"]["questionnaire_declic_ziplo"]["files"] == ["questionnaire_declic_ziplo.csv"]
    landing_config = json.loads((tmp_path / "landing_config.json").read_text(encoding="utf-8"))
    vague1_metadata = next(
        item
        for item in landing_config["experiments"]
        if item["experiment_name"] == "declic-mobility-vague1"
    )
    assert vague1_metadata["excursion_area_path"] == str(tmp_path / "zones" / "vague1.geojson")
    assert not (tmp_path / "user_statistics.parquet").exists()
    assert (tmp_path / "sensitive_columns_report.csv").exists()
    assert (tmp_path / "column_contract_report.csv").exists()
    assert (tmp_path / "column_alias_report.csv").exists()
    assert not (tmp_path / "questionnaire_declic_prefig.parquet").exists()
    assert not (tmp_path / "questionnaire_declic.parquet").exists()


def test_landing_alias_report_detects_identical_and_divergent_identifier_aliases():
    tables = {
        "trips": pd.DataFrame(
            {
                "id": ["trip-1", "trip-2"],
                "trip_id": ["trip-1", "trip-2"],
                "user_id": ["u1", "u2"],
            }
        ),
        "journeys": pd.DataFrame(
            {
                "id": ["journey-1", "journey-source-2"],
                "journey_id": ["journey-1", "journey-2"],
                "user_id": ["u1", "u2"],
            }
        ),
    }

    report = scan_landing_alias_columns(tables, {
        "id_columns": {
            "trips": {
                "user_id": "user_id",
                "user_id_candidates": ["user_id", "id"],
                "entry_id": "trip_id",
                "entry_id_candidates": ["trip_id", "id"],
            },
            "journeys": {
                "user_id": "user_id",
                "user_id_candidates": ["user_id", "id"],
                "entry_id": "journey_id",
                "entry_id_candidates": ["journey_id", "id"],
            },
        }
    })

    trip_alias = report.loc[
        report["table_name"].eq("trips")
        & report["canonical_column"].eq("trip_id")
        & report["alias_column"].eq("id")
    ].iloc[0]
    journey_alias = report.loc[
        report["table_name"].eq("journeys")
        & report["canonical_column"].eq("journey_id")
        & report["alias_column"].eq("id")
    ].iloc[0]

    assert trip_alias["relationship"] == "identical_values"
    assert trip_alias["action"] == "drop_alias_after_standardization"
    assert journey_alias["relationship"] == "different_values"
    assert journey_alias["mismatch_count"] == 1
