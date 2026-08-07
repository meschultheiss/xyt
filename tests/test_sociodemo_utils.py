import sys
from pathlib import Path

import pandas as pd
import pytest


NOTEBOOK_UTILS_DIR = Path(__file__).resolve().parents[2] / "Notebooks" / "utils"
if not NOTEBOOK_UTILS_DIR.exists():
    pytest.skip(
        "Projet-specific notebook utility tests require Notebooks/utils.",
        allow_module_level=True,
    )
sys.path.insert(0, str(NOTEBOOK_UTILS_DIR))

from sociodemo_utils import build_sociodemo_outputs  # noqa: E402


def test_build_sociodemo_outputs_harmonises_three_questionnaire_schemas():
    questionnaires = {
        "concat": {
            "questionnaire_prefig": pd.DataFrame(
                {
                    "experiment_name": ["prefig"],
                    "questionnaire_match_value": ["u-prefig"],
                    "genre": ["F"],
                    "annee_naissance": [1980],
                    "commune": ["Genève"],
                    "equipement_menage_voitures": [1],
                    "structmen_26_64": [2],
                    "structmen_0_15": [1],
                }
            ),
            "questionnaire_declic": pd.DataFrame(
                {
                    "experiment_name": ["vague1"],
                    "questionnaire_match_value": ["i-v1"],
                    "etesvousunefemmeouunhomme": ["Un homme"],
                    "quelleestvotreanneedenaissance": [1990],
                    "dansquellecommunedegenevevotredomicileestilsitue": ["Carouge"],
                    "decombiendevoituresdisposezvous_auseindevotremenage": [2],
                    "combiendepersonnesde26a64anscomposentvotremenage": [2],
                }
            ),
            "questionnaire_ziplo": pd.DataFrame(
                {
                    "experiment_name": ["ziplo"],
                    "questionnaire_match_value": ["i-ziplo"],
                    "a2_genre": ["Une homme"],
                    "a3_age": [32],
                    "ville_commune": ["Lausanne"],
                    "b4_voit_nb": ["Plus de trois voitures"],
                    "a4_compo_menage": ["En couple, avec enfants"],
                }
            ),
        }
    }
    user_expe = pd.DataFrame(
        {
            "experiment_name": ["prefig", "vague1", "ziplo", "temoin"],
            "user_id": ["u-prefig", "u-v1", "u-ziplo", "u-temoin"],
            "in_questionnaire": [True, True, True, False],
        }
    )

    respondents, outputs = build_sociodemo_outputs(
        questionnaires,
        user_expe,
        experiment_names=["prefig", "vague1", "ziplo", "temoin"],
        reference_year_by_experiment={
            "prefig": 2025,
            "vague1": 2026,
            "ziplo": 2026,
            "temoin": 2026,
        },
        min_cell_count=1,
    )

    by_experiment = respondents.set_index("experiment_name")
    assert by_experiment.loc["prefig", "gender"] == "Femme"
    assert by_experiment.loc["vague1", "gender"] == "Homme"
    assert by_experiment.loc["ziplo", "gender"] == "Homme"
    assert by_experiment.loc["prefig", "age"] == 45
    assert by_experiment.loc["vague1", "age"] == 36
    assert by_experiment.loc["ziplo", "age"] == 32
    assert by_experiment.loc["ziplo", "car_count_group"] == "4+"
    assert by_experiment.loc["prefig", "household_size_group"] == "3"
    assert by_experiment.loc["vague1", "household_size_group"] == "2"
    assert pd.isna(by_experiment.loc["ziplo", "household_size_group"])

    coverage = outputs["180_80"].set_index("experiment_name")
    assert coverage.loc["prefig", "coverage_rate"] == 1
    assert coverage.loc["temoin", "questionnaire_respondents"] == 0
    assert coverage.loc["temoin", "coverage_rate"] == 0
    assert set(outputs) == {
        "180_80",
        "180_81",
        "180_82",
        "180_83",
        "180_84",
        "180_85",
    }


def test_sociodemo_outputs_group_cells_below_threshold():
    questionnaires = {
        "concat": {
            "questionnaire": pd.DataFrame(
                {
                    "experiment_name": ["vague1", "vague1", "vague1"],
                    "questionnaire_match_value": ["a", "b", "c"],
                    "genre": ["F", "F", "H"],
                }
            )
        }
    }
    user_expe = pd.DataFrame(
        {
            "experiment_name": ["vague1"] * 3,
            "user_id": ["a", "b", "c"],
            "in_questionnaire": [True] * 3,
        }
    )

    _, outputs = build_sociodemo_outputs(
        questionnaires,
        user_expe,
        experiment_names=["vague1"],
        reference_year_by_experiment={"vague1": 2026},
        min_cell_count=2,
    )

    gender = outputs["180_81"].set_index("gender")
    assert gender.loc["Femme", "n_respondents"] == 2
    assert gender.loc["Autres / effectif faible", "n_respondents"] == 1


def test_sociodemo_outputs_support_updated_declic_questionnaire_columns():
    questionnaires = {
        "concat": {
            "questionnaire_declic": pd.DataFrame(
                {
                    "experiment_name": ["vague2"],
                    "questionnaire_match_value": ["code-v2"],
                    "genre": ["Femme"],
                    "annee_naissance": [1988],
                    "commune_origine": ["Onex"],
                    "nb_voitures": [1],
                    "combien_moins16ans": [1],
                    "combien_16_17ans": [0],
                    "combien_18_25ans": [0],
                    "combien_26_64ans": [2],
                    "combien_65plus": [0],
                }
            )
        }
    }
    user_expe = pd.DataFrame(
        {
            "experiment_name": ["vague2"],
            "user_id": ["u-v2"],
            "in_questionnaire": [True],
        }
    )

    respondents, _outputs = build_sociodemo_outputs(
        questionnaires,
        user_expe,
        experiment_names=["vague2"],
        reference_year_by_experiment={"vague2": 2026},
        min_cell_count=1,
    )

    row = respondents.iloc[0]
    assert row["gender"] == "Femme"
    assert row["age"] == 38
    assert row["commune"] == "Onex"
    assert row["car_count_group"] == "1"
    assert row["household_size_group"] == "3"
