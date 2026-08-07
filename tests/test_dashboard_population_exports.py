import sys
from pathlib import Path

import pandas as pd
import pytest


NOTEBOOK_UTILS_DIR = Path(__file__).resolve().parents[2] / "Notebooks" / "utils"
if not NOTEBOOK_UTILS_DIR.exists():
    pytest.skip(
        "Projet-specific dashboard export tests require Notebooks/utils.",
        allow_module_level=True,
    )
sys.path.insert(0, str(NOTEBOOK_UTILS_DIR))

from dashboard_exports.population import _merge_questionnaire  # noqa: E402


def test_population_questionnaire_merge_supports_updated_declic_columns():
    user_stats = pd.DataFrame(
        {
            "experiment_name": ["declic-mobility-vague2"],
            "user_id": ["u-v2"],
            "registration_code": ["code-v2"],
        }
    )
    questionnaires = {
        "declic-mobility-vague2": {
            "questionnaire_declic": pd.DataFrame(
                {
                    "experiment_name": ["declic-mobility-vague2"],
                    "questionnaire_match_value": ["code-v2"],
                    "registrationcode": ["code-v2"],
                    "annee_naissance": [1988],
                    "structure_menage": ["Couple avec enfant"],
                    "commune_origine": ["Onex"],
                    "nb_voitures": [1],
                }
            )
        }
    }

    merged = _merge_questionnaire("declic-mobility-vague2", user_stats, questionnaires)

    assert merged is not None
    assert merged.loc[0, "commune"] == "Onex"
    assert merged.loc[0, "structmen"] == "Couple avec enfant"
    assert merged.loc[0, "nb_voitures"] == 1
