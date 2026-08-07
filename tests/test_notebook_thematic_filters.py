import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_UTILS = REPO_ROOT / "Notebooks" / "utils"
if not NOTEBOOK_UTILS.exists():
    pytest.skip(
        "Projet-specific notebook utility tests require Notebooks/utils.",
        allow_module_level=True,
    )
sys.path.insert(0, str(NOTEBOOK_UTILS))

from thematic_utils import (  # noqa: E402
    align_reference_by_experiment_phase,
    split_table_by_experiment_phase,
)


def test_split_table_keeps_requested_empty_combinations():
    table = pd.DataFrame(
        {
            "experiment_name": ["intervention"],
            "phase": ["Phase1"],
            "value": [1],
        }
    )

    split = split_table_by_experiment_phase(
        table,
        experiment_names=["intervention", "temoin"],
        phases=["Phase1", "Phase2"],
    )

    assert split["intervention"]["Phase1"]["value"].tolist() == [1]
    assert split["intervention"]["Phase2"].empty
    assert split["temoin"]["Phase1"].empty
    assert split["temoin"]["Phase2"].empty


def test_reference_alignment_uses_protocol_sequence_weeks():
    analysis = pd.DataFrame(
        {
            "experiment_name": ["intervention"] * 4,
            "phase": ["Phase1", "Phase1", "Phase2", "Phase2"],
            "n_week_sequence": [1, 2, 3, 4],
        }
    )
    reference = pd.DataFrame(
        {
            "experiment_name": ["temoin"] * 4,
            "phase": ["Phase1"] * 4,
            "n_week_sequence": [1, 2, 3, 4],
        }
    )

    aligned = align_reference_by_experiment_phase(
        analysis,
        reference,
        experiment_names=["intervention"],
        phases=["Phase1", "Phase2"],
    )

    phase_1 = aligned["intervention"]["Phase1"]
    phase_2 = aligned["intervention"]["Phase2"]
    assert phase_1["n_week_sequence"].tolist() == [1, 2]
    assert phase_2["n_week_sequence"].tolist() == [3, 4]
    assert set(phase_2["phase"]) == {"Phase2"}
    assert set(phase_2["reference_for_experiment"]) == {"intervention"}
