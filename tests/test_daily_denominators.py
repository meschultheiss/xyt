import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_UTILS_DIR = REPO_ROOT / "Notebooks" / "utils"
if not NOTEBOOK_UTILS_DIR.exists():
    pytest.skip(
        "Projet-specific dashboard export tests require Notebooks/utils.",
        allow_module_level=True,
    )
sys.path.insert(0, str(NOTEBOOK_UTILS_DIR))

from dashboard_exports.co2 import export_co2_total_par_phase  # noqa: E402
from dashboard_exports.daily_denominators import (  # noqa: E402
    denominator_metadata,
    select_user_days,
    value_per_user_day,
)
from dashboard_exports.population import export_grille_participation  # noqa: E402
from dashboard_exports.rythme import export_temps_trajet_journalier  # noqa: E402


def _coverage():
    return pd.DataFrame(
        {
            "experiment_name": ["test"] * 4,
            "user_id": ["u1"] * 4,
            "date": pd.date_range("2025-01-01", periods=4),
            "phase": ["Phase1"] * 4,
            "n_week_sequence": [1] * 4,
            "day_in_range": [1, 1, 1, 1],
            "tracked_day": [1, 1, 0, 1],
            "active_day": [1, 0, 0, 1],
        }
    )


def test_person_day_denominators_are_distinct_and_explicit():
    coverage = _coverage()

    assert len(select_user_days(coverage, experiment_name="test", day_basis="days_in_range")) == 4
    assert len(select_user_days(coverage, experiment_name="test", day_basis="tracked_days")) == 3
    assert len(select_user_days(coverage, experiment_name="test", day_basis="active_days")) == 2

    metadata = denominator_metadata(
        coverage,
        experiment_name="test",
        phase="Phase1",
        day_basis="tracked_days",
    )
    assert metadata == {
        "day_basis": "tracked_days",
        "n_user_days": 3,
        "n_users": 1,
        "n_calendar_days": 3,
    }

    with pytest.raises(ValueError, match="Aucun personne-jour"):
        value_per_user_day(1.0, {"n_user_days": 0})


def test_daily_export_uses_selected_person_day_denominator(tmp_path):
    coverage = _coverage()
    legs = pd.DataFrame(
        {
            "phase": ["Phase1"],
            "excursion": [False],
            "co2_g": [4000.0],
            "co2_direct_g": [2000.0],
        }
    )

    range_result = export_co2_total_par_phase(
        "test",
        legs,
        tmp_path / "range.csv",
        user_day_coverage=coverage,
        day_basis="days_in_range",
    )
    active_result = export_co2_total_par_phase(
        "test",
        legs,
        tmp_path / "active.csv",
        user_day_coverage=coverage,
        day_basis="active_days",
    )

    assert range_result.loc[0, "co2_total_kg_jour"] == 1.0
    assert range_result.loc[0, "n_user_days"] == 4
    assert active_result.loc[0, "co2_total_kg_jour"] == 2.0
    assert active_result.loc[0, "n_user_days"] == 2


def test_participation_export_keeps_tracking_and_movement_days(tmp_path):
    result = export_grille_participation(
        "test",
        pd.DataFrame(),
        tmp_path / "participation.csv",
        user_day_coverage=_coverage(),
    )

    assert result.loc[0, "n_days_in_range"] == 4
    assert result.loc[0, "n_tracked_days"] == 3
    assert result.loc[0, "n_active_days"] == 2
    assert result.loc[0, "level"] == 2
    assert result.loc[0, "active_level"] == 1


def test_missing_aligned_reference_is_exported_as_unavailable(tmp_path):
    intervention_coverage = _coverage().assign(
        experiment_name="intervention",
        phase="Phase2",
    )
    reference_coverage = _coverage().assign(
        experiment_name="declic-mobility-temoin",
        phase="Phase1",
    )
    legs = pd.DataFrame(
        {"phase": ["Phase2"], "excursion": [False], "duration": [600.0]}
    )
    reference_legs = pd.DataFrame(
        {"phase": ["Phase1"], "duration": [600.0]}
    )

    result = export_temps_trajet_journalier(
        "intervention",
        legs,
        reference_legs,
        tmp_path / "rythme.csv",
        user_day_coverage=intervention_coverage,
        user_day_coverage_temoin=reference_coverage,
    )

    assert not result.loc[0, "temoin_available"]
    assert pd.isna(result.loc[0, "temoin_duree_min"])
    assert result.loc[0, "temoin_n_user_days"] == 0
