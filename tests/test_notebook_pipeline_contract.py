import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "Notebooks"
if not NOTEBOOKS_DIR.exists():
    pytest.skip(
        "Projet-specific notebook contract tests require Notebooks/.",
        allow_module_level=True,
    )
HEADER_SECTIONS = (
    "## Paramètres à relire",
    "## Structure du notebook",
    "## Points d'attention",
    "## Prise en main",
)
DECLIC_VAGUE1_PHASE_WEEK_SEQUENCES = {
    "phase_1": [1, 2],
    "phase_2": [3, 4, 5, 6],
    "phase_3": [8, 9, 10],
}
DECLIC_VAGUE2_PHASE_WEEK_SEQUENCES = {
    "phase_1": [1, 2, 3],
    "phase_2": [4, 5, 6, 7],
    "phase_3": [9, 10, 11],
}
DECLIC_VAGUE3_PHASE_WEEK_SEQUENCES = {
    "phase_1": [1, 2, 3],
    "phase_2": [4, 5, 6, 7],
    "phase_3": [9, 10, 11],
}
PREFIG_PHASE_WEEK_SEQUENCES = {
    "phase_1": [1, 2, 3],
    "phase_2": [5, 6, 7, 8],
    "phase_3": [10, 11, 12],
}
TEMOIN_PHASE_WEEK_SEQUENCES = {
    "phase_1": list(range(1, 23)),
    "phase_2": [],
    "phase_3": [],
}
ZIPLO_PHASE_WEEK_SEQUENCES = {
    "phase_1": [1, 2, 3],
    "phase_2": list(range(4, 19)),
    "phase_3": [20, 21, 22],
}


def _code_source(notebook_name: str) -> str:
    notebook = json.loads((NOTEBOOKS_DIR / notebook_name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _experiment_config(experiment_name: str) -> dict:
    path = NOTEBOOKS_DIR / "config" / "experiments" / f"{experiment_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_active_notebooks_have_reviewed_operational_headers_and_no_outputs():
    notebook_paths = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))

    assert notebook_paths
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        first_markdown = next(
            cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"
        )
        header = "".join(first_markdown.get("source", []))

        assert re.search(r"> Dernière relecture : \d{1,2} [^.\n]+ \d{4}\.", header), path.name
        for section in HEADER_SECTIONS:
            assert section in header, f"{path.name}: section manquante {section}"
        assert not any(
            cell.get("outputs")
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ), f"{path.name}: sorties de cellules à nettoyer"


def test_030_uses_spatial_prepared_tables_and_exports_experiment_scopes():
    source = _code_source("030_enrich_indicators.ipynb")

    assert 'spatial_prepared_dir = transformed_dir / "spatial-cleaned"' in source
    assert 'gpd.read_parquet(spatial_prepared_dir / "legs.parquet")' in source
    assert "read_parquet(transformed_dir /" not in source
    assert 'indicators_dir = enriched_dir / "indicators"' in source
    assert "component_experiments" in source
    assert 'f"population_indicators_{mode_col}_long.csv"' in source
    assert 'figures_dir / "population_indicators_mode_niv1.html"' in source
    assert 'GOOGLE_REPAIR_APPLY_QUALITY_FILTER = "low_quality_legs_2"' in source
    assert "google_repair_quality_mask" in source
    assert "applicable_repairs" in source
    assert "google_repair_audit" in source
    assert "google_repair_apply_quality_filter" in source
    assert "MAKE_GOOGLE_REPAIR_PREVIEW_MAP = True" in source
    assert 'GOOGLE_REPAIR_PREVIEW_SAMPLE_N = 5' in source
    assert 'figures_dir / "google_repair_preview_map.html"' in source
    assert "geometry_original_wkt" in source


def test_010_exports_landing_and_transformed_volume_summaries():
    source = _code_source("010_data_loading.ipynb")

    assert "landing_track_summary_by_experiment = summarize_track_events_by_experiment" in source
    assert "transformed_leg_summary_by_experiment = summarize_track_events_by_experiment" in source
    assert "landing_to_transformed_track_summary = compare_track_event_summaries_by_experiment" in source
    assert '"landing_track_summary_by_experiment": landing_track_summary_by_experiment' in source
    assert '"transformed_leg_summary_by_experiment": transformed_leg_summary_by_experiment' in source
    assert '"landing_to_transformed_track_summary": landing_to_transformed_track_summary' in source


def test_040_uses_spatial_tables_as_base_and_requires_matching_enriched_legs():
    source = _code_source("040_export_cleaned_dataset.ipynb")

    assert 'gpd.read_parquet(spatial_cleaned_dir / "legs.parquet")' in source
    assert "read_parquet(transformed_dir /" not in source
    assert "if spatial_leg_ids != enriched_leg_ids" in source
    assert "missing_spatial_columns" in source
    assert "add_new_columns_from_spatial" not in source
    assert 'metadata_dir = clean_dir / "metadata"' in source
    assert "cleaned_leg_summary_by_experiment" in source
    assert "landing_to_clean_track_summary" in source
    assert "transformed_to_clean_track_summary" in source
    assert "metadata_ready_for_export" in source
    assert "stringify_metadata_value" in source
    assert 'od_zone_correspondence = pd.read_parquet(spatial_cleaned_dir / "od_zone_correspondence.parquet")' in source
    assert '"od_zone_correspondence": od_zone_correspondence' in source


def test_100_exports_global_declic_and_ziplo_clean_bases():
    source = _code_source("100_concat_experiments.ipynb")

    assert 'EXPERIMENT_NAMES = ["declic-mobility-prefig", "declic-all"]' in source
    assert "CONCAT_EXPORT_GROUPS = [" in source
    assert '"name": "declic-all-2025-2026"' in source
    assert '"name": "declic-mobilite-2025-2026"' in source
    assert '"name": "declic-ziplo-2026"' in source
    assert "DECLIC_MOBILITE_EXPERIMENTS = [" in source
    assert '"declic-mobility-temoin"' in source
    assert '"declic-mobility-vague3"' in source
    assert 'ZIPLO_EXPERIMENTS = ["declic-mobility-ziplo"]' in source
    assert "filter_cleaned_payload_by_experiments" in source
    assert 'NON_DELIVERED_EXPERIMENT_NAMES = {"unassigned"}' in source
    assert "available_cleaned_payload_experiments(loaded)" in source
    assert 'EXPORT_TABLE_FORMATS = ("parquet", "csv")' in source
    assert 'QUESTIONNAIRE_EXPORT_FORMATS = ("csv", "parquet")' in source
    assert "write_dataset_readme" in source
    assert 'metadata_dir / "export_manifest.csv"' in source
    assert 'metadata_dir / "concat_manifest.json"' in source
    assert 'metadata_dir / "cleaned_leg_summary_by_experiment.csv"' in source


def test_020_propagates_participation_tables_to_spatial_output():
    source = _code_source("020_spatial_cleaning.ipynb")

    assert '["participation_grid", "participation_summary"]' in source
    assert "shutil.copy2(source_path, target_path)" in source
    assert "NETWORK_PROCESSING_IN_020_ENABLED = False" in source
    assert "from graphhopper_config import (" not in source
    assert "xyt.map_match_legs(" not in source
    assert "build_origin_destination_zone_correspondence" in source
    assert '"od_zone_correspondence": od_zone_correspondence' in source
    assert '"spatial_geometry_exclusions": spatial_geometry_exclusions' in source
    assert source.index("drop_discontinuous=False") < source.index("drop_discontinuous=True")


def test_011_reports_quality_by_experiment():
    source = _code_source("011_quality_check.ipynb")

    assert 'participation_grid.groupby("experiment_name", sort=False)' in source
    assert "participation_heatmaps = {}" in source
    assert 'title=f"Participation hebdomadaire - {experiment_name}"' in source
    assert '"legs_total": legs_total' in source
    assert '"legs_share": float(legs_count / legs_total)' in source
    assert "selection_reason_summary_by_experiment" in source
    assert "quality_key_figures_by_experiment = pd.DataFrame" in source
    assert 'quality_key_figures_by_experiment.to_csv' in source


def test_021_loads_existing_google_results_and_disables_paid_calls_by_default():
    source = _code_source("021_gps_network_processing.ipynb")

    assert "LOAD_EXISTING_GOOGLE_RESULTS = True" in source
    assert "RUN_GOOGLE_MAPS_REQUESTS = False" in source
    assert "EXPORT_TEST_RESULTS = False" in source
    assert "from google_maps_config import (" in source
    assert "MAX_LEGS = 300" in source
    assert "SAMPLE_LOW_QUALITY_ONLY = True" in source
    assert 'GOOGLE_MAPS_OUTPUT_DIRNAME = "google-repaired-geometries"' in source
    assert 'GOOGLE_MAPS_CACHE_ROOT = DATA_OUTPUT_DIR.resolve() / "Cache" / "google-repaired-geometries"' in source
    assert "GOOGLE_MAPS_CACHE_VERSION = None" in source
    assert "SYNC_GOOGLE_CACHE_TO_TRANSFORMED = True" in source
    assert "ALLOW_OVERWRITE_GOOGLE_OUTPUT = False" in source
    assert "ALLOW_OVERWRITE_GOOGLE_CACHE_VERSION = False" in source
    assert "GOOGLE_KEY_SPLIT_COUNT = 2" in source
    assert "if LOAD_EXISTING_GOOGLE_RESULTS:" in source
    assert "if LOAD_EXISTING_GOOGLE_RESULTS and not RUN_GOOGLE_MAPS_REQUESTS:" not in source
    assert "GOOGLE_CAR_MODES" in source
    assert '"Mode::Car"' in source
    assert '"Mode::Ecar"' in source
    assert "low_quality_legs_1" in source
    assert "low_quality_legs_2" in source
    assert "repair_candidate" in source
    assert "network_processing_candidate" in source
    assert "network_processing_reason" in source
    assert "sample_pool.sample(" in source
    assert "get_google_maps_api_keys(enabled=RUN_GOOGLE_MAPS_REQUESTS)" in source
    assert "planned_requests > GOOGLE_MAPS_MAX_REQUESTS" in source
    assert "find_google_cache_dir" in source
    assert "sync_google_results" in source
    assert "load_existing_google_results" in source
    assert "Résultats Google existants chargés" in source
    assert "Résultats Google synchronisés vers" in source
    assert "tqdm(total=planned_requests" in source
    assert "graphhopper_enabled=False" in source
    assert "osrm_enabled=False" in source
    assert "google_fallback_enabled=True" in source
    assert "google_api_keys=[api_key]" in source
    assert "Export cache bloqué pour ne pas écraser" in source
    assert "google_repaired_leg_geometries.parquet" in source
    assert "geometry_origin" in source
    assert "geometry_repair_applied" in source
    assert "geometry_repaired_google_wkt" in source
    assert "line_weight = 3.2 if version == \"original\" else 2.4" in source


def test_declic_phase_week_sequences_are_declared_in_configs():
    vague1 = _experiment_config("declic-mobility-vague1")
    assert vague1["phase_week_sequences"] == DECLIC_VAGUE1_PHASE_WEEK_SEQUENCES
    assert vague1["phase_1"] == ["2026-02-23", "2026-03-08"]
    assert vague1["phase_2"] == ["2026-03-09", "2026-04-08"]
    assert vague1["phase_3"] == ["2026-04-09", "2026-04-28"]
    assert vague1["start_expe"] == "2026-02-23"
    assert vague1["end_expe"] == "2026-05-03"
    assert vague1["analysis_phase_1"] == ["2026-02-23", "2026-03-08"]
    assert vague1["analysis_phase_2"] == ["2026-03-09", "2026-04-05"]
    assert vague1["analysis_phase_3"] == ["2026-04-13", "2026-05-03"]

    vague2 = _experiment_config("declic-mobility-vague2")
    assert vague2["phase_week_sequences"] == DECLIC_VAGUE2_PHASE_WEEK_SEQUENCES
    assert vague2["phase_1"] == ["2026-03-30", "2026-04-12"]
    assert vague2["phase_2"] == ["2026-04-13", "2026-05-13"]
    assert vague2["phase_3"] == ["2026-05-14", "2026-05-31"]
    assert vague2["start_expe"] == "2026-03-23"
    assert vague2["end_expe"] == "2026-06-07"
    assert vague2["analysis_phase_1"] == ["2026-03-23", "2026-04-12"]
    assert vague2["analysis_phase_2"] == ["2026-04-13", "2026-05-10"]
    assert vague2["analysis_phase_3"] == ["2026-05-18", "2026-06-07"]

    prefig = _experiment_config("declic-mobility-prefig")
    assert prefig["phase_week_sequences"] == PREFIG_PHASE_WEEK_SEQUENCES
    assert prefig["phase_1"] == ["2025-04-12", "2025-05-02"]
    assert prefig["phase_2"] == ["2025-05-03", "2025-06-02"]
    assert prefig["phase_3"] == ["2025-06-04", "2025-06-28"]
    assert prefig["start_expe"] == "2025-04-07"
    assert prefig["end_expe"] == "2025-06-29"
    assert prefig["analysis_phase_1"] == ["2025-04-07", "2025-04-27"]
    assert prefig["analysis_phase_2"] == ["2025-05-05", "2025-06-01"]
    assert prefig["analysis_phase_3"] == ["2025-06-09", "2025-06-29"]

    vague3 = _experiment_config("declic-mobility-vague3")
    assert vague3["phase_week_sequences"] == DECLIC_VAGUE3_PHASE_WEEK_SEQUENCES
    assert vague3["phase_1"] == ["2026-04-27", "2026-05-17"]
    assert vague3["phase_2"] == ["2026-05-18", "2026-06-17"]
    assert vague3["phase_3"] == ["2026-06-18", "2026-07-05"]
    assert vague3["start_expe"] == "2026-04-27"
    assert vague3["end_expe"] == "2026-07-12"
    assert vague3["analysis_phase_1"] == ["2026-04-27", "2026-05-17"]
    assert vague3["analysis_phase_2"] == ["2026-05-18", "2026-06-14"]
    assert vague3["analysis_phase_3"] == ["2026-06-22", "2026-07-12"]

    temoin = _experiment_config("declic-mobility-temoin")
    assert temoin["phase_week_sequences"] == TEMOIN_PHASE_WEEK_SEQUENCES
    assert temoin["phase_1"] == ["2026-02-23", "2026-06-28"]
    assert temoin["phase_2"] == [None, None]
    assert temoin["phase_3"] == [None, None]
    assert temoin["start_expe"] == "2026-02-23"
    assert temoin["end_expe"] == "2026-07-26"
    assert temoin["analysis_phase_1"] == ["2026-02-23", "2026-07-26"]
    assert temoin["analysis_phase_2"] == [None, None]
    assert temoin["analysis_phase_3"] == [None, None]

    ziplo = _experiment_config("declic-mobility-ziplo")
    assert ziplo["phase_week_sequences"] == ZIPLO_PHASE_WEEK_SEQUENCES
    assert ziplo["phase_1"] == ["2026-03-02", "2026-03-15"]
    assert ziplo["phase_2"] == ["2026-03-16", "2026-07-03"]
    assert ziplo["phase_3"] == ["2026-07-04", "2026-07-18"]
    assert ziplo["start_expe"] == "2026-02-23"
    assert ziplo["end_expe"] == "2026-07-26"
    assert ziplo["analysis_phase_1"] == ["2026-02-23", "2026-03-15"]
    assert ziplo["analysis_phase_2"] == ["2026-03-16", "2026-06-28"]
    assert ziplo["analysis_phase_3"] == ["2026-07-06", "2026-07-26"]

    for config in [vague1, vague2, prefig, vague3, temoin, ziplo]:
        weeks = [
            week
            for phase_weeks in config["phase_week_sequences"].values()
            for week in phase_weeks
        ]
        assert len(weeks) == len(set(weeks))

    declic_all = _experiment_config("declic-all")
    components = {
        component["experiment_name"]: component
        for component in declic_all["component_experiments"]
    }
    assert (
        components["declic-mobility-vague1"]["phase_week_sequences"]
        == DECLIC_VAGUE1_PHASE_WEEK_SEQUENCES
    )
    assert (
        components["declic-mobility-vague2"]["phase_week_sequences"]
        == DECLIC_VAGUE2_PHASE_WEEK_SEQUENCES
    )
    assert (
        components["declic-mobility-vague3"]["phase_week_sequences"]
        == DECLIC_VAGUE3_PHASE_WEEK_SEQUENCES
    )
    assert (
        components["declic-mobility-temoin"]["phase_week_sequences"]
        == TEMOIN_PHASE_WEEK_SEQUENCES
    )
    assert (
        components["declic-mobility-ziplo"]["phase_week_sequences"]
        == ZIPLO_PHASE_WEEK_SEQUENCES
    )
    for experiment_name, component in components.items():
        standalone = _experiment_config(experiment_name)
        for key in (
            "phase_1",
            "phase_2",
            "phase_3",
            "analysis_phase_1",
            "analysis_phase_2",
            "analysis_phase_3",
            "phase_week_sequences",
            "start_expe",
            "end_expe",
        ):
            assert component[key] == standalone[key]


def test_declic_2026_configs_use_updated_questionnaire_export():
    expected_questionnaire = "../../Data/dumps_bmh/260804_export_questionnaire_declic.xlsx"
    declic_2026_experiments = (
        "declic-mobility-temoin",
        "declic-mobility-vague1",
        "declic-mobility-vague2",
        "declic-mobility-vague3",
    )
    for experiment_name in declic_2026_experiments:
        config = _experiment_config(experiment_name)
        assert config["paths"]["questionnaire"] == expected_questionnaire

    declic_all = _experiment_config("declic-all")
    components = {
        component["experiment_name"]: component
        for component in declic_all["component_experiments"]
    }
    for experiment_name in declic_2026_experiments:
        assert components[experiment_name]["paths"]["questionnaire"] == expected_questionnaire


def test_002_keeps_public_export_disabled_during_review():
    source = _code_source("002_build_anonymous_test_set.ipynb")

    assert "EXPORT_TEST_SET = False" in source
    assert 'if EXPORT_TEST_SET else None' in source
