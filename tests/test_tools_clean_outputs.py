import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_OUTPUTS_PATH = REPO_ROOT / "tools" / "clean_outputs.py"
if not CLEAN_OUTPUTS_PATH.exists():
    pytest.skip(
        "Projet-specific output cleaning tests require tools/clean_outputs.py.",
        allow_module_level=True,
    )


def _load_clean_outputs_module():
    spec = importlib.util.spec_from_file_location("clean_outputs", CLEAN_OUTPUTS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_clean_outputs_preserves_google_cache_by_default(tmp_path):
    module = _load_clean_outputs_module()
    output_dir = tmp_path / "Data" / "Output"
    generated = output_dir / "2-transformed-data" / "declic-all"
    google_cache = output_dir / "Cache" / "google-repaired-geometries" / "declic-all" / "v1"
    generated.mkdir(parents=True)
    google_cache.mkdir(parents=True)
    (generated / "legs.parquet").write_text("generated")
    (google_cache / "google_repaired_leg_geometries.parquet").write_text("cached")

    deletions = module.clean_outputs(output_dir, execute=False)

    assert any(path == generated.resolve() or path in generated.resolve().parents for path in deletions)
    assert not any(google_cache.resolve() == path or google_cache.resolve() in path.parents for path in deletions)


def test_clean_outputs_can_delete_google_cache_without_other_cache(tmp_path):
    module = _load_clean_outputs_module()
    output_dir = tmp_path / "Data" / "Output"
    google_cache = output_dir / "Cache" / "google-repaired-geometries"
    other_cache = output_dir / "Cache" / "other-cache"
    google_cache.mkdir(parents=True)
    other_cache.mkdir(parents=True)
    (google_cache / "result.parquet").write_text("cached")
    (other_cache / "keep.txt").write_text("cached")

    deletions = module.clean_outputs(
        output_dir,
        delete_google_api_data=True,
        execute=False,
    )

    assert google_cache.resolve() in deletions
    assert not any(other_cache.resolve() == path or other_cache.resolve() in path.parents for path in deletions)
