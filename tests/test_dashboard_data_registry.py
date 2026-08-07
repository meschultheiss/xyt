import csv
import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "Notebooks"
REGISTRY_PATH = NOTEBOOKS_DIR / "config" / "dashboard_data.csv"
if not REGISTRY_PATH.exists():
    pytest.skip(
        "Projet-specific dashboard registry tests require Notebooks/config.",
        allow_module_level=True,
    )
OUTPUT_ID_PATTERN = re.compile(r'OUTPUT_DIR\s*/\s*"(\d{3}_\d{2})\.csv"')


def _notebook_output_ids() -> dict[str, str]:
    outputs: dict[str, str] = {}
    for notebook_path in sorted(NOTEBOOKS_DIR.glob("1[1-8]0_*.ipynb")):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        for output_id in OUTPUT_ID_PATTERN.findall(source):
            assert output_id not in outputs or outputs[output_id] == notebook_path.name, (
                f"{output_id} est exporté par plusieurs notebooks"
            )
            outputs[output_id] = notebook_path.name
    return outputs


def test_dashboard_registry_documents_every_thematic_export():
    with REGISTRY_PATH.open(encoding="utf-8", newline="") as registry_file:
        rows = list(csv.DictReader(registry_file))

    assert rows
    assert set(rows[0]) == {
        "numero",
        "titre",
        "type",
        "onglet",
        "notebook",
        "commentaire",
    }

    registry = {row["numero"]: row for row in rows}
    assert len(registry) == len(rows), "Les identifiants du registre doivent être uniques"

    notebook_outputs = _notebook_output_ids()
    assert set(registry) == set(notebook_outputs)
    for output_id, notebook_name in notebook_outputs.items():
        assert registry[output_id]["notebook"] == notebook_name
        assert registry[output_id]["titre"]
        assert registry[output_id]["type"]
        assert registry[output_id]["onglet"]
        assert registry[output_id]["commentaire"]
