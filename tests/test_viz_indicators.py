"""Tests for notebook-friendly indicator visualizations."""

from __future__ import annotations

import pandas as pd
import pytest

import xyt_gps as xyt


def test_plot_indicator_bars_writes_empty_state_for_empty_table(tmp_path):
    output_path = tmp_path / "empty_indicators.html"

    result = xyt.plot_indicator_bars(
        pd.DataFrame(),
        title="Indicateurs vides",
        group_col="mode",
        save_path=output_path,
    )

    assert result is not None
    html = output_path.read_text(encoding="utf-8")
    assert "Aucun indicateur disponible" in html


def test_plot_indicator_bars_keeps_schema_error_for_non_empty_table():
    indicators = pd.DataFrame({"distance_km_per_day": [1.2]})

    with pytest.raises(KeyError, match="mode"):
        xyt.plot_indicator_bars(indicators, group_col="mode")
