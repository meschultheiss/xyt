"""Tests for the recommended end-to-end pipeline entry point."""

from __future__ import annotations

from fixtures_core import minimal_project_config, minimal_raw_gps_data
from xyt_gps import MobilityPipelineResult, run_mobility_pipeline


def test_run_mobility_pipeline_returns_named_and_unpackable_result():
    result = run_mobility_pipeline(
        minimal_project_config(),
        raw=minimal_raw_gps_data(),
        resample_missing_days=False,
    )

    raw, dataset, indicators = result

    assert isinstance(result, MobilityPipelineResult)
    assert raw is result.raw
    assert dataset is result.dataset
    assert indicators is result.indicators
    assert len(dataset.legs) == 1
    assert indicators is not None
    assert not indicators.population.empty


def test_run_mobility_pipeline_can_skip_indicator_computation():
    result = run_mobility_pipeline(
        minimal_project_config(),
        raw=minimal_raw_gps_data(),
        compute_indicators=False,
    )

    assert result.indicators is None
    assert len(result.dataset.legs) == 1
