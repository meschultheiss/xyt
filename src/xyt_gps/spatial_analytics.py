"""Spatial analytics public facade.

The implementation is split across `spatial_time`, `spatial_h3` and
`spatial_exports` so the dashboard preparation workflow remains readable.
Importing from this module remains supported for the high-level API.
"""

from .spatial_time import add_time_slices
from .spatial_h3 import (
    DEFAULT_H3_COUNT_DIMENSION_SETS,
    DEFAULT_H3_COUNT_METRICS,
    DEFAULT_H3_METADATA_COLUMNS,
    aggregate_h3_frequencies,
    build_h3_count_matrix,
    legs_to_h3_points,
)
from .spatial_exports import (
    build_spatial_analytics_tables,
    write_duckdb_spatial_database,
    write_spatial_analytics_exports,
    write_spatial_analytics_tables,
)

__all__ = [
    "DEFAULT_H3_COUNT_DIMENSION_SETS",
    "DEFAULT_H3_COUNT_METRICS",
    "DEFAULT_H3_METADATA_COLUMNS",
    "add_time_slices",
    "aggregate_h3_frequencies",
    "build_h3_count_matrix",
    "build_spatial_analytics_tables",
    "legs_to_h3_points",
    "write_duckdb_spatial_database",
    "write_spatial_analytics_exports",
    "write_spatial_analytics_tables",
]
