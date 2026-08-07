"""Spatial façade for geometry, GPS signal quality and zone helpers."""

from __future__ import annotations

from .spatial_geometry import clean_leg_geometries, max_consecutive_point_distance
from .spatial_matching import (
    DEFAULT_OSRM_SERVICES,
    GOOGLE_ROADS_SNAP_URL,
    GOOGLE_ROUTES_URL,
    GRAPHHOPPER_MATCH_URL,
    LegMapMatchingResult,
    map_match_legs,
)
from .spatial_quality import (
    MAX_SIGNAL_LOSS_COL,
    RELATIVE_SIGNAL_LOSS_COL,
    SIGNAL_QUALITY_REQUIRED_COLUMNS,
    add_signal_loss_metrics,
    add_signal_quality_flags,
    add_signal_quality_to_user_stats,
    build_user_signal_quality_stats,
    flag_low_quality_legs_by_mode,
    identify_bad_signal_users,
)
from .spatial_zones import (
    add_excursion_flags,
    add_journey_origin_destination_from_trips,
    add_leg_origin_destination_zones,
    add_spatial_zone_labels,
    add_trip_origin_destination_from_legs,
    build_origin_destination_zone_correspondence,
    classify_leg_relation_to_area,
)

__all__ = [
    "MAX_SIGNAL_LOSS_COL",
    "RELATIVE_SIGNAL_LOSS_COL",
    "SIGNAL_QUALITY_REQUIRED_COLUMNS",
    "clean_leg_geometries",
    "max_consecutive_point_distance",
    "DEFAULT_OSRM_SERVICES",
    "GOOGLE_ROADS_SNAP_URL",
    "GOOGLE_ROUTES_URL",
    "GRAPHHOPPER_MATCH_URL",
    "LegMapMatchingResult",
    "map_match_legs",
    "add_signal_loss_metrics",
    "add_signal_quality_flags",
    "add_signal_quality_to_user_stats",
    "build_user_signal_quality_stats",
    "flag_low_quality_legs_by_mode",
    "identify_bad_signal_users",
    "add_excursion_flags",
    "add_journey_origin_destination_from_trips",
    "add_leg_origin_destination_zones",
    "add_spatial_zone_labels",
    "add_trip_origin_destination_from_legs",
    "build_origin_destination_zone_correspondence",
    "classify_leg_relation_to_area",
]
