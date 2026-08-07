"""Public API for xyt_gps.

The top-level namespace keeps two public usage levels:

- a short analyst-facing path for demos, onboarding and scripts;
- an explicit construction path for production notebooks that need inspectable
  intermediate states.

It is organized around five practical blocks:

0. Recommended end-to-end entry point.
1. Prepare raw inputs.
2. Build structured mobility tables.
3. Control quality and spatial consistency.
4. Compute mobility indicators.
5. Export dashboard-ready analytical tables.
"""

# 0. Short path: recommended end-to-end entry point.
from .pipeline import MobilityPipelineResult, run_mobility_pipeline

# 1. Short path and construction path: prepare raw inputs.
from .config import (
    MatchingThresholds,
    Phase,
    ProjectConfig,
    SignalLossThreshold,
    SpatialQualityThresholds,
    TimeSlice,
    TrackingThresholds,
    default_time_slices,
)
from .io import (
    GpsExportPaths,
    RawSampleConfig,
    concat_raw_gps_data,
    infer_gps_export_paths,
    load_gps_export,
    load_gps_source,
    load_gps_sources,
    load_sociodemo,
    sample_raw_gps_data,
    source_id_for_config,
)
from .mappings import MobilityMappings, mode_purpose_mapping
from .models import (
    IndicatorResult,
    MobilityDataset,
    RawGpsData,
    SchemaValidationResult,
    ValidationIssue,
)
from .privacy import (
    GeofenceAnonymizationConfig,
    GpsAnonymizationResult,
    anonymize_landed_gps_tables,
    anonymize_storyline_with_geofencing,
    apply_user_pseudonym_mapping,
    build_user_pseudonym_mapping,
    detect_staypoint_geofences,
    drop_direct_identifier_columns,
    sample_tables_by_user_count,
    swap_sensitive_purpose_columns,
)
from .references import available_reference_tables, load_reference_table
from .sample_data import find_sample_gps_path, load_sample_gps
from .schema import check_raw_import_columns, expected_gps_schema, validate_gps_raw, validate_schema
from .synthetic import (
    SyntheticAnomalyRates,
    SyntheticExperiment,
    SyntheticGpsDataset,
    default_declic_synthetic_experiments,
    generate_synthetic_declic_gps,
    write_synthetic_gps_dataset,
)

# 2. Build structured mobility tables.
from .export import (
    export_clean_dataset,
    export_mobility_tables,
    mobility_dataset_tables,
    prepare_public_export_table,
    write_mobility_dataset,
)
from .mobility_tables import add_length_quantile_flags, add_user_id_day, split_storyline
from .parsing import assign_phase, drop_nans_if_low_rate, parse_date_columns, parse_ewkb
# Explicit construction path used by production notebooks.
from .prepare_tables import (
    apply_storyline_mappings,
    apply_trip_journey_mappings,
    prepare_journeys,
    prepare_storyline,
    prepare_trips,
)
from .relations import (
    add_excursion_flags_to_trips_journeys,
    add_journey_to_trips,
    add_trip_destination_activity,
    build_legs_staypoints_map,
    build_track_trip_journey_map,
)
from .temporal import build_relative_time_index
from .table_dictionary import (
    TABLE_DESCRIPTIONS,
    WORKING_TABLE_ORDER,
    build_attribute_dictionary,
    build_table_descriptions,
    column_description,
    table_description,
)
from .transform import (
    concat_mobility_datasets,
    prepare_mobility_dataset,
    prepare_mobility_datasets,
)
from .user_stats import (
    add_excursion_stats_to_user_stats,
    build_user_stats,
)

# 3. Control quality and spatial consistency.
from .quality import (
    build_daily_tracking_presence,
    build_mode_detection_precision,
    build_tracking_gap_report,
    build_tracking_quality_report,
    build_user_day_coverage,
    build_user_confirmation_rates,
    build_user_selection_table,
    build_weekly_participation_grid,
    calculate_tracking_periods,
    calculate_user_tracking_stats,
    filter_mobility_dataset_by_users,
    filter_table_by_users,
    flag_tracking_quality,
    get_extreme_legs_by_mode,
    resample_missing_stays,
    select_analysis_users,
    select_valid_tracking_users,
    summarize_leg_lengths_by_mode,
    summarize_participation_grid,
    summarize_phase_window_calendar_balance,
    summarize_phase_tracking,
    summarize_phase_window_weekday_balance,
    summarize_weekday_balance,
)
from .spatial import (
    DEFAULT_OSRM_SERVICES,
    GOOGLE_ROADS_SNAP_URL,
    GOOGLE_ROUTES_URL,
    GRAPHHOPPER_MATCH_URL,
    LegMapMatchingResult,
    add_excursion_flags,
    add_journey_origin_destination_from_trips,
    add_leg_origin_destination_zones,
    add_signal_loss_metrics,
    add_signal_quality_flags,
    add_signal_quality_to_user_stats,
    add_spatial_zone_labels,
    add_trip_origin_destination_from_legs,
    build_origin_destination_zone_correspondence,
    build_user_signal_quality_stats,
    classify_leg_relation_to_area,
    clean_leg_geometries,
    flag_low_quality_legs_by_mode,
    identify_bad_signal_users,
    max_consecutive_point_distance,
    map_match_legs,
)
from .viz import plot_gps_on_map, plot_gps_traces, plot_participation_heatmap

# 4. Compute mobility indicators.
from .enrichment import (
    CO2OccupancyConfig,
    HealthConfig,
    add_co2_occupancy_metrics,
    add_health_metrics,
    build_leg_enrichment_tables,
)
from .demand import build_daily_demand_profile
from .indicators import (
    build_person_day_indicators,
    build_person_phase_indicators,
    build_population_indicators,
    compute_mobility_indicators,
    population_indicator_summary,
)
from .motifs import (
    assign_mobility_motif_ids,
    build_mobility_motif_sequences,
    build_mobility_motifs,
    summarize_mobility_motifs,
)
from .viz import plot_indicator_bars, plot_mobility_motif_graphs

# 5. Export dashboard-ready analytical tables.
from .export import write_indicator_result
from .spatial_analytics import (
    add_time_slices,
    aggregate_h3_frequencies,
    build_h3_count_matrix,
    build_spatial_analytics_tables,
    legs_to_h3_points,
    write_duckdb_spatial_database,
    write_spatial_analytics_exports,
    write_spatial_analytics_tables,
)
from .viz import plot_h3_frequency_map

__all__ = [
    # 0. Recommended end-to-end entry point.
    "MobilityPipelineResult",
    "run_mobility_pipeline",
    # 1. Prepare raw inputs.
    "GpsExportPaths",
    "GeofenceAnonymizationConfig",
    "GpsAnonymizationResult",
    "MatchingThresholds",
    "MobilityMappings",
    "Phase",
    "ProjectConfig",
    "RawGpsData",
    "RawSampleConfig",
    "SchemaValidationResult",
    "SignalLossThreshold",
    "SpatialQualityThresholds",
    "SyntheticAnomalyRates",
    "SyntheticExperiment",
    "SyntheticGpsDataset",
    "TimeSlice",
    "TrackingThresholds",
    "ValidationIssue",
    "check_raw_import_columns",
    "anonymize_landed_gps_tables",
    "anonymize_storyline_with_geofencing",
    "apply_user_pseudonym_mapping",
    "available_reference_tables",
    "build_user_pseudonym_mapping",
    "concat_raw_gps_data",
    "default_declic_synthetic_experiments",
    "default_time_slices",
    "detect_staypoint_geofences",
    "drop_direct_identifier_columns",
    "expected_gps_schema",
    "generate_synthetic_declic_gps",
    "infer_gps_export_paths",
    "load_gps_export",
    "load_gps_source",
    "load_gps_sources",
    "load_reference_table",
    "load_sociodemo",
    "mode_purpose_mapping",
    "sample_raw_gps_data",
    "sample_tables_by_user_count",
    "swap_sensitive_purpose_columns",
    "source_id_for_config",
    "validate_gps_raw",
    "write_synthetic_gps_dataset",
    # 2. Build structured mobility tables.
    "MobilityDataset",
    "add_excursion_flags_to_trips_journeys",
    "add_excursion_stats_to_user_stats",
    "add_journey_to_trips",
    "add_length_quantile_flags",
    "add_trip_destination_activity",
    "add_user_id_day",
    "apply_storyline_mappings",
    "apply_trip_journey_mappings",
    "build_legs_staypoints_map",
    "build_relative_time_index",
    "build_attribute_dictionary",
    "build_table_descriptions",
    "build_track_trip_journey_map",
    "build_user_stats",
    "column_description",
    "concat_mobility_datasets",
    "export_clean_dataset",
    "mobility_dataset_tables",
    "prepare_journeys",
    "prepare_mobility_dataset",
    "prepare_mobility_datasets",
    "prepare_storyline",
    "prepare_trips",
    "prepare_public_export_table",
    "split_storyline",
    "table_description",
    "TABLE_DESCRIPTIONS",
    "WORKING_TABLE_ORDER",
    "write_mobility_dataset",
    # 3. Control quality and spatial consistency.
    "DEFAULT_OSRM_SERVICES",
    "GOOGLE_ROADS_SNAP_URL",
    "GOOGLE_ROUTES_URL",
    "GRAPHHOPPER_MATCH_URL",
    "LegMapMatchingResult",
    "add_excursion_flags",
    "add_journey_origin_destination_from_trips",
    "add_leg_origin_destination_zones",
    "add_signal_loss_metrics",
    "add_signal_quality_flags",
    "add_signal_quality_to_user_stats",
    "add_spatial_zone_labels",
    "add_trip_origin_destination_from_legs",
    "build_origin_destination_zone_correspondence",
    "build_daily_tracking_presence",
    "build_mode_detection_precision",
    "build_tracking_gap_report",
    "build_tracking_quality_report",
    "build_user_day_coverage",
    "build_user_confirmation_rates",
    "build_user_selection_table",
    "build_user_signal_quality_stats",
    "build_weekly_participation_grid",
    "calculate_tracking_periods",
    "calculate_user_tracking_stats",
    "classify_leg_relation_to_area",
    "clean_leg_geometries",
    "filter_mobility_dataset_by_users",
    "filter_table_by_users",
    "flag_low_quality_legs_by_mode",
    "flag_tracking_quality",
    "get_extreme_legs_by_mode",
    "identify_bad_signal_users",
    "max_consecutive_point_distance",
    "map_match_legs",
    "plot_gps_traces",
    "plot_gps_on_map",
    "plot_participation_heatmap",
    "resample_missing_stays",
    "select_analysis_users",
    "select_valid_tracking_users",
    "summarize_leg_lengths_by_mode",
    "summarize_participation_grid",
    "summarize_phase_window_calendar_balance",
    "summarize_phase_tracking",
    "summarize_phase_window_weekday_balance",
    "summarize_weekday_balance",
    # 4. Compute mobility indicators.
    "CO2OccupancyConfig",
    "HealthConfig",
    "IndicatorResult",
    "add_co2_occupancy_metrics",
    "add_health_metrics",
    "assign_mobility_motif_ids",
    "build_daily_demand_profile",
    "build_leg_enrichment_tables",
    "build_mobility_motif_sequences",
    "build_mobility_motifs",
    "build_person_day_indicators",
    "build_person_phase_indicators",
    "build_population_indicators",
    "compute_mobility_indicators",
    "plot_indicator_bars",
    "plot_mobility_motif_graphs",
    "population_indicator_summary",
    "summarize_mobility_motifs",
    # 5. Export dashboard-ready analytical tables.
    "add_time_slices",
    "aggregate_h3_frequencies",
    "build_h3_count_matrix",
    "build_spatial_analytics_tables",
    "legs_to_h3_points",
    "plot_h3_frequency_map",
    "write_duckdb_spatial_database",
    "write_indicator_result",
    "write_spatial_analytics_exports",
    "write_spatial_analytics_tables",
]
