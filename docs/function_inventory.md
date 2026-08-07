# Inventaire actuel de l'API `xyt_gps`

Date : 2026-06-19

Cette page décrit l'état courant du package avant publication PyPI. Elle ne
liste pas les fonctions envisagées, les idées issues des archives, ni les
éléments de conception non implémentés.

La page d'usage à privilégier reste [API recommandée](reference/recommended-api.md).
Le présent inventaire sert surtout à vérifier le périmètre public et à éviter
que `__all__` redevienne un fourre-tout.

## Règles de contrat

Deux niveaux d'usage sont conservés :

- chemin court recommandé : charger, préparer, sélectionner, calculer, exporter ;
- chemin explicite notebook : préparer les tables et inspecter les états
  intermédiaires étape par étape.

`import xyt_gps as xyt` reste supporté pour les notebooks de production. En
revanche, `from xyt_gps import *` doit exposer une surface plus courte et plus
lisible. Les helpers internes restent accessibles par attribut ou par
sous-module lorsqu'ils sont nécessaires à un diagnostic.

Aucun fallback implicite ne doit masquer une incohérence de configuration. Si
un filtre strict vide un périmètre d'analyse, la correction doit passer par une
configuration explicite, un paramètre visible ou une erreur contrôlée.

## Chemin court recommandé

Symboles à mettre en avant dans les guides de démarrage :

```python
ProjectConfig
Phase
TimeSlice
RawGpsData
load_gps_export
check_raw_import_columns
prepare_mobility_dataset
mobility_dataset_tables
build_user_selection_table
filter_mobility_dataset_by_users
compute_mobility_indicators
build_relative_time_index
export_clean_dataset
plot_gps_traces
plot_participation_heatmap
plot_indicator_bars
plot_mobility_motif_graphs
```

Exemple minimal :

```python
import xyt_gps as xyt

config = xyt.ProjectConfig()
raw = xyt.load_gps_export(config)
dataset = xyt.prepare_mobility_dataset(raw, config)
selection = xyt.build_user_selection_table(dataset.user_stats)
indicators = xyt.compute_mobility_indicators(dataset)
manifest = xyt.export_clean_dataset(dataset, indicators, output_dir, selection_table=selection)
```

Par défaut, l'export propre ajoute aussi `table_descriptions` et
`attribute_dictionary` au manifeste.

## Chemin explicite notebook

Ces symboles restent dans l'API racine car ils sont utilisés pour rendre les
choix méthodologiques visibles dans les notebooks de production :

```python
prepare_storyline
prepare_trips
prepare_journeys
apply_storyline_mappings
apply_trip_journey_mappings
split_storyline
add_user_id_day
add_signal_quality_flags
build_track_trip_journey_map
build_legs_staypoints_map
add_journey_to_trips
add_trip_destination_activity
build_user_stats
add_signal_quality_to_user_stats
MobilityDataset
write_mobility_dataset
write_indicator_result
```

`MobilityDataset` reste volontairement mutable. Les notebooks peuvent enrichir
une table, par exemple `dataset.legs`, si cette opération est visible dans le
notebook.

## API racine exportée par `__all__`

### Configuration et modèles

```python
ProjectConfig
Phase
TimeSlice
TrackingThresholds
SpatialQualityThresholds
SignalLossThreshold
MatchingThresholds
MobilityMappings
RawGpsData
MobilityDataset
IndicatorResult
SchemaValidationResult
ValidationIssue
GpsExportPaths
RawSampleConfig
MobilityPipelineResult
```

### Import, validation et jeux d'essai

```python
infer_gps_export_paths
load_gps_export
load_gps_source
load_gps_sources
concat_raw_gps_data
load_sociodemo
source_id_for_config
check_raw_import_columns
expected_gps_schema
validate_gps_raw
sample_raw_gps_data
SyntheticAnomalyRates
SyntheticExperiment
SyntheticGpsDataset
default_declic_synthetic_experiments
generate_synthetic_declic_gps
write_synthetic_gps_dataset
```

### Anonymisation de données landed

```python
GeofenceAnonymizationConfig
GpsAnonymizationResult
anonymize_landed_gps_tables
anonymize_storyline_with_geofencing
apply_user_pseudonym_mapping
build_user_pseudonym_mapping
detect_staypoint_geofences
drop_direct_identifier_columns
sample_tables_by_user_count
swap_sensitive_purpose_columns
```

### Construction des tables de mobilité

```python
mode_purpose_mapping
prepare_mobility_dataset
prepare_mobility_datasets
concat_mobility_datasets
prepare_storyline
prepare_trips
prepare_journeys
apply_storyline_mappings
apply_trip_journey_mappings
split_storyline
add_user_id_day
add_length_quantile_flags
build_track_trip_journey_map
build_legs_staypoints_map
add_journey_to_trips
add_trip_destination_activity
add_excursion_flags_to_trips_journeys
add_excursion_stats_to_user_stats
build_user_stats
mobility_dataset_tables
prepare_public_export_table
write_mobility_dataset
export_clean_dataset
```

### Qualité, participation et spatial

```python
calculate_user_tracking_stats
calculate_tracking_periods
flag_tracking_quality
build_tracking_quality_report
build_daily_tracking_presence
build_weekly_participation_grid
summarize_participation_grid
summarize_phase_tracking
build_tracking_gap_report
build_user_confirmation_rates
build_mode_detection_precision
build_user_selection_table
select_analysis_users
select_valid_tracking_users
filter_table_by_users
filter_mobility_dataset_by_users
resample_missing_stays
add_signal_loss_metrics
flag_low_quality_legs_by_mode
add_signal_quality_flags
add_signal_quality_to_user_stats
build_user_signal_quality_stats
identify_bad_signal_users
get_extreme_legs_by_mode
summarize_leg_lengths_by_mode
max_consecutive_point_distance
clean_leg_geometries
add_excursion_flags
add_spatial_zone_labels
add_leg_origin_destination_zones
build_origin_destination_zone_correspondence
classify_leg_relation_to_area
add_trip_origin_destination_from_legs
add_journey_origin_destination_from_trips
```

### Indicateurs, enrichissements et motifs

```python
CO2OccupancyConfig
HealthConfig
add_co2_occupancy_metrics
add_health_metrics
available_reference_tables
build_leg_enrichment_tables
build_person_day_indicators
build_person_phase_indicators
build_population_indicators
compute_mobility_indicators
load_reference_table
population_indicator_summary
build_daily_demand_profile
assign_mobility_motif_ids
build_mobility_motifs
build_mobility_motif_sequences
summarize_mobility_motifs
```

### Exports spatiaux et visualisations

```python
add_time_slices
legs_to_h3_points
aggregate_h3_frequencies
build_h3_count_matrix
build_spatial_analytics_tables
write_spatial_analytics_tables
write_spatial_analytics_exports
write_duckdb_spatial_database
build_relative_time_index
plot_gps_on_map
plot_gps_traces
plot_participation_heatmap
plot_indicator_bars
plot_mobility_motif_graphs
plot_h3_frequency_map
```

## Helpers compatibles hors `__all__`

Ces fonctions restent importées dans le namespace racine pour ne pas casser les
notebooks explicites, mais elles ne sont plus exposées par `from xyt_gps import
*`. Lorsqu'elles sont utilisées directement, préférer l'import par sous-module.

```python
xyt_gps.parsing.assign_phase
xyt_gps.parsing.drop_nans_if_low_rate
xyt_gps.parsing.parse_date_columns
xyt_gps.parsing.parse_ewkb
xyt_gps.sample_data.find_sample_gps_path
xyt_gps.sample_data.load_sample_gps
xyt_gps.schema.validate_schema
xyt_gps.export.export_mobility_tables
```

## Sous-modules à privilégier pour lecture du code

| Responsabilité | Sous-modules |
|---|---|
| configuration et modèles | `config`, `models`, `mappings` |
| import et validation | `io`, `schema` |
| tests et données internes | `sample_data`, `synthetic` |
| préparation des tables | `prepare_tables`, `mobility_tables`, `relations`, `transform`, `user_stats` |
| qualité et participation | `quality`, `quality_reports`, `quality_tracking`, `quality_resampling`, `quality_selection`, `quality_diagnostics` |
| spatial | `spatial`, `spatial_geometry`, `spatial_quality`, `spatial_zones` |
| indicateurs | `indicators`, `enrichment`, `demand`, `motifs` |
| exports | `export`, `temporal`, `spatial_analytics`, `spatial_h3`, `spatial_time`, `spatial_exports` |
| visualisation | `viz`, `viz_maps`, `viz_participation`, `viz_indicators`, `viz_demand`, `viz_motifs` |
| confidentialité | `privacy` |

## Hors contrat de publication initiale

Les éléments suivants ne doivent pas être présentés comme API actuelle tant
qu'ils ne sont pas implémentés, testés et documentés dans le package :

- segmentation depuis points GPS bruts non MotionTag ;
- graphes d'espace d'action ;
- objets `FilterConfig` ou `ExternalGeodataConfig` ;
- fonctions historiques des archives `xyt_dynamite`.
