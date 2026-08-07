# Migration des notebooks historiques

!!! note "Page de maintenance"
    Cette page sert à vérifier la migration des anciens notebooks. Elle n'est pas un tutoriel d'usage du package.

Cette page sert de document de travail pour vérifier que les transformations issues des anciens notebooks `Notebooks/260618_Archives/WP1` et `Notebooks/260618_Archives/WP2` sont bien reprises dans le package et dans les notebooks `Notebooks/`.

Les termes `WP1` et `WP2` sont conservés ici uniquement comme repères historiques. Dans l'usage du package, on parle plutôt de landing, préparation des données, qualité, spatial, enrichissements, indicateurs et export analytique.

## Synthèse

| Notebook historique | Rôle historique | Notebook package-ready | Statut | Remarques |
|---|---|---|---|---|
| `WP1/module0_1_1_parserawstoryline.ipynb` | chargement, parsing, mappings, resampling, découpage storyline, matching et export | `Notebooks/010_data_loading.ipynb` | couvert | notebook principal de remplacement |
| `WP1/module0_1_2_run_tracking_qual_check.ipynb` | qualité du suivi, confirmations, valeurs extrêmes | `Notebooks/011_quality_check.ipynb` | couvert | diagnostics séparés pour éviter une boîte noire |
| `WP1/module0_2_1_spatial_cleaning.ipynb` | pertes de signal, jointures spatiales, origine-destination | `Notebooks/020_spatial_cleaning.ipynb` | couvert partiellement | dépend des couches territoriales fournies par projet |
| `WP1/module0_4_filter_data.ipynb` | filtrage utilisateurs et données analysables | `Notebooks/011_quality_check.ipynb` + fonctions `filter_*` | partiel | logique disponible, règles finales à fixer par projet |
| `WP1/module1_1_co2_occupancy.ipynb` | occupation et émissions CO2 | `Notebooks/030_enrich_indicators.ipynb` | couvert | facteurs et taux d'occupation paramétrables via `Notebooks/config/reference/*.csv` |
| `WP1/module1_2_health.ipynb` | enrichissements santé | `Notebooks/030_enrich_indicators.ipynb` | couvert | métriques élémentaires, pas d'interprétation thématique |
| `WP1/module1_3_1_osrm_processor.ipynb` | OSRM, map matching, plus court chemin | aucun notebook coeur | hors noyau | à garder comme extension optionnelle si nécessaire |
| `WP1/module1_3_2_google_direction.ipynb` | Google Directions pour cas à faible qualité | aucun notebook coeur | hors noyau | dépend d'API externe et de clés |
| `WP2/1_base_personne_jour.ipynb` | indicateurs personne-jour, personne-phase, population | `Notebooks/030_enrich_indicators.ipynb` | couvert | via `compute_mobility_indicators()` |
| `WP2/2_export_data.ipynb` | documentation et export analytique | `Notebooks/030_enrich_indicators.ipynb` | couvert | sortie `Data/Output/3-enriched-data/<experiment_name>` |

## Notebooks package-ready

| Dossier | Notebook | Remplace |
|---|---|---|
| `Notebooks` | `000_data_landing.ipynb` | préparation manuelle des exports bruts |
| `Notebooks` | `010_data_loading.ipynb` | `WP1/module0_1_1_parserawstoryline.ipynb` |
| `Notebooks` | `011_quality_check.ipynb` | `WP1/module0_1_2_run_tracking_qual_check.ipynb` |
| `Notebooks` | `020_spatial_cleaning.ipynb` | `WP1/module0_2_1_spatial_cleaning.ipynb` |
| `Notebooks` | `030_enrich_indicators.ipynb` | `WP1/module1_1`, `WP1/module1_2`, `WP2/1_base_personne_jour.ipynb`, `WP2/2_export_data.ipynb` |

## Séquence détaillée

### Préparation des données

| Étape historique | Fonction ou notebook actuel | Sortie contrôlable |
|---|---|---|
| charger les exports bruts d'une expérience | `Notebooks/000_data_landing.ipynb` | `Data/Output/0-landed-data/<EXPERIMENT_NAME>/{complete,anonymized_altered}` |
| concaténer plusieurs fichiers d'une même table | `build_landed_tables()` | tables landed consolidées |
| supprimer les colonnes sensibles directes | `scan_sensitive_columns()` et logique du notebook landing | export sans `email` |
| vérifier les colonnes attendues | `expected_gps_schema()`, `check_raw_import_columns()` | rapport de colonnes |
| détecter les alias de colonnes identiques | `scan_landing_alias_columns()` | `column_alias_report.csv` |
| parser les dates | `prepare_storyline()`, `prepare_trips()`, `prepare_journeys()` | dates typées |
| convertir EWKB en géométrie | `prepare_storyline()` | `GeoDataFrame` en EPSG:4326 |
| agréger modes et motifs | `mode_purpose_mapping()`, `apply_*_mappings()` | colonnes `mode_*` et `purpose_*` |
| gérer phases et expériences | `ProjectConfig`, `Phase`, `concat_mobility_datasets()` | phase/expérience dans les tables |
| resampler les jours manquants | `resample_missing_stays()` | lignes `Resampled_stay` |
| produire les statistiques utilisateur | `build_user_stats()` | `user_stats` enrichi |
| scinder storyline | `split_storyline()` | `legs`, `staypoints` |
| ajouter `user_id_day` | `add_user_id_day()` | clé personne-jour |
| ajouter durées et valeurs extrêmes | `split_storyline()`, `add_length_quantile_flags()` | flags et durées inspectables |
| relier legs, trips, journeys | `build_track_trip_journey_map()` | `map_track_trip_journey` |
| relier legs et staypoints | `build_legs_staypoints_map()` | `map_legs_staypoints` |
| propager excursions | `add_excursion_flags*()` | flags dans legs, trips, journeys, user_stats |
| exporter les états | `write_mobility_dataset()` | manifest + fichiers parquet/csv/pkl |

### Qualité et spatial

| Étape historique | Fonction ou notebook actuel | Statut |
|---|---|---|
| jours suivis et jours manquants | `build_weekly_participation_grid()`, `summarize_phase_tracking()` | couvert |
| confirmations des arrêts | `build_user_confirmation_rates()` | couvert |
| confirmation et précision des modes | `build_user_confirmation_rates()` | couvert si colonnes disponibles |
| valeurs extrêmes par mode | `summarize_leg_lengths_by_mode()` | couvert |
| pertes de signal | `add_signal_quality_flags()` | couvert |
| métriques utilisateur de signal GPS | `add_signal_quality_to_user_stats()` | couvert |
| jointures spatiales activités | `add_spatial_zone_labels()` | couvert, couches à fournir |
| origine-destination des legs | `add_leg_origin_destination_zones()` | couvert, couches à fournir |
| origine-destination trips/journeys | `propagate_leg_zones_to_trips_journeys()` | couvert |

### Enrichissements et indicateurs

| Étape historique | Fonction ou notebook actuel | Statut |
|---|---|---|
| mapping occupation | `CO2OccupancyConfig.from_reference_files()`, `add_co2_occupancy_metrics()` | couvert |
| facteurs CO2 | `CO2OccupancyConfig.from_reference_files()`, `add_co2_occupancy_metrics()` | couvert |
| mapping santé | `HealthConfig.from_reference_files()`, `add_health_metrics()` | couvert |
| indicateurs personne-jour | `compute_mobility_indicators()` | couvert |
| indicateurs personne-phase | `compute_mobility_indicators()` | couvert |
| indicateurs population | `compute_mobility_indicators()` | couvert |
| figures HTML | `plot_indicator_bars()`, `plot_participation_heatmap()`, `plot_gps_traces()` | couvert |
| export analytique final | `write_indicator_result()` | couvert |

## Points à surveiller

| Point | État actuel | Décision proposée |
|---|---|---|
| `public_transport_legs` | conservé au landing | définir un contrat de jointure stable avec `legs` ou `trips` |
| questionnaires | exportés séparément car hétérogènes | construire une table sociodémo consolidée seulement si nécessaire |
| OSRM et Google Directions | non intégrés au noyau | garder comme extension optionnelle, hors package principal |
| phases par utilisateur | support via `user_presence` | privilégier `user_presence` pour les calendriers d'analyse |
| tests de non-régression | tests unitaires + notebooks exécutés | ajouter un petit jeu anonymisé ou synthétique stable pour CI |
