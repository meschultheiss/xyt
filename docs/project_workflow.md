# Workflow projet xyt_gps

Cette page décrit la structure cible du projet et la correspondance entre les notebooks prêts à l'emploi, les tables multi-fichiers et les fonctions du package.

## Principe général

Le package se lit plus simplement en cinq blocs :

| Bloc | Objet principal | Sortie typique |
|---|---|---|
| 1. Préparer l’entrée | homogénéiser les sources GPS avant usage du package | `landing_dir` |
| 2. Transformer | produire les tables de mobilité structurées | `MobilityDataset`, `transformed_dir` |
| 3. Contrôler | qualifier le suivi, le signal GPS, les utilisateurs et les états intermédiaires | rapports qualité, participation, filtres utilisateurs |
| 4. Produire les indicateurs | enrichir CO2/santé et calculer les indicateurs génériques | `IndicatorResult`, `enriched_dir` |
| 5. Exporter la base propre | consolider les tables transformées, enrichies et cartographiques | `clean_dir`, GeoParquet, H3, DuckDB, HTML |

Le package `xyt_gps` doit rester utilisable étape par étape. L'orchestrateur `prepare_mobility_dataset()` est utile, mais chaque étape importante doit aussi exister comme fonction explicite afin de pouvoir contrôler les états intermédiaires.

Les notebooks de production de ce dépôt sont rangés dans
`Notebooks/` avec une numérotation continue : landing, loading,
qualité, spatial, enrichissement et indicateurs. Pour un nouveau projet, voir
la page [Structure de projet recommandée](getting-started/project-structure.md).

À partir du loading, une exécution traite une seule expérience. Les sorties sont
rangées sous `2-transformed-data/<experiment_name>`,
`3-enriched-data/<experiment_name>` et `4-clean-data/<experiment_name>`. Cette
organisation réduit les ambiguïtés de configuration lorsque les phases ou les
règles de suivi diffèrent entre expériences. Les analyses multi-expériences
concatènent ensuite les bases propres au début des notebooks thématiques.

Le package ne couvre pas les analyses thématiques finales. Il doit aller
jusqu'à la production d'une base de mobilité analysable : tables de mobilité
structurées, indicateurs génériques, enrichissements transversaux documentés et
exports.

## Périmètre cible

| Bloc | Rôle | Dans le package |
|---|---|---|
| 1. Préparer l’entrée | homogénéiser les sources et produire une base prête à ingérer | support par notebooks et utils hors coeur package |
| 2. Transformer | transformer les exports GPS en objets de mobilité contrôlés | oui |
| 3. Contrôler | diagnostiquer suivi, signal GPS, outliers, zones et participation | oui |
| 4. Produire les indicateurs | préparer les enrichissements CO2/santé et les indicateurs génériques | oui |
| 5. Préparer les exports dashboard | produire GeoParquet, H3, DuckDB et cartes HTML | oui, comme couche optionnelle |
| analyses thématiques | figures, interprétations et analyses spécifiques | non |

Les enrichissements CO2 et santé doivent être intégrés avec le même principe que la préparation des données :
fonctions petites, paramètres explicites, docstrings courtes, documentation des
hypothèses et notebooks d'exemple. Les facteurs CO2, taux d'occupation et METs
doivent être lus depuis des référentiels CSV versionnés, pas cachés dans le
code. Ils ne doivent pas contenir l'interprétation thématique finale.

## Repères historiques

La documentation d’usage emploie les termes `landing`, `préparation`,
`contrôle qualité`, `spatial`, `enrichissement`, `indicateurs` et `export
analytique`. Les correspondances avec les anciens notebooks sont conservées dans
la page de maintenance [Migration des notebooks historiques](notebook_migration_matrix.md).

## Structure des données GPS

La base GPS est multi-fichier. Les tables principales sont :

| Table | Rôle | Identifiants importants |
|---|---|---|
| `storyline` | table brute ou landed contenant les segments `Stay` et `Track` | `storyline_id`, `user_id`, `trip_id` si disponible |
| `staypoints` | activités ou arrêts dérivés de `storyline` | `activity_id`, `user_id` |
| `legs` | déplacements élémentaires dérivés de `storyline` | `leg_id`, `user_id`, `user_id_day`, `trip_id` si disponible |
| `trips` | niveau déplacement fourni ou reconstruit | `trip_id`, `user_id`, `journey_id` après matching |
| `journeys` | niveau journée ou chaîne de déplacements fourni | `journey_id`, `user_id` |
| `user_stats` | statistiques et informations utilisateur | `user_id` |
| `user_day_coverage` | calendrier canonique personne-jour et états de suivi | `experiment_name`, `user_id`, `date`, `phase` |
| `user_expe` | calendrier et rattachement projet au niveau utilisateur | `experiment_name`, `user_id` |
| `public_transport_legs` | enrichissement optionnel | `leg_id` |
| `map_track_trip_journey` | table de construction | `leg_id`, `trip_id`, `journey_id` |
| `map_legs_staypoints` | table de construction | `previous_leg_id`, `activity_id`, `next_leg_id` |

Les tables doivent pouvoir être liées par identifiants stables. Les tables de construction sont conservées pour rendre les jointures vérifiables. Quand une relation source est incomplète, la table de mapping garde le `leg_id` mais laisse la clé parente à `NA` plutôt que de créer un lien invalide.

`user_stats` est la synthèse à l'échelle utilisateur. `user_day_coverage`
distingue systématiquement les jours calendaires de la plage individuelle
(`day_in_range`), les jours avec une observation réelle (`tracked_day`) et les
jours avec un déplacement (`active_day`). Les métriques éventuellement livrées
par le fournisseur dans `user_statistics` sont remplacées par ces calculs.

## Sorties de production

La chaîne actuelle produit plusieurs niveaux de sortie. Le package n'impose pas
leur emplacement : les notebooks doivent définir des variables de chemin au
début du traitement.

| Variable | Exemple de dossier | Contenu | Usage |
|---|---|---|---|
| `landing_dir` | `data/outputs/0-landed-data/<experiment_name>/complete` | tables homogénéisées, manifest, README, `user_expe`, `column_alias_report`, rapports de colonnes | base prête à ingérer |
| `anonymized_altered_landing_dir` | `data/outputs/0-landed-data/<experiment_name>/anonymized_altered` | copie pseudonymisée et spatialement altérée si le projet en produit une | validation technique sans exposer les données personnelles directes ; les trajectoires ne doivent pas être interprétées scientifiquement |
| `transformed_dir` | `data/outputs/2-transformed-data/<experiment_name>` | `storyline`, `legs`, `staypoints`, `trips`, `journeys`, `user_stats`, `user_day_coverage`, `user_expe`, mappings, qualité et participation | base structurée de mobilité |
| `spatial_analytics_dir` | `data/outputs/2-transformed-data/<experiment_name>/spatial-analytics` | points de legs indexés H3 et fréquentation par cellule | carte de fréquentation simple à partir des legs transformés |
| `enriched_dir` | `data/outputs/3-enriched-data/<experiment_name>` | `legs_enriched`, tables CO2/santé, indicateurs, rapports QA, figures HTML | enrichissements et indicateurs génériques |
| `clean_dir` | `data/outputs/4-clean-data/<experiment_name>` | `cleaned-base` uniquement | base analysable pour les notebooks thématiques |

Les contrôles clés exportés avec `3-enriched-data` sont :

- `qa_key_checks.csv` : clés manquantes, uniques et dupliquées ;
- `qa_link_checks.csv` : liens invalides ou absents entre tables ;
- `qa_phase_coverage.csv` : lignes par table et par phase, y compris `Other` ;
- `qa_analysis_scope.csv` : volume conservé dans le périmètre analytique.

## Data-landing

Bloc 1 du package.

Objectif : ouvrir des sources hétérogènes et produire des tables landed homogènes.

| Étape | Fonction ou notebook | Statut |
|---|---|---|
| ouvrir les fichiers sources | `Notebooks/000_data_landing.ipynb` et `Notebooks/utils/data_landing.py` | en place |
| concaténer plusieurs fichiers bruts | `build_landed_tables()` | en place |
| conserver les configs de base | `landing_config.json` | en place |
| produire un manifest | `landing_manifest.json` | en place |
| produire `user_expe` | `build_user_expe_table()` | en place |
| produire un README de livraison | `write_landing_output()` | en place |
| détecter les colonnes alias ou dupliquées | `scan_landing_alias_columns()` | en place |
| vérifier le schéma attendu | `xyt.expected_gps_schema()` | en place dans le notebook landing |
| harmoniser ou compléter les colonnes attendues | fonctions landing utils | partiel, à documenter source par source |

Le landing doit accepter des données avec ou sans expérience et avec ou sans phase. Si les phases existent, elles sont décrites dans la configuration. Si elles n'existent pas, le loading doit fonctionner avec un `ProjectConfig()` minimal.

## Data-loading

Bloc 2 du package.

Objectif : charger les tables landed et construire les tables de mobilité inspectables.

| Étape | Fonction package | Paramètres principaux | Statut |
|---|---|---|---|
| charger les tables landed | `pandas.read_parquet()`, `pandas.read_csv()` | chemins `landing_dir` | notebook |
| revérifier le schéma | `expected_gps_schema()`, `check_raw_import_columns()`, `validate_gps_raw()` | `trips`, `journeys`, `raise_on_error` | en place |
| contrôler et nettoyer les nulls critiques | `prepare_storyline()` | `drop_nan_threshold=0.01` | en place |
| parser EWKB en Shapely | `prepare_storyline()` | `target_crs` | en place |
| parser les dates | `prepare_storyline()`, `prepare_trips()`, `prepare_journeys()` | colonnes connues | en place |
| agréger modes et motifs | `mode_purpose_mapping()`, `apply_storyline_mappings()`, `apply_trip_journey_mappings()` | mappings par table et niveau | en place |
| gérer phases et expériences | `ProjectConfig`, `Phase`, `concat_mobility_datasets()` | `experiment_name`, `phases` | en place pour phases par expérience |
| resampler les jours manquants | `resample_missing_stays()` | `resample_missing_days` | en place |
| distinguer jours de plage, trackés, actifs et resamplés | `build_user_day_coverage()`, `build_user_stats()` | `days_in_range_count`, `tracked_days_count`, `active_days_count`, `resampled_days_count` | en place |
| enrichir `user_stats` | `build_user_stats()` | `user_statistics`, `sociodemo`, `weights` | en place partiel |
| pondérer | `build_user_stats()` | `weights`, `weight_col`, `default_weight=1.0` | en place |
| scinder storyline | `split_storyline()` | aucun | en place |
| ajouter `user_id_day` | `add_user_id_day()` | aucun | en place |
| ajouter durée aux legs et staypoints | `split_storyline()` | dates de début/fin | en place |
| flagger valeurs extrêmes | `add_length_quantile_flags()` | `quantiles`, `group_col`, `length_col` | en place |
| matcher legs, trips, journeys | `build_track_trip_journey_map()` | `tolerance`, `trip_id` si disponible | en place |
| matcher legs et staypoints | `build_legs_staypoints_map()` | `tolerance` | en place |
| propager excursions | `add_excursion_flags_to_trips_journeys()`, `add_excursion_stats_to_user_stats()` | `excursion_col` | en place |
| calculer excursions géographiques | `add_excursion_flags()` | `area`, `area_path`, `center`, `radius_m`, `mode` | en place |
| exporter les tables transformées | `write_mobility_dataset()` | `formats=("parquet", "csv", "pickle")`, `extra_tables={...}` | en place |
| exporter les tables spatiales dashboard | `legs_to_h3_points()`, `aggregate_h3_frequencies()`, `write_spatial_analytics_exports()` | `h3_resolution`, `sample_distance_m`, `frequency_group_cols` | en place optionnel |
| produire une base SQL locale | `write_duckdb_spatial_database()` | `load_spatial`, `require_spatial_extension` | en place optionnel |
| exporter participation | `build_weekly_participation_grid()` | `exclude_resampled=True` | en place dans notebook |
| contrôler la qualité du suivi | `build_tracking_gap_report()`, `summarize_leg_lengths_by_mode()`, `build_user_confirmation_rates()` | notebooks de diagnostic | en place |
| recaler/nettoyer/enrichir spatialement | `map_match_legs()`, `add_spatial_zone_labels()`, `add_leg_origin_destination_zones()`, `classify_leg_relation_to_area()` | clé GraphHopper optionnelle, plafond, zones, CRS et seuils géométriques | client GraphHopper en place dans `020` et banc d'essai `021`; appels désactivés par défaut |

## Enrichissements et indicateurs

Bloc 4 du package.

Après la préparation des données, le package doit préparer les enrichissements
nécessaires aux indicateurs génériques, sans basculer dans l'analyse thématique.

| Étape | Entrées | Sorties attendues | Principe |
|---|---|---|---|
| enrichissement CO2 | `legs`, `trips`, `journeys`, mappings de modes, référentiel CO2, référentiel d'occupation | émissions par leg et agrégations personne-jour ou période | facteurs visibles, sourcés et remplaçables |
| enrichissement santé | `legs`, modes, durées, distances, vitesses, référentiel METs, hypothèses de calories | indicateurs santé élémentaires par leg et agrégations | hypothèses paramétrables |
| indicateurs et export analytique | tables enrichies | base de mobilité analysable et export final | indicateurs génériques, non interprétation thématique |

Le notebook de référence pour ce niveau est :

```text
Notebooks/030_enrich_indicators.ipynb
```

Dans la chaîne projet Déclic, il lit obligatoirement
`transformed_dir / "spatial-cleaned"`, calcule les enrichissements, exporte
`enriched_dir` et produit des visualisations HTML compatibles avec le design du
dashboard. Les indicateurs agrégés et par expérimentation sont regroupés dans
`enriched_dir / "indicators"`.

Le notebook `Notebooks/040_export_cleaned_dataset.ipynb` consolide ensuite
les tables `spatial-cleaned` et les enrichissements utiles de `enriched_dir`
dans `clean_dir / "cleaned-base"`. Les identifiants des legs spatiaux et
enrichis doivent être strictement identiques. Il ne produit pas de figures,
rapports QA, exports H3, référentiels, manifests ou base DuckDB.

Lorsque `BUILD_RELATIVE_TIME_INDEX=True` dans `040_export_cleaned_dataset.ipynb`,
`cleaned-base` contient aussi
`relative_time_index`, une table de correspondance temporelle avec une ligne par
événement de `storyline`, les dates absolues et un calendrier relatif pour
superposer plusieurs vagues. L’ancre temporelle relative doit représenter le
jour 1 de la semaine 1 du calendrier de comparaison. Avec la convention actuelle
du package, elle doit être un lundi ; sinon le package émet un warning pour
éviter de décaler les jours de semaine.

## Exports dashboard

Bloc 5 du package.

Ce bloc transforme les données déjà préparées en formats adaptés aux
plateformes de visualisation et aux requêtes analytiques :

| Besoin | Fonction | Sortie |
|---|---|---|
| points de legs indexés H3 | `legs_to_h3_points()` | `leg_points_h3` |
| fréquentation par cellule et dimensions | `aggregate_h3_frequencies()` | `h3_frequency` |
| counts larges par mode, motif et tranche horaire | `build_h3_count_matrix()` | `h3_count_matrix` |
| construction complète des tables spatiales | `build_spatial_analytics_tables()` | dictionnaire de tables |
| export spatial dashboard | `write_spatial_analytics_tables()` | fichiers `.parquet` et `.csv` par défaut, `.pkl` optionnel |
| base SQL locale | `write_duckdb_spatial_database()` | fichier `.duckdb` |
| carte de contrôle | `plot_h3_frequency_map()` | HTML interactif |

Les paramètres structurants sont `h3_resolution`, `time_slices`,
`frequency_group_cols`, `count_dimension_sets` et `sample_distance_m`.

## Paramètres structurants

Les paramètres à définir explicitement au début du loading sont :

| Paramètre | Exemple | Rôle |
|---|---|---|
| `project_root` | `Path("..").resolve()` | racine du projet d'analyse, pas du package |
| `data_root` | `project_root / "data"` | racine locale des données, ignorée par git |
| `output_root` | `data_root / "outputs"` | racine des tables produites |
| `experiment_name` | `declic-mobility-vague1` | identifie l'expérience traitée par l'exécution courante |
| `landing_dir` | `output_root / "0-landed-data" / experiment_name / "anonymized_altered"` | source des tables landed de contrôle |
| `transformed_dir` | `output_root / "2-transformed-data" / experiment_name` | destination des tables structurées de mobilité |
| `enriched_dir` | `output_root / "3-enriched-data" / experiment_name` | destination des enrichissements, indicateurs et figures |
| `clean_dir` | `output_root / "4-clean-data" / experiment_name` | destination de la base propre finale pour analyses thématiques |
| `spatial_analytics_dir` | `transformed_dir / "spatial-analytics"` | destination des points H3 et agrégats spatiaux |
| `h3_resolution` | `9` | taille des cellules pour les cartes de fréquentation |
| `sample_distance_m` | `None` ou `50` | utilise les sommets des legs ou échantillonne les lignes |
| `phases` | `Phase("Phase1", "2025-01-01", "2025-01-21")` | optionnel, découpage analytique |
| `mappings` | `xyt.mode_purpose_mapping()` | regroupement modes et motifs |
| `drop_nan_threshold` | `0.01` | seuil de nulls critiques acceptable |
| `RESAMPLE_MISSING_DAYS` | `True` | ajoute des `Resampled_stay` |
| `ADD_EXCURSION_FLAGS` | `False` par défaut | active une règle géographique d'excursion |
| `EXCURSION_CENTER` | `(6.1558, 46.2073)` | centre si règle par cercle |
| `EXCURSION_RADIUS_M` | `350000` | rayon en mètres |
| `weight_col` | `weight` | pondération utilisateur |

## Nommage des fonctions

Le package suit les conventions Python :

| Élément | Convention | Exemple |
|---|---|---|
| fonctions | `snake_case` | `prepare_storyline()` |
| classes et dataclasses | `PascalCase` | `ProjectConfig` |
| constantes | `UPPER_CASE` | `GPS_SCHEMAS` |
| helpers internes | `_snake_case` | `_phase_week_calendar()` |

Les noms publics doivent rester génériques lorsque la fonction ne dépend pas strictement d’un fournisseur GPS. Par exemple :

- utiliser `expected_gps_schema()` dans les notebooks ;
- utiliser `validate_gps_raw()` pour valider les tables brutes ;
- utiliser `prepare_mobility_dataset()` pour construire le `MobilityDataset` ;
- utiliser `mode_purpose_mapping()` pour créer une configuration de mapping lisible ;
- garder `MobilityMappings` comme dataclass de configuration avancée.

## Couverture de la séquence demandée

| Point demandé | Couverture actuelle |
|---|---|
| ouvrir, concaténer et documenter les sources | couvert dans `Notebooks/000_data_landing.ipynb` et `Notebooks/utils/data_landing.py` |
| produire `manifest`, `README`, `user_expe` | couvert dans le landing |
| détecter colonnes équivalentes ou divergentes | couvert avec `column_alias_report.csv` |
| vérifier le contrat de colonnes en landing et loading | couvert avec `expected_gps_schema()` et `check_raw_import_columns()` |
| gérer des données avec ou sans expérimentation | couvert par `ProjectConfig()` minimal et par le notebook loading |
| gérer des phases si elles existent | couvert par `Phase` et `ProjectConfig(phases=...)` |
| parser EWKB et dates | couvert par `prepare_storyline()`, `prepare_trips()`, `prepare_journeys()` |
| agréger modes et motifs | couvert par `mode_purpose_mapping()` et les fonctions `apply_*_mappings()` |
| resampler les jours manquants | couvert par `resample_missing_stays()` ; les jours observés et resamplés sont séparés dans `user_stats` |
| enrichir `user_stats` | couvert pour suivi, qualité GPS, excursions, poids par défaut et tables utilisateur simples |
| questionnaires hétérogènes | partiel : exportés séparément au landing ; consolidation analytique à définir par projet |
| pondération | couvert techniquement avec `weights`, `weight_col`, `default_weight=1.0` ; méthode de pondération à documenter si utilisée |
| scinder `storyline` en `staypoints` et `legs` | couvert par `split_storyline()` |
| ajouter `user_id_day`, durées, outliers | couvert par `add_user_id_day()`, `split_storyline()`, `add_length_quantile_flags()` |
| matcher legs, trips, journeys | couvert par `build_track_trip_journey_map()` ; utilise `trip_id` s'il existe |
| matcher legs et staypoints | couvert par `build_legs_staypoints_map()` |
| motifs de boucle quotidiens | couvert par `build_mobility_motifs()` et `summarize_mobility_motifs()` |
| excursions depuis zone ou cercle | couvert par `add_excursion_flags()` ; activation et zone à définir par projet |
| propager excursions aux trips, journeys et user_stats | couvert par `add_excursion_flags_to_trips_journeys()` et `add_excursion_stats_to_user_stats()` |
| public transport | partiel : table landing conservée ; contrat de jointure stable à finaliser |
| export transformé parquet/pickle/csv | couvert par `write_mobility_dataset()` |
| export spatial pour dashboard | couvert par `legs_to_h3_points()`, `aggregate_h3_frequencies()`, `write_spatial_analytics_exports()` et `write_duckdb_spatial_database()` |
| qualité de suivi | couvert par `build_tracking_gap_report()`, `summarize_phase_tracking()`, `build_user_confirmation_rates()` et le notebook `Notebooks/011_quality_check.ipynb` |
| nettoyage spatial | couvert par `spatial_geometry.py`, `spatial_quality.py`, `spatial_zones.py` et le notebook `Notebooks/020_spatial_cleaning.ipynb` ; les couches territoriales restent à fournir par projet |

## Points encore à finaliser

| Point | Décision proposée |
|---|---|
| phases propres à chaque utilisateur | utiliser `user_expe` pour produire une table de périodes utilisateur, puis adapter participation et indicateurs |
| public transport enrichi | définir le contrat de jointure stable avec `legs` ou `trips` |
| questionnaires | conserver des exports séparés, puis créer une table sociodémo consolidée si nécessaire |
| comparaison historique package/notebook | remplacée par `Notebooks/010_data_loading.ipynb` et les contrôles d'exports anonymisés |
| exports pickle | pris en charge via `formats=("pickle",)` ou `formats=("pkl",)` ; Parquet reste préférable pour les tables tabulaires durables |
| tests de non-régression historiques | comparer les exports package avec des sorties notebook de référence anonymisées |
