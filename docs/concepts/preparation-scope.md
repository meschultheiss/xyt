# Préparation des données

Cette page distingue les transformations indispensables, les contrôles fortement recommandés et les traitements optionnels pour préparer une base de mobilité analysable.

La préparation des données couvre pour l'instant les exports GPS déjà segmentés : `Storyline`, `Trips`, `Journeys` et, si disponible, `UserStatistics`. Elle ne couvre pas encore la segmentation complète de points GPS non structurés.

## Statut actuel

Le socle de préparation est considéré comme disponible pour le flux suivant :

```text
exports GPS structurés
-> validation du schéma brut
-> parsing dates et géométries
-> mappings modes et motifs
-> staypoints, legs, trips, journeys
-> tables de correspondance
-> user_stats et qualité de suivi
-> MobilityDataset
-> exports intermédiaires de contrôle
```

Ce socle suffit pour comparer progressivement les sorties du package avec les notebooks historiques de préparation des données.

Ce qui reste hors de cette étape obligatoire, mais pas nécessairement hors package :

- traitement de points GPS non segmentés ;
- map matching GraphHopper, disponible comme traitement spatial optionnel ;
- enrichissements territoriaux avancés par spatial join ;
- indicateurs, CO2 et santé, qui appartiennent aux étapes d’enrichissement et d’export analytique.

Les analyses thématiques restent hors package. Elles doivent utiliser les
exports analytiques comme entrée, sans faire remonter leurs hypothèses
spécifiques dans le coeur de transformation GPS.

## Socle obligatoire

Ces étapes sont nécessaires pour transformer des exports GPS structurés en tables de mobilité structurées.

| Étape | Fonctions principales | Pourquoi c'est nécessaire |
|---|---|---|
| Configuration projet | `ProjectConfig`, `Phase`, `mode_purpose_mapping()` | fixe les phases, chemins, CRS, mappings et seuils |
| Import brut | `load_gps_export()`, `load_gps_sources()`, `RawSampleConfig.by_users()` | charge les tables sources en conservant leur traçabilité |
| Validation brute | `check_raw_import_columns()`, puis validation interne | bloque les exports incomplets avant transformation |
| Parsing | `prepare_storyline()`, `prepare_trips()`, `prepare_journeys()` | convertit géométries, dates et types |
| Mappings | `apply_storyline_mappings()`, `apply_trip_journey_mappings()` | produit les catégories analytiques de modes et motifs |
| États mobilité | `split_storyline()`, `add_user_id_day()` | sépare `staypoints` et `legs`, ajoute les identifiants personne-jour |
| Correspondances | `build_track_trip_journey_map()`, `build_legs_staypoints_map()` | relie legs, trips, journeys et activités |
| Statistiques utilisateur | `build_user_stats()`, `calculate_tracking_periods()`, `flag_tracking_quality()` | documente durée de suivi, phases et qualité minimale |
| Objet final | `prepare_mobility_dataset()` | regroupe les tables dans `MobilityDataset` |

## Contrôles recommandés

Ces étapes ne changent pas toujours la structure minimale des données, mais elles sont importantes avant analyse.

| Étape | Fonctions principales | Statut |
|---|---|---|
| Qualité du suivi | `build_tracking_gap_report()`, `build_weekly_participation_grid()` | recommandé pour contrôler l’assiduité du tracking |
| Qualité GPS | `add_signal_quality_flags()`, `add_signal_quality_to_user_stats()` | recommandé par défaut, désactivable si la qualité a déjà été certifiée |
| Outliers de longueur | `add_length_quantile_flags()`, `summarize_leg_lengths_by_mode()`, `get_extreme_legs_by_mode()` | recommandé pour inspecter les legs extrêmes |
| Confirmation et modes | `build_user_confirmation_rates()`, `build_mode_detection_precision()` | utile lorsque les corrections utilisateur disponibles sont disponibles |
| Nettoyage spatial optionnel | `add_spatial_zone_labels()`, `add_leg_origin_destination_zones()`, `classify_leg_relation_to_area()` | utile si des couches territoriales de référence sont disponibles |
| Table de sélection | `build_user_selection_table()`, `select_analysis_users()` | recommandé avant tout calcul d'indicateur |
| Filtrage cohérent | `filter_mobility_dataset_by_users()` | recommandé pour appliquer une sélection à toutes les tables |
| Exports de contrôle | `write_mobility_dataset()` | recommandé pour comparer package et notebooks |
| Visualisation | `plot_gps_traces()` | utile en notebook, optionnel et dépendant de `folium` |

## Options de transformation

`prepare_mobility_dataset()` garde un comportement complet par défaut. Pour un export déjà nettoyé, il est possible de désactiver certaines étapes avec des paramètres explicites.

```python
dataset = xyt.prepare_mobility_dataset(
    raw,
    config,
    resample_missing_days=False,
    clean_leg_geometries=True,
    add_length_outlier_flags=False,
    add_signal_quality_flags=False,
)
```

Les options disponibles sont :

| Option | Défaut | Quand la désactiver |
|---|---:|---|
| `resample_missing_days` | `False` | à activer seulement si l'analyse exige un calendrier continu |
| `clean_leg_geometries` | `True` | si les géométries sont déjà homogènes et validées |
| `add_length_outlier_flags` | `True` | si les longueurs ont déjà été contrôlées en amont |
| `add_signal_quality_flags` | `True` | si les données reçues sont déjà nettoyées ou certifiées pour la perte de signal |

Même si les enrichissements optionnels sont désactivés, le package conserve les tables principales du `MobilityDataset`. Les colonnes de qualité signal absentes sont alors interprétées comme non calculées, pas comme une preuve de bonne qualité. Dans `user_stats`, la colonne `signal_quality_computed` garde cette information. Si elle vaut `False`, `build_user_selection_table(..., exclude_bad_signal_users=True)` renvoie une erreur explicite.

## Règle pratique

Pour un nouveau projet ou un nouvel export, utiliser le comportement par défaut.

Pour une donnée déjà préparée par un fournisseur ou une chaîne externe, désactiver seulement les étapes dont la méthode amont est documentée. Le notebook de démonstration doit alors indiquer explicitement quelles options ont été désactivées et pourquoi.
