# API recommandée

Cette page donne le chemin d’usage à privilégier. Les fonctions internes restent documentées dans la référence complète, mais l’apprentissage du package doit commencer par ces cinq blocs.

```python
from pathlib import Path
import xyt_gps as xyt

project_root = Path("..").resolve()
output_root = project_root / "data" / "outputs"
experiment_name = "experiment-a"
clean_dir = output_root / "4-clean-data" / experiment_name
```

## Chemin court recommandé

Pour découvrir le package, commencer par un fil explicite : charger, préparer,
contrôler, calculer, exporter.

```python
config = xyt.ProjectConfig()

raw = xyt.load_gps_export(config)
dataset = xyt.prepare_mobility_dataset(raw, config)
quality = xyt.build_user_selection_table(dataset.user_stats)
indicators = xyt.compute_mobility_indicators(dataset)

manifest = xyt.export_clean_dataset(
    dataset,
    indicators,
    clean_dir,
    selection_table=quality,
)
```

Quand les tables GPS sont déjà chargées avec `pandas`, construire `RawGpsData`
directement :

```python
raw = xyt.RawGpsData(storyline=storyline, trips=trips, journeys=journeys)
dataset = xyt.prepare_mobility_dataset(raw, config)
```

`run_mobility_pipeline()` reste disponible pour les scripts qui veulent un
raccourci compact. Il charge les données si `raw` n'est pas fourni, construit le
`MobilityDataset`, puis calcule les indicateurs génériques.

```python
result = xyt.run_mobility_pipeline(config, raw=raw)

result.raw
result.dataset
result.indicators
```

## Chemin explicite pour notebooks de production

Les notebooks de production peuvent déplier les étapes au lieu d'utiliser la
façade `prepare_mobility_dataset()`. C'est le bon choix lorsque les états
intermédiaires doivent rester inspectables ou lorsque les tables proviennent de
plusieurs sources déjà préparées.

Le contrat public de construction couvre notamment :

```python
storyline = xyt.prepare_storyline(raw.storyline, config)
trips = xyt.prepare_trips(raw.trips, config)
journeys = xyt.prepare_journeys(raw.journeys, config)
staypoints, legs = xyt.split_storyline(storyline)

legs = xyt.add_user_id_day(legs)
legs = xyt.add_signal_quality_flags(legs, config)
map_track_trip_journey = xyt.build_track_trip_journey_map(legs, trips, journeys)
map_legs_staypoints = xyt.build_legs_staypoints_map(staypoints, legs)
user_stats = xyt.build_user_stats(storyline, config)

dataset = xyt.MobilityDataset(
    storyline=storyline,
    staypoints=staypoints,
    legs=legs,
    trips=trips,
    journeys=journeys,
    user_stats=user_stats,
    map_track_trip_journey=map_track_trip_journey,
    map_legs_staypoints=map_legs_staypoints,
)
```

`MobilityDataset` est volontairement mutable. Les notebooks peuvent enrichir
une table après construction, par exemple remplacer `dataset.legs` par une
version enrichie, si cette opération reste visible dans le notebook. Pour les
scripts et les démonstrations, préférer `prepare_mobility_dataset()` puis les
fonctions de filtrage ou d'export qui retournent des objets ou des manifests
explicites.

## Import par sous-module

L'import `import xyt_gps as xyt` reste pratique pour les notebooks. Pour lire le
code ou explorer l'API par responsabilité, préférer les sous-modules :

```python
import xyt_gps.io as xyt_io
import xyt_gps.transform as xyt_transform
import xyt_gps.quality as xyt_quality
import xyt_gps.spatial_quality as xyt_spatial_quality
import xyt_gps.indicators as xyt_indicators
```

Ce pattern évite de dépendre de l'import étoile. Le namespace racine conserve
des helpers de compatibilité pour les notebooks de production, mais `__all__`
reste plus court : les fonctions internes comme `parse_ewkb()` ou
`validate_schema()` doivent être importées depuis leur sous-module si elles sont
utilisées directement.

## 1. Préparer l’entrée

Objectif : charger ou construire des tables GPS prêtes à être transformées.

| Besoin | Fonction ou objet | Sortie |
|---|---|---|
| définir le projet | `ProjectConfig`, `Phase`, `TimeSlice` | configuration explicite |
| déclarer les mappings | `mode_purpose_mapping()` | `MobilityMappings` |
| vérifier les colonnes attendues | `expected_gps_schema()`, `check_raw_import_columns()` | rapport de colonnes |
| échantillonner un gros export | `RawSampleConfig.by_users(n)`, `RawSampleConfig.random_rows(n)` | configuration d’échantillon |
| charger des fichiers sources | `load_gps_export()`, `load_gps_sources()` | `RawGpsData` |
| valider les tables brutes | `validate_gps_raw()` | rapports de validation |
| anonymiser un profil landed pour démonstration | `GeofenceAnonymizationConfig`, `anonymize_landed_gps_tables()` | tables pseudonymisées et geofences |

Cas minimal :

```python
config = xyt.ProjectConfig()

raw = xyt.RawGpsData(
    storyline=storyline,
    trips=trips,
    journeys=journeys,
    user_statistics=user_statistics,
)

xyt.check_raw_import_columns(
    raw.storyline,
    raw.user_statistics,
    trips=raw.trips,
    journeys=raw.journeys,
    raise_on_error=True,
)
```

Repères de nommage :

| Fonction | Usage |
|---|---|
| `load_gps_export()` | charge un export unique à partir de `ProjectConfig` |
| `load_gps_source()` | charge un export unique et ajoute des métadonnées de source |
| `load_gps_sources()` | charge et concatène plusieurs sources |

### Anonymisation pour démonstration

L'anonymisation géographique n'est pas une étape du pipeline analytique
standard. Elle sert à produire un profil de démonstration ou de test à partir
de tables landed locales. La méthode recommandée évite les déplacements
aléatoires de points : elle détecte les principaux clusters de staypoints par
utilisateur, construit des geofences, tronque les legs aux entrées et sorties de
ces geofences avec une distance variable, puis remplace les `user_id` par des
pseudonymes. Les legs trop courts sont traités explicitement avec
`short_leg_policy`.

```python
privacy_config = xyt.GeofenceAnonymizationConfig(
    top_staypoint_clusters=4,
    dbscan_eps_m=250,
    trim_buffer_min_m=20,
    trim_buffer_max_m=120,
    short_leg_policy="remove_geometry",
    sample_user_count=15,
    swap_sensitive_purpose_values=True,
)

result = xyt.anonymize_landed_gps_tables(tables, privacy_config)

anonymized_tables = result.tables
geofences = result.geofences
report = result.report
```

`swap_sensitive_purpose_values=True` ne mélange pas tous les motifs : seuls les
motifs sensibles configurés, par exemple domicile, travail et formation, sont
échangés entre eux. Les motifs non sensibles restent inchangés.

Cette fonction ne garantit pas à elle seule une conformité juridique ou
éthique. Avant de partager un test set, contrôler la carte, les colonnes
exportées, les volumes par utilisateur et l'absence de table de correspondance
entre anciens et nouveaux identifiants.

## 2. Transformer

Objectif : convertir les tables GPS en tables de mobilité liées entre elles.

| Besoin | Fonction ou objet | Sortie |
|---|---|---|
| lancer la préparation complète | `prepare_mobility_dataset()` | `MobilityDataset` |
| traiter plusieurs sources | `prepare_mobility_datasets()` | `MobilityDataset` concaténé |
| inspecter les tables | `mobility_dataset_tables()` | dictionnaire de tables |
| construire l’index temporel relatif | `build_relative_time_index()` | table de correspondance temporelle |
| écrire l’export propre recommandé | `export_clean_dataset()` | manifest unifié + tables propres |
| écrire les tables structurées intermédiaires | `write_mobility_dataset()` | Parquet, CSV ou pickle |

```python
dataset = xyt.prepare_mobility_dataset(
    raw,
    config,
    resample_missing_days=True,
    clean_leg_geometries=True,
    add_length_outlier_flags=True,
    add_signal_quality_flags=True,
)

tables = xyt.mobility_dataset_tables(dataset)
```

`export_clean_dataset()` est le point de sortie simple à privilégier en usage
courant. Il écrit les tables du `MobilityDataset`, les indicateurs, l’index
temporel relatif, les tables additionnelles déclarées par le projet et, par
défaut, `table_descriptions` et `attribute_dictionary`. Le résultat est un
manifest unique qui rend explicites les fichiers produits.

```python
manifest = xyt.export_clean_dataset(
    dataset,
    indicators,
    clean_dir,
    formats=("parquet", "csv"),
    selection_table=quality,
    extra_tables={
        "questionnaires": questionnaires,
        "occupancy_co2": occupancy_co2,
        "health": health,
    },
)
```

`build_relative_time_index()` est utile après export propre, notamment pour
comparer plusieurs vagues décalées dans le temps. La table garde une ligne par
événement de `storyline`, les identifiants de correspondance (`leg_id`,
`activity_id`, `trip_id`, `journey_id`), les dates locales absolues, puis ajoute
une semaine et une date relatives. On peut ainsi agréger, par exemple, tous les
mercredis de la semaine 4 de plusieurs vagues sans perdre la trace des dates
réelles. Le paramètre `relative_anchor` représente le jour 1 de la semaine 1 du
calendrier relatif. Avec la convention actuelle, les semaines commencent le
lundi : utiliser une ancre non-lundi décalerait les jours de semaine et déclenche
un warning. Le paramètre `default_timezone` est seulement un fallback quand les
colonnes timezone sont absentes ; les dates absolues utilisent les timezones
originales lorsqu’elles sont disponibles.

## 3. Contrôler

Objectif : rendre visibles les limites de suivi et les choix de nettoyage.

| Besoin | Fonction | Sortie |
|---|---|---|
| présence journalière | `build_daily_tracking_presence()` | table user-jour |
| participation hebdomadaire | `build_weekly_participation_grid()` | score 0-7 par semaine |
| équilibre calendaire lundi-dimanche | `summarize_phase_window_calendar_balance()` | `n_lun` à `n_dim` par fenêtre configurée |
| équilibre observé lundi-dimanche | `summarize_phase_window_weekday_balance()` | `n_lun` à `n_dim` par expérience/phase |
| rapport qualité | `build_tracking_quality_report()` | rapports utilisateur |
| trous de suivi | `build_tracking_gap_report()` | jours observés, manquants et consécutifs |
| confirmation utilisateur | `build_user_confirmation_rates()` | taux de confirmation |
| précision mode détecté/confirmé | `build_mode_detection_precision()` | matrice et taux de précision |
| sélection utilisateur | `build_user_selection_table()` | table de décision |
| qualité GPS | `add_signal_quality_flags()`, `build_user_signal_quality_stats()` | flags leg/user |
| contrôle cartographique structuré | `plot_gps_traces()` | carte HTML/Folium |
| contrôle cartographique exploratoire | `plot_gps_on_map()` | carte HTML/Folium |

```python
participation = xyt.build_weekly_participation_grid(dataset.storyline, config)
xyt.plot_participation_heatmap(participation)
weekday_balance = xyt.summarize_phase_window_weekday_balance(
    dataset.user_day_coverage,
    phase_windows,
    day_basis="day_in_range",
)
```

`plot_participation_heatmap()` ajoute par défaut un séparateur rouge entre deux
semaines lorsque la colonne de phase change. Cela permet de lire la
participation relativement aux périodes du protocole, sans devoir recalculer les
semaines calendaires. Les paramètres `summary` et `notes` ajoutent un court
contexte de contrôle dans l'export HTML ; `phase_col` indique la colonne utilisée
pour les séparateurs et les infobulles de phase.

`plot_gps_traces()` accepte `use_antpath=True` pour animer les legs avec Folium
et `show_staypoints=False` pour produire une carte centrée sur les traces sans
points d'arrêt. Ces options sont utiles dans les contrôles visuels des notebooks
de production.

Les enrichissements spatiaux utilisés en production restent dans l'API racine :
`add_spatial_zone_labels()` accepte `predicate` et `fill_value` pour contrôler
la jointure spatiale ; `add_leg_origin_destination_zones()` expose `origin_col`,
`destination_col` et `fill_value` ; `classify_leg_relation_to_area()` expose
`relation_col`, `code_col` et `operations_crs`.
`build_origin_destination_zone_correspondence()` construit une table légère de
correspondance OD pour plusieurs découpages polygonaux. Les entrées peuvent être
des chemins de fichiers, des GeoDataFrames ou des dictionnaires de configuration
avec `path`, `layer`, `zone_id_col`, `zone_name_col` et `fill_value`.

## 4. Produire les indicateurs

Objectif : enrichir les legs et produire des indicateurs génériques, sans faire l’analyse thématique finale.

| Besoin | Fonction | Sortie |
|---|---|---|
| lister les référentiels attendus | `available_reference_tables()` | noms des tables |
| inspecter un référentiel projet | `load_reference_table(name, path)` | `DataFrame` |
| enrichir CO2 | `add_co2_occupancy_metrics()` | legs enrichis |
| enrichir santé | `add_health_metrics()` | legs enrichis |
| construire les motifs quotidiens | `build_mobility_motifs()` | motifs par jour |
| visualiser les motifs quotidiens | `plot_mobility_motif_graphs()` | graphes HTML/SVG |
| construire le profil horaire | `build_daily_demand_profile()` | courbes 5 minutes |
| calculer les indicateurs | `compute_mobility_indicators()` | `IndicatorResult` |
| écrire les indicateurs | `write_indicator_result()` | tables exportées |
| visualiser les indicateurs | `plot_indicator_bars()` | HTML |

Les facteurs CO2, taux d'occupation et METs sont des tables CSV du projet,
stockées dans `Notebooks/config/reference/`. Le package sait les lire et les
valider, mais ne fournit pas de valeurs par défaut.

Pour la santé, les intensités actives sont déduites de `speed_kmh`, calculée à
partir de la distance et de la durée du leg. Le fichier
`metabolic_equivalent_tasks.csv` peut définir les colonnes `min_speed_kmh` et
`max_speed_kmh`; elles sont reprises dans les sorties
`intensity_min_speed_kmh` et `intensity_max_speed_kmh`.

```python
xyt.available_reference_tables()

co2_factors = xyt.load_reference_table("co2_factors", "Notebooks/config/reference/co2_factors.csv")
occupancy_rates = xyt.load_reference_table("occupancy_rates", "Notebooks/config/reference/occupancy_rates.csv")
met_values = xyt.load_reference_table(
    "metabolic_equivalent_tasks",
    "Notebooks/config/reference/metabolic_equivalent_tasks.csv",
)
```

Passer ensuite ces CSV projet aux configs d'enrichissement :

```python
co2_config = xyt.CO2OccupancyConfig.from_reference_files(
    co2_factors_path="Notebooks/config/reference/co2_factors.csv",
    occupancy_rates_path="Notebooks/config/reference/occupancy_rates.csv",
)
health_config = xyt.HealthConfig.from_reference_files(
    metabolic_equivalent_tasks_path="Notebooks/config/reference/metabolic_equivalent_tasks.csv",
)

dataset.legs = xyt.add_co2_occupancy_metrics(dataset.legs, config=co2_config)
dataset.legs = xyt.add_health_metrics(dataset.legs, config=health_config)
```

```python
indicators = xyt.compute_mobility_indicators(
    dataset,
    mode_col="mode_niv1",
    include_zero_days=True,
    include_excursions=True,
    include_airplane=False,
    use_weights=True,
    weight_col="weight",
)
```

Les paramètres à rendre visibles dans un rapport d’indicateurs sont :

| Paramètre | Rôle |
|---|---|
| `include_zero_days` | inclut les jours suivis sans déplacement dans les moyennes journalières |
| `include_excursions` | inclut ou exclut les legs/trips marqués comme excursions |
| `include_airplane` | inclut ou exclut les étapes et déplacements avion ; par défaut ils sont exclus |
| `use_weights` | calcule les moyennes population avec la pondération utilisateur |
| `weight_col` | nom de la colonne de pondération dans `user_stats` |

`plot_indicator_bars()` lit ces informations depuis `IndicatorResult.metadata` et les affiche dans une carte d’identité de l’export HTML.
L’export ajoute aussi une ligne `Tous modes`, en rose, pour chaque indicateur affiché. Cette ligne donne le total ou la moyenne tous modes confondus selon la métrique calculée, puis les modes détaillés restent visibles en dessous. La largeur de `Tous modes` et celle des modes détaillés utilisent deux échelles de référence distinctes, afin que le total tous modes ne réduise pas artificiellement les barres par mode.

Lorsque les legs contiennent des heures de début et de fin, `compute_mobility_indicators()` ajoute aussi un profil de demande par tranche de 5 minutes. `plot_indicator_bars()` l’affiche dans la carte d’identité sous forme de courbes par phase : une courbe tous modes confondus et des courbes par mode. La valeur affichée correspond au nombre moyen de personnes en déplacement sur une journée de la phase, ce qui rend les phases comparables même lorsqu’elles n’ont pas la même durée.

La carte d’identité contient aussi une heatmap horaire des fréquentations. Elle agrège par défaut les tranches de 5 minutes en heures pour rester lisible dans un export HTML. Un sélecteur permet de passer de `Tous modes` à un mode spécifique. Lorsque les legs contiennent une colonne de motif ou de purpose, la heatmap peut aussi être lue par motifs agrégés, en complément de la lecture par jours de semaine.

Si les données ont été filtrées avec `filter_mobility_dataset_by_users()`, la carte d’identité peut afficher un ratio du type `35/67` : le premier chiffre correspond aux utilisateurs conservés dans le calcul, le second aux utilisateurs présents dans l’export GPS avant filtre.

## 5. Préparer les exports dashboard

Objectif : produire des tables interrogeables et cartographiables.

| Besoin | Fonction | Sortie |
|---|---|---|
| ajouter des tranches horaires | `add_time_slices()` | colonne `time_slice` |
| convertir les legs en points H3 | `legs_to_h3_points()` | `leg_points_h3` |
| agréger la fréquentation | `aggregate_h3_frequencies()` | `h3_frequency` |
| produire des counts larges | `build_h3_count_matrix()` | `h3_count_matrix` |
| construire toutes les tables spatiales | `build_spatial_analytics_tables()` | dictionnaire de tables |
| écrire les exports | `write_spatial_analytics_tables()` | Parquet + CSV par défaut, pickle optionnel |
| inclure les exports spatiaux dans l’export propre | `export_clean_dataset(..., include_spatial_analytics=True)` | manifest unifié |
| créer une base SQL locale | `write_duckdb_spatial_database()` | `.duckdb` |
| cartographier H3 | `plot_h3_frequency_map()` | carte Folium |

```python
spatial_tables = xyt.build_spatial_analytics_tables(
    dataset,
    h3_resolution=[8, 9],
    frequency_group_cols=["h3_resolution", "h3_cell", "mode_niv1", "time_slice"],
    parallel=True,
    max_workers=None,
    chunk_size=250,
)

xyt.write_spatial_analytics_tables(
    spatial_tables,
    output_root / "spatial-analytics" / experiment_name,
    formats=("parquet", "csv"),
)
```

Dans `export_clean_dataset()`, les exports spatiaux sont désactivés par défaut
car ils peuvent être volumineux. Si `include_spatial_analytics=True` et que les
formats par défaut sont utilisés, la table détaillée `leg_points_h3` est écrite
en Parquet mais pas en CSV ; les tables agrégées restent exportées en Parquet et
CSV.

Pour les exports volumineux, `parallel=True` accélère l'indexation H3 en
traitant les legs par lots. Garder `max_workers=None` laisse Python choisir le
nombre de threads ; fixer un entier permet de limiter l'usage CPU sur une
machine partagée. `chunk_size` règle le nombre de legs par lot.

## À garder en tête

Le cas par défaut ne suppose ni expérimentation, ni phase. Les phases, pondérations, questionnaires et découpages temporels sont des couches de configuration à ajouter seulement lorsqu’elles existent dans le projet.
