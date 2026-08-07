# Exports spatiaux pour dashboard

Cette page décrit les formats utiles pour rendre les données GPS transformées
faciles à interroger, cartographier et intégrer dans une plateforme de
visualisation.

## Principe

Les tables de mobilité restent la référence scientifique du traitement :
`legs`, `staypoints`, `trips`, `journeys`, `user_stats` et indicateurs. Les
exports spatiaux ajoutent une couche d'usage pour les cartes et dashboards :

- GeoParquet pour conserver des tables géographiques propres et durables ;
- H3 pour agréger les traces dans une grille spatiale stable ;
- DuckDB pour requêter localement les tables avec SQL ;
- PostGIS si les données doivent être servies par une base multi-utilisateur ;
- PMTiles si une carte web doit consommer des tuiles vectorielles statiques.

Les exemples ci-dessous supposent que les dossiers du projet sont définis en
début de notebook :

```python
from pathlib import Path

project_root = Path("..").resolve()
output_root = project_root / "data" / "outputs"
experiment_name = "experiment-a"
transformed_dir = output_root / "2-transformed-data" / experiment_name
enriched_dir = output_root / "3-enriched-data" / experiment_name
clean_dir = output_root / "4-clean-data" / experiment_name
```

## GeoParquet

GeoParquet est un format Parquet avec métadonnées géographiques. La
spécification définit comment stocker les colonnes de géométrie dans Parquet
et où placer les métadonnées `geo`. Elle recommande aussi de déclarer le CRS ;
en l'absence de CRS explicite, le défaut attendu est `OGC:CRS84`, proche de
WGS84 pour les coordonnées longitude/latitude.

Dans `xyt_gps`, GeoParquet est le format de base recommandé pour les tables
spatiales transformées. Les exports peuvent aussi être écrits en CSV et en
pickle lorsque ces formats facilitent le contrôle ou l'échange :

```python
transformed_dir = output_root / "2-transformed-data" / experiment_name

xyt.write_mobility_dataset(
    dataset,
    transformed_dir,
    formats=("parquet", "csv"),
)
```

Pour beaucoup d'usages Python, QGIS, notebooks et échanges analytiques,
GeoParquet suffit. Il garde les géométries, les colonnes métier et les types
de données dans un format compact.

## H3

H3 est une grille hexagonale hiérarchique. Le package `h3-py` expose
`latlng_to_cell(lat, lng, res)`, qui associe une coordonnée à une cellule H3.
Plus la résolution est élevée, plus les cellules sont petites et plus la table
produite est volumineuse. Le package accepte une résolution unique ou une liste
de résolutions, par exemple `[8, 9]`.

Dans `xyt_gps`, la fonction `legs_to_h3_points()` transforme les géométries de
legs en points indexés H3 :

```python
leg_points_h3 = xyt.legs_to_h3_points(
    dataset.legs,
    h3_resolution=[8, 9],
)

h3_frequency = xyt.aggregate_h3_frequencies(
    leg_points_h3,
    group_cols=["mode_niv1", "purpose_mrmt", "time_slice"],
)
```

Par défaut, les points sont les sommets déjà présents dans la géométrie des
legs. Si les lignes sont peu denses, il est possible d'échantillonner la ligne
à intervalle régulier :

```python
leg_points_h3 = xyt.legs_to_h3_points(
    dataset.legs,
    h3_resolution=9,
    sample_distance_m=50,
)
```

Cet échantillonnage sert à produire une représentation cartographique de la
fréquentation. Il ne doit pas être présenté comme les points GPS bruts si la
source ne contient que des géométries de legs.

Les tranches horaires réutilisables sont définies dans la configuration projet :

```python
config = xyt.ProjectConfig(
    time_slices=(
        xyt.TimeSlice("HPM", "07:10", "09:00"),
        xyt.TimeSlice("HPS", "17:30", "20:00"),
    )
)
```

Les observations hors HPM/HPS sont étiquetées `HC`.

Construction et export recommandés :

```python
spatial_tables = xyt.build_spatial_analytics_tables(
    dataset,
    config=config,
    h3_resolution=[8, 9],
    frequency_group_cols=["mode_niv1", "purpose_mrmt", "time_slice", "phase"],
    count_dimension_sets=(
        ("mode_niv1",),
        ("purpose_mrmt",),
        ("time_slice",),
        ("mode_niv1", "time_slice"),
        ("purpose_mrmt", "time_slice"),
    ),
    count_metrics=("point_count", "trip_count"),
    parallel=True,
    max_workers=None,
    chunk_size=250,
)

manifest = xyt.write_spatial_analytics_tables(
    spatial_tables,
    transformed_dir / "spatial-analytics",
    formats=("parquet", "csv"),
)
```

Sur des exports longs, l'étape coûteuse est l'indexation H3 des points issus
des legs. `parallel=True` découpe cette étape en lots de legs et utilise un pool
de threads. `max_workers=None` laisse Python choisir le nombre de threads ;
utiliser un entier si la machine doit rester disponible pour d'autres tâches.
`chunk_size` contrôle le nombre de legs par lot : une valeur plus basse réduit
la taille de chaque tâche, une valeur plus haute limite l'overhead de
parallélisation.

Les fonctions `write_spatial_analytics_tables()` et
`write_spatial_analytics_exports()` écrivent par défaut les sorties en Parquet
et en CSV. Le CSV est pratique pour inspecter rapidement les tables H3 ou les
ouvrir dans un tableur ; le Parquet reste le format recommandé pour conserver
les types et limiter la taille des fichiers. Pour `leg_points_h3`, le Parquet
seul est recommandé dans les gros projets, car le CSV détaillé peut devenir très
volumineux. Pour ajouter un export pickle :

```python
manifest = xyt.write_spatial_analytics_tables(
    spatial_tables,
    output_root / "spatial-analytics" / experiment_name,
    formats=("parquet", "csv", "pkl"),
)
```

Tables produites :

| Table | Grain | Usage |
|---|---|---|
| `leg_points_h3` | point extrait ou échantillonné dans un leg et résolution H3 | contrôle, debug, agrégations fines |
| `h3_frequency` | cellule H3, résolution et dimensions demandées | table longue pour requêtes SQL |
| `h3_count_matrix` | cellule H3 et résolution | table large avec colonnes de counts par mode, motif et tranche horaire |

Exemples de colonnes dans `h3_count_matrix` :

```text
point_count__total
trip_count__total
point_count__mode_niv1__marche
trip_count__time_slice__hpm
point_count__mode_niv1__voiture__time_slice__hps
point_count__purpose_mrmt__travail
```

Carte de contrôle :

```python
h3_map = xyt.plot_h3_frequency_map(
    h3_count_matrix.loc[h3_count_matrix["h3_resolution"].eq(9)].rename(
        columns={"point_count__total": "point_count"}
    ),
    value_col="point_count",
    max_cells=2500,
    save_path=enriched_dir / "spatial-analytics" / "h3_frequency_map.html",
)
h3_map
```

La limite `max_cells` évite de rendre plusieurs dizaines de milliers de
polygones dans le navigateur. Pour une carte web de production, il faudra
plutôt passer par des tuiles vectorielles ou PMTiles.

## DuckDB spatial

DuckDB est une base analytique locale. Elle permet de requêter des Parquet ou
une base `.duckdb` avec SQL. L'extension officielle `spatial` ajoute des
fonctions géométriques, par exemple `ST_Point` ou `ST_MakePoint` pour créer des
points depuis des colonnes longitude/latitude.

Le package peut écrire une base DuckDB locale :

```python
manifest = xyt.write_duckdb_spatial_database(
    {
        "legs": dataset.legs,
        "leg_points_h3": leg_points_h3,
        "h3_frequency": h3_frequency,
    },
    enriched_dir / "mobility.duckdb",
)
```

Les géométries sont stockées en WKB pour rester interrogeables même si
l'extension `spatial` n'est pas disponible. Quand l'extension peut être chargée,
des vues `*_spatial` sont créées avec une colonne géométrique DuckDB.

DuckDB est adapté pour :

- explorer les données localement ;
- alimenter un notebook ou une application de dashboard mono-fichier ;
- requêter rapidement des tables Parquet ou GeoParquet.

Ce n'est pas le meilleur choix si plusieurs utilisateurs doivent écrire dans
la base ou si les données doivent être servies par une API durable.

## PostGIS

PostGIS est l'extension géographique de PostgreSQL. C'est une bonne option
lorsqu'il faut une base serveur, des index spatiaux, plusieurs utilisateurs,
des permissions, une API ou des requêtes géographiques en production.

Pour `xyt_gps`, PostGIS peut venir après le MVP :

1. produire les tables propres en GeoParquet ;
2. contrôler les clés et le CRS ;
3. charger les tables dans PostGIS ;
4. exposer les agrégations ou tuiles nécessaires au dashboard.

## PMTiles

PMTiles est un format de tuiles cartographiques dans un seul fichier. Il est
utile pour publier une carte web statique et rapide. Ce n'est pas une base de
données transactionnelle : les tuiles sont générées en aval à partir de tables
ou d'agrégats déjà prêts.

Flux recommandé :

```text
GeoParquet / DuckDB / PostGIS
        -> agrégation H3 ou tuiles vectorielles
        -> PMTiles
        -> carte web
```

## Où placer ces exports

| Niveau | Contenu conseillé |
|---|---|
| `transformed_dir / "spatial-analytics"` | `leg_points_h3`, `h3_frequency` calculés directement depuis les legs transformés |
| `enriched_dir / "spatial-analytics"` | agrégations H3 enrichies avec modes, CO2, santé, qualité ou variables de dashboard |
| `enriched_dir / "mobility.duckdb"` | base locale SQL pour prototype de dashboard |
| dossier dédié choisi par l'analyste | version consolidée pour un prototype cartographique ou dashboard |

Le niveau `2-transformed-data` suffit pour une carte de fréquentation simple.
Le niveau `3-enriched-data` est préférable si la carte doit afficher des
indicateurs, modes agrégés, filtres de qualité ou dimensions analytiques.
Le niveau `4-clean-data/<experiment_name>/cleaned-base` est le point d'entrée
recommandé lorsque les analyses thématiques doivent réutiliser les tables déjà
enrichies et contrôlées. Les agrégats H3, DuckDB et autres exports dashboard
sont optionnels et doivent être produits dans un dossier dédié lorsqu'une
analyse ou une application en a besoin.

## Zones OD Grand Genève

Pour construire des matrices origine-destination, le notebook
`020_spatial_cleaning.ipynb` peut utiliser un découpage territorial fourni par
le projet, par exemple des communes, secteurs statistiques ou zones de modèle :

```text
external_data_dir / "od-zones"
    zones.geojson
    centroids.geojson
```

La table des zones doit porter une clé stable, par exemple `ID_Zone`, utilisée
pour ajouter des
colonnes d'origine et de destination :

| Table | Colonnes OD ajoutées |
|---|---|
| `staypoints` | `activity_id_zone` |
| `legs` | `origin_id_zone`, `destination_id_zone` |
| `trips` | `trip_origin_id_zone`, `trip_destination_id_zone` |
| `journeys` | `journey_origin_id_zone`, `journey_destination_id_zone` |

La table des centroïdes contient les points représentatifs des zones. Elle peut être exportée
avec les données propres pour relier les matrices OD à des points de carte ou à
un graphe territorial.

## Dépendances

Les fonctions H3 et DuckDB sont optionnelles :

```bash
python -m pip install -r requirements-analytics.txt
python -m pip install -r requirements-export.txt
python -m pip install -r requirements-viz.txt
```

`requirements-export.txt` reste nécessaire pour écrire les fichiers Parquet.
`requirements-viz.txt` est nécessaire pour les cartes Folium.

## Références

- DuckDB, extension spatial : <https://duckdb.org/docs/current/core_extensions/spatial/overview>
- DuckDB, fonctions `ST_Point` et `ST_MakePoint` : <https://duckdb.org/docs/current/core_extensions/spatial/functions>
- h3-py, API `latlng_to_cell` : <https://uber.github.io/h3-py/api_quick.html>
- GeoParquet 1.1.0 : <https://geoparquet.org/releases/v1.1.0/>
- PMTiles : <https://docs.protomaps.com/pmtiles/>
