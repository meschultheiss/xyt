# xyt_gps

Package Python produit par Bureau Action Située pour transformer des exports GPS
en données de mobilité structurées, contrôlables et exploitables pour l'analyse.

Le développement de ce package a bénéficié du soutien financier de la Fondation
MODUS, du SNSF et de la faculté ENAC de l'EPFL.

## État du package

`xyt_gps` couvre aujourd'hui la chaîne principale de préparation :

1. préparer l'entrée : charger les tables GPS, contrôler le schéma, appliquer
   les mappings modes et motifs ;
2. transformer : parser les dates et géométries, produire `storyline`, `legs`,
   `staypoints`, `trips`, `journeys`, `user_stats`, `user_day_coverage` et
   tables de correspondance ;
3. contrôler et nettoyer : suivre la participation, la qualité GPS, les filtres
   utilisateurs et les enrichissements spatiaux optionnels ;
4. enrichir et produire les indicateurs : CO2, santé, motifs, indicateurs
   personne-jour, personne-phase et population ;
5. préparer les exports dashboard : Parquet, CSV, H3, DuckDB et cartes HTML.

Le cas par défaut ne suppose ni expérimentation ni phase. Les phases,
pondérations, questionnaires, excursions et découpages territoriaux sont des
couches de configuration à ajouter seulement lorsqu'elles existent dans le
projet.

Les référentiels qui influencent directement les résultats, notamment facteurs
CO2, taux d'occupation et METs, sont fournis comme CSV versionnés dans la
configuration du projet et passés explicitement au package.

## Installation

Version recommandée : Python 3.12. Le package cible Python `>=3.10,<3.13`, afin
de rester dans une zone stable pour les dépendances géospatiales.

Installation utilisateur, après publication :

```bash
python -m pip install xyt-gps
```

Avec les exports Parquet/Excel, les cartes et les agrégats spatiaux :

```bash
python -m pip install "xyt-gps[export,viz,analytics]"
```

Installation de développement local.

Depuis la racine du dépôt `xyt-gps` :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Dépendances optionnelles :

```bash
python -m pip install -r requirements-export.txt
python -m pip install -r requirements-viz.txt
python -m pip install -r requirements-analytics.txt
python -m pip install -r requirements-docs.txt
```

Voir aussi `docs/dependencies.md`.

## Documentation

Prévisualiser la documentation :

```bash
mkdocs serve
```

Construire la documentation :

```bash
python -m mkdocs build
```

Les pages à lire en priorité :

- `docs/getting-started/sample-data.md` : récupérer et placer le test set ;
- `examples/quickstart-analyse-gps.ipynb` : premier parcours
  analyste ;
- `examples/diagnostiquer-ses-donnees.ipynb` : vérifier les
  tables avant transformation ;
- `docs/reference/recommended-api.md` : API recommandée en 5 blocs ;
- `docs/examples_inventory.md` : rôle des notebooks de production et de démo ;
- `docs/project_workflow.md` : séquence de transformation du projet.

## Exemple minimal

```python
from pathlib import Path
import xyt_gps as xyt

config = xyt.ProjectConfig()

raw = xyt.RawGpsData(
    storyline=storyline,
    trips=trips,
    journeys=journeys,
    user_statistics=user_statistics,
)

dataset = xyt.prepare_mobility_dataset(raw, config)
quality = xyt.build_user_selection_table(dataset.user_stats)
indicators = xyt.compute_mobility_indicators(dataset)

xyt.export_clean_dataset(
    dataset,
    indicators,
    Path("demo-output"),
    selection_table=quality,
)
```

Le raccourci `run_mobility_pipeline()` reste disponible :

```python
raw, dataset, indicators = xyt.run_mobility_pipeline(config, raw=raw)
```

Pour explorer l’API par responsabilité :

```python
import xyt_gps.io as xyt_io
import xyt_gps.quality as xyt_quality
import xyt_gps.indicators as xyt_indicators
```

## Exemple détaillé

```python
xyt.check_raw_import_columns(
    raw.storyline,
    raw.user_statistics,
    trips=raw.trips,
    journeys=raw.journeys,
    raise_on_error=True,
)

dataset = xyt.prepare_mobility_dataset(
    raw,
    config,
    resample_missing_days=True,
    clean_leg_geometries=True,
    add_length_outlier_flags=True,
    add_signal_quality_flags=True,
)
```

Contrôler la qualité et sélectionner explicitement les utilisateurs :

```python
quality_report = xyt.build_tracking_quality_report(dataset.user_stats)

selection_table = xyt.build_user_selection_table(
    dataset.user_stats,
    require_tracking_quality=False,
    exclude_bad_signal_users=True,
    max_low_quality_legs_share=0.25,
)

analysis_users = xyt.select_analysis_users(selection_table)
analysis_dataset = xyt.filter_mobility_dataset_by_users(dataset, analysis_users)
```

Produire les indicateurs :

```python
indicators = xyt.compute_mobility_indicators(
    analysis_dataset,
    mode_col="mode_niv1",
    include_zero_days=True,
    include_excursions=True,
    use_weights=True,
    weight_col="weight",
)

xyt.write_indicator_result(indicators, "Data/Output/3-enriched-data/mode_niv1")
```

Préparer des exports spatiaux dashboard :

```python
spatial_tables = xyt.build_spatial_analytics_tables(
    analysis_dataset,
    h3_resolution=[8, 9],
    frequency_group_cols=["mode_niv1", "purpose_mrmt", "time_slice"],
    count_metrics=("point_count", "trip_count"),
)

xyt.write_spatial_analytics_tables(
    spatial_tables,
    "Data/Output/4-clean-data/spatial-analytics",
    formats=("parquet", "csv"),
)
```

## Notebooks

La chaîne projet prête à exécuter est rangée dans :

```text
Notebooks/
```

Séquence principale :

- `000_data_landing.ipynb` : préparer une base landed homogène ;
- `010_data_loading.ipynb` : transformer les tables GPS en tables de mobilité ;
- `011_quality_check.ipynb` : contrôler la qualité du suivi ;
- `020_spatial_cleaning.ipynb` : ajouter les enrichissements spatiaux optionnels ;
- `030_enrich_indicators.ipynb` : enrichir et calculer les indicateurs ;
- `040_export_cleaned_dataset.ipynb` : produire `Data/Output/4-clean-data`.

Les notebooks de démonstration autonomes restent dans `examples/`.
Le parcours d'apprentissage recommandé est :

- `quickstart-analyse-gps.ipynb` ;
- `diagnostiquer-ses-donnees.ipynb` ;
- `demo-fil-rouge-pipeline.ipynb` ;
- `demo-qualite-visualisations-test-set.ipynb`.

## Données de test

Des données de démonstration sont disponibles pour apprendre le package sans
accéder à un export GPS complet. Voir :

```text
docs/getting-started/sample-data.md
```

La génération synthétique reste un outil de maintenance et de tests, pas le
parcours recommandé pour apprendre le package.

## Licence et citation

Le package utilise une structure de licence en trois niveaux :

- code source : PolyForm Noncommercial License 1.0.0 ;
- documentation : Creative Commons Attribution-NonCommercial 4.0 International ;
- jeu de données sample : conditions spécifiques dans `LICENSE.sample-data.md`.

Les usages commerciaux du code, de la documentation ou du jeu de données sample
nécessitent un accord écrit préalable de Bureau Action Située.

Citation recommandée :

```text
Bureau Action Située. xyt_gps, package Python pour transformer des exports GPS
en données de mobilité structurées et analysables. Développement : Marc-Edouard
Schultheiss. Version 1.0.0.
```
