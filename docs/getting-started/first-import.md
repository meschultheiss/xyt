# Premier import

Cette page montre deux chemins courts : transformer des tables déjà chargées, ou charger des exports GPS structurés par noms de fichiers.

## Chemin recommandé

```python
import xyt_gps as xyt

config = xyt.ProjectConfig()

raw = xyt.RawGpsData(
    storyline=storyline,
    trips=trips,
    journeys=journeys,
    user_statistics=user_statistics,
)

result = xyt.run_mobility_pipeline(config, raw=raw)

raw = result.raw
dataset = result.dataset
indicators = result.indicators
```

La même fonction peut être déballée en trois objets :

```python
raw, dataset, indicators = xyt.run_mobility_pipeline(config, raw=raw)
```

Ce chemin est le plus simple pour découvrir le package. Les fonctions plus
fines restent disponibles lorsque l'on veut inspecter ou modifier une étape.

## Cas générique : données déjà chargées

Si les tables `storyline`, `trips` et `journeys` sont déjà lues avec `pandas`, aucune expérience ni phase n'est nécessaire.

```python
import xyt_gps as xyt

raw = xyt.RawGpsData(
    storyline=storyline,
    trips=trips,
    journeys=journeys,
    user_statistics=user_statistics,
)

config = xyt.ProjectConfig()

dataset = xyt.prepare_mobility_dataset(raw, config)
```

Dans ce cas, les tables de mobilité ne reçoivent pas de colonne `phase`. Les indicateurs utilisent automatiquement une période analytique `All`.

```python
indicators = xyt.compute_mobility_indicators(dataset)
```

## Charger des exports structurés par noms de fichiers

Pour utiliser `load_gps_export()`, il faut fournir les informations qui permettent d'inférer les noms des fichiers source. Le champ `motiontag_project_name` reste présent pour les exports issus de ce fournisseur.

```python
from pathlib import Path
import xyt_gps as xyt

project_root = Path("..").resolve()
raw_data_dir = project_root / "data" / "raw" / "gps"

config = xyt.ProjectConfig(
    experiment_name="mobility-study",
    motiontag_project_name="gps-provider-project",
    period="2026-04-01--2026-06-30",
    raw_data_dir=raw_data_dir,
    phases=(
        xyt.Phase("Phase1", "2026-04-01", "2026-04-21"),
        xyt.Phase("Phase2", "2026-04-22", "2026-05-31"),
        xyt.Phase("Phase3", "2026-06-01", "2026-06-30"),
    ),
)

raw = xyt.load_gps_export(config, validate=True)
```

Les phases sont optionnelles. Elles doivent être renseignées seulement si l'analyse doit comparer des périodes du protocole.

## Vérifier avant de transformer

```python
raw.validation["storyline"].to_frame()
```

Les erreurs bloquantes doivent être corrigées avant transformation. Les avertissements indiquent des colonnes recommandées manquantes ou des enrichissements qui pourraient être ignorés.

## Transformer

```python
dataset = xyt.prepare_mobility_dataset(
    raw,
    config,
    resample_missing_days=True,
)
```

Par défaut, `prepare_mobility_dataset()` exécute la préparation complète : parsing, mappings, séparation `staypoints` / `legs`, tables de correspondance, statistiques de suivi, flags de longueur et qualité GPS. Si une partie de ces contrôles a déjà été réalisée en amont, passer les options directement à `prepare_mobility_dataset()` et documenter les étapes désactivées.

Pour suivre les étapes sur un export volumineux, activer les logs Python avant
la transformation :

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
```

Les tables principales sont ensuite accessibles :

```python
dataset.legs
dataset.staypoints
dataset.trips
dataset.journeys
dataset.user_stats
```

## Export lourd

Pour travailler vite sur un export volumineux :

```python
raw = xyt.load_gps_export(
    config,
    sample=xyt.RawSampleConfig.by_users(5, random_state=42, chunksize=100_000),
)
```

Ce mode conserve toutes les lignes de quelques utilisateurs et reste le plus cohérent pour tester la chaîne complète. Il lit les CSV par chunks après avoir sélectionné les utilisateurs, ce qui évite de charger tout l’export pour un premier contrôle.

## Qualité du suivi

Après transformation, les filtres de suivi et de qualité GPS sont visibles dans `dataset.user_stats` :

```python
quality_report = xyt.build_tracking_quality_report(dataset.user_stats)
dataset.user_stats[["user_id", "tracking_quality_ok", "signal_quality_computed"]].head()
```

Si `signal_quality_computed` vaut `False`, le filtre `exclude_bad_signal_users=True`
ne doit pas être appliqué. Le package renvoie une erreur explicite pour éviter
de confondre une qualité GPS non calculée avec une bonne qualité GPS.

Pour préparer un sous-ensemble d'analyse, construire d'abord une table de sélection. Elle garde une ligne par utilisateur, les critères appliqués, le flag final et la raison d'exclusion.

```python
selection_table = xyt.build_user_selection_table(
    dataset.user_stats,
    require_tracking_quality=False,
    exclude_bad_signal_users=True,
    max_low_quality_legs_share=0.25,
)

selection_table[
    [
        "user_id",
        "tracking_filter_ok",
        "bad_signal_user_filter_ok",
        "low_quality_legs_share_filter_ok",
        "analysis_user_ok",
        "analysis_user_reason",
    ]
]
```

Dans ce réglage, les mauvais signaux GPS sont exclus, mais les utilisateurs
avec une assiduité de tracking insuffisante restent disponibles pour l'analyse.
Leur statut reste visible dans `tracking_quality_ok` et
`tracking_quality_reason`.

Le filtrage reste ensuite explicite et reproductible :

```python
analysis_users = xyt.select_analysis_users(selection_table)
analysis_dataset = xyt.filter_mobility_dataset_by_users(dataset, analysis_users)
```

`analysis_dataset` conserve les mêmes tables que `dataset`, mais limitées aux utilisateurs sélectionnés. Les tables de correspondance sont aussi filtrées pour ne conserver que les `leg_id`, `trip_id`, `journey_id` et `activity_id` encore présents.

## Calculer les premiers indicateurs

Les premiers indicateurs utilisent les tables de mobilité structurées. Ils produisent une table personne-jour, une table personne-période et une synthèse population par période analytique et mode. Si aucune phase n'est définie, la période vaut `All`.

```python
indicators = xyt.compute_mobility_indicators(
    analysis_dataset,
    mode_col="mode_niv1",
    include_zero_days=True,
    config=config,
)

indicators.person_day.head()
```

Pour lire directement la synthèse population :

```python
xyt.population_indicator_summary(indicators)
```

La version actuelle calcule :

```text
distance_km
travel_time_min
trip_count
```

Si les legs ont été enrichis avec `add_co2_occupancy_metrics()` et
`add_health_metrics()`, les indicateurs propagent aussi les métriques
disponibles, par exemple `co2_kg`, `calories_kcal` et `mets`.

## Exporter les états intermédiaires

Pour comparer avec les anciens notebooks, exporter les tables structurées et les rapports :

```python
manifest = xyt.write_mobility_dataset(
    dataset,
    config.export_path or "exports/loaded-data",
    formats=("csv", "geojson", "parquet"),
    selection_table=selection_table,
)

manifest
```

Le manifest liste les fichiers écrits : `storyline`, `legs`, `staypoints`, `trips`, `journeys`, `user_stats`, tables de correspondance, validation et rapports qualité.

## Visualiser les traces

La visualisation Folium est optionnelle :

```bash
python -m pip install -r requirements-viz.txt
```

Dans un notebook :

```python
m = xyt.plot_gps_traces(
    dataset,
    user_ids=analysis_users[:3],
    color_by="mode_niv1",
    sample_n=200,
    save_path="exports/loaded-data/gps_traces_preview.html",
)
m
```

Cette carte sert au contrôle visuel, pas à produire un indicateur.
