# Structure de projet recommandée

`xyt_gps` n'impose pas une arborescence de projet. Les fonctions acceptent des
`Path` Python ou des chemins texte. Pour garder les notebooks lisibles et
réutilisables, il est recommandé de définir les chemins une seule fois au début
du projet.

## Principe

Un projet d'analyse GPS peut être organisé ainsi :

```text
project_root/
├── config/
│   ├── project_config.json
│   └── mappings.yml
├── data/
│   ├── raw/                     # exports GPS originaux, non versionnés
│   ├── external/                # zones, questionnaires, pondérations
│   └── outputs/
│       ├── 0-landed-data/       # données homogénéisées avant package
│       │   └── experiment-a/
│       │       ├── complete/
│       │       └── anonymized_altered/
│       ├── 2-transformed-data/
│       │   └── experiment-a/    # tables de mobilité structurées
│       ├── 3-enriched-data/
│       │   └── experiment-a/    # enrichissements et indicateurs
│       └── 4-clean-data/
│           └── experiment-a/    # base propre pour analyses thématiques
│               └── cleaned-base/
├── notebooks/
│   ├── 000_data_landing.ipynb
│   ├── 010_data_loading.ipynb
│   ├── 011_quality_check.ipynb
│   ├── 020_spatial_cleaning.ipynb
│   ├── 030_enrich_indicators.ipynb
│   └── 040_export_cleaned_dataset.ipynb
└── reports/
    └── figures/
```

Les dossiers `data/raw/` et `data/outputs/` doivent rester ignorés par git,
surtout lorsque les données contiennent des traces GPS ou des informations
personnelles.

## Variables de chemins

Dans un notebook, définir explicitement :

```python
from pathlib import Path

project_root = Path("..").resolve()
data_root = project_root / "data"
output_root = data_root / "outputs"
experiment_name = "experiment-a"

raw_data_dir = data_root / "raw" / "gps"
external_data_dir = data_root / "external"
landing_dir = output_root / "0-landed-data" / experiment_name / "anonymized_altered"
transformed_dir = output_root / "2-transformed-data" / experiment_name
enriched_dir = output_root / "3-enriched-data" / experiment_name
clean_dir = output_root / "4-clean-data" / experiment_name
```

Ces noms sont des conventions de travail, pas des paramètres imposés par le
package. Ils permettent de lire la chaîne de traitement sans chemins absolus et
sans hypothèse propre à un ordinateur.

Dans `4-clean-data/<experiment_name>`, le sous-dossier à utiliser en priorité
pour les analyses thématiques est `cleaned-base/`. Il contient les tables de
mobilité stabilisées : `storyline`, `legs`, `staypoints`, `trips`, `journeys`,
`user_stats`, `map_track_trip_journey`, `map_legs_staypoints`, et si
disponibles `public_transport_legs`, `user_expe`, `occupancy_co2` et
`health`.

Lorsqu'une analyse doit couvrir plusieurs expériences, il est préférable de les
préparer séparément jusqu'à `4-clean-data/<experiment_name>`, puis de les
concaténer au moment de l'analyse. Cette convention évite de mélanger des
calendriers de phases ou des règles de projet différentes pendant la
transformation.

## Lien avec `ProjectConfig`

`ProjectConfig.raw_data_dir` décrit le dossier d'entrée utilisé par
`load_gps_export()`. `ProjectConfig.export_dir` peut servir de dossier de sortie
par défaut, mais les fonctions d'export acceptent aussi un chemin explicite.

```python
import xyt_gps as xyt

config = xyt.ProjectConfig(
    experiment_name="mobility-study",
    raw_data_dir=raw_data_dir,
    export_dir=transformed_dir,
)
```

Si les tables sont déjà chargées avec `pandas`, `raw_data_dir` n'est pas
nécessaire :

```python
config = xyt.ProjectConfig()
dataset = xyt.prepare_mobility_dataset(raw, config)
```

## Convention utilisée dans ce dépôt

Les notebooks historiques de ce dépôt utilisent encore des dossiers nommés
`Notebooks/` et `Data/Output/`. Ce sont des conventions du projet Déclic, pas
des contraintes du package. Pour un nouveau projet destiné à être partagé, une
arborescence en minuscules comme `notebooks/`, `data/` et `config/` est plus
standard.
