# Configuration projet

La configuration projet sert à rendre explicites les choix qui structurent la
transformation des données : expérience traitée, chemins d'entrée, phases,
contrat de colonnes, seuils qualité, mappings modes/motifs et ressources
géographiques éventuelles.

Dans les notebooks `Notebooks`, cette configuration est écrite en JSON. Le
package Python utilise ensuite un objet `ProjectConfig`, construit à partir de
ces valeurs. Les deux niveaux ont un rôle différent :

| Niveau | Fichier ou objet | Rôle |
|---|---|---|
| configuration notebook | `Notebooks/config/*.json` | décrire les sources, les phases, les chemins et le contrat de colonnes d'une expérience |
| configuration package | `xyt.ProjectConfig` | transmettre au package les paramètres nécessaires à la transformation et aux indicateurs |

Le cas par défaut du package reste volontairement générique : pas d'expérience
analytique et pas de découpage en phases.

```python
config = xyt.ProjectConfig()
```

Avec cette configuration minimale, `prepare_mobility_dataset()` produit les
tables de mobilité sans colonne `phase`. Les indicateurs regroupent alors les
résultats dans une période analytique nommée `All`.

## Organisation des fichiers JSON

La structure recommandée est la suivante :

```text
Notebooks/config/
├── shared.json
└── experiments/
    ├── experiment-a.json
    └── experiment-b.json
```

| Fichier | Rôle |
|---|---|
| `experiments/<experiment_name>.json` | configuration d'une expérience |
| `shared.json` | paramètres communs : table de correspondance utilisateurs, contrat de colonnes, ressources partagées |

Les chemins déclarés dans ces JSON sont résolus relativement au dossier
`Notebooks/config/`. Cela évite de figer des chemins absolus
propres à une machine.

Les expériences disponibles sont découvertes automatiquement à partir des
fichiers `experiments/*.json`. Ajouter une expérience revient donc à ajouter un
fichier JSON dans ce dossier.

## JSON minimal d'une expérience

Une expérience sans phases peut être décrite avec peu de champs :

```json
{
  "experiment_name": "mobility-study",
  "dump_date_range": "2026-04-01--2026-06-30",
  "paths": {
    "gps_dump_dir": "../../Data/raw/gps/mobility-study/",
    "storyline": "../../Data/raw/gps/mobility-study/storyline.csv",
    "trips": "../../Data/raw/gps/mobility-study/trips.csv",
    "journeys": "../../Data/raw/gps/mobility-study/journeys.csv",
    "user_statistics": "../../Data/raw/gps/mobility-study/user_statistics.csv",
    "public_transport_legs": null,
    "questionnaire": null,
    "excursion_area": null
  },
  "start_expe": "2026-04-01",
  "end_expe": "2026-06-30"
}
```

Ce cas produit des tables sans découpage analytique par phase. Les indicateurs
utilisent alors la période `All`.

## JSON avec phases et ressources spatiales

Lorsque le protocole comporte des périodes distinctes, les phases sont déclarées
dans le JSON d'expérience :

```json
{
  "experiment_name": "mobility-study-phases",
  "dump_date_range": "2026-04-01--2026-07-15",
  "paths": {
    "gps_dump_dir": "../../Data/raw/gps/mobility-study-phases/",
    "storyline": "../../Data/raw/gps/mobility-study-phases/StorylineExport.mobility-study-phases.2026-04-01--2026-07-15.csv",
    "trips": "../../Data/raw/gps/mobility-study-phases/Trips.2026-04-01--2026-07-15.csv",
    "journeys": "../../Data/raw/gps/mobility-study-phases/Journeys.2026-04-01--2026-07-15.csv",
    "user_statistics": "../../Data/raw/gps/mobility-study-phases/UserStatistics.2026-04-01--2026-07-15.csv",
    "public_transport_legs": "../../Data/raw/gps/mobility-study-phases/PublicTransportLegs.2026-04-01--2026-07-15.csv",
    "questionnaire": "../../Data/raw/questionnaires/mobility-study-phases.xlsx",
    "excursion_area": "../../Data/reference/zones/excursion_area.geojson"
  },
  "phase_1": ["2026-04-01", "2026-04-21"],
  "phase_2": ["2026-04-22", "2026-06-15"],
  "phase_3": ["2026-06-16", "2026-07-15"],
  "analysis_phase_1": ["2026-04-06", "2026-04-19"],
  "analysis_phase_2": ["2026-04-27", "2026-06-07"],
  "analysis_phase_3": ["2026-06-22", "2026-07-12"],
  "phase_week_sequences": {
    "phase_1": [1, 2, 3],
    "phase_2": [4, 5, 6, 7, 8, 9, 10, 11],
    "phase_3": [12, 13, 14, 15]
  },
  "start_expe": "2026-04-01",
  "end_expe": "2026-07-15"
}
```

Les phases sont inclusives : une ligne datée du premier ou du dernier jour d'une
phase appartient à cette phase.

## Champs d'une expérience

| Champ | Obligatoire | Rôle |
|---|---:|---|
| `experiment_name` | oui | nom analytique stable de l'expérience |
| `dump_date_range` | non | information de provenance du dump ; ne doit pas être interprétée comme période d'analyse |
| `paths.gps_dump_dir` | recommandé | dossier source du dump, pour documentation et diagnostic |
| `paths.storyline` | oui | fichier storyline exact à charger |
| `paths.trips` | oui | fichier trips exact à charger |
| `paths.journeys` | oui | fichier journeys exact à charger |
| `paths.user_statistics` | oui | fichier user statistics exact à charger |
| `paths.public_transport_legs` | non | fichier transports publics si disponible |
| `paths.questionnaire` | non | fichier questionnaire de l'expérience si disponible |
| `paths.excursion_area` | non | zone géographique utilisée pour flaguer des excursions |
| `phase_1`, `phase_2`, `phase_3`, ... | non | dates officielles de phase, sous forme `[date_debut, date_fin]` |
| `analysis_phase_1`, `analysis_phase_2`, `analysis_phase_3`, ... | non | fenêtres ajustées pour les agrégations, notamment après retrait de semaines de transition |
| `phase_week_sequences` | non | correspondance entre phases analytiques et semaines relatives `n_week_sequence`, utile aux analyses multi-vagues |
| `start_expe`, `end_expe` | recommandé | bornes du protocole ou de l'observation utile; elles peuvent encadrer des fenêtres analytiques plus larges que les dates officielles |

Le nombre de phases n'est pas imposé par le package. Les cas à trois phases sont
un usage possible, pas une contrainte générale.

`n_week_sequence` est calculé depuis `start_expe` et commence à 1. Cette
semaine relative permet de comparer des expériences lancées à des dates
différentes sur les mêmes semaines de protocole. `phase_week_sequences` ne
remplace pas les dates de phases pour le filtrage ; il documente la
correspondance attendue entre fenêtres analytiques et semaines relatives. Les
fenêtres `analysis_phase_*` peuvent viser des semaines complètes
lundi-dimanche, exclure les semaines de transition et, lorsque les données le
permettent, étendre les périodes avant/après au-delà du protocole officiel.
Elles sont relues après contrôle calendaire dans `000_data_landing.ipynb` et
contrôle de l'équilibre observé dans le notebook qualité. Dans
`011_quality_check.ipynb`, la cellule `analysis_phase_windows_to_copy` donne les
valeurs `analysis_phase_*` à recopier dans le JSON après arbitrage.

Les helpers de data landing ne devinent pas les noms de fichiers fournisseur.
Si un export est retéléchargé et que le nom du CSV change, modifier uniquement
la valeur correspondante dans `paths`.

## `shared.json`

`shared.json` contient les règles communes à plusieurs expériences. Deux blocs
sont particulièrement importants.

### Table de correspondance utilisateurs

```json
{
  "user-mapping-table": {
    "path": "../../../Data/raw/users.csv",
    "csv_sep": ",",
    "ignored_project_values": ["example-project"],
    "rename_columns": {
      "User": "id",
      "Project": "project_raw",
      "Invite code": "invite_code"
    },
    "project_value_mapping": {
      "Provider project A": "experiment-a",
      "Provider project B": "experiment-b"
    },
    "link_columns": [
      "id",
      "user_id",
      "project_raw",
      "experiment_name",
      "invite_code"
    ]
  }
}
```

Ce bloc est utile lorsqu'un fournisseur livre une table commune à plusieurs
projets et que les noms internes doivent être rattachés aux noms analytiques des
expériences. La table est utilisée pour construire `user_expe`, mais elle n'est
pas destinée à être exportée comme donnée d'analyse.

### Contrat de colonnes

Le contrat de colonnes décrit ce que le landing doit produire avant que le
package lise les données.

```json
{
  "landing-column-contract": {
    "manual_rename": {
      "storyline": {
        "IDNO": "user_id"
      },
      "trips": {},
      "journeys": {},
      "user_statistics": {},
      "public_transport_legs": {},
      "user_expe": {}
    },
    "id_columns": {
      "storyline": {
        "user_id": "user_id",
        "user_id_candidates": ["user_id", "userid", "idno", "user"],
        "entry_id": "storyline_id",
        "entry_id_candidates": ["storyline_id", "id"]
      }
    },
    "must_have": {
      "storyline": [
        "storyline_id",
        "user_id",
        "type",
        "started_at",
        "finished_at",
        "started_on",
        "geometry",
        "experiment_name"
      ]
    },
    "nice_to_have": {
      "storyline": [
        "trip_id",
        "phase",
        "phase_number",
        "phase_start",
        "phase_end",
        "analysis_phase",
        "analysis_phase_number",
        "analysis_phase_start",
        "analysis_phase_end",
        "n_week_sequence",
        "purpose",
        "mode",
        "detected_mode",
        "length"
      ]
    }
  }
}
```

Règles pratiques :

- `manual_rename` sert à déclarer explicitement les renommages nécessaires pour
  une source particulière ;
- `id_columns` évite de conserver des colonnes génériques comme `id` lorsque des
  identifiants explicites existent ;
- `must_have` contient les colonnes bloquantes ;
- `nice_to_have` contient les colonnes utiles mais non bloquantes ;
- les colonnes directes d'identification, par exemple `email`, doivent être
  retirées au landing avant les exports de travail.

## Profils de sortie du landing

Le profil de sortie n'est pas un champ du JSON d'expérience. Il se règle dans les
notebooks avec `LANDING_PROFILE`.

| Profil | Dossier | Usage |
|---|---|---|
| `complete` | `Data/Output/0-landed-data/<experiment_name>/complete/` | version complète locale, sans colonnes de contact direct |
| `anonymized_altered` | `Data/Output/0-landed-data/<experiment_name>/anonymized_altered/` | version pseudonymisée et spatialement altérée pour contrôles techniques |
| `anonymous_test_set` | `Data/Output/anonymous-test-set-gps/` | jeu anonymisé utilisé par les tutoriels package |

Les traces de `anonymized_altered` ne doivent pas être interprétées
scientifiquement : les identifiants sont pseudonymisés et les géométries ont été
modifiées pour protéger les origines et destinations.

## Construire `ProjectConfig` depuis le JSON

Le JSON sert à cadrer le projet. Au moment de transformer les tables, les
notebooks construisent un objet `ProjectConfig`.

```python
import xyt_gps as xyt

phases = []
for index in (1, 2, 3):
    value = experiment_config.get(f"phase_{index}")
    if value:
        start, end = value
        phases.append(xyt.Phase(f"Phase{index}", start, end))

config = xyt.ProjectConfig(
    experiment_name=experiment_config["experiment_name"],
    motiontag_project_name=experiment_config.get("motiontag_project_name"),
    raw_data_dir=landing_dir,
    export_dir=transformed_dir,
    phases=tuple(phases),
    start_expe=experiment_config.get("start_expe"),
    end_expe=experiment_config.get("end_expe"),
    excursion_area_path=experiment_config.get("paths", {}).get("excursion_area"),
)
```

`ProjectConfig` ne charge pas les données. Il fixe seulement les paramètres qui
seront utilisés par les fonctions du package.

Pour charger des fichiers structurés par noms inférés, il faut en revanche renseigner les champs utilisés dans les noms de fichiers :

```python
from pathlib import Path

project_root = Path("..").resolve()
raw_data_dir = project_root / "data" / "raw" / "gps"
transformed_dir = project_root / "data" / "outputs" / "2-transformed-data"

config = xyt.ProjectConfig(
    experiment_name="mobility-study",
    motiontag_project_name="gps-provider-project",
    period="2026-04-01--2026-06-30",
    raw_data_dir=raw_data_dir,
    export_dir=transformed_dir,
    target_crs="EPSG:4326",
    operations_crs="EPSG:2056",
)
```

Voir aussi [Structure de projet recommandée](../getting-started/project-structure.md)
pour organiser `project_root`, `data/`, `config/` et `notebooks/`.

## Paramètres principaux

| Paramètre | Rôle | Exemple |
|---|---|---|
| `experiment_name` | nom analytique optionnel du projet | `"mobility-study"` |
| `motiontag_project_name` | nom fournisseur utilisé dans les fichiers, requis pour `load_gps_export()` | `"gps-provider-project"` |
| `period` | période encodée dans les noms de fichiers, requise pour `load_gps_export()` | `"2026-04-01--2026-06-30"` |
| `raw_data_dir` | dossier des exports bruts | `project_root / "data" / "raw" / "gps"` |
| `export_dir` | dossier de sortie optionnel | `project_root / "data" / "outputs" / "2-transformed-data"` |
| `target_crs` | CRS des données exportées | `"EPSG:4326"` |
| `operations_crs` | CRS métrique pour les opérations spatiales | `"EPSG:2056"` |
| `phases` | périodes analytiques optionnelles | `Phase("Phase1", "2026-04-01", "2026-04-21")` |
| `tracking_thresholds` | seuils de suivi | `TrackingThresholds(min_total_tracked_days=7)` |
| `spatial_quality_thresholds` | seuils qualité spatiale | `SpatialQualityThresholds(outlier_quantiles_by_mode=(0.98, 0.99))` |
| `matching_thresholds` | seuils de matching | `MatchingThresholds(leg_trip_journey_tolerance="5s")` |
| `mappings` | modes et motifs | `mode_purpose_mapping()` ou mapping propre au projet |
| `time_slices` | tranches horaires réutilisables | `TimeSlice("HPM", "07:10", "09:00")` |

Par défaut, le package définit deux périodes de pointe et une période
résiduelle :

| Code | Intervalle | Rôle |
|---|---|---|
| `HPM` | 07:10-09:00 | heure de pointe du matin |
| `HPS` | 17:30-20:00 | heure de pointe du soir |
| `HC` | reste de la journée | heures creuses |

## Exemple complet avec qualité GPS

```python
from pathlib import Path
import xyt_gps as xyt

project_root = Path("..").resolve()

config = xyt.ProjectConfig(
    experiment_name="mobility-study",
    motiontag_project_name="gps-provider-project",
    period="2026-04-01--2026-06-30",
    raw_data_dir=project_root / "data" / "raw" / "gps",
    export_dir=project_root / "data" / "outputs" / "2-transformed-data",
    phases=(
        xyt.Phase("Phase1", "2026-04-01", "2026-04-21"),
        xyt.Phase("Phase2", "2026-04-22", "2026-05-31"),
        xyt.Phase("Phase3", "2026-06-01", "2026-06-30"),
    ),
    tracking_thresholds=xyt.TrackingThresholds(
        min_days_by_phase={"Phase1": 7, "Phase2": 21, "Phase3": 7},
        min_total_tracked_days=7,
    ),
    spatial_quality_thresholds=xyt.SpatialQualityThresholds(
        outlier_quantiles_by_mode=(0.98, 0.99),
        bad_signal_user_quantile=0.995,
        signal_loss_mode_column="mode",
    ),
)
```

## Découper ou non l'analyse par phase

Sans phase :

```python
config = xyt.ProjectConfig()
dataset = xyt.prepare_mobility_dataset(raw, config)
indicators = xyt.compute_mobility_indicators(dataset)
```

Avec deux ou trois phases :

```python
config = xyt.ProjectConfig(
    phases=(
        xyt.Phase("Phase1", "2026-04-01", "2026-04-21"),
        xyt.Phase("Phase2", "2026-04-22", "2026-05-31"),
    ),
)
```

Le nombre de phases n'est pas fixé par le package. Les cas Déclic à trois phases sont un usage particulier, pas une contrainte générale.

Dans le dictionnaire livré le 2025-08-15, les colonnes de sortie observées incluent par exemple `relative_signal_loss`, `low_quality_legs_1` et `bad_signal_user`. Elles correspondent aux fonctions de qualité GPS intégrées dans `xyt_gps.spatial`.

## Options de transformation

`ProjectConfig` décrit le projet : nom, phases, mappings, seuils et systèmes de coordonnées. Les options d'exécution sont passées directement à `prepare_mobility_dataset()` pour rester visibles au moment où la transformation est lancée.

Le comportement par défaut est le plus prudent : il applique le nettoyage géométrique léger, les flags de longueurs extrêmes et les flags de qualité GPS.

```python
dataset = xyt.prepare_mobility_dataset(
    raw,
    config,
    resample_missing_days=False,
    clean_leg_geometries=True,
    add_length_outlier_flags=True,
    add_signal_quality_flags=True,
)
```

Pour un export déjà nettoyé et documenté en amont :

```python
dataset = xyt.prepare_mobility_dataset(
    raw,
    config,
    add_length_outlier_flags=False,
    add_signal_quality_flags=False,
)
```

Il faut éviter de désactiver une étape uniquement parce qu’elle ralentit ou complique l’analyse. Une étape optionnelle peut être désactivée lorsque son équivalent a déjà été réalisé et documenté.

Quand `add_signal_quality_flags=False`, le package écrit `signal_quality_computed=False` dans `dataset.user_stats`. Les colonnes comme `bad_signal_user` ne doivent alors pas être interprétées comme un résultat de qualité GPS. Pour filtrer les utilisateurs dans ce cas, il faut appeler `build_user_selection_table(..., exclude_bad_signal_users=False)` et citer le contrôle amont utilisé.

## Règle pratique

Si un paramètre change l’interprétation des résultats, il doit être visible dans `ProjectConfig` ou documenté dans les pages d’hypothèses. Les phases de `ProjectConfig` sont aussi utilisées par `compute_mobility_indicators(..., include_zero_days=True)` pour construire le calendrier personne-jour, notamment les jours sans mouvement.
