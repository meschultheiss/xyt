# Schéma brut d'import GPS

Cette page décrit le contrat minimal utilisé par `xyt_gps` pour charger des exports GPS déjà structurés en plusieurs tables.

Les données GPS réelles ne sont pas documentées ligne à ligne. La page décrit uniquement les tables et colonnes nécessaires au chargement.

## Fichiers attendus

Les chemins sont construits a partir de `ProjectConfig` :

```text
{raw_data_dir}/StorylineWithTripId.{motiontag_project_name}.{period}.csv
{raw_data_dir}/StorylineWithUserAnnotations.{motiontag_project_name}.{period}.csv
{raw_data_dir}/StorylineExport.{motiontag_project_name}.{period}.csv
{raw_data_dir}/Trips.{motiontag_project_name}.{period}.csv
{raw_data_dir}/Journeys.{motiontag_project_name}.{period}.csv
{raw_data_dir}/UserStatistics.{motiontag_project_name}.{period}.csv
```

Pour `storyline`, `ProjectConfig.storyline_prefix_candidates` définit les
préfixes acceptés. Par défaut, le package teste `StorylineWithTripId`,
`StorylineWithUserAnnotations`, puis `StorylineExport`. Dans les notebooks de
data landing du projet, les fichiers source sont plutôt déclarés explicitement
dans `Notebooks/config/experiments/<experiment_name>.json`, section `paths`.

## Import multi-projets

Le package permet de charger plusieurs expériences ou périodes en une seule passe, par exemple :

```python
raw = xyt.load_gps_sources([declic_config, declic_ziplo_config])
dataset = xyt.prepare_mobility_datasets([declic_config, declic_ziplo_config])
```

Chaque `ProjectConfig` porte ses propres valeurs :

- `experiment_name` ;
- `motiontag_project_name` ;
- `period` ;
- `raw_data_dir` ;
- `phases` et seuils de suivi ;
- `mappings` modes et motifs.

Cette structure évite d'appliquer par erreur les dates de phase d'une expérimentation à une autre. Elle évite aussi d'appliquer implicitement une nomenclature de modes ou de motifs qui ne correspondrait pas au projet.

Lors d'un import multi-source, les colonnes suivantes peuvent être ajoutées aux tables brutes :

```text
experiment_name
motiontag_project_name
period
```

`source_id` reste utilisé comme paramètre interne lorsque `namespace_ids=True`,
mais il n'est pas ajouté comme colonne de données. Par défaut, les identifiants
sont préfixés pour éviter les collisions entre projets :

```text
{experiment_name}__{period}::{identifiant_original}
```

Les valeurs originales sont conservees lorsque possible dans des colonnes
`raw_*` :

```text
raw_storyline_id
raw_user_id
raw_trip_id
raw_journey_id
```

## Sampling au chargement

Lorsque les exports bruts sont trop lourds, `load_gps_export()` et `load_gps_sources()` acceptent un parametre `sample`.

Deux modes sont prevus :

```python
# Garde toutes les lignes de 5 utilisateurs tires au hasard.
raw = xyt.load_gps_export(
    config,
    sample=xyt.RawSampleConfig.by_users(5, random_state=42, chunksize=100_000),
)

# Garde 1000 lignes aleatoires par table.
# Pour storyline, les lignes Track sont privilegiees afin d'inspecter des legs.
raw = xyt.load_gps_export(
    config,
    sample=xyt.RawSampleConfig.random_rows(1000, random_state=42),
)
```

Le mode `by_users` est le plus coherent pour tester la chaine de transformation, car il conserve toutes les traces des utilisateurs selectionnes dans `storyline`, `trips`, `journeys` et `user_statistics`.

Dans la version actuelle, `by_users` est aussi le mode adapté aux exports lourds : le package lit d'abord uniquement la colonne `user_id` de `storyline`, sélectionne les utilisateurs, puis filtre les tables CSV par chunks. Le paramètre `chunksize` contrôle la taille de lecture en mémoire.

Le mode `random_rows` sert surtout a inspecter rapidement le format et obtenir des legs aleatoires parmi l'ensemble des utilisateurs. Il ne garantit pas une coherence relationnelle complete entre tables.

## Mappings modes et motifs

Les mappings de modes et de motifs sont stockes dans `ProjectConfig.mappings`.

Ils couvrent actuellement :

```text
storyline_purpose_niv1
storyline_mode_niv1
storyline_mode_niv2
storyline_mode_mrmt
trip_purpose_mrmt
trip_mode_niv1
trip_mode_niv2
trip_mode_mrmt
default_mode
default_purpose
```

Ces mappings sont structurants : ils definissent les categories produites par les colonnes `mode_niv1`, `mode_niv2`, `mode_mrmt`, `purpose_niv1` et `purpose_mrmt`.

Leur modification doit donc etre traçable au niveau de la configuration projet, et non cachee dans une fonction de transformation.

## Données de démonstration

Le jeu `anonymous-test-set-gps` suit ce contrat de tables, avec des labels
d'expérience génériques et une table `sociodemographics` synthétique. Voir
[Données de démonstration](getting-started/sample-data.md) pour le placer dans
un projet local.

## Tables et colonnes minimales

### storyline

Colonnes obligatoires :

```text
storyline_id
user_id
type
started_at
finished_at
started_on
geometry
```

Colonnes recommandees :

```text
created_at
updated_at
started_at_in_timezone
finished_at_in_timezone
confirmed_at_in_timezone
created_at_in_timezone
updated_at_in_timezone
purpose
mode
detected_mode
length
trip_id
```

La colonne `geometry` peut contenir des géométries WKB/EWKB hexadécimales, des
bytes WKB ou des chaînes WKT. Les exports CSV du test set utilisent WKT ; les
exports Parquet peuvent conserver des géométries WKB.

### trips

Colonnes obligatoires :

```text
trip_id
user_id
started_at
finished_at
```

Colonnes recommandees :

```text
duration
distance
tracks_count
created_at
purpose
main_mode
modes
```

### journeys

Colonnes obligatoires :

```text
journey_id
user_id
started_at
finished_at
```

Colonnes recommandees :

```text
duration
distance
tracks_count_without_walks
trips_count
tracks_count
created_at
purpose
main_mode
modes
```

### user_statistics

Colonnes obligatoires :

```text
user_id
```

Colonnes recommandees :

```text
first_tracked_at
latest_tracked_at
days_in_range_count
tracked_days_count
active_days_count
zero_movement_tracked_days_count
untracked_days_count
inactive_days_count
```

Ces compteurs issus de `user_statistics` sont informatifs à l'import. Le
pipeline les recalcule depuis `storyline` et les remplace dans `user_stats` afin
de garantir des définitions identiques entre fournisseurs et expériences.

## Validation implementee

Pour un notebook de chargement, commencer par `check_raw_import_columns()`. Cette fonction vérifie la structure des colonnes de `storyline`, `trips`, `journeys` et `user_statistics`, sans parser les dates ou les géométries :

```python
structure_report = xyt.check_raw_import_columns(
    storyline_raw,
    user_statistics_raw,
    trips=trips_raw,
    journeys=journeys_raw,
    raise_on_error=True,
)
```

La fonction plus complète `validate_gps_raw()` est utilisée en interne par la transformation. Elle vérifie :

- les colonnes obligatoires manquantes ;
- les colonnes recommandees manquantes ;
- les taux de valeurs manquantes sur `geometry`, `started_at` et `finished_at` ;
- un seuil de 1 % de valeurs manquantes sur les colonnes critiques, avant parsing.

La fonction `expected_gps_schema()` retourne le schéma attendu sous forme de `DataFrame`, pour inspection dans un notebook.

## Limites actuelles

Le schema reste un premier contrat. Il doit être confirmé sur chaque nouveau format fournisseur avec :

- les types exacts des colonnes ;
- les variantes de noms selon les exports ;
- les colonnes liees aux annotations utilisateur ;
- les cas ou `trip_id` est absent de la storyline.
