# Notebooks de production et de démonstration

Cette page résume le rôle des notebooks prêts à l'emploi. Elle sert à choisir
le bon point d'entrée sans devoir ouvrir tous les fichiers.

## Notebooks de production

| Notebook | Entrée | Sortie | Rôle |
|---|---|---|---|
| `Notebooks/000_data_landing.ipynb` | sources brutes d'une expérience | `Data/Output/0-landed-data/<EXPERIMENT_NAME>/complete` | prépare une base homogène prête à être ingérée |
| `Notebooks/001_data_anonymisation.ipynb` | `0-landed-data/<EXPERIMENT_NAME>/complete` | `0-landed-data/<EXPERIMENT_NAME>/anonymized_altered` | produit une version pseudonymisée et spatialement altérée |
| `Notebooks/002_build_anonymous_test_set.ipynb` | `0-landed-data/<EXPERIMENT_NAME>/anonymized_altered` | `Data/Output/anonymous-test-set-gps` | construit le test set anonymisé utilisé par les tutoriels package |
| `Notebooks/010_data_loading.ipynb` | base landed d'une expérience et d'un profil | `Data/Output/2-transformed-data/<EXPERIMENT_NAME>` | transforme les tables GPS en tables de mobilité pour une expérience |
| `Notebooks/011_quality_check.ipynb` | `Data/Output/2-transformed-data/<EXPERIMENT_NAME>` | `quality-reports/` | contrôle le suivi, les confirmations et les filtres utilisateurs |
| `Notebooks/020_spatial_cleaning.ipynb` | `Data/Output/2-transformed-data/<EXPERIMENT_NAME>` | `spatial-cleaned/` | applique les contrôles et enrichissements spatiaux requis par `030` et `040` |
| `Notebooks/030_enrich_indicators.ipynb` | `Data/Output/2-transformed-data/<EXPERIMENT_NAME>/spatial-cleaned` | `Data/Output/3-enriched-data/<EXPERIMENT_NAME>` | ajoute CO2/santé et calcule les indicateurs agrégés et par expérimentation |
| `Notebooks/040_export_cleaned_dataset.ipynb` | tables `spatial-cleaned` + enrichissements `3` de la même expérience | `Data/Output/4-clean-data/<EXPERIMENT_NAME>/cleaned-base` | consolide uniquement les tables propres utiles aux analyses thématiques |
| `Notebooks/100_concat_experiments.ipynb` | un ou plusieurs dossiers `4-clean-data/<EXPERIMENT_NAME>` | `5-clean-data-concat/<CONCAT_NAME>/cleaned-base/*.parquet`, questionnaires CSV par expérience | prépare les analyses multi-expériences |

Le dossier unique `Notebooks/` évite la confusion entre les
anciens lots de travail et la chaîne technique réelle.

## Démonstrations package

Les notebooks de démonstration sont dans `examples/`.
Ils utilisent le jeu `anonymous-test-set-gps`, disponible sur demande à
`contact@situee.ch`.

| Notebook | Entrée | Sortie | Rôle |
|---|---|---|---|
| `quickstart-analyse-gps.ipynb` | `Data/Output/anonymous-test-set-gps` | figure et tables affichées dans le notebook | premier parcours court pour analyste |
| `diagnostiquer-ses-donnees.ipynb` | `Data/Output/anonymous-test-set-gps` | rapports affichés dans le notebook | diagnostic des colonnes, utilisateurs et transformation minimale |
| `demo-fil-rouge-pipeline.ipynb` | `Data/Output/anonymous-test-set-gps` | export propre local dans `examples/_outputs/fil-rouge-clean-export` | parcours complet de l'API recommandée |
| `demo-qualite-visualisations-test-set.ipynb` | `Data/Output/anonymous-test-set-gps` | figures HTML locales dans `examples/_outputs/quality-visualisations` | contrôles qualité, carte GPS, indicateurs et motifs |

Le test set contient 12 utilisateurs anonymisés. Les labels d'expérience sont
génériques (`test-expe-1`, `test-expe-2`, `test-expe-3`) et la table
`sociodemographics` est synthétique. Les questionnaires réels ne sont pas
exportés.

## Détail par notebook

### `Notebooks/000_data_landing.ipynb`

Prépare les données avant usage du package. Le notebook ouvre les sources
déclarées dans la section `paths` de la configuration d'une expérience,
homogénéise les noms de colonnes, concatène les tables GPS de cette expérience,
construit `user_expe`, contrôle le contrat de colonnes et exporte :

- `Data/Output/0-landed-data/<EXPERIMENT_NAME>/complete` : version complète
  sans colonnes sensibles directes comme `email`.

La version `anonymized_altered` est produite ensuite par
`001_data_anonymisation.ipynb`. Ces traces ne doivent pas être interprétées
scientifiquement comme trajectoires réelles.

Le notebook `002_build_anonymous_test_set.ipynb` peut ensuite construire un
petit test set anonymisé pour les tutoriels package. Il ne fait pas partie du
traitement analytique d'une expérience réelle.

Le notebook exporte aussi un `README.md`, un `landing_manifest.json` et
`column_alias_report.csv`, qui signale les colonnes équivalentes ou divergentes.

La configuration est séparée entre `Notebooks/config/shared.json`
et `Notebooks/config/experiments/*.json`. Les fonctions propres
au landing sont dans `Notebooks/utils/data_landing.py`, hors
package Python.

### `Notebooks/010_data_loading.ipynb`

Il charge les tables landed, vérifie la structure, parse les dates et
géométries, applique les mappings modes/motifs, gère les phases si elles
existent, resample les jours manquants, construit `legs`, `staypoints`,
`trips`, `journeys`, `user_stats` et les tables de correspondance.

Les sorties principales sont exportées dans
`Data/Output/2-transformed-data/<EXPERIMENT_NAME>`, avec `user_expe`,
`participation_grid`, `participation_summary`, les rapports de validation et
les rapports qualité.

### `Notebooks/011_quality_check.ipynb`

Isole les contrôles qualité pour ne pas alourdir le notebook principal. Il
vérifie la couverture temporelle du suivi, les jours manquants, les
confirmations, les longueurs extrêmes, la disponibilité du taux d'occupation
déclaré pour les legs voiture et produit une table de sélection utilisateur.

### `Notebooks/020_spatial_cleaning.ipynb`

Regroupe les traitements spatiaux de préparation : nettoyage des géométries,
qualité GPS, zones territoriales, origine-destination et classification par
rapport à une zone de référence. Dans la configuration Déclic, il utilise les
zones communales GIREC du Grand Genève et ajoute `origin_id_zone` /
`destination_id_zone` aux legs, puis les propage aux trips et journeys.

### `Notebooks/030_enrich_indicators.ipynb`

Ajoute les enrichissements transversaux nécessaires avant analyse : occupation,
émissions CO2, métriques santé simples, indicateurs personne-jour,
personne-phase et population. Pour une source consolidée, il répète le calcul
par expérimentation, regroupe les tables dans `indicators/` et produit une fiche
HTML avec sélecteur d'expérimentation.

### `Notebooks/040_export_cleaned_dataset.ipynb`

Assemble la base propre finale pour analyses thématiques. Il prend les tables
`spatial-cleaned` comme base, exige les legs enrichis correspondants, puis ajoute
les tables latérales CO2/santé, les questionnaires liés à l'expérience et, si
demandé, la table `relative_time_index`.

La sortie principale est
`Data/Output/4-clean-data/<EXPERIMENT_NAME>/cleaned-base`.

Les formats multiples sont rangés dans des sous-dossiers `parquet/`, `csv/` et
`pkl/` pour éviter de mélanger plusieurs versions d'une même table. Les
notebooks thématiques doivent commencer par `cleaned-base/`, ou par
`100_concat_experiments.ipynb` lorsqu'ils couvrent plusieurs expériences.
`relative_time_index` est concaténé comme les autres tables de `cleaned-base` :
une ligne par événement de `storyline`, dates absolues conservées, semaine de
séquence et identifiants de liaison.
Le concat multi-expériences reste minimal : il n'exporte pas de copie
concaténée des dossiers `indicators/` et `spatial-analytics/`. Dans le dossier
concaténé, les fichiers sont écrits directement sous `cleaned-base/` et
`cleaned-base/questionnaires/`, sans sous-dossier `parquet/` ou `csv/`.

## Propositions d'amélioration

| Sujet | Proposition |
|---|---|
| noms des notebooks | conserver la séquence `000_data_landing`, `010_data_loading`, `011_quality_check`, `020_spatial_cleaning`, `030_enrich_indicators`, `040_export_cleaned_dataset` |
| landing | garder les helpers dans `Notebooks/utils/`, hors package, car ils sont propres aux fournisseurs et aux exports projet |
| anonymisation | extraire les helpers du notebook dans `Notebooks/utils/data_landing_privacy.py` si le bloc grossit |
| configuration | ajouter un exemple YAML ou JSON minimal pour projets sans phase et projets avec phases |
| validation | ajouter une fonction publique `validate_mobility_dataset_integrity()` pour contrôler clés, liens, CRS et volumes |
| exports | conserver `write_mobility_dataset(extra_tables=...)` pour ajouter des tables de construction sans modifier `MobilityDataset` |
| données de démonstration | créer un jeu stable, zippé, hébergé et documenté avant de l'exposer dans la doc |
| documentation | conserver peu de pages, mais avec des tableaux d'entrées/sorties et une séquence claire |
