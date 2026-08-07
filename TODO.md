# TODO — xyt_gps : indépendance du package avant PyPI

Date : 2026-08-07
Statut : plan de travail actif

Ce plan stabilise le contrat public du package avant publication. Il remplace
la logique de refactorisation interne immédiate par une séquence plus sûre :
définir le contrat, le tester, le documenter, puis seulement simplifier les
modules internes.

## Décisions de cadrage

### Deux chemins d'usage à conserver

Le package doit assumer deux usages publics distincts.

Chemin court, pour onboarding, démonstration et scripts simples :

```text
load_gps_export
→ prepare_mobility_dataset
→ build_user_selection_table
→ compute_mobility_indicators
→ export_clean_dataset
```

Chemin explicite, pour notebooks de production :

```text
lecture manuelle
→ prepare_storyline / prepare_trips / prepare_journeys
→ split_storyline
→ enrichissements et contrôles étape par étape
→ MobilityDataset(...)
→ write_mobility_dataset / write_indicator_result / build_relative_time_index
```

Les deux chemins sont légitimes. Le chemin court ne doit pas remplacer les
notebooks de production lorsque ceux-ci doivent rendre visibles les choix
méthodologiques.

### Nom d'installation

La documentation publique doit recommander :

```bash
pip install xyt-gps
```

Le nom du module Python reste :

```python
import xyt_gps as xyt
```

Le nom de distribution préparé est `xyt-gps`, cohérent avec le futur dépôt
`action-situee/xyt-gps`. Les outils Python normalisent tirets et underscores,
mais la documentation utilisateur doit rester sur `pip install xyt-gps`.

Le projet PyPI historique `xyt` existe déjà. Deux stratégies restent possibles :

- publier la nouvelle génération comme `xyt-gps`, recommandé pour éviter la
  confusion entre deux contrats de package ;
- publier une version de compatibilité sous `xyt`, seulement si les accès PyPI
  du projet historique sont disponibles et si la rupture de contrat est assumée.

### Intervention prioritaire

Priorité au package `xyt-gps`. Les notebooks viennent ensuite comme
validation ergonomique et documentation vivante. Ne pas convertir les notebooks
de production vers les façades tant que le contrat package n'est pas stabilisé.

### Pas de fallback implicite

Les notebooks et les façades publiques ne doivent pas masquer une incohérence de
configuration par une sélection alternative silencieuse. Si un filtre strict
vide un périmètre d'analyse, la correction doit rester explicite : configuration,
paramètre visible ou erreur contrôlée. Cette règle vaut en particulier pour les
phases d'expérience et les filtres de qualité utilisateur.

## P0 — Sécuriser l'état de travail

- Garder hors périmètre les modifications existantes dans
  `Notebooks/002_build_anonymous_test_set.ipynb`, sauf demande explicite.
- Versionner le nouveau test `tests/test_public_api_ergonomics.py` avec les
  autres changements du package.
- Lancer après chaque étape de code :

```bash
python -m pytest tests -q
```

## P1 — Stabiliser les tests du contrat public

Compléter `tests/test_public_api_ergonomics.py` avec les cas suivants :

- [x] les symboles recommandés sont importables depuis `import xyt_gps as xyt` ;
- [x] le chemin explicite des notebooks de production reste supporté :
  `prepare_storyline`, `prepare_trips`, `prepare_journeys`,
  `apply_trip_journey_mappings`, `split_storyline`, construction manuelle de
  `MobilityDataset`, puis `write_mobility_dataset` ;
- [x] `export_clean_dataset()` écrit un manifest unifié avec indicateurs ;
- [x] `export_clean_dataset(..., indicators=None)` fonctionne ;
- [x] collision explicite avec `extra_tables={"relative_time_index": ...}` lève
  `ValueError` ;
- [x] format non supporté lève `ValueError` ;
- [x] `separate_format_dirs=True` écrit bien dans les sous-dossiers de format ;
- [x] `include_spatial_analytics=True` fonctionne sur un petit jeu synthétique ou
  fixture.

Ne pas encore tester que des modules internes ne sont plus importables : cela
relève d'une refactorisation ultérieure.

## P2 — Clarifier le contrat de `MobilityDataset`

Décision proposée : `MobilityDataset` reste mutable.

Justification : les notebooks de production enrichissent parfois une table
après construction, notamment `legs`, puis reconstruisent ou réutilisent un
`MobilityDataset`. Cette mutabilité est acceptable si elle est documentée comme
un état de travail inspectable, et non comme un objet immuable.

Actions :

- documenter cette politique dans `recommended-api.md` ;
- éviter pour l'instant un helper `replace_legs()` tant qu'un cas de bug concret
  ne le justifie pas ;
- conserver les dataclasses existantes sans `frozen=True`.

## P3 — Documenter les deux niveaux d'API

Dans `docs/reference/recommended-api.md` :

- mettre en première section le chemin court ;
- ajouter une section "Chemin explicite pour notebooks de production" ;
- documenter les paramètres utilisés en production mais peu visibles :
  - `plot_participation_heatmap(summary=, notes=, phase_col=)` ;
  - `plot_gps_traces(use_antpath=, show_staypoints=)` ;
  - `add_spatial_zone_labels(fill_value=, predicate=)` ;
  - `add_leg_origin_destination_zones(origin_col=, destination_col=,
    fill_value=)` ;
  - `classify_leg_relation_to_area(relation_col=, code_col=,
    operations_crs=)` ;
  - `build_tracking_gap_report()` ;
  - `build_user_confirmation_rates()` ;
  - `build_mode_detection_precision()`.

Dans `src/xyt_gps/__init__.py` :

- conserver les exports nécessaires aux notebooks ;
- organiser les commentaires en distinguant chemin court et construction
  explicite ;
- ne retirer de `__all__` que les symboles confirmés inutiles et non nécessaires
  aux notebooks.

## P4 — Réduire le bruit de `__all__` sans casser les notebooks

Retraits effectués de `__all__` après vérification par `rg` :

- [x] `export_mobility_tables` : alias de `write_mobility_dataset` ;
- [x] `parse_ewkb` ;
- [x] `parse_date_columns` ;
- [x] `assign_phase` ;
- [x] `drop_nans_if_low_rate` ;
- [x] `validate_schema`.

Ces fonctions restent accessibles via `import xyt_gps as xyt` pour compatibilité,
mais ne sont plus exposées par `from xyt_gps import *`.

Décision avant publication initiale : ne pas réduire davantage `__all__` sans
usage problématique identifié. Le chemin explicite des notebooks de production
reste un contrat public assumé.

À conserver tant que les notebooks de production les utilisent :

- `prepare_public_export_table` ;
- `split_storyline` ;
- `add_user_id_day` ;
- `add_length_quantile_flags` ;
- `apply_trip_journey_mappings` ;
- `build_track_trip_journey_map` ;
- `build_legs_staypoints_map` ;
- `add_journey_to_trips` ;
- `add_trip_destination_activity` ;
- `add_excursion_flags` ;
- `add_excursion_flags_to_trips_journeys` ;
- `add_excursion_stats_to_user_stats` ;
- `add_signal_quality_to_user_stats` ;
- `build_tracking_gap_report` ;
- `build_user_confirmation_rates` ;
- `build_mode_detection_precision` ;
- `summarize_leg_lengths_by_mode` ;
- `get_extreme_legs_by_mode` ;
- `add_spatial_zone_labels` ;
- `add_leg_origin_destination_zones` ;
- `classify_leg_relation_to_area` ;
- `add_trip_origin_destination_from_legs` ;
- `add_journey_origin_destination_from_trips` ;
- `build_leg_enrichment_tables`.

## P5 — Nettoyer la documentation obsolète

- [x] Remplacer `docs/function_inventory.md` par un inventaire actuel :
  API recommandée, chemin explicite notebook, `__all__`, helpers compatibles
  hors `__all__` et sous-modules existants.
- [x] Retirer de l'API actuelle les éléments non implémentés ou anciens :
  `filter_users_by_tracking`, `filter_legs`, `build_filter_report`,
  `FilterConfig`, `map_match_osrm`, `map_match_google`, `OSRMConfig`,
  `GoogleDirectionsConfig`, `load_external_geodata`,
  `ExternalGeodataConfig`.

## P6 — Améliorer `run_mobility_pipeline`

La fonction a beaucoup de paramètres. Évaluer une dataclass `PipelineOptions`
seulement après stabilisation des tests et de la documentation du contrat.

Contrainte : ne pas casser les appels existants. Ajouter `options=` comme
nouvelle voie lisible, tout en gardant les paramètres actuels pendant la phase
pré-PyPI.

## P7 — Renommer les modules internes, plus tard

Le renommage `quality_tracking.py` → `_quality_tracking.py` et équivalents est
possible, mais non prioritaire. Il doit venir après :

- tests du chemin court ;
- test de régression du chemin explicite ;
- documentation claire des façades publiques `quality.py`, `spatial.py`,
  `spatial_analytics.py`.

Ne pas faire ce renommage dans la même étape que la réduction de `__all__`.

## P8 — Notebook fil rouge

Créer ou mettre à jour `examples/demo-fil-rouge-pipeline.ipynb` après
stabilisation du package.

Objectif : démontrer le chemin court sur données sample ou synthétiques, sans
accès aux données réelles.

- [x] `examples/demo-fil-rouge-pipeline.ipynb` utilise le test set anonymisé,
  prépare un `MobilityDataset`, calcule les indicateurs et écrit un export
  propre.
- [x] `examples/demo-qualite-visualisations-test-set.ipynb` illustre les
  contrôles qualité, la carte GPS, les indicateurs HTML et les motifs
  journaliers.

## P9 — Préparation PyPI

À faire uniquement après P1 à P5.

Actions :

- [x] décider le nom de distribution préparé : `xyt-gps`, import
  `xyt_gps` ;
- [x] mettre à jour les métadonnées licence `pyproject.toml` au format moderne :
  `license = "PolyForm-Noncommercial-1.0.0"` et `project.license-files` ;
- [x] compléter `pyproject.toml` avec les URLs confirmées :
  documentation `https://xyt.situee.ch` et dépôt cible
  `https://github.com/action-situee/xyt-gps` ;
- ne pas inventer d'email auteur : laisser `[À COMPLÉTER]` ou omettre ;
- [x] mettre à jour le README avec `pip install xyt-gps` ;
- [x] reconstruire wheel et sdist depuis le dépôt indépendant ;
- [x] vérifier les artefacts avec `twine check` ;
- [x] installer la wheel construite dans un environnement vierge Python 3.12 ;
- [ ] publier sur TestPyPI, si un token est disponible ;
- [ ] publier sur PyPI, après validation explicite.

## P10 — Indépendantisation du dépôt

Objectif : sortir le package du repo Déclic sans casser
`projet-declic-mobilite`.

État au 7 août 2026 :

- [x] conserver `package-xyt-gps/` dans `projet-declic-mobilite` pour ne pas
  casser les notebooks existants ;
- [x] préparer les métadonnées pour le dépôt cible `action-situee/xyt-gps` ;
- [x] créer ou mettre à jour le dossier local
  `/Users/marc-ed/BAS-local/repos/xyt_gps` ;
- [x] initialiser ce dossier à partir de l'historique `meschultheiss/xyt` si
  l'historique doit être conservé ;
- [x] copier l'état actuel de `package-xyt-gps` dans ce dépôt indépendant, sans
  copier `.venv`, `build`, `dist`, `site` ni caches ;
- [x] lancer les tests et la construction documentaire dans le dépôt
  indépendant ;
- [x] configurer localement le remote cible
  `https://github.com/action-situee/xyt-gps.git` ;
- [ ] créer le dépôt GitHub `action-situee/xyt-gps` ;
- [ ] pousser la branche principale vers `action-situee/xyt-gps` ;
- [ ] configurer PyPI/TestPyPI avec un token de publication.

Notes :

- le dépôt GitHub `action-situee/xyt-gps` n'existe pas encore au moment de cette
  vérification ; le remote local est prêt, mais le push échouera tant que le
  dépôt distant n'est pas créé ;
- `meschultheiss/xyt` existe et peut servir de base d'historique si l'on veut
  conserver la continuité de la première version ;
- la migration des notebooks Déclic vers `pip install xyt-gps` doit être faite
  plus tard, après publication et validation du package indépendant.

Validation locale du dépôt indépendant :

- `PYTHONPATH=src python -m pytest tests -q` : 111 tests passés, 12 tests
  projet ignorés hors `projet-declic-mobilite`, 2 warnings `pyproj` connus ;
- `PYTHONPATH=src python -m mkdocs build --strict` : documentation construite.
- `python -m build` : wheel et sdist construits ;
- `python -m twine check dist/*` : artefacts valides ;
- installation de `dist/xyt_gps-0.2.0-py3-none-any.whl` dans une venv Python
  3.12 vierge : import `xyt_gps` et symboles publics principaux OK.

## Ordre d'exécution recommandé

1. P1 — compléter les tests du contrat public.
2. P2 — documenter la mutabilité de `MobilityDataset`.
3. P3 — documenter chemin court et chemin explicite.
4. P5 — corriger la documentation obsolète.
5. P4 — réduire `__all__` seulement après vérification des usages.
6. P6 — évaluer `PipelineOptions`.
7. P8 — notebook fil rouge.
8. P9 — préparation PyPI.
9. P10 — indépendance du dépôt.
10. P7 — renommage interne `_*.py`, si encore utile.
