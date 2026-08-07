# xyt_gps - cadrage initial du package

!!! note "Page de maintenance"
    Cette page documente le cadrage historique du package. Elle n'est pas nécessaire pour lancer une transformation GPS.

Date : 2026-06-09

## Périmètre

Objectif initial : transformer les notebooks Declic Mobilite en package Python simple, progressif et testable.

Le package doit permettre de :

1. charger des exports GPS bruts, principalement MotionTag ;
2. nettoyer, structurer et enrichir ces donnees en objets de mobilite ;
3. produire des indicateurs personne-jour et populationnels ;
4. exporter des tables lisibles et documentees.

Le package doit s'arreter a l'export d'une base de mobilite analysable :
tables structurees, indicateurs generiques et enrichissements transversaux
documentes. Les analyses thematiques ne sont pas integrees au package.

Terminologie cible :

| Ancien repere | Terme package |
|---|---|
| WP1 / module0 | preparation des donnees |
| module1_1 | enrichissement CO2 |
| module1_2 | enrichissement sante |
| WP2 | indicateurs et export analytique |
| WP3 | analyses thematiques hors package |

Les anciens noms restent utiles pour retrouver les notebooks sources, mais ils
ne doivent pas devenir la terminologie publique du package.

Les donnees du dossier `Data/` ne doivent pas etre lues par le package pendant les tests. Les tests devront utiliser de petits jeux synthetiques.

## Elements existants a reprendre

### `archive/legacy-code/xyt_dynamite`

Fonctions reutilisables ou a refactoriser :

| Fichier | Role actuel | Reprise probable |
|---|---|---|
| `xyt_0_parsing.py` | parsing EWKB, parsing dates, affectation de phases, suppression controlee de NA | `xyt_gps.transform.parsing`, `xyt_gps.project.phases` |
| `config.py` | couleurs, CRS, dates de phases, perimetres, constantes projet | remplacer par `ProjectConfig`, garder quelques valeurs par defaut documentees |
| `signal_quality.py` | qualite GPS, perte de signal, utilisateurs a mauvais signal | `xyt_gps.transform.quality` |
| `utils.py` | distance maximale entre points successifs | `xyt_gps.transform.geometry` |
| `xyt_panel.py` | filtres et indicateurs personne-jour | scinder entre `xyt_gps.indicators.filters` et `xyt_gps.indicators.person_day` |
| `xyt_to_map.py` | visualisation Folium | a garder hors coeur, dans `xyt_gps.viz` ou a differer |
| `xyt_rhythm.py` | ACP, CAH, rythmes | hors MVP GPS, a differer |

### Notebooks historiques de preparation des donnees

Le coeur actuel est compose de :

| Notebook | Role | Statut package |
|---|---|---|
| `module0_1_1_parserawstoryline.ipynb` | charge MotionTag, parse geometries/dates, cree storyline/staypoints/legs/trips/journeys/user_stats, cree tables de correspondance | priorite 1 |
| `module0_1_2_run_tracking_qual_check.ipynb` | controles de suivi, confirmations, distribution des modes, qualite temporelle | priorite 2, surtout diagnostics |
| `module0_2_1_spatial_cleaning.ipynb` | perte de signal, spatial joins, cantons O/D, indicateurs intra/extra | priorite 1 |
| `module0_4_filter_data.ipynb` | filtres utilisateurs : duree de suivi, couverture, activite, problemes | priorite 2 |
| `module1_1_co2_occupancy.ipynb` | taux d'occupation et facteurs CO2 par mode | priorite 2 |
| `module1_2_health.ipynb` | intensite, METs, calories | priorite 2 |
| `module1_3_1_osrm_processor.ipynb` | ancien map matching OSRM | remplacé dans le flux actif par GraphHopper |
| `module1_3_2_google_direction.ipynb` | ancienne correction Google Directions | conservé seulement comme historique |

### Notebooks historiques d'indicateurs et d'export

| Notebook | Role | Statut package |
|---|---|---|
| `1_base_personne_jour.ipynb` | agregations par utilisateur, periode, phase et mode ; distance, temps, trips, CO2, calories, METs | priorite 1 pour les indicateurs |
| `2_export_data.ipynb` | export GeoJSON/CSV/XLSX, dictionnaire de variables | priorite 2 |

### `for_after_marced_vacation`

Ces fichiers ressemblent a un brouillon utile mais moins propre :

| Element | Reprise proposee |
|---|---|
| `xyt_panel.py` | version proche de `archive/legacy-code/xyt_dynamite/xyt_panel.py`, avec `user_id_fors`. À utiliser pour identifier les divergences, pas comme base principale. |
| notebooks Streamlit | bonnes idees de filtres et d'indicateurs interactifs, mais a differer. |
| map snapping / shortest path | utile pour module experimental `xyt_gps.enrich.map_matching`, pas pour le MVP. |

## Chaine de transformation optimisee

| Etape | Action package | Inputs | Outputs |
|---|---|---|---|
| 0 | Definir les parametres projet | `ProjectConfig` : nom experimentation, fournisseur, periode, dates des phases, chemins, CRS, seuils, fichiers externes | configuration validee |
| 0bis | Données de démonstration | en construction provisoire | jeu stable à créer, compresser, héberger et documenter avant usage dans la doc |
| 1 | Charger les exports bruts | `user_statistics`, `storyline`, `trips`, `journeys`, option `public_transport_legs`, table sociodemo, zones geographiques | `RawGpsDataset` |
| 1bis | Echantillonner les exports lourds si necessaire | `RawSampleConfig.by_users(n)` pour garder toutes les traces de quelques IDNO/user_id, ou `RawSampleConfig.random_rows(n)` pour lignes/legs aleatoires | `RawGpsDataset` reduit pour inspection rapide |
| 2 | Controler les colonnes minimales | schemas attendus, taux de NA acceptables | erreurs explicites ou donnees conservees |
| 3 | Parser geometries et dates | EWKB, colonnes `started_at`, `finished_at`, fuseaux horaires | dataframes avec `geometry`, datetimes UTC/locales |
| 4 | Harmoniser les identifiants | `id`, `trip_id`, `journey_id`, `user_id`, `leg_id`, `activity_id` | identifiants stables |
| 5 | Harmoniser modes et motifs | mappings MotionTag vers `mode_niv1`, `mode_niv2`, `mode_mrmt`, `purpose_mrmt` | colonnes de modes et motifs standardisees |
| 6 | Affecter les phases | dates de phases et dates de suivi utilisateur | colonne `phase`, dates de phases effectives par utilisateur |
| 7 | Reconstituer les objets | `storyline` | `staypoints`, `legs`, `trips`, `journeys` |
| 8 | Nettoyer les geometries de legs | MultiLineString, LineString, CRS operationnel | legs LineString propres, longueurs et durees |
| 9 | Construire les tables de correspondance | legs, trips, journeys, staypoints | `map_track_trip_journey`, `map_legs_staypoints` |
| 10 | Produire `user_stats` enrichi | UserStatistics MotionTag, sociodemo, dates de suivi, phases | suivi effectif, nb jours par phase, variables socio-demo |
| 11 | Calculer la qualite GPS | legs en EPSG:2056, seuils par mode | perte de signal absolue/relative, flags qualite, mauvais utilisateurs |
| 12 | Enrichir spatialement | TAZ/cantons, pays, perimetres | pays/canton des staypoints, O/D canton des legs/trips/journeys, intra/exchange |
| 13 | Filtrer la base | seuils de suivi, couverture, activite, outliers, mauvais signal | dataset nettoye et journal de filtrage |
| 14 | Ajouter CO2 et occupation | legs, journeys, facteurs CO2, taux occupation | `occupancy_co2` par leg |
| 15 | Ajouter sante | legs, vitesse, mode, duree, poids moyen parametre | `health` par leg |
| 16 | Corriger les geometries optionnelles | GraphHopper, clé d'environnement, plafond et seuils géométriques | géométrie retenue et tables d'audit |
| 17 | Calculer indicateurs | legs/trips/enrichissements/user_stats | tables personne-jour par phase, mode et indicateur |
| 18 | Exporter | dataset transforme, indicateurs, config | Parquet interne, CSV/GeoJSON/XLSX livrables |

## Parametres projet a formaliser

Parametres minimaux :

| Parametre | Type | Usage |
|---|---|---|
| `experiment_name` | str | nom analytique de l'etude |
| `provider_project_name` | str | nom utilise dans les exports MotionTag |
| `period` | str ou intervalle date | reperage des fichiers bruts |
| `raw_dir` | Path | dossier d'import |
| `work_dir` | Path | sorties intermediaires package |
| `export_dir` | Path | livrables |
| `target_crs` | str | CRS final, actuellement `EPSG:4326` |
| `metric_crs` | str | CRS metrique, actuellement `EPSG:2056` |
| `phase_dates` | dict | debut/fin des phases |
| `tracking_thresholds` | dict | jours minimum par phase, arrondi semaine |
| `spatial_quality_thresholds` | dict | distance entre points consecutifs, perte de signal, outliers Q98/Q99 par mode |
| `matching_thresholds` | dict | tolérance legs/trips/journeys, paramètres techniques de matching |
| `mode_mapping` | dict | niveaux de modes |
| `purpose_mapping` | dict | motifs MRMT |
| `external_geodata` | dict | TAZ, pays, perimetres, zones excursions |
| `sociodemo_path` | Path optionnel | lien aux donnees sociodemographiques |
| `map_matching` | dict optionnel | GraphHopper, URL, variable de clé, profils, plafonds et seuils |

## Architecture recommandee

Variante A : pipeline fonctionnel simple.

Cette variante est recommandee pour demarrer. Elle garde une API intuitive, des fonctions testables, et permet de reprendre progressivement les notebooks.

```python
import xyt_gps as xyt

config = xyt.ProjectConfig(...)
raw = xyt.load_gps_export(config)
mobility = xyt.transform(raw, config)
enriched = xyt.enrich(mobility, config, modules=["co2", "health"])
tables = xyt.indicators(enriched, config, mode_col="mode_niv2")
xyt.export(enriched, tables, config)
```

Structure proposee :

```text
xyt-gps/
  pyproject.toml
  src/xyt_gps/
    __init__.py
    config.py
    schemas.py
    datasets.py
    io/
      motiontag.py
      geodata.py
    transform/
      parsing.py
      modes.py
      temporal.py
      quality.py
      spatial_geometry.py
      spatial_quality.py
      spatial_zones.py
      mapping.py
    enrich/
      co2.py
      health.py
      map_matching.py
    indicators/
      person_day.py
      filters.py
      export_tables.py
    export/
      files.py
      dictionary.py
    validation.py
  tests/
  examples/
  docs/
```

Variante B : objet `Project`.

Cette variante est plus ergonomique pour automatiser une experimentation complete, mais elle risque de figer trop vite les choix.

```python
from xyt_gps import Project

project = Project.from_config("declic.yml")
project.load()
project.transform()
project.enrich(["co2", "health"])
project.compute_indicators()
project.export()
```

Recommandation : developper d'abord la variante A, puis ajouter une facade `Project` quand les schemas et les sorties sont stabilises.

## Plan de packaging

1. Initialiser le package : `pyproject.toml`, `src/xyt_gps`, imports publics minimaux, dependances reduites.
2. Creer les dataclasses : `ProjectConfig`, `PhaseConfig`, `GpsExportPaths`, `MobilityDataset`, `IndicatorResult`.
3. Porter les fonctions de parsing : EWKB, dates, suppression NA, modes/motifs.
4. Porter le loader MotionTag avec detection robuste des noms de fichiers.
5. Porter `transform_storyline()` : storyline vers staypoints/legs/trips/journeys/mappings/user_stats.
6. Ajouter des validations de schemas avant et apres transformation.
7. Porter qualite GPS et spatial cleaning.
8. Porter les enrichissements CO2 et sante, avec référentiels documentes et parametrables.
9. Porter les indicateurs et exports analytiques en fonctions pures.
10. Ajouter les exports propres et dictionnaires de variables.
11. Ajouter tests unitaires sur donnees synthetiques, sans acces a `Data/`.
12. Ajouter un exemple minimal reproductible.
13. Comparer les sorties package avec les sorties notebooks sur un echantillon controle.
14. Traiter l'ancien package PyPI `xyt` seulement apres stabilisation de la publication `xyt-gps`.

## Points de vigilance

1. Plusieurs chemins et dates sont actuellement hardcodes.
2. Les conventions `user_id` et `user_id_fors` divergent entre versions.
3. Certains seuils de perte de signal semblent incoherents entre commentaires et valeurs effectives, par exemple plusieurs seuils absolus commentes en kilometres valent `1`.
4. Les facteurs CO2, METs et taux d'occupation doivent etre sources, documentes et stockes hors code dans des fichiers de reference versionnes.
5. GraphHopper doit rester optionnel tant que les coûts, conditions contractuelles et seuils ne sont pas validés sur des traces réelles.
6. Les exports actuels melangent donnees internes, donnees livrables et dictionnaire. Le package doit separer sorties intermediaires et livrables.
7. Les notebooks contiennent des controles exploratoires utiles mais pas tous necessaires dans le MVP.
8. Le package PyPI `xyt` existe deja en version `0.1.2` publiee le 2023-12-04. La strategie recommandee est de publier la nouvelle generation sous `xyt-gps`, avec `import xyt_gps`, puis de traiter `xyt` comme un eventuel canal de compatibilite.
