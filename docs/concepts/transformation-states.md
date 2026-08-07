# États de transformation

Le package doit rendre les états intermédiaires inspectables. Le détail de la préparation des données est décrit dans [Préparation des données](preparation-scope.md).

## Chaîne actuelle

| État | Niveau | Fonction principale | Description | Contrôle attendu |
|---|---|---|---|---|
| `RawGpsData` | obligatoire | `load_gps_export()` | tables brutes chargées, non transformées | chemins, dimensions, colonnes |
| validation brute | obligatoire | `check_raw_import_columns()` puis validation interne | contrat minimal GPS structuré avant parsing | erreurs bloquantes, colonnes recommandées, taux de nulls |
| `storyline` préparée | obligatoire | `prepare_storyline()` | géométries parsées, dates converties, phases et mappings ajoutés | modes, motifs, dates, CRS |
| `trips` préparée | obligatoire | `prepare_trips()` | trips fournis ou reconstruits avec `trip_id`, dates et compteurs | types numériques, dates, phases |
| `journeys` préparée | obligatoire | `prepare_journeys()` | journeys fournis ou reconstruits avec `journey_id`, dates et compteurs | types numériques, dates, phases |
| jours manquants | optionnel | `resample_missing_stays()` | option méthodologique pour ajouter des `Resampled_stay` | nombre de jours ajoutés par utilisateur |
| `staypoints` | obligatoire | `split_storyline()` | lignes `Stay` et `Resampled_stay` | motifs, durées, phases |
| motifs quotidiens | optionnel | `build_mobility_motifs()` | structure journalière des lieux visités par utilisateur | signatures de motifs, fréquences, séquences |
| `legs` | obligatoire | `split_storyline()` puis `clean_leg_geometries()` | lignes `Track`, géométries nettoyées, identifiants personne-jour | longueurs, géométries, outliers par mode |
| qualité de suivi détaillée | recommandé | `build_tracking_gap_report()` | jours actifs, jours manquants, maximum de jours consécutifs | rapport utilisateur-période |
| diagnostics legs | recommandé | `summarize_leg_lengths_by_mode()`, `get_extreme_legs_by_mode()` | distribution et cas extrêmes par mode | distances, vitesses si disponibles |
| confirmation utilisateur | optionnel | `build_user_confirmation_rates()`, `build_mode_detection_precision()` | confirmations des stays/tracks et précision des modes détectés | taux par utilisateur ou mode |
| excursions géographiques | optionnel | `add_excursion_flags()` | flag `excursion` depuis zone GeoJSON, géométrie ou cercle | convention `outside` / `inside`, CRS, rayon |
| labels territoriaux | optionnel | `add_spatial_zone_labels()`, `add_leg_origin_destination_zones()` | zones des activités et origine-destination des legs | pays, canton, zone projet |
| relation à une zone | optionnel | `classify_leg_relation_to_area()` | classe les legs en `intra`, `extra`, `exchange` | zone, rayon, CRS |
| `map_track_trip_journey` | obligatoire | `build_track_trip_journey_map()` | correspondance legs, trips, journeys | couverture du mapping |
| `map_legs_staypoints` | obligatoire | `build_legs_staypoints_map()` | correspondance staypoints et legs voisins | legs précédents et suivants |
| `user_day_coverage` | obligatoire | `build_user_day_coverage()` | calendrier personne-jour entre première et dernière observation réelle | jours de plage, trackés, actifs, non trackés et resamplés |
| `user_stats` qualité | obligatoire | `build_user_stats()` | statistiques de suivi, jours observés, jours resamplés, pondération | `tracking_quality_ok`, `resampled_days_count`, `weight` |
| qualité GPS des legs | recommandé | `add_signal_quality_flags()` | perte de signal absolue et relative, flags par mode | `max_signal_loss_m`, `relative_signal_loss`, `low_quality_legs_*`, `bad_signal_user` |
| traitement réseau des legs | expérimental et optionnel | `map_match_legs()` | GraphHopper Map Matching pour tous les legs `car`, `bike`, `foot`, avant exclusion des discontinuités | plafond d'appels, métriques géométriques, tables `map_leg_geometry_correspondence` et `spatial_geometry_exclusions` |
| excursions utilisateur | optionnel | `add_excursion_stats_to_user_stats()` | agrège les flags excursion à l'échelle utilisateur | nombre et ratio de legs excursion |
| table de sélection utilisateur | recommandé | `build_user_selection_table()` | combine qualité de suivi, signal GPS et seuils d'exclusion | `analysis_user_ok`, `analysis_user_reason` |
| dataset filtré | recommandé | `filter_mobility_dataset_by_users()` | applique une sélection d'utilisateurs à toutes les tables | cohérence des identifiants conservés |
| `MobilityDataset` | obligatoire | `prepare_mobility_dataset()` | objet final qui regroupe les tables et la provenance | cohérence inter-tables, `metadata` |
| export des états | recommandé | `write_mobility_dataset()` | écrit les tables et rapports pour comparaison notebook/package | manifest des fichiers écrits |
| carte de contrôle | optionnel | `plot_gps_traces()` | visualisation Folium des legs et staypoints | contrôle visuel d'un échantillon |
| enrichissement CO2 | enrichissement | à intégrer depuis le notebook historique CO2 | émissions et hypothèses d'occupation documentées | facteurs visibles, unités, sources |
| enrichissement santé | enrichissement | à intégrer depuis le notebook historique santé | métriques santé simples dérivées des legs | hypothèses METs, calories ou intensités documentées |
| indicateurs personne-jour | indicateurs | `build_person_day_indicators()` | distance, temps de déplacement et nombre de trips par utilisateur, date, phase et mode | unités, modes, jours sans mouvement |
| indicateurs personne-phase | indicateurs | `build_person_phase_indicators()` | moyennes journalières par utilisateur, phase et mode | cohérence avec la table personne-jour |
| indicateurs population | indicateurs | `build_population_indicators()` | moyennes de population par phase et mode | nombre d'utilisateurs contributeurs |

## Contrôle qualité

Le contrôle qualité est porté par `xyt_gps.quality`.

Les colonnes produites dans `user_stats` rendent le filtrage explicite :

```text
days_in_range_count
tracked_days_count
active_days_count
zero_movement_tracked_days_count
untracked_days_count
inactive_days_count
tracking_completeness
active_days_share
n_days_phase_*
low_quality_legs_count
low_quality_legs_share
mean_relative_signal_loss
bad_signal_user
signal_quality_computed
tracked_days_ok
phase*_tracked_days_ok
tracking_quality_ok
tracking_quality_reason
analysis_user_ok
analysis_user_reason
```

Le filtre recommandé reste en trois temps :

1. inspecter `build_tracking_quality_report(user_stats)` ;
2. construire une table de sélection avec `build_user_selection_table(user_stats)` ;
3. sélectionner les utilisateurs avec `select_analysis_users(selection_table)`, puis filtrer le dataset avec `filter_mobility_dataset_by_users(dataset, user_ids)`.

Les diagnostics complémentaires sont également disponibles :
`build_tracking_gap_report()`, `summarize_phase_tracking()`,
`summarize_leg_lengths_by_mode()`, `get_extreme_legs_by_mode()`,
`build_user_confirmation_rates()` et `build_mode_detection_precision()`.

## Nettoyage spatial

Le nettoyage spatial est porté par `xyt_gps.spatial`. Il reste optionnel et
paramétrable, car il dépend des couches territoriales disponibles et des
hypothèses du projet.

Les fonctions principales sont `clean_leg_geometries()`,
`add_signal_quality_flags()`, `map_match_legs()`, `add_spatial_zone_labels()`,
`add_leg_origin_destination_zones()`, `classify_leg_relation_to_area()`,
`add_trip_origin_destination_from_legs()` et
`add_journey_origin_destination_from_trips()`.

`map_match_legs()` peut traiter un flag qualité ou tous les modes pris en charge.
Dans `020`, GraphHopper reçoit tous les sommets ordonnés des profils `car`,
`bike` et `foot`, par blocs avec chevauchement. La clé seule ne déclenche aucun
appel et un plafond est contrôlé avant chaque leg. La table `legs` ne contient
qu'une géométrie active; la table longue `geometry_correspondence` conserve les
versions originale et propre. Les discontinuités non résolues sont consignées
avant leur retrait. Cette option doit rester désactivée tant qu'elle n'a pas été
validée sur un échantillon réel.

## Export et visualisation

`write_mobility_dataset()` sert à figer un état intermédiaire sur disque. C'est la fonction à utiliser pour comparer une sortie package avec une sortie historique de notebook.

`plot_gps_traces()` sert au contrôle visuel dans un notebook. Elle reste optionnelle, dépend de `folium`, et ne doit pas être utilisée comme étape obligatoire du pipeline.

## Mobility motifs

Les mobility motifs décrivent la forme quotidienne des lieux visités : par exemple domicile, travail, domicile, ou domicile, achats. La fonction `build_mobility_motifs()` part des `staypoints`, supprime les doublons consécutifs, relabellise les lieux dans leur ordre d'apparition, puis produit une signature canonique comparable entre utilisateurs.

Cette étape est optionnelle, car elle dépend de la définition de lieu. Si une colonne `location_id` existe déjà, elle peut être utilisée. Sinon, le package dérive un identifiant simple à partir du motif d'activité et des coordonnées arrondies. Les fonctions `summarize_mobility_motifs()` et `build_mobility_motif_sequences()` permettent ensuite d'analyser les fréquences de motifs et les séquences quotidiennes par utilisateur.

## Indicateurs

Le premier socle d’indicateurs est volontairement limité aux métriques déjà calculables à partir des tables de mobilité préparées :

```text
distance_km
travel_time_min
trip_count
leg_count
```

La fonction `compute_mobility_indicators()` produit trois tables :

| Table | Grain | Usage |
|---|---|---|
| `person_day` | utilisateur, jour, phase, mode | contrôler les valeurs journalières et les jours sans mouvement |
| `person_phase` | utilisateur, phase, mode | comparer les comportements moyens par phase |
| `population` | phase, mode | produire les premiers tableaux de synthèse |

Avec `include_zero_days=True`, le calendrier personne-jour vient de
`dataset.user_day_coverage`. `day_basis` choisit `days_in_range`,
`tracked_days` ou `active_days`. Le fallback historique fondé sur les phases de
`ProjectConfig` reste disponible pour les datasets construits manuellement sans
table de couverture.

Les autres paramètres structurants sont explicites dans `compute_mobility_indicators()` :

| Paramètre | Effet |
|---|---|
| `include_excursions` | inclut ou exclut les lignes marquées `excursion` avant le calcul |
| `include_airplane` | inclut ou exclut les étapes et déplacements avion ; par défaut ils sont exclus |
| `use_weights` | active la pondération population à partir de `user_stats` |
| `weight_col` | définit la colonne de pondération, par défaut `weight` |

Ces options sont stockées dans `IndicatorResult.metadata`. L’export HTML `plot_indicator_bars()` les affiche avec le nombre d’utilisateurs, le nombre de jours-utilisateurs, le nombre moyen d’étapes par jour-utilisateur, le périmètre de calcul et les informations disponibles sur le taux d’occupation. Le graphe reste ainsi interprétable hors notebook.

Dans les graphes par mode, une ligne `Tous modes` est ajoutée en tête de chaque indicateur. Elle donne le total ou la moyenne tous modes confondus selon la métrique calculée, avant le détail par mode. `Tous modes` et les modes détaillés ont chacun leur propre barre de référence, ce qui évite que le total tous modes écrase visuellement les différences entre modes.

Lorsque les horaires des legs sont disponibles, les métadonnées contiennent aussi un profil de demande par phase et par tranche de 5 minutes. L’export HTML l’affiche comme une courbe de demande journalière moyenne : axe horizontal de 00:00 à 24:00, axe vertical en nombre moyen de personnes en déplacement. Cette lecture complète les indicateurs agrégés en montrant à quel moment de la journée la mobilité se concentre.

Le même profil alimente une heatmap horaire. Elle montre les pics par jours de semaine et, si une colonne de motif est disponible, par motifs agrégés. Le sélecteur de mode permet de comparer `Tous modes` avec un mode particulier sans recalculer les indicateurs.

Les enrichissements CO2 et santé s'ajoutent avant le calcul des indicateurs :

- `add_co2_occupancy_metrics()` ajoute les facteurs, taux d'occupation et émissions par leg à partir de `CO2OccupancyConfig`, lui-même alimenté par les CSV projet stockés dans `Notebooks/config/reference/` ;
- `add_health_metrics()` ajoute vitesse, intensité, bornes de vitesse de l'intensité, METs, calories et pas à partir de `HealthConfig`, lui-même alimenté par le CSV projet des METs ;
- `build_leg_enrichment_tables()` permet d'exporter des tables latérales lisibles par `leg_id`.

Ces enrichissements préparent la base analysable. Ils ne remplacent pas les analyses thématiques, qui restent hors package.

Les analyses thématiques restent hors package. Les notebooks ou applications
dédiées utiliseront les exports analytiques comme entrée.

## Principe

Une transformation doit pouvoir être relue table par table. L’orchestrateur `prepare_mobility_dataset()` est utile, mais il ne doit pas devenir la seule façon de comprendre le traitement.
