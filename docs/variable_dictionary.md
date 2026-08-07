# Dictionnaire des variables

Date : 2026-06-09

Source de contexte : `Data/dumps_situee/RENDU250815_gps_prefiguration_clean/dictionnaire_variables.xlsx`.

Ce document rend le dictionnaire livré dans le précédent projet accessible dans le dépôt du package. Il décrit les tables exportées, les variables, les types observés, les taux de valeurs manquantes et les descriptions disponibles.

Limite : les types et taux de valeurs manquantes reflètent le dump de préfiguration livré le 2025-08-15. Ils ne doivent pas être interprétés comme un contrat universel pour toutes les expérimentations.

## Comment lire une ligne du dictionnaire

Chaque ligne décrit une variable observée dans une table de sortie. Exemple tiré de la table `legs` :

| variable | dtype | type_variable | missing_percent | modalites | interprétation package |
|---|---|---|---|---|---|
| `length` | `int32` | Continue | `0%` | `Min: 3.00, Max: 9114420.00, Moyenne: 7887.43` | longueur du leg fournie ou conservée dans la table de mobilité |
| `relative_signal_loss` | `float64` | Continue | `0%` | `Min: 0.00, Max: 1.00, Moyenne: 0.25` | plus grand saut GPS divisé par la longueur métrique du leg |
| `low_quality_legs_1` | `int64` | Dummy/Binaire | `0%` | `[0, 1]` | flag de qualité GPS selon le niveau 1 de seuils par mode |
| `bad_signal_user` | `int64` | Catégorielle | `0%` | `[0]` dans le dump livré | flag utilisateur, calculé à partir des pertes de signal moyennes |

Un dictionnaire de variables n’est donc pas seulement une liste de colonnes. Il indique aussi le type observé, la complétude et l’usage analytique attendu. Dans `xyt_gps`, les variables dérivées doivent rester reliées à une fonction explicite ou à une hypothèse documentée.

## Synthèse par table

| dataset | variables |
| --- | --- |
| journey | 19 |
| legs | 28 |
| legs_info | 17 |
| occupancy_co2 | 4 |
| public_transport | 8 |
| staypoints | 30 |
| trips | 19 |
| user_stats | 49 |

## Legs

| variable | dtype | type_variable | missing_count | missing_percent | modalites | description |
| --- | --- | --- | --- | --- | --- | --- |
| leg_id | object | Identifiant | 0 | 0% | 27334 identifiants uniques | Variable identifiant unique |
| user_id_day | object | Identifiant | 0 | 0% | 2584 identifiants uniques | Variable identifiant unique |
| legs_date | datetime64[ns] | Temporelle | 0 | 0% | 2025-04-07 00:00:00 à 2025-06-29 00:00:00 | Variable temporelle (dates/heures) |
| user_id | object | Identifiant | 0 | 0% | 50 identifiants uniques | Variable identifiant unique |
| type | object | Catégorielle nominale | 0 | 0% | ['Track'] | Variable catégorielle (1 modalités) |
| started_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 16:41:31+00:00 à 2025-06-29 22:18:07+00:00 | Variable temporelle (dates/heures) |
| started_at_timezone | object | Temporelle avec timezone | 0 | 0% | Africa/Casablanca à Pacific/Tahiti | Variable temporelle avec information de fuseau horaire |
| finished_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 16:44:40+00:00 à 2025-06-29 22:55:32+00:00 | Variable temporelle (dates/heures) |
| finished_at_timezone | object | Temporelle avec timezone | 0 | 0% | Africa/Casablanca à Pacific/Tahiti | Variable temporelle avec information de fuseau horaire |
| length | int32 | Continue | 0 | 0% | Min: 3.00, Max: 9114420.00, Moyenne: 7887.43 | Variable numérique continue |
| mode | object | Texte | 0 | 0% | 24 valeurs uniques | Variable texte libre |
| geometry | geometry | Géospatiale | 0 | 0% | LineString GPS tracks | Géométrie du segment de déplacement |
| started_on | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 00:00:00+00:00 à 2025-06-30 00:00:00+00:00 | Variable temporelle (dates/heures) |
| started_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 16:41:31+00:00 à 2025-06-29 22:18:07+00:00 | Variable temporelle (dates/heures) |
| finished_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 16:44:40+00:00 à 2025-06-29 22:55:32+00:00 | Variable temporelle (dates/heures) |
| trip_id | object | Identifiant | 173 | 1% | 12342 identifiants uniques | Variable identifiant unique |
| Phase | object | Catégorielle nominale | 0 | 0% | ['Before_Phase1', 'Phase1', 'Other', 'Phase2', 'Phase3'] | Variable catégorielle (5 modalités) |
| mode_niv1 | object | Catégorielle nominale | 0 | 0% | ['Marche', 'TP', 'Vélo', 'Voiture', 'Autres'] | Variable catégorielle (5 modalités) |
| mode_niv2 | object | Catégorielle nominale | 0 | 0% | ['Marche', 'Train', 'TPU', 'Vélo', 'Voiture', 'Autres', 'Carsharing'] | Variable catégorielle (7 modalités) |
| date_only | datetime64[ns] | Temporelle | 0 | 0% | 2025-04-07 00:00:00 à 2025-06-29 00:00:00 | Variable temporelle (dates/heures) |
| duration | Int64 | Autre | 0 | 0% | 3432 valeurs uniques | Type de données: Int64 |
| extreme98_length_mode | bool | Dummy/Binaire | 0 | 0% | [np.False_, np.True_] | Variable binaire (True/False) |
| extreme99_length_mode | bool | Dummy/Binaire | 0 | 0% | [np.False_, np.True_] | Variable binaire (True/False) |
| relative_signal_loss | float64 | Continue | 0 | 0% | Min: 0.00, Max: 1.00, Moyenne: 0.25 | Variable numérique continue |
| low_quality_legs_1 | int64 | Dummy/Binaire | 0 | 0% | [np.int64(0), np.int64(1)] | Variable binaire numérique (0/1 ) |
| intra_GE | float64 | Catégorielle | 0 | 0% | [np.float64(1.0), np.float64(0.5), np.float64(0.0)] | Variable catégorielle (3 modalités) |
| origin_canton | object | Catégorielle nominale | 0 | 0% | ['GE', 'VD', 'NE', 'FR', 'ZH', 'BE', 'VS', 'HorsCHE', 'LU', 'SO'] | Variable catégorielle (20 modalités) |
| destination_canton | object | Catégorielle nominale | 0 | 0% | ['GE', 'VD', 'NE', 'FR', 'ZH', 'BE', 'VS', 'HorsCHE', 'LU', 'SO'] | Variable catégorielle (20 modalités) |

## Trips

| variable | dtype | type_variable | missing_count | missing_percent | modalites | description |
| --- | --- | --- | --- | --- | --- | --- |
| trip_id | object | Identifiant | 0 | 0% | 12344 identifiants uniques | Variable identifiant unique |
| user_id | object | Identifiant | 0 | 0% | 50 identifiants uniques | Variable identifiant unique |
| started_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 16:41:31+00:00 à 2025-06-29 22:18:07+00:00 | Variable temporelle (dates/heures) |
| finished_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 17:02:09+00:00 à 2025-06-29 22:55:32+00:00 | Variable temporelle (dates/heures) |
| main_mode | object | Texte | 0 | 0% | 24 valeurs uniques | Variable texte libre |
| modes | object | Texte | 0 | 0% | 1321 valeurs uniques | Variable texte libre |
| long_distance | bool | Dummy/Binaire | 0 | 0% | [np.False_, np.True_] | Variable binaire (True/False) |
| tracks_count | Int64 | Autre | 0 | 0% | 24 valeurs uniques | Type de données: Int64 |
| tracks_count_without_walks | int64 | Catégorielle | 0 | 0% | [np.int64(1), np.int64(3), np.int64(0), np.int64(2), np.int64(4), np.int64(5), np.int64(6), np.int64(8), np.int64(10), np.int64(7)] | Variable catégorielle (12 modalités) |
| purpose | object | Catégorielle nominale | 64 | 1% | ['Family & Friends', 'At home', 'Shopping', 'Eat out', 'Other', 'Leisure', 'Sports', 'Errand', 'Medical visit', 'Working'] | Variable catégorielle (14 modalités) |
| duration | Int64 | Autre | 0 | 0% | 4428 valeurs uniques | Type de données: Int64 |
| distance | Int64 | Autre | 0 | 0% | 6713 valeurs uniques | Type de données: Int64 |
| created_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-08 07:18:41+00:00 à 2025-07-06 14:09:11+00:00 | Variable temporelle (dates/heures) |
| purpose_mrmt | object | Catégorielle nominale | 0 | 0% | ['Loisir', 'Domicile', 'Achat et services', 'Tout motifs', 'Travail', 'Accompagnement', 'Formation'] | Variable catégorielle (7 modalités) |
| main_mode_niv1 | object | Catégorielle nominale | 0 | 0% | ['Voiture', 'TP', 'Marche', 'Vélo', 'Autres'] | Variable catégorielle (5 modalités) |
| modes_niv1 | object | Texte | 0 | 0% | 724 valeurs uniques | Variable texte libre |
| leading_activity_id | object | Identifiant | 228 | 2% | 12116 identifiants uniques | Variable identifiant unique |
| trip_origin_canton | object | Catégorielle nominale | 0 | 0% | ['GE', 'HorsCHE', 'BS', 'ZH', 'VS', 'VD', 'NE', 'LU', 'BE', 'FR'] | Variable catégorielle (20 modalités) |
| trip_destination_canton | object | Catégorielle nominale | 0 | 0% | ['GE', 'VS', 'HorsCHE', 'TI', 'NE', 'VD', 'ZH', 'BE', 'FR', 'JU'] | Variable catégorielle (20 modalités) |

## Staypoints

| variable | dtype | type_variable | missing_count | missing_percent | modalites | description |
| --- | --- | --- | --- | --- | --- | --- |
| activity_id | object | Identifiant | 0 | 0% | 19172 identifiants uniques | Variable identifiant unique |
| user_id | object | Identifiant | 0 | 0% | 50 identifiants uniques | Variable identifiant unique |
| type | object | Dummy/Binaire | 0 | 0% | ['Stay', 'Resampled_stay'] | Variable binaire textuelle (Stay/Resampled_stay ) |
| started_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 15:33:51+00:00 à 2025-06-29 22:55:32+00:00 | Variable temporelle (dates/heures) |
| started_at_timezone | object | Temporelle avec timezone | 0 | 0% | 2025-04-11 00:00:01+0000 à Pacific/Tahiti | Variable temporelle avec information de fuseau horaire |
| finished_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 16:41:31+00:00 à 2025-06-30 17:03:50+00:00 | Variable temporelle (dates/heures) |
| finished_at_timezone | object | Temporelle avec timezone | 0 | 0% | 2025-04-11 23:59:59+0000 à Pacific/Tahiti | Variable temporelle avec information de fuseau horaire |
| purpose | object | Catégorielle nominale | 163 | 1% | ['home', 'leisure', 'shopping', 'eat', 'work', 'wait', 'unknown', 'errand', 'family_friends', 'sport'] | Variable catégorielle (14 modalités) |
| geometry | geometry | Géospatiale | 163 | 1% | LineString GPS tracks | Géométrie du segment de déplacement |
| confirmed_at | datetime64[ns, UTC] | Temporelle | 9350 | 49% | 2025-04-09 06:38:53.380000+00:00 à 2025-07-06 14:10:30.118000+00:00 | Variable temporelle (dates/heures) |
| started_on | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 00:00:00+00:00 à 2025-06-30 00:00:00+00:00 | Variable temporelle (dates/heures) |
| misdetected_completely | object | Dummy/Binaire | 0 | 0% | ['f', 't'] | Variable binaire textuelle (f/t ) |
| merged | object | Catégorielle nominale | 0 | 0% | ['f'] | Variable catégorielle (1 modalités) |
| created_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 18:14:11.813000+00:00 à 2025-08-15 10:02:23+00:00 | Variable temporelle (dates/heures) |
| updated_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-08 07:18:41.633000+00:00 à 2025-08-15 10:02:23+00:00 | Variable temporelle (dates/heures) |
| started_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 15:33:51+00:00 à 2025-06-29 22:55:32+00:00 | Variable temporelle (dates/heures) |
| finished_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 16:41:31+00:00 à 2025-06-30 17:03:50+00:00 | Variable temporelle (dates/heures) |
| confirmed_at_in_timezone | datetime64[ns, UTC] | Temporelle | 9187 | 48% | 2025-04-09 06:38:53+00:00 à 2025-08-15 10:02:23+00:00 | Variable temporelle (dates/heures) |
| created_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 18:14:11+00:00 à 2025-08-15 10:02:23+00:00 | Variable temporelle (dates/heures) |
| updated_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-08 07:18:41+00:00 à 2025-08-15 10:02:23+00:00 | Variable temporelle (dates/heures) |
| comment_feedback | object | Texte | 18977 | 99% | 28 valeurs uniques | Variable texte libre |
| trip_id | object | Identifiant | 12735 | 66% | 4093 identifiants uniques | Variable identifiant unique |
| Phase | object | Catégorielle nominale | 0 | 0% | ['Before_Phase1', 'Phase1', 'Other', 'Phase2', 'Phase3'] | Variable catégorielle (5 modalités) |
| purpose_niv1 | object | Catégorielle nominale | 163 | 1% | ['Domicile', 'Loisir', 'Achat', 'Travail', 'Attente', 'Autre', 'Santé'] | Variable catégorielle (7 modalités) |
| date_only | datetime64[ns] | Temporelle | 0 | 0% | 2025-04-07 00:00:00 à 2025-06-29 00:00:00 | Variable temporelle (dates/heures) |
| lon | float64 | Continue | 163 | 1% | Min: -149.61, Max: 30.74, Moyenne: 6.17 | Variable numérique continue |
| lat | float64 | Continue | 163 | 1% | Min: -17.56, Max: 55.70, Moyenne: 45.89 | Variable numérique continue |
| duration | int32 | Continue | 0 | 0% | Min: 31.00, Max: 86398.00, Moyenne: 10758.51 | Variable numérique continue |
| activity_in_country | object | Texte | 0 | 0% | 22 valeurs uniques | Variable texte libre |
| activity_in_canton | object | Texte | 0 | 0% | 41 valeurs uniques | Variable texte libre |

## Occupancy Co2

| variable | dtype | type_variable | missing_count | missing_percent | modalites | description |
| --- | --- | --- | --- | --- | --- | --- |
| leg_id | object | Identifiant | 0 | 0% | 27334 identifiants uniques | Variable identifiant unique |
| occupancy_rate | float64 | Catégorielle | 0 | 0% | [np.float64(1.0), np.float64(2.08), np.float64(1.81), np.float64(1.8), np.float64(1.91), np.float64(1.92), np.float64(2.16), np.float64(2.41), np.float64(2.15), np.float64(2.64)] | Variable catégorielle (17 modalités) |
| g_CO2_km | float64 | Catégorielle | 326 | 1% | [np.float64(0.0), np.float64(7.0), np.float64(42.8), np.float64(8.0), np.float64(5.6), np.float64(90.5), np.float64(298.3), np.float64(36.2), np.float64(161.3), np.float64(10.6)] | Variable catégorielle (15 modalités) |
| g_CO2_direct_km | float64 | Catégorielle | 326 | 1% | [np.float64(0.0), np.float64(0.1), np.float64(0.4), np.float64(63.5), np.float64(180.8), np.float64(0.2), np.float64(123.7), np.float64(223.6), np.float64(90.4), np.float64(100.0)] | Variable catégorielle (10 modalités) |

## Legs Info

| variable | dtype | type_variable | missing_count | missing_percent | modalites | description |
| --- | --- | --- | --- | --- | --- | --- |
| leg_id | object | Identifiant | 0 | 0% | 27334 identifiants uniques | Variable identifiant unique |
| geometry_map_matched | geometry | Autre | 0 | 0% | 26310 valeurs uniques | Type de données: geometry |
| map_match_source | object | Catégorielle nominale | 0 | 0% | ['osrm', 'motiontag', 'google'] | Variable catégorielle (3 modalités) |
| mode | object | Texte | 0 | 0% | 24 valeurs uniques | Variable texte libre |
| detected_mode | object | Catégorielle nominale | 0 | 0% | ['Mode::Walk', 'Mode::Train', 'Mode::Tram', 'Mode::RegionalTrain', 'Mode::Bicycle', 'Mode::Bus', 'Mode::Car', 'Mode::LightRail', 'Mode::Ferry', 'Mode::Carsharing'] | Variable catégorielle (16 modalités) |
| origin_osm_id | float64 | Continue | 4236 | 15% | Min: 0.00, Max: 13016377360.00, Moyenne: 5679758120.53 | Variable numérique continue |
| destination_osm_id | float64 | Continue | 4236 | 15% | Min: 0.00, Max: 13016377360.00, Moyenne: 5664341555.77 | Variable numérique continue |
| started_on | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 00:00:00+00:00 à 2025-06-30 00:00:00+00:00 | Variable temporelle (dates/heures) |
| confirmed_at | object | Temporelle avec timezone | 12805 | 47% | 14529 valeurs uniques | Variable temporelle avec information de fuseau horaire |
| created_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 18:14:11.854000+00:00 à 2025-07-01 21:14:49.785000+00:00 | Variable temporelle (dates/heures) |
| updated_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 18:14:11.854000+00:00 à 2025-07-06 14:10:30.101000+00:00 | Variable temporelle (dates/heures) |
| confirmed_at_in_timezone | datetime64[ns, UTC] | Temporelle | 12805 | 47% | 2025-04-09 06:38:53+00:00 à 2025-07-06 14:10:30+00:00 | Variable temporelle (dates/heures) |
| created_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 18:14:11+00:00 à 2025-07-01 21:14:49+00:00 | Variable temporelle (dates/heures) |
| updated_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 18:14:11+00:00 à 2025-07-06 14:10:30+00:00 | Variable temporelle (dates/heures) |
| misdetected_completely | object | Dummy/Binaire | 0 | 0% | ['f', 't'] | Variable binaire textuelle (f/t ) |
| merged | object | Dummy/Binaire | 0 | 0% | ['f', 't'] | Variable binaire textuelle (f/t ) |
| comment_feedback | object | Texte | 27265 | 100% | 32 valeurs uniques | Variable texte libre |

## Public Transport

| variable | dtype | type_variable | missing_count | missing_percent | modalites | description |
| --- | --- | --- | --- | --- | --- | --- |
| leg_id | object | Identifiant | 0 | 0% | 3841 identifiants uniques | Variable identifiant unique |
| departure_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-09 10:54:18+00:00 à 2025-06-29 21:24:10+00:00 | Variable temporelle (dates/heures) |
| departure_station_name | object | Texte | 1 | 0% | 623 valeurs uniques | Variable texte libre |
| arrival_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-09 11:08:02+00:00 à 2025-06-29 21:29:23+00:00 | Variable temporelle (dates/heures) |
| arrival_station_name | object | Texte | 0 | 0% | 640 valeurs uniques | Variable texte libre |
| line_number | object | Texte | 12 | 0% | 264 valeurs uniques | Variable texte libre |
| departure_station | object | Texte | 0 | 0% | 1092 valeurs uniques | Variable texte libre |
| arrival_station | object | Texte | 0 | 0% | 1095 valeurs uniques | Variable texte libre |

## User Stats

Les statistiques descriptives ci-dessous proviennent du snapshot qui a servi à
générer ce dictionnaire. Le contrat courant recalcule depuis `storyline`
`days_in_range_count`, `tracked_days_count`, `active_days_count`,
`zero_movement_tracked_days_count` et `untracked_days_count`. Les distributions
doivent être régénérées après un nouveau run du pipeline.

| variable | dtype | type_variable | missing_count | missing_percent | modalites | description |
| --- | --- | --- | --- | --- | --- | --- |
| user_id | object | Identifiant | 0 | 0% | 50 identifiants uniques | Variable identifiant unique |
| first_tracked_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 00:00:00+00:00 à 2025-05-07 00:00:00+00:00 | Variable temporelle (dates/heures) |
| latest_tracked_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-12 00:00:00+00:00 à 2025-06-29 00:00:00+00:00 | Variable temporelle (dates/heures) |
| days_in_range_count | int64 | Continue | 0 | 0% | Min: 2.00, Max: 81.00, Moyenne: 55.40 | Variable numérique continue |
| active_days_count | int64 | Continue | 0 | 0% | Min: 2.00, Max: 81.00, Moyenne: 52.14 | Variable numérique continue |
| max_gap | int64 | Continue | 0 | 0% | Min: 0.00, Max: 21.00, Moyenne: 1.72 | Variable numérique continue |
| inactive_days_count | int64 | Continue | 0 | 0% | Min: 0.00, Max: 26.00, Moyenne: 3.26 | Variable numérique continue |
| tracking_completeness | float64 | Continue | 0 | 0% | Min: 0.55, Max: 1.00, Moyenne: 0.94 | Variable numérique continue |
| activated_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-03-19 13:42:59.499000+00:00 à 2025-05-07 09:49:52.503000+00:00 | Variable temporelle (dates/heures) |
| activated_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-03-19 13:42:59+00:00 à 2025-05-07 09:49:52+00:00 | Variable temporelle (dates/heures) |
| confirmed_days_count | int64 | Continue | 0 | 0% | Min: 0.00, Max: 81.00, Moyenne: 26.94 | Variable numérique continue |
| confirmed_tracks_count | int64 | Continue | 0 | 0% | Min: 0.00, Max: 1825.00, Moyenne: 290.58 | Variable numérique continue |
| created_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-03-19 13:42:33.338000+00:00 à 2025-05-07 09:46:28.316000+00:00 | Variable temporelle (dates/heures) |
| created_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-03-19 13:42:33+00:00 à 2025-05-07 09:46:28+00:00 | Variable temporelle (dates/heures) |
| first_tracked_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-07 15:33:51+00:00 à 2025-05-07 16:12:05+00:00 | Variable temporelle (dates/heures) |
| handset | object | Texte | 0 | 0% | 30 valeurs uniques | Variable texte libre |
| latest_confirmed_finished_at | datetime64[ns, UTC] | Temporelle | 9 | 18% | 2025-04-08 18:34:46+00:00 à 2025-06-30 08:09:17+00:00 | Variable temporelle (dates/heures) |
| latest_confirmed_finished_at_in_timezone | datetime64[ns, UTC] | Temporelle | 9 | 18% | 2025-04-08 18:34:46+00:00 à 2025-06-30 08:09:17+00:00 | Variable temporelle (dates/heures) |
| latest_data_transmission_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-12 12:35:53+00:00 à 2025-07-22 09:30:53+00:00 | Variable temporelle (dates/heures) |
| latest_data_transmission_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-12 12:35:53+00:00 à 2025-07-22 09:30:53+00:00 | Variable temporelle (dates/heures) |
| latest_storyline_finished_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-12 12:35:32+00:00 à 2025-06-30 17:03:50+00:00 | Variable temporelle (dates/heures) |
| latest_storyline_finished_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-12 12:35:32+00:00 à 2025-06-30 17:03:50+00:00 | Variable temporelle (dates/heures) |
| latest_tracked_at_in_timezone | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-12 12:35:32+00:00 à 2025-07-22 09:30:52+00:00 | Variable temporelle (dates/heures) |
| operating_system | object | Dummy/Binaire | 0 | 0% | ['ios', 'android'] | Variable binaire textuelle (ios/android ) |
| os_version | object | Catégorielle nominale | 0 | 0% | ['iOS 15.6.1', '15', '14', 'iOS 18.3.1', 'iOS 18.1.1', 'iOS 18.5', 'iOS 18.4.1', '10', 'iOS 15.8.4', 'iOS 18.3.2'] | Variable catégorielle (15 modalités) |
| tracks_count | int64 | Continue | 0 | 0% | Min: 13.00, Max: 1880.00, Moyenne: 546.66 | Variable numérique continue |
| unconfirmed_days_count | int64 | Continue | 0 | 0% | Min: 0.00, Max: 81.00, Moyenne: 25.16 | Variable numérique continue |
| unconfirmed_tracks_count | int64 | Continue | 0 | 0% | Min: 0.00, Max: 1062.00, Moyenne: 256.08 | Variable numérique continue |
| weight | float64 | Catégorielle ordinale | 0 | 0% | [np.float64(1.0)] | Variable catégorielle ordonnée (1 niveaux) |
| id_Gen | float64 | Continue | 5 | 10% | Min: 1001.00, Max: 20022.00, Moyenne: 5665.84 | Variable numérique continue |
| commune | object | Dummy/Binaire | 5 | 10% | ['Ville de Carouge', 'Ville de Genève'] | Variable binaire textuelle (Ville de Carouge/Ville de Genève ) |
| anciennete_commune | object | Catégorielle nominale | 5 | 10% | ['plus de 10 ans', '5 à 10 ans', '2 à 5 ans', 'moins de 2 ans'] | Variable catégorielle (4 modalités) |
| genre | object | Dummy/Binaire | 5 | 10% | ['F', 'H'] | Variable binaire textuelle (F/H ) |
| année_naissance | float64 | Continue | 5 | 10% | Min: 1952.00, Max: 1999.00, Moyenne: 1980.29 | Variable numérique continue |
| statut_pro | object | Catégorielle nominale | 6 | 12% | ['tps partiel', 'Retraité·e', 'tps plein', 'chômage', 'autre', 'Autres'] | Variable catégorielle (6 modalités) |
| abo_TP | object | Catégorielle nominale | 5 | 10% | ['Abonnement demi-tarif CFF', 'Abonnement général (AG) CFF', 'Abonnement de zone unireso TPG', 'Aucun', 'Abonnement de zone unireso TPG,  Abonnement demi-tarif CFF', ' Abonnement demi-tarif CFF', 'Abonnement de parcours CFF', 'Abonnement de zone unireso TPG,  Abonnement de parcours CFF', 'Abonnement demi-tarif CFF,  Abonnement de zone unireso TPG'] | Variable catégorielle (9 modalités) |
| possessions | object | Catégorielle nominale | 5 | 10% | ['Aucun', ' Vélo conventionnel (mécanique)', 'Vélo à assistance électrique', 'Vélo conventionnel (mécanique)', 'Vélo à assistance électrique, Vélo conventionnel (mécanique)'] | Variable catégorielle (5 modalités) |
| ouverture à l'évolution des pratiques modales | object | Dummy/Binaire | 5 | 10% | ['Oui tout à fait', 'Plutôt oui'] | Variable binaire textuelle (Oui tout à fait/Plutôt oui ) |
| structmen 65+ | float64 | Catégorielle ordinale | 5 | 10% | [np.float64(0.0), np.float64(1.0), np.float64(2.0)] | Variable catégorielle ordonnée (3 niveaux) |
| structmen 26-64 | float64 | Catégorielle ordinale | 5 | 10% | [np.float64(0.0), np.float64(1.0), np.float64(2.0)] | Variable catégorielle ordonnée (3 niveaux) |
| structmen 18-25 | float64 | Dummy/Binaire | 5 | 10% | [np.float64(0.0), np.float64(1.0)] | Variable binaire numérique (0.0/1.0 ) |
| structmen 16-17 | float64 | Dummy/Binaire | 5 | 10% | [np.float64(0.0), np.float64(1.0)] | Variable binaire numérique (0.0/1.0 ) |
| structmen 0-15 | float64 | Catégorielle ordinale | 5 | 10% | [np.float64(0.0), np.float64(1.0), np.float64(2.0), np.float64(3.0)] | Variable catégorielle ordonnée (4 niveaux) |
| equipement menage motos | float64 | Catégorielle ordinale | 5 | 10% | [np.float64(0.0), np.float64(1.0), np.float64(2.0), np.float64(3.0)] | Variable catégorielle ordonnée (4 niveaux) |
| freq_moto | object | Catégorielle nominale | 5 | 10% | ['Une à plusieurs fois par semaine', 'Moins d’une fois par semaine', 'Tous les jours ou presque'] | Variable catégorielle (3 modalités) |
| equipement menage voitures | float64 | Catégorielle ordinale | 5 | 10% | [np.float64(0.0), np.float64(1.0), np.float64(2.0)] | Variable catégorielle ordonnée (3 niveaux) |
| freq_voiture | object | Catégorielle nominale | 5 | 10% | ['Moins d’une fois par semaine', 'Une à plusieurs fois par semaine', 'Tous les jours ou presque'] | Variable catégorielle (3 modalités) |
| freq_accompagnements | object | Catégorielle nominale | 38 | 76% | ['Une à plusieurs fois par semaine', 'Moins d’une fois par semaine', 'Tous les jours ou presque'] | Variable catégorielle (3 modalités) |
| bad_signal_user | int64 | Catégorielle | 0 | 0% | [np.int64(0)] | Variable catégorielle (1 modalités) |

## Journey

| variable | dtype | type_variable | missing_count | missing_percent | modalites | description |
| --- | --- | --- | --- | --- | --- | --- |
| journey_id | object | Identifiant | 0 | 0% | 5151 identifiants uniques | Variable identifiant unique |
| user_id | object | Identifiant | 0 | 0% | 49 identifiants uniques | Variable identifiant unique |
| started_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-08 07:57:55+00:00 à 2025-06-28 22:29:27+00:00 | Variable temporelle (dates/heures) |
| finished_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-08 18:34:46+00:00 à 2025-06-29 10:10:34+00:00 | Variable temporelle (dates/heures) |
| main_mode | object | Texte | 0 | 0% | 22 valeurs uniques | Variable texte libre |
| modes | object | Texte | 0 | 0% | 1526 valeurs uniques | Variable texte libre |
| long_distance | bool | Dummy/Binaire | 0 | 0% | [np.False_, np.True_] | Variable binaire (True/False) |
| trips_count | Int64 | Autre | 0 | 0% | 27 valeurs uniques | Type de données: Int64 |
| tracks_count | Int64 | Autre | 0 | 0% | 51 valeurs uniques | Type de données: Int64 |
| tracks_count_without_walks | Int64 | Autre | 0 | 0% | 29 valeurs uniques | Type de données: Int64 |
| main_purpose | object | Catégorielle nominale | 2414 | 47% | ['Leisure', 'Shopping', 'Eat out', 'Errand', 'Family & Friends', 'Sports', 'Unknown', 'Waiting', 'Working', 'Medical visit'] | Variable catégorielle (13 modalités) |
| duration | Int64 | Autre | 0 | 0% | 4113 valeurs uniques | Type de données: Int64 |
| distance | Int64 | Autre | 0 | 0% | 3934 valeurs uniques | Type de données: Int64 |
| created_at | datetime64[ns, UTC] | Temporelle | 0 | 0% | 2025-04-09 07:49:26+00:00 à 2025-07-06 14:09:11+00:00 | Variable temporelle (dates/heures) |
| main_purpose_mrmt | object | Catégorielle nominale | 0 | 0% | ['Loisir', 'Tout motifs', 'Achat et services', 'Travail', 'Accompagnement', 'Formation'] | Variable catégorielle (6 modalités) |
| main_mode_niv1 | object | Catégorielle nominale | 0 | 0% | ['Marche', 'TP', 'Vélo', 'Voiture', 'Autres'] | Variable catégorielle (5 modalités) |
| modes_niv1 | object | Texte | 0 | 0% | 1252 valeurs uniques | Variable texte libre |
| journey_origin_canton | object | Catégorielle nominale | 0 | 0% | ['GE', 'VD', 'FR', 'ZH', 'HorsCHE', 'LU', 'VS', 'BE', 'TI', 'NE'] | Variable catégorielle (11 modalités) |
| journey_destination_canton | object | Catégorielle nominale | 0 | 0% | ['GE', 'VD', 'FR', 'ZH', 'HorsCHE', 'LU', 'VS', 'BE', 'TI', 'NE'] | Variable catégorielle (11 modalités) |
