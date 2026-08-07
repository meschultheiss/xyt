# Hypothèses et seuils de traitement

Date : 2026-08-03

Source de contexte : `Data/dumps_situee/RENDU250815_gps_prefiguration_clean/seuils_hypotheses.xlsx`.

Ce document liste les hypothèses et seuils formalisés dans le précédent export livré. Les valeurs chiffrées reprises dans `ProjectConfig` sont intégrées seulement lorsqu’elles sont explicites. Les seuils indiqués comme paramétrables restent à définir projet par projet.

Le traitement réseau reste expérimental. Le client GraphHopper est conservé,
mais l'essai hébergé n'est pas exploitable avec le free plan testé. Le notebook
`021` utilise désormais Google Roads/Routes sur un échantillon de legs voiture
candidats au traitement réseau.
Aucune validation complète sur les traces réelles n'est encore documentée. Les
seuils ci-dessous décrivent le comportement à évaluer; ils ne constituent pas
des paramètres de production validés. Le quickstart désactive les appels.

## Table source

| Domaine | Paramètre | Valeur / règle | Impact | Réglage / source |
| --- | --- | --- | --- | --- |
| CRS & géo | Système d’analyse | EPSG:2056 | Distances/vitesses métriques fiables | Config projets / tous notebooks |
| CRS & géo | Système d’export | EPSG:4326 | Compatibilité GeoJSON / web | export_data.ipynb |
| Temps | Format temporel | ISO-8601 + UTC + colonnes locales | Agrégations quotidiennes, mapping temporel | préparation des dates |
| Qualité spatiale | Filtre points aberrants (dist. consécutive max) | Seuil paramétrable (m) | Nettoyage sauts GPS | nettoyage spatial |
| Qualité spatiale | Perte de signal | relative_signal_loss(0–1) | Flag legs à faible qualité | qualité GPS |
| Outliers | Seuils longueur/vitesse | Quantiles Q98 & Q99par mode | Flags extreme98_*, extreme99_* | préparation des legs |
| Intégrité | Mapping legs↔trips↔journeys | Chevauchement horaire + tolérance | Réaffectation robuste des IDs | tables de correspondance |
| Suivi minimal | Utilisateur à écarter/pondérer | < 7 jours de données | Qualité échantillon | README source / indicateurs |
| Traitement Google `021` | Sélection | legs `Mode::Car` et `Mode::Ecar` uniquement | Concentrer le test sur les traces voiture les plus structurantes | `GOOGLE_CAR_MODES` |
| Google Roads | Découpage | 100 points par requête, avec deux points de chevauchement | Respecter la limite Roads API et traiter les longues traces | `GOOGLE_MAPS_MAX_POINTS_PER_ROADS_REQUEST` |
| Map matching | Validation | longueur 0,50–1,75, extrémités, couverture, forme dense et boucles | Rejeter les résultats incompatibles | paramètres de `020` et `021` |
| CO₂ | Facteurs d’émission par mode | MOBITOOL (gCO₂e/km) | Calcul émissions par leg | enrichissement CO2 |
| CO₂ | Taux d’occupation | Par mode (jeu interne) | Ajustement des émissions | enrichissement CO2 |
| Admin | Localisation cantonale | origin_canton / destination_canton | Agrégations territoriales | Ajout pendant la préparation spatiale ou l’enrichissement |
| TP | Stations départ/arrivée | Recodage des libellés | Cohérence analyses TP | PublicTransportLegs + recode |

## Traduction dans la configuration `xyt_gps`

| Sujet | Paramètre package | Valeur actuelle | Commentaire |
|---|---|---|---|
| CRS d’analyse | `ProjectConfig.operations_crs` | `EPSG:2056` | Utilisé pour les distances et opérations métriques en Suisse. |
| CRS d’export | `ProjectConfig.target_crs` | `EPSG:4326` | Compatible GeoJSON et usages web. |
| Fuseau horaire | `ProjectConfig.timezone` | `Europe/Zurich` | Les timestamps bruts restent lus en UTC, avec colonnes locales si disponibles. |
| Suivi minimal utilisateur | `TrackingThresholds.min_total_tracked_days` | `7` | Seuil du README livré : les suivis `<7 jours` sont à écarter ou pondérer. |
| Seuils par phase | `TrackingThresholds.min_days_by_phase` | `{}` | À définir selon le protocole de chaque expérimentation. |
| Points GPS aberrants | `SpatialQualityThresholds.max_consecutive_point_distance_m` | `None` | Le fichier source mentionne un seuil paramétrable en mètres, sans valeur chiffrée. |
| Perte de signal | `SpatialQualityThresholds.signal_loss_thresholds_by_level` | seuils historiques par mode | Les seuils restent configurables par projet. |
| Mauvais signal utilisateur | `SpatialQualityThresholds.bad_signal_user_quantile` | `0.995` | Utilisateurs dont la perte de signal moyenne est très élevée relativement aux autres utilisateurs. |
| Outliers longueur/vitesse | `SpatialQualityThresholds.outlier_quantiles_by_mode` | `(0.98, 0.99)` | Flags Q98 et Q99 par mode. |
| Matching legs/trips/journeys | `MatchingThresholds.leg_trip_journey_tolerance` | `5s` | Valeur par défaut actuelle, à valider si un projet utilise une autre convention. |
| Points Google Roads | `GOOGLE_MAPS_MAX_POINTS_PER_ROADS_REQUEST` | `100` | Limite de Roads API `snapToRoads`. |
| Waypoints Google Routes | `GOOGLE_MAPS_MAX_INTERMEDIATE_WAYPOINTS` | `10` | Reste sous le seuil de 11 waypoints qui peut changer la catégorie de facturation. |
| Plafond Google `021` | `GOOGLE_MAPS_MAX_REQUESTS` | `100` par défaut local | Vérifié avant l'envoi de l'échantillon. |
| Validation map-matching | `MAP_MATCH_MIN_LENGTH_RATIO`, `MAP_MATCH_MAX_LENGTH_RATIO` dans `020` | `0.50`, `1.75` | Une réponse hors intervalle est rejetée et la géométrie originale est conservée. |
| Couverture map-matching | tolérance et part minimale dans `020` | `100 m`, `0.80` | Au moins 80 % des sommets observés doivent rester proches du résultat. |
| Extrémités map-matching | déplacement maximal dans `020` | `250 m` | Limite le déplacement des origines et destinations. |
| Facteurs CO2 | `Notebooks/config/reference/co2_factors.csv` lu par `CO2OccupancyConfig` | CSV projet | Facteurs par mode en g/km. Les modes non strictement identiques sont rapprochés par proxy explicite et documenté dans la table. |
| Occupation automobile | `Notebooks/config/reference/occupancy_rates.csv` lu par `CO2OccupancyConfig` | CSV projet | Taux d'occupation par classe de distance et motif, appliqué aux modes voiture par défaut. |
| Santé | `Notebooks/config/reference/metabolic_equivalent_tasks.csv` lu par `HealthConfig` | CSV projet | METs par mode et intensité. Les intensités actives sont définies par `min_speed_kmh` et `max_speed_kmh`. `weight_kg=70` et `step_length_m=0.75` restent des paramètres de calcul explicites. |

## Points à confirmer

- Valeur numérique du seuil de distance entre points GPS consécutifs.
- Robustesse métier des seuils absolus de perte de signal. Certaines valeurs historiques valent `1` mètre alors que les commentaires des notebooks évoquent des distances plus élevées ; dans la pratique, le seuil relatif devient alors le critère déterminant.
- Tolérance exacte utilisée historiquement pour le mapping `legs` ↔ `trips` ↔ `journeys` si elle diffère de `5s`.
- Liste officielle des facteurs CO2, taux d’occupation et METs à valider pour une version de production scientifique. Les valeurs actuelles reprennent les hypothèses historiques du projet, stockées dans `Notebooks/config/reference/`.
- Règles de recodage des stations de transport public.
- Validation empirique des rapports de longueurs 0,50 et 1,75 sur un échantillon de legs, notamment les boucles et les fortes pertes de signal.

## Seuils de perte de signal

Les seuils utilisent deux critères combinés :

- PSA : plus grande perte de signal absolue entre deux points consécutifs, en mètres ;
- PSR : perte de signal relative, soit `PSA / length_leg`, stockée dans `relative_signal_loss`.

La colonne de sortie pour la perte de signal absolue est `max_signal_loss_m`.

Un leg est flaggé lorsque les deux seuils sont dépassés pour son groupe de modes.

| Niveau | Groupe | Modes | PSA | PSR |
|---|---|---|---:|---:|
| 1 | road | Car, Motorbike, Bus, Tram, Subway, KickScooter, Other, TaxiUber, Carsharing, Ecar | 1 | 0.60 |
| 1 | rail | Train, RegionalTrain, LightRail | 2000 | 0.70 |
| 1 | cycling | Bicycle, Ebicycle | 1 | 0.65 |
| 1 | walking | Walk | 1 | 0.95 |
| 1 | plane_boat | Boat, Airplane | 100000000 | 1.00 |
| 2 | road | Car, Motorbike, Bus, Tram, Subway, KickScooter, Other, TaxiUber, Carsharing, Ecar | 1 | 0.45 |
| 2 | rail | Train, RegionalTrain, LightRail | 1500 | 0.55 |
| 2 | cycling | Bicycle, Ebicycle | 1 | 0.50 |
| 2 | walking | Walk | 1 | 0.75 |
| 2 | plane_boat | Boat, Airplane | 100000000 | 1.00 |

Ces valeurs sont des valeurs de compatibilité avec le traitement historique, pas une vérité générale sur la qualité GPS. Elles doivent pouvoir être ajustées dans `ProjectConfig` pour une autre expérimentation.

## Enrichissements CO2 et santé

Les enrichissements transversaux sont calculés au niveau du leg avant agrégation :

- `add_co2_occupancy_metrics()` ajoute `occupancy_rate`, les facteurs CO2 par km et les émissions totales `co2_g` et `co2_direct_g` ;
- `add_health_metrics()` ajoute `speed_kmh`, `step_count`, `intensity`, les bornes de vitesse de l'intensité, `mets` et `calories_burned` ;
- `compute_mobility_indicators()` agrège ensuite ces métriques en tables personne-jour, personne-phase et population.

Le calcul d’indicateurs expose les paramètres qui changent directement l’interprétation :

- `include_zero_days=True` inclut les jours suivis sans déplacement dans les moyennes ;
- `include_excursions=True` conserve les legs et trips marqués comme excursions ;
- `include_airplane=False` exclut par défaut les étapes et déplacements avion afin d'éviter qu'ils dominent les distances, émissions CO2 et profils de demande ;
- `use_weights=True` utilise la pondération utilisateur si `user_stats` contient la colonne définie par `weight_col`.

`plot_indicator_bars()` reprend ces informations dans l’export HTML pour éviter qu’un graphe soit lu sans connaître son périmètre.

Les facteurs CO2 historiques sont exprimés en g/km et stockés dans `Notebooks/config/reference/co2_factors.csv`. Les colonnes `co2_g` et `co2_direct_g` sont des émissions totales par leg, calculées à partir de la distance. Pour les modes voiture, le facteur est divisé par le taux d'occupation lorsque `apply_occupancy_to_car_modes=True`.

Le taux d'occupation est recalculé à partir de `Notebooks/config/reference/occupancy_rates.csv`, selon la distance et le motif, car la colonne source `occupancy` peut être absente ou très incomplète. Si cette colonne est jugée exploitable pour un projet, `add_co2_occupancy_metrics(..., prefer_observed_occupancy=True, occupancy_col="occupancy")` utilise la valeur observée quand elle est positive et utilise sinon la valeur calculée. La colonne `occupancy_source` indique alors `observed`, `computed` ou `computed_fallback`.

Les intensités santé sont traduites à partir de `speed_kmh`, calculée comme
`distance / durée`. Dans `metabolic_equivalent_tasks.csv`, les bornes
`min_speed_kmh` et `max_speed_kmh` définissent les classes. La borne basse est
incluse, la borne haute est exclue, et une borne haute vide signifie “et plus”.
Avec les valeurs actuelles, la marche est classée en `calme` de 0 à 3 km/h,
`moderee` de 3 à 5 km/h, puis `intense` à partir de 5 km/h. Les colonnes
`intensity_min_speed_kmh` et `intensity_max_speed_kmh` sont ajoutées aux legs
enrichis pour garder cette règle traçable.

Pour remplacer les références sans modifier le package :

```python
co2_config = xyt.CO2OccupancyConfig.from_reference_files(
    co2_factors_path="Notebooks/config/reference/co2_factors.csv",
    occupancy_rates_path="Notebooks/config/reference/occupancy_rates.csv",
)
health_config = xyt.HealthConfig.from_reference_files(
    metabolic_equivalent_tasks_path="Notebooks/config/reference/metabolic_equivalent_tasks.csv",
)
```

Certains modes MotionTag rares sont rapprochés d'un mode proxy afin d'éviter des facteurs manquants :

| Mode source | Proxy actuel |
|---|---|
| `Mode::Ridepooling`, `Mode::CarPassenger` | voiture |
| `Mode::Scooter` | trottinette / kick scooter |
| `Mode::EcargoBike` | vélo électrique |
| `Mode::Ferry` | bateau |
| `Mode::Coach` | bus |
| `Mode::Cablecar` | tram / électrique |
| `Mode::Hiking`, `Mode::Ski` | 0 g/km direct dans l'hypothèse actuelle |

Ces proxies doivent être explicitement revus si les indicateurs CO2 deviennent un livrable scientifique central.

## Périmètre des phases

Quand des phases expérimentales sont configurées, les lignes situées hors périodes connues sont conservées avec `phase = "Other"`. Elles sont utiles pour auditer le périmètre du dump GPS, mais elles ne doivent pas être mélangées automatiquement aux indicateurs principaux du dashboard.

Le notebook `Notebooks/030_enrich_indicators.ipynb` applique actuellement :

```python
ANALYSIS_PHASES = ["Phase1", "Phase2", "Phase3"]
```

Les lignes `Other` sont exportées dans `qa_phase_coverage.csv`, mais exclues des indicateurs principaux.
