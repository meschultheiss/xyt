# Qualité et nettoyage spatial

Cette page décrit les contrôles qualité et les enrichissements spatiaux disponibles après la préparation des tables de mobilité.

Les notebooks historiques de qualité du suivi et de nettoyage spatial ont servi de base, mais les fonctions du package utilisent des noms génériques.

L'objectif n'est pas d'ajouter une grande fonction opaque. Ces étapes restent séparées, car elles servent surtout à contrôler, documenter et décider si certaines données doivent être filtrées ou enrichies avant les indicateurs.

## Qualité du suivi

Le contrôle qualité porte sur trois niveaux :

| Niveau | Fonctions | Sortie |
|---|---|---|
| suivi journalier | `build_user_day_coverage()`, `build_tracking_gap_report()`, `build_weekly_participation_grid()` | jours de plage, jours trackés, jours actifs, jours manquants, semaines de participation |
| qualité utilisateur | `build_tracking_quality_report()`, `build_user_selection_table()` | règles de sélection explicites |
| contrôles de cohérence | `summarize_leg_lengths_by_mode()`, `get_extreme_legs_by_mode()`, `build_user_confirmation_rates()`, `build_mode_detection_precision()` | tableaux de diagnostic |

Exemple minimal :

```python
tracking_gap_report = xyt.build_tracking_gap_report(
    mobility.storyline,
    config,
    user_ids=mobility.user_stats["user_id"],
)

length_summary = xyt.summarize_leg_lengths_by_mode(mobility.legs)
extreme_legs = xyt.get_extreme_legs_by_mode(mobility.legs, top_n=5)
confirmation_rates = xyt.build_user_confirmation_rates(mobility.storyline)
mode_precision = xyt.build_mode_detection_precision(mobility.storyline)

selection_table = xyt.build_user_selection_table(
    mobility.user_stats,
    require_tracking_quality=False,
    exclude_bad_signal_users=True,
    max_low_quality_legs_share=0.25,
)
```

Ces fonctions ne suppriment pas de lignes. Elles produisent des rapports inspectables. Le filtrage effectif doit rester une décision explicite :

La qualité de suivi et la qualité GPS doivent rester distinguées. Le filtre
strict `require_tracking_quality=True` utilise uniquement les phases déclarées
dans `ProjectConfig`. Si la config contient seulement `Phase1` et `Phase2`, la
sélection ne demandera pas de `Phase3`. En revanche, si une phase est déclarée
mais que le dump ne la couvre pas, `tracking_quality_ok` peut être faux pour de
bonnes raisons calendaires. Dans ce cas, filtrer d'abord le signal GPS et garder
l'assiduité comme diagnostic.

```python
analysis_users = xyt.select_analysis_users(selection_table)
analysis_dataset = xyt.filter_mobility_dataset_by_users(mobility, analysis_users)
```

## Nettoyage spatial optionnel

Le nettoyage spatial est optionnel, car il dépend des données déjà livrées par le fournisseur et des ressources territoriales disponibles.

| Besoin | Fonctions | Paramètres importants |
|---|---|---|
| normaliser les géométries de legs | `clean_leg_geometries()` | `drop_discontinuous` |
| calculer la perte de signal GPS | `add_signal_quality_flags()` | `SpatialQualityThresholds` |
| identifier les utilisateurs à mauvais signal | `add_signal_quality_to_user_stats()` | `bad_signal_user_quantile` |
| labelliser des points d'activité | `add_spatial_zone_labels()` | `zones`, `zone_label_col`, `output_col` |
| labelliser origine et destination des legs | `add_leg_origin_destination_zones()` | `origin_col`, `destination_col` |
| classer un leg par rapport à une zone | `classify_leg_relation_to_area()` | `area`, `area_path`, `center`, `radius_m` |
| propager les zones aux trips | `add_trip_origin_destination_from_legs()` | table `map_track_trip_journey` |
| propager les zones aux journeys | `add_journey_origin_destination_from_trips()` | table `map_track_trip_journey` |

Exemple minimal :

```python
legs = xyt.clean_leg_geometries(mobility.legs)
legs = xyt.add_signal_quality_flags(legs, config)

staypoints = xyt.add_spatial_zone_labels(
    mobility.staypoints,
    canton_zones,
    zone_label_col="N_KT",
    output_col="activity_canton",
)

legs = xyt.add_leg_origin_destination_zones(
    legs,
    canton_zones,
    zone_label_col="N_KT",
    origin_col="origin_canton",
    destination_col="destination_canton",
)

trips = xyt.add_trip_origin_destination_from_legs(
    mobility.trips,
    legs,
    mobility.map_track_trip_journey,
    leg_origin_col="origin_canton",
    leg_destination_col="destination_canton",
    output_origin_col="trip_origin_canton",
    output_destination_col="trip_destination_canton",
)
```

## Décision méthodologique

Ces étapes doivent rester paramétrables :

- les seuils de perte de signal peuvent varier selon le mode ;
- les zones territoriales dépendent du projet ;
- une donnée déjà map-matchée ou nettoyée peut désactiver certains contrôles ;
- les rapports doivent être exportés avant tout filtrage définitif.

Le notebook `Notebooks/011_quality_check.ipynb` sert à inspecter la qualité. Le notebook `Notebooks/020_spatial_cleaning.ipynb` sert à appliquer les enrichissements spatiaux optionnels.

Dans la configuration Déclic, `020_spatial_cleaning.ipynb` utilise les zones
communales GIREC du Grand Genève. La clé `ID_Zone` est ajoutée aux départs et
arrivées des legs, puis propagée aux trips et journeys. Les centroïdes
`Nodes_commune.geojson` sont exportables avec la base propre pour préparer des
matrices origine-destination et leur visualisation cartographique.
