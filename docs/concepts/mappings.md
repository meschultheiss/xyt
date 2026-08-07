# Mappings modes et motifs

Les mappings transforment les valeurs fournies par le fournisseur GPS en catégories
analytiques utilisées dans les tables et les indicateurs.

Par défaut, `xyt_gps` produit plusieurs niveaux :

| Colonne produite | Usage |
|---|---|
| `mode_niv1` | regroupement simple : voiture, transports publics, marche, vélo, autres |
| `mode_niv2` | regroupement un peu plus détaillé : carsharing, TPU, train |
| `mode_mrmt` | nomenclature proche des usages MRMT : bus, train, tram/métro, vélo électrique, etc. |
| `purpose_niv1` | motifs d’activité issus de la `storyline` |
| `purpose_mrmt` | motifs de déplacement issus des tables `trips` et `journeys` |

Ces mappings sont définis par `mode_purpose_mapping()` et stockés dans
`ProjectConfig.mappings`. Ils peuvent être adaptés manuellement pour un projet.

```python
config = xyt.ProjectConfig(
    mappings=xyt.mode_purpose_mapping(),
)
```

## Modes de la storyline

La colonne source est généralement `mode`. Certains exports utilisent la forme
`Mode::...`.

| Valeur source | `mode_niv1` | `mode_niv2` | `mode_mrmt` |
|---|---|---|---|
| `Mode::Car` | Voiture | Voiture | Voiture |
| `Mode::Ridepooling` | Voiture | Voiture | Voiture |
| `Mode::Ecar` | Voiture | Voiture | Voiture |
| `Mode::Carsharing` | Voiture | Carsharing | Voiture |
| `Mode::TaxiUber` | Voiture | Voiture | Taxi |
| `Mode::Motorbike` | Autres | Autres | 2RM |
| `Mode::Bus` | TP | TPU | Bus |
| `Mode::Coach` | TP | TPU | Train |
| `Mode::Cablecar` | TP | TPU | Bus |
| `Mode::Tram` | TP | TPU | Tram/Métro |
| `Mode::Subway` | TP | TPU | Tram/Métro |
| `Mode::LightRail` | TP | TPU | Train |
| `Mode::Train` | TP | Train | Train |
| `Mode::RegionalTrain` | TP | Train | Train |
| `Mode::Boat` | TP | TPU | Bateau |
| `Mode::Ferry` | TP | TPU | Bateau |
| `Mode::Walk` | Marche | Marche | Marche |
| `Mode::Hiking` | Marche | Marche | Marche |
| `Mode::Scooter` | Marche | Marche | Marche |
| `Mode::Bicycle` | Vélo | Vélo | Vélo conventionnel |
| `Mode::Bikesharing` | Vélo | Vélo | Vélo conventionnel |
| `Mode::Ebicycle` | Vélo | Vélo | Vélo électrique |
| `Mode::Other` | Autres | Autres | Autres |
| `Mode::Airplane` | Autres | Autres | Airplane |

## Modes des trips et journeys

Dans les tables `trips` et `journeys`, les valeurs source sont généralement des
libellés plus lisibles que dans la `storyline`.

| Valeur source | `mode_niv1` | `mode_niv2` | `mode_mrmt` |
|---|---|---|---|
| `Car` | Voiture | Voiture | Voiture |
| `Electric car` | Voiture | Voiture | Voiture |
| `Ridepooling` | Voiture | Voiture | Voiture |
| `Carsharing` | Voiture | Carsharing | Voiture |
| `Taxi/Uber` | Voiture | Voiture | Taxi |
| `Motorbike` | Voiture | Voiture | 2RM |
| `Bus` | TP | TPU | Bus |
| `Coach` | TP | TPU | Bus |
| `Cable car` | TP | TPU | Bus |
| `Tram` ou `Tramway` | TP | TPU | Tram/Métro |
| `Subway` | TP | TPU | Tram/Métro |
| `LightRail` | TP | TPU | Tram/Métro |
| `Train` | TP | Train | Train |
| `Regional train` ou `Regional Train` | TP | Train | Train |
| `Rapid transit railway` | TP | Train | Train |
| `Boat` | TP | TPU | Bateau |
| `Ferry` | TP | TPU | Bateau |
| `Walking` | Marche | Marche | Marche |
| `Hiking` | Marche | Marche | Marche |
| `Scooter` | Marche | Marche | Marche |
| `Bicycle` | Vélo | Vélo | Vélo conventionnel |
| `Bikesharing` | Vélo | Vélo | Vélo conventionnel |
| `E-Bike` ou `Ebicycle` | Vélo | Vélo | Vélo électrique |
| `Other` | Autres | Autres | Autres |
| `Airplane` | Autres | Autres | Airplane |

## Motifs de la storyline

La colonne source est généralement `purpose`.

| Valeur source | `purpose_niv1` |
|---|---|
| `home` | Domicile |
| `work`, `study` | Travail |
| `leisure`, `eat`, `family_friends`, `sport` | Loisir |
| `shopping`, `errand` | Achat |
| `wait` | Attente |
| `medical_visit`, `assistance` | Santé |
| `other`, `unknown`, valeur manquante | Autre |

## Motifs des trips et journeys

| Valeur source | `purpose_mrmt` |
|---|---|
| `At home` | Domicile |
| `Working` | Travail |
| `Education` | Formation |
| `Leisure`, `Eat out`, `Family & Friends`, `Sports` | Loisir |
| `Shopping`, `Errand`, `Medical visit` | Achat et services |
| `Drop off / pick up someone` | Accompagnement |
| `Other`, `Unknown`, `unknown`, `Waiting`, valeur manquante | Tout motifs |

## Adapter un mapping

Un projet peut modifier un niveau sans réécrire toute la configuration. Par
exemple, pour regrouper marche et vélo dans une catégorie de mobilité active :

```python
mappings = xyt.mode_purpose_mapping(
    storyline_mode_niv1={
        "Mobilité active": ("Mode::Walk", "Mode::Hiking", "Mode::Scooter", "Mode::Bicycle", "Mode::Bikesharing", "Mode::Ebicycle"),
        "Voiture": ("Mode::Car", "Mode::Carsharing", "Mode::Ridepooling", "Mode::Ecar", "Mode::TaxiUber"),
        "TP": ("Mode::Train", "Mode::RegionalTrain", "Mode::Bus", "Mode::Tram", "Mode::Subway"),
        "Autres": ("Mode::Other", "Mode::Airplane"),
    },
)
```

Une modification de mapping peut changer les indicateurs. Elle doit donc être
visible dans la configuration du projet ou dans le notebook qui produit les
exports.
