# Contexte du dump Déclic préfiguration

!!! note "Page de maintenance"
    Cette page documente un contexte source du projet Déclic. Elle n'est pas nécessaire pour utiliser `xyt_gps` sur d'autres données GPS.

Date : 2026-06-09

Source de contexte : `Data/dumps_situee/RENDU250815_gps_prefiguration_clean/_README.txt`.

Ce document reprend uniquement les éléments de contexte utiles à la construction du package. Les listes nominatives ou identifiantes d’utilisateurs remarquables ne sont pas recopiées ici.

## Export livré

| Élément | Information |
|---|---|
| Date de création | 2025-08-15 15:03:57 |
| Projet | Action Située - Dynamite 2025 |
| Notebook source historique | `/Notebooks/260618_Archives/WP2/export_data.ipynb` |
| Période des données | 2025-04-07 à 2025-06-29 |
| Utilisateurs uniques | 50 |
| Legs | 27 334 |
| Trips | 12 344 |
| Staypoints | 19 172 |
| Journeys | 5 151 |
| Variables documentées | 174 |

## Tables exportées

| Fichier | Contenu | Format |
|---|---|---|
| `legs.geojson` | étapes de déplacement | GeoJSON |
| `staypoints.geojson` | points d’arrêt | GeoJSON |
| `trips.csv` | déplacements | CSV |
| `journey.csv` | journées | CSV |
| `user_stats.csv` | statistiques utilisateurs | CSV |
| `occupancy_co2.csv` | émissions CO2 et occupation | CSV |
| `legs_info.csv` | étapes map-matchées | CSV |
| `public_transport.csv` | étapes de transport public | CSV |
| `dictionnaire_variables.xlsx` | documentation structurée des variables | XLSX |

## Phases Déclic observées

| Phase | Début | Fin | Durée inclusive |
|---|---|---|---|
| Phase 1 | 2025-04-12 | 2025-05-02 | 21 jours |
| Phase 2 | 2025-05-03 | 2025-06-02 | 31 jours |
| Phase 3 | 2025-06-04 | 2025-06-28 | 25 jours |

## Implications pour `xyt_gps`

- Les géométries sont exportées en `EPSG:4326`, mais les opérations spatiales doivent être faites en `EPSG:2056`.
- Les dates doivent être traitées comme timestamps UTC, avec conservation des colonnes locales si elles existent.
- Les identifiants doivent rester stables pour les jointures entre `legs`, `trips`, `journey`, `staypoints`, `occupancy_co2`, `legs_info` et `public_transport`.
- Le package doit garder le dictionnaire des variables accessible sous forme Markdown versionnée.
- Les utilisateurs à suivi court sont un cas de qualité de données, pas une règle cachée : le seuil doit rester configurable.
