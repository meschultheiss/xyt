# Données de démonstration

Le package peut être testé avec le jeu `anonymous-test-set-gps`.
Ce dossier n'est pas versionné dans le dépôt public. Il est disponible sur
demande à `contact@situee.ch`.

Dans le dépôt de travail, ce dossier est régénéré par
`Notebooks/002_build_anonymous_test_set.ipynb` à partir des exports landed
`anonymized_altered`.

## Installation locale du test set

Après réception du dossier, deux options sont possibles.

Option 1 : le placer dans l'arborescence attendue par les notebooks d'exemple.

```text
Data/Output/anonymous-test-set-gps/
```

Option 2 : laisser le dossier ailleurs et définir une variable d'environnement.

```bash
export XYT_GPS_DEMO_DATA_DIR=/chemin/vers/anonymous-test-set-gps
```

## Contenu

Le dossier contient les tables suivantes en CSV et Parquet lorsque le format est
pertinent :

| Table | Rôle |
|---|---|
| `storyline` | événements GPS anonymisés : tracks et stays |
| `trips` | déplacements agrégés |
| `journeys` | chaînes de déplacements |
| `user_statistics` | statistiques de suivi par utilisateur |
| `public_transport_legs` | étapes de transport public disponibles |
| `user_expe` | présence des utilisateurs dans les labels d'expérience de test |
| `sociodemographics` | table synthétique : âge, structure du ménage, nombre de voitures et CSP |

Le test set contient 12 utilisateurs anonymisés. Les labels d'expérience sont
`test-expe-1`, `test-expe-2` et `test-expe-3`. Les questionnaires réels ne sont
pas exportés ; les variables sociodémographiques sont générées aléatoirement
pour les tutoriels.

## Notebooks associés

Les notebooks prêts à l'emploi sont dans `examples/` :

- `quickstart-analyse-gps.ipynb` ;
- `diagnostiquer-ses-donnees.ipynb` ;
- `demo-fil-rouge-pipeline.ipynb` ;
- `demo-qualite-visualisations-test-set.ipynb`.
