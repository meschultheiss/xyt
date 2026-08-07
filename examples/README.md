# Exemples `xyt_gps`

Ces notebooks illustrent l'usage du package sur le jeu
`anonymous-test-set-gps`.

Le jeu de données n'est pas versionné dans le dépôt. Il est disponible sur
demande à `contact@situee.ch` pour tester le package. Après téléchargement,
placer le dossier dans l'un des emplacements suivants :

- `Data/Output/anonymous-test-set-gps` depuis la racine du dépôt ;
- `anonymous-test-set-gps` dans le dossier courant ;
- ou définir `XYT_GPS_DEMO_DATA_DIR=/chemin/vers/anonymous-test-set-gps`.

## Parcours recommandés

| Notebook | Rôle |
|---|---|
| `quickstart-analyse-gps.ipynb` | premier parcours court : charger un échantillon, lancer l'API recommandée, lire les indicateurs et produire une figure |
| `diagnostiquer-ses-donnees.ipynb` | contrôler les tables, colonnes, utilisateurs et erreurs fréquentes avant transformation |
| `demo-fil-rouge-pipeline.ipynb` | chemin complet : charger, configurer, transformer, contrôler, calculer les indicateurs et exporter |
| `demo-qualite-visualisations-test-set.ipynb` | participation, carte GPS, indicateurs HTML et motifs journaliers |

Le test set contient 12 utilisateurs anonymisés, trois labels d'expérience
génériques (`test-expe-1`, `test-expe-2`, `test-expe-3`) et une table
`sociodemographics` synthétique. Les questionnaires réels ne sont pas exportés.
