# Notebooks demo

Les notebooks de démonstration utilisent le jeu `anonymous-test-set-gps`.
Ce jeu n'est pas versionné dans le dépôt. Il est disponible sur demande à
`contact@situee.ch` pour tester le package.

Après téléchargement, placer le dossier dans `Data/Output/anonymous-test-set-gps`
depuis la racine du dépôt, ou définir :

```bash
export XYT_GPS_DEMO_DATA_DIR=/chemin/vers/anonymous-test-set-gps
```

## Notebooks de démonstration

| Notebook | Rôle |
|---|---|
| `examples/quickstart-analyse-gps.ipynb` | premier parcours court : charger un échantillon, préparer les tables, calculer les indicateurs et produire une figure |
| `examples/diagnostiquer-ses-donnees.ipynb` | vérifier les tables, colonnes, utilisateurs et erreurs fréquentes avant transformation |
| `examples/demo-fil-rouge-pipeline.ipynb` | charger le test set, préparer un `MobilityDataset`, contrôler les utilisateurs, calculer les indicateurs et exporter une base propre |
| `examples/demo-qualite-visualisations-test-set.ipynb` | produire les contrôles qualité, la carte GPS, les indicateurs HTML et les motifs journaliers |

Le test set contient 12 utilisateurs anonymisés, trois labels d'expérience
génériques (`test-expe-1`, `test-expe-2`, `test-expe-3`) et une table
`sociodemographics` synthétique. Les questionnaires réels ne sont pas exportés.

## Chaîne de production

Pour reproduire la chaîne projet sur des données préparées, utiliser la
séquence suivante :

```text
Notebooks/000_data_landing.ipynb
Notebooks/010_data_loading.ipynb
Notebooks/011_quality_check.ipynb
Notebooks/020_spatial_cleaning.ipynb
Notebooks/030_enrich_indicators.ipynb
Notebooks/040_export_cleaned_dataset.ipynb
```

| Notebook | Rôle | Sortie principale |
|---|---|---|
| `000_data_landing.ipynb` | homogénéiser les sources d'une expérience, vérifier les colonnes, produire les manifests et `user_presence` | `Data/Output/0-landed-data/<EXPERIMENT_NAME>/{complete,anonymized_altered}` |
| `010_data_loading.ipynb` | parser les dates et géométries, construire `legs`, `staypoints`, `trips`, `journeys`, `user_stats`, mappings et participation | `Data/Output/2-transformed-data/<EXPERIMENT_NAME>` |
| `011_quality_check.ipynb` | contrôler la qualité de suivi et préparer la sélection utilisateurs | rapports qualité |
| `020_spatial_cleaning.ipynb` | ajouter les contrôles et enrichissements spatiaux requis par les étapes suivantes | tables spatial-cleaned |
| `030_enrich_indicators.ipynb` | ajouter CO2/santé, calculer les indicateurs et produire les figures HTML | `Data/Output/3-enriched-data/<EXPERIMENT_NAME>` |
| `040_export_cleaned_dataset.ipynb` | consolider uniquement les tables propres nécessaires aux analyses thématiques | `Data/Output/4-clean-data/<EXPERIMENT_NAME>/cleaned-base` |

Les notebooks de démonstration ne remplacent pas cette chaîne de production :
ils montrent l'API package sur un jeu public réduit. Les notebooks de production
restent le bon support lorsque les choix méthodologiques doivent être inspectés
étape par étape.
