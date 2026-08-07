# Notebooks qualité et spatial

Après `Notebooks/010_data_loading.ipynb`, deux notebooks prolongent la préparation des données par des contrôles qualité et spatiaux.

## Qualité de suivi

Notebook :

```text
Notebooks/011_quality_check.ipynb
```

Il produit :

- un rapport séparant jours trackés, jours actifs, jours manquants et jours consécutifs ;
- une grille de participation hebdomadaire ;
- un résumé des longueurs par mode ;
- les legs extrêmes par mode ;
- les taux de confirmation des stays et tracks ;
- la précision des modes détectés si les colonnes sont disponibles ;
- une table de sélection utilisateur inspectable.

## Nettoyage spatial

Notebook :

```text
Notebooks/020_spatial_cleaning.ipynb
```

Il applique, de façon optionnelle :

- le nettoyage des géométries de legs ;
- les flags de perte de signal GPS ;
- les labels territoriaux des activités ;
- les zones d'origine et de destination des legs ;
- la classification `intra`, `extra`, `exchange` par rapport à une zone ;
- la propagation des origines-destinations vers trips et journeys.

Les couches territoriales ne sont pas incluses dans le package. Elles doivent être fournies par le projet, par exemple sous forme de GeoJSON ou de GeoPackage.
Dans la chaîne Déclic, le notebook utilise `Zones_commune.geojson` et
`Nodes_commune.geojson` du dossier `Data/dumps_border/grandgeneve-matricesod-girec`
pour associer un `ID_Zone` aux départs et arrivées.
