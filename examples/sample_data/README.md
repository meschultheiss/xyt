# Données sample locales

Ce dossier est l'emplacement attendu pour les données de démonstration locales
utilisées par `xyt.load_sample_gps()`.

Fichier attendu :

```text
sample_data_marced_motiontag.pkl
```

Le fichier `.pkl` reste volontairement ignoré par git. Il peut être utilisé en
local pour apprendre le package ou générer des données synthétiques, mais il ne
doit pas être ajouté au dépôt.

Le code actif ne dépend pas des archives historiques. Si le fichier sample n'est
pas présent ici, fournir un chemin explicite à `load_sample_gps(path=...)` ou
utiliser les scripts de génération de test set documentés dans MkDocs.
