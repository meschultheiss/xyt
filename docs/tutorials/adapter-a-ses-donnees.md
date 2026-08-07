# Adapter le package à ses données

Cette page décrit le passage du test set anonymisé à un projet réel. Elle sert
de checklist avant de lancer les notebooks de production ou l'API
`xyt_gps` dans un script.

## 1. Stabiliser les fichiers d'entrée

Le package attend des tables GPS structurées autour de quatre objets :

| Table | Obligatoire | Rôle |
|---|---:|---|
| `storyline` | oui | événements GPS : stays et tracks |
| `trips` | oui | déplacements agrégés fournis par la source |
| `journeys` | oui | chaînes de déplacements |
| `user_statistics` | recommandé | suivi utilisateur et dates de couverture |
| `sociodemographics` | optionnel | variables individuelles jointes par `user_id` |

Avant de transformer les données, lancer un contrôle de colonnes :

```python
report = xyt.check_raw_import_columns(
    storyline,
    user_statistics,
    trips=trips,
    journeys=journeys,
)
report.query("status == 'missing_required'")
```

Une colonne obligatoire manquante doit être corrigée dans la donnée source, dans
le landing ou dans le mapping de colonnes. Elle ne doit pas être compensée par un
fallback implicite.

## 2. Définir explicitement le projet

Les choix méthodologiques doivent être visibles dans `ProjectConfig` :

```python
config = xyt.ProjectConfig(
    experiment_name="mon-projet",
    motiontag_project_name="mon-export-source",
    start_expe="2026-04-01",
    end_expe="2026-06-30",
    phases=(
        xyt.Phase("Phase1", "2026-04-01", "2026-04-21"),
        xyt.Phase("Phase2", "2026-04-22", "2026-05-19"),
    ),
)
```

Si le projet n'a pas de phase, utiliser `phases=()`. Si une phase future n'a pas
encore commencé, ne pas l'instancier dans `ProjectConfig`. Dans un fichier JSON
de configuration projet, utiliser `null` pour documenter une date absente, mais
le code qui construit `ProjectConfig` doit ignorer cette phase tant que ses deux
dates ne sont pas connues.

## 3. Charger les données

Lorsque les fichiers suivent le nommage attendu par le fournisseur, le package
peut les trouver depuis `ProjectConfig`. Pour un notebook, il est souvent plus
lisible de charger les tables explicitement :

```python
raw = xyt.RawGpsData(
    storyline=storyline,
    trips=trips,
    journeys=journeys,
    user_statistics=user_statistics,
)
```

Le diagnostic utile avant transformation :

```python
validation = xyt.validate_gps_raw(raw)
summary = {
    name: {
        "ok": report.ok,
        "errors": sum(issue.severity == "error" for issue in report.issues),
        "warnings": sum(issue.severity == "warning" for issue in report.issues),
    }
    for name, report in validation.items()
}
```

## 4. Transformer sur un petit échantillon

Avant un traitement complet, filtrer quelques utilisateurs complets et lancer
une transformation minimale :

```python
result = xyt.run_mobility_pipeline(
    config,
    raw=raw_sample,
    sociodemo=sociodemographics_sample,
    resample_missing_days=False,
    clean_leg_geometries=False,
    add_length_outlier_flags=False,
    add_signal_quality_flags=False,
    compute_indicators=False,
)
```

Ce test vérifie les dates, les géométries, les liens `trips` / `journeys` et les
mappings de modes sans lancer toute la chaîne d'indicateurs.

## 5. Passer au traitement complet

Une fois le diagnostic validé :

- réactiver les contrôles utiles au projet ;
- expliciter les phases réellement disponibles ;
- documenter les seuils retenus ;
- conserver les sorties intermédiaires et les rapports de validation ;
- relancer les notebooks de production dans l'ordre.

Les notebooks `quickstart-analyse-gps.ipynb` et
`diagnostiquer-ses-donnees.ipynb` montrent ces étapes sur le test set anonymisé.
