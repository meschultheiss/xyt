# Import multi-projets

Le package permet de charger plusieurs expérimentations ou périodes en une seule passe.

```python
configs = [declic_config, declic_ziplo_config]

raw = xyt.load_gps_sources(configs)
dataset = xyt.prepare_mobility_datasets(configs)
```

Chaque configuration garde ses propres :

- dates ;
- phases ;
- chemins ;
- mappings ;
- seuils.

Après transformation, `dataset.metadata` conserve la provenance des sources et
leurs phases. Les indicateurs peuvent donc construire un calendrier personne-jour
par projet, même si les phases portent les mêmes noms mais ne couvrent pas les
mêmes dates.

## Identifiants

Par défaut, les identifiants sont namespacés pour éviter les collisions :

```text
{experiment_name}__{period}::{identifiant_original}
```

Les colonnes originales sont conservées lorsque possible :

```text
raw_storyline_id
raw_user_id
raw_trip_id
raw_journey_id
```

## Export lourd

Le sampling peut aussi être appliqué au multi-projets :

```python
raw = xyt.load_gps_sources(
    configs,
    sample=xyt.RawSampleConfig.by_users(5, random_state=42),
)
```

Ce tirage est appliqué à chaque source avant concaténation.
