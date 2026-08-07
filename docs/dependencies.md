# Dépendances Python

Date : 2026-06-09

## Principe

Le package doit garder un noyau de dépendances réduit. Les bibliothèques utilisées seulement pour les exports, les notebooks ou les tests sont déclarées comme dépendances optionnelles.

Version Python recommandée : 3.12. Le contrat du package est `>=3.10,<3.13`. Les versions 3.13+ sont exclues pour l’instant, car les piles géospatiales peuvent avoir des décalages de wheels ou de compatibilité sur les versions Python très récentes.

La borne haute n'est donc pas une limite fonctionnelle définitive : elle indique
les versions effectivement validées pour le package et les notebooks
`Notebooks`. Elle pourra être relâchée après une passe de tests et
d'exécution des notebooks en Python 3.13.

## Dépendances du noyau

| Package | Version | Rôle |
|---|---:|---|
| `pandas` | `>=2.2,<3.0` | tables, dates, agrégations |
| `geopandas` | `>=0.14,<2.0` | GeoDataFrames, CRS, opérations géospatiales |
| `shapely` | `>=2.0,<3.0` | parsing et manipulation des géométries |

## Dépendances optionnelles

| Extra | Package | Version | Usage |
|---|---|---:|---|
| `export` | `pyarrow` | `>=14,<22` | lecture/écriture Parquet |
| `export` | `openpyxl` | `>=3.1,<4.0` | lecture/écriture Excel |
| `analytics` | `duckdb` | `>=1.1,<2.0` | base SQL locale pour exports dashboard |
| `analytics` | `h3` | `>=4.1,<5.0` | indexation hexagonale des points GPS |
| `viz` | `folium` | `>=0.16,<1.0` | cartes interactives HTML dans les notebooks |
| `notebook` | `ipykernel` | `>=6.29,<7.0` | noyau Jupyter pour le notebook tutoriel |
| `docs` | `mkdocs-material` | `>=9.5,<10.0` | site documentaire |
| `docs` | `mkdocstrings[python]` | `>=0.25,<1.0` | référence API depuis les docstrings |
| `docs` | `mkdocs-jupyter` | `>=0.25,<1.0` | support des notebooks dans MkDocs |
| `dev` | `pytest` | `>=8.2,<9.0` | tests |
| `dev` | `nbformat` | `>=5.10,<6.0` | validation structurelle des notebooks |

## Pourquoi plusieurs fichiers `requirements`

Les dépendances sont séparées par usage pour éviter d’installer des paquets
lourds ou inutiles dans tous les environnements.

`pyproject.toml` reste la source de vérité pour les bornes de versions et les
extras optionnels. Les fichiers `requirements-*.txt` ne dupliquent pas ces
bornes : ils appellent simplement les extras, par exemple `-e .[analytics]`,
pour faciliter la création d'environnements reproductibles.

| Fichier | Usage | À installer quand |
|---|---|---|
| `requirements.txt` | noyau minimal du package | utiliser `xyt_gps` pour charger et transformer des données |
| `requirements-dev.txt` | tests et développement local | modifier le code ou lancer `pytest` |
| `requirements-docs.txt` | documentation MkDocs | prévisualiser ou construire la documentation |
| `requirements-export.txt` | exports Parquet et Excel | écrire des fichiers `.parquet` ou `.xlsx` |
| `requirements-analytics.txt` | H3 et DuckDB | produire des agrégats spatiaux et une base SQL locale |
| `requirements-viz.txt` | cartes interactives | utiliser `plot_gps_traces()` avec Folium |

Cette séparation est utile parce que certaines dépendances sont optionnelles :
`pyarrow` est nécessaire pour Parquet mais relativement lourd, `duckdb` et `h3`
ne servent qu'aux exports spatiaux avancés, `folium` ne sert qu’aux cartes, et
`mkdocs` ne sert qu’à la documentation. Le noyau du package reste donc plus
simple à installer et à maintenir.

La règle pratique est la suivante : installer seulement ce qui correspond à
l’usage prévu. Pour un environnement de travail complet, installer les fichiers
optionnels en plus du développement.

## Installation recommandée

Après publication, installation minimale :

```bash
python -m pip install xyt-gps
```

Avec les dépendances optionnelles les plus courantes :

```bash
python -m pip install "xyt-gps[export,viz,analytics]"
```

Pour travailler depuis le checkout local.

Depuis la racine du dépôt `xyt-gps` :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name xyt-gps --display-name "Python (xyt-gps)"
```

Pour un environnement complet de développement, documentation, exports et visualisation :

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-docs.txt
python -m pip install -r requirements-export.txt
python -m pip install -r requirements-viz.txt
python -m pip install -r requirements-analytics.txt
```

Pour un usage sans outils de développement :

```bash
python -m pip install -r requirements.txt
```

Les exports Parquet et Excel restent optionnels :

```bash
python -m pip install -r requirements-export.txt
```

Les cartes interactives Folium restent aussi optionnelles :

```bash
python -m pip install -r requirements-viz.txt
```

Les exports H3 et DuckDB restent optionnels :

```bash
python -m pip install -r requirements-analytics.txt
```

Remarque : sur les versions Python très récentes, `pyarrow` peut ne pas disposer immédiatement de wheels précompilés et se compiler localement. C'est une raison supplémentaire pour ne pas l'inclure dans l'environnement de développement de base.

Pour construire ou prévisualiser la documentation :

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

## Dépendances volontairement non ajoutées

Le client GraphHopper utilise l'API HTTP et la bibliothèque standard Python; il
n'ajoute pas de dépendance au noyau. Les moteurs OSRM, Google Directions, les
tableaux de bord et les outils de publication PyPI ne sont pas inclus.

## Protection des notebooks dans git

Les notebooks peuvent contenir des sorties de cellules avec des aperçus de données. Pour réduire le risque d'ajouter ces sorties dans git, le dépôt contient :

- `.gitattributes`, qui applique un filtre aux fichiers `.ipynb` lors de `git add` ;
- `tools/strip_notebook_outputs.py`, qui supprime les sorties et les compteurs d'exécution ;
- `.githooks/pre-commit`, qui nettoie les notebooks déjà staged avant le commit ;
- `.githooks/pre-push`, qui vérifie qu'aucun notebook suivi ne contient de sortie.

À configurer une fois dans chaque clone du dépôt :

```bash
git config filter.strip-notebook-output.clean 'python3 tools/strip_notebook_outputs.py --stdin'
git config filter.strip-notebook-output.smudge cat
git config filter.strip-notebook-output.required true
git config core.hooksPath .githooks
```

Cette protection ne remplace pas la règle principale : les dossiers `Data/` et `Data/Output/` restent ignorés par git et ne doivent pas être versionnés.
