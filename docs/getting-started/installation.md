# Installation

## Créer l’environnement

Version recommandée : Python 3.12. Le package accepte Python 3.10 à 3.12, mais exclut Python 3.13+ pour limiter les problèmes de compatibilité des dépendances géospatiales.

## Installer le package

Après publication :

```bash
python -m pip install xyt-gps
```

Avec les dépendances optionnelles les plus courantes :

```bash
python -m pip install "xyt-gps[export,viz,analytics]"
```

## Créer un environnement de développement local

Depuis la racine du dépôt `xyt-gps` :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Pour lancer le notebook tutoriel :

```bash
python -m ipykernel install --user --name xyt-gps --display-name "Python (xyt-gps)"
```

## Installer la documentation

Les dépendances de documentation sont séparées du noyau du package :

```bash
python -m pip install -r requirements-docs.txt
```

Lancer le site local :

```bash
mkdocs serve
```

Construire le site statique :

```bash
python -m mkdocs build
```

## Pourquoi des extras séparés

Le cœur du package doit rester léger : `pandas`, `geopandas`, `shapely`.

Les bibliothèques d’export, de notebook, de test et de documentation sont installées seulement quand elles sont nécessaires.
