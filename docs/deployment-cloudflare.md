# Déployer la documentation sur Cloudflare Pages

!!! note "Page de maintenance"
    Cette page concerne le déploiement de la documentation et n'est pas nécessaire pour utiliser le package.

Cette page décrit le déploiement automatique de la documentation MkDocs lorsque
la branche `main` est poussée sur GitHub.

## Principe

Cloudflare Pages peut être connecté à un dépôt GitHub. À chaque push sur la
branche de production, Cloudflare exécute la commande de build, publie le
dossier `site`, puis met à jour le site. Les branches et pull requests peuvent
aussi produire des déploiements de prévisualisation.

Dans le dépôt indépendant `xyt-gps`, la documentation MkDocs vit à la racine du
dépôt. Cloudflare Pages peut donc construire le site directement depuis cette
racine.

## Configuration recommandée

Dans Cloudflare Pages :

| Paramètre | Valeur |
|---|---|
| Framework preset | `MkDocs` ou `None` |
| Production branch | `main` |
| Root directory | vide ou `/` |
| Build command | `python -m pip install -r requirements-docs.txt && python -m mkdocs build` |
| Build output directory | `site` |
| Variable d'environnement | `PYTHON_VERSION=3.12` |

Cette configuration installe uniquement les dépendances nécessaires à la
documentation, puis construit le site statique.

Le dossier de sortie doit être `site`, sans slash initial. Une valeur comme
`/site` peut pointer au mauvais endroit selon la configuration du build.

La commande utilise `python -m mkdocs build` plutôt que `mkdocs build` parce
que, dans l'environnement Cloudflare, le script `mkdocs` installé par `pip`
peut ne pas être disponible dans le `PATH` alors que le module Python est bien
installé.

## Contrôle local avant push

Depuis la racine du dépôt `xyt-gps` :

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs build
```

Pour prévisualiser localement :

```bash
mkdocs serve
```

La documentation est ensuite disponible sur `http://127.0.0.1:8000/`.

## Points de vigilance

- Ne pas publier `Data/`, `Data/Output/`, `Output/`, ni les notebooks avec
  sorties de cellules.
- Garder `requirements-docs.txt` léger : il sert seulement à construire la doc.
- Mettre à jour la doc avant le push si l'API publique ou les notebooks
  `Notebooks/` changent.
- Vérifier que la branche de production est bien `main`.

## Références

- Guide Cloudflare Pages pour MkDocs :
  <https://developers.cloudflare.com/pages/framework-guides/deploy-an-mkdocs-site/>
- Configuration des builds Cloudflare Pages :
  <https://developers.cloudflare.com/pages/configuration/build-configuration/>
- Intégration Git Cloudflare Pages :
  <https://developers.cloudflare.com/pages/configuration/git-integration/>
