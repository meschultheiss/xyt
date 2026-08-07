# Données synthétiques

Cette page est une note de maintenance. Le parcours recommandé pour apprendre
le package utilise le jeu `anonymous-test-set-gps`, documenté dans
[Données de démonstration](sample-data.md).

Les fonctions de génération synthétique restent utiles pour les tests automatisés
et certains diagnostics internes, mais elles ne remplacent pas les données de
démonstration anonymisées. Elles ne doivent pas être présentées comme un exemple
analytique réaliste.

## Usage actuel

| Besoin | Support recommandé |
|---|---|
| apprendre le package | `anonymous-test-set-gps` et notebooks dans `examples/` |
| tester la robustesse du code | tests unitaires autour de `generate_synthetic_declic_gps()` |
| créer un nouveau test set partageable | `Notebooks/002_build_anonymous_test_set.ipynb` |

Les fonctions synthétiques sont donc conservées dans le package, mais restent un
outil de développement. Pour un tutoriel analyste, utiliser les notebooks
`quickstart-analyse-gps.ipynb` et `diagnostiquer-ses-donnees.ipynb`.
