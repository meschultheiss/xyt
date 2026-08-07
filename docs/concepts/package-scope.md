# Périmètre du package

`xyt_gps` transforme des exports GPS en données de mobilité analysables. Le
package couvre la préparation technique de la base, les contrôles qualité, les
enrichissements transversaux et les indicateurs génériques.

Il ne couvre pas les analyses thématiques finales : interprétation, figures de
rapport, messages de synthèse ou comparaisons propres à un mandat.

## Terminologie

Dans la documentation du package, les noms génériques décrivent les opérations
réellement réalisées sur les données.

| Terme | Rôle |
|---|---|
| préparation des données | charger, valider, parser et structurer les tables GPS |
| contrôle qualité | vérifier suivi, trous de tracking, confirmations, modes et outliers |
| nettoyage spatial | qualifier les legs, les zones, les pertes de signal et les relations spatiales |
| enrichissements transversaux | ajouter CO2, occupation, santé ou métriques équivalentes |
| indicateurs et export analytique | produire les tables personne-jour, personne-période et synthèses |
| analyses thématiques | analyses spécifiques hors package |

Les correspondances avec les anciens notebooks sont conservées dans les pages de
maintenance du projet.

## Limite fonctionnelle

| Étape | Dans le package | Sortie attendue |
|---|---|---|
| Data-landing | support par notebooks et utilitaires | base homogène prête à charger |
| Préparation des données | oui | `MobilityDataset` contrôlable |
| Contrôle qualité et spatial | oui | flags, rapports, tables filtrables |
| Enrichissements CO2 et santé | oui, progressivement | variables documentées par leg, personne-jour ou période |
| Indicateurs et export analytique | oui | base de mobilité analysable et exports |
| Analyses thématiques | non | notebooks, figures, rapports ou applications dédiées |

La règle pratique est simple : une fonction appartient au package si elle
transforme, contrôle ou enrichit la base de mobilité de manière générique. Une
fonction reste hors package si elle produit une interprétation thématique ou une
visualisation propre à un rapport.

## Chaîne cible

```text
Data-landing
-> Préparation des données
-> Contrôle qualité et spatial
-> Enrichissements transversaux
-> Indicateurs et export analytique
-> Analyses thématiques hors package
```

Les enrichissements CO2 et santé doivent rester paramétrables et documentés. Les
facteurs CO2, taux d’occupation, METs ou hypothèses de calories ne doivent pas
être cachés dans le code.
