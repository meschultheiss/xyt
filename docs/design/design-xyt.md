# Design xyt

!!! note "Page de maintenance"
    Cette page sert de référence visuelle pour les exports HTML, cartes et figures. Elle n'est pas nécessaire pour utiliser les fonctions du package.

Document de référence mis à jour le 16 juin 2026.

Ce document décrit le design actuellement implémenté dans le dashboard Déclic
Mobilité. Il sert de référence pour préserver l'identité visuelle,
l'architecture d'interface et les comportements UI avant toute évolution.

Dans le cadre de `xyt_gps`, il sert surtout à cadrer les exports HTML, cartes,
figures et tables destinées à être intégrées dans un dashboard. Les notebooks
`Notebooks/030_enrich_indicators.ipynb` et
`Notebooks/040_export_cleaned_dataset.ipynb` doivent produire des
figures compatibles avec cette grammaire visuelle : typographie Arial, palette
Action Située, cartes sobres, légendes explicites et densité lisible.

En cas d'écart entre une maquette future et ce document, le code actuel du
dashboard fait foi jusqu'à validation explicite d'un nouveau design.

## 1. Positionnement visuel

Le dashboard adopte une interface institutionnelle douce, claire et dense sans être austère.

- Fond d'application chaud et légèrement cassé, non blanc pur.
- Cartes blanches très lisibles sur fond beige clair.
- Accent principal violet, soutenu par un rose vif, un bleu canard, un vert anis et un orange.
- Iconographie simple, arrondie, systématiquement utilisée pour guider la lecture.
- Hiérarchie sobre, avec peu d'effets décoratifs et peu d'ombres.
- Design orienté lisibilité, consultation, comparaison de phases et accessibilité des graphiques.

## 2. Tokens visuels figés

### Palette principale

- Fond application: `#f8f7f4`
- Texte principal: `#2b2340`
- Texte secondaire: `#7e7494`
- Surface carte / header / footer: `#ffffff`
- Violet principal: `#6b4c9a`
- Rose accent: `#e84b8a`
- Vert anis: `#d4e64a`
- Bleu canard: `#4a90a4`
- Orange: `#f5a623`
- Fond secondaire lavande: `#f3f0f8`
- Bordure standard: `rgba(107, 76, 154, 0.12)`
- Bordure explicite secondaire: `#e0dce6`

### Rayons et densité

- Rayon standard: `0.625rem`
- Rayon dominant sur cartes: `rounded-xl`
- Boutons pill: `rounded-full`
- Densité générale: compacte à moyenne, sans grands espaces vides.

### Typographie

- Police réellement appliquée dans la coquille principale: `Arial, sans-serif`
- Police de secours et popups carte: `Arial, sans-serif`
- `Inter` est importée dans les styles mais n'est pas la police dominante de l'interface actuelle.
- Titre produit dans le header: graisse forte, tracking négatif léger, taille fluide avec `clamp`.
- Titres de page: violet, semi-gras à gras.
- Corps de texte: tailles entre `0.8rem` et `0.95rem` dans la majorité des composants.
- Chiffres KPI: `1.3rem` à `1.6rem`, poids `700`.

## 3. Structure globale de l'application

L'application repose sur une coquille unique, centrée et constante sur toutes les pages.

### Largeur et grille

- Conteneur principal limité à `max-w-5xl`.
- Padding horizontal standard: `px-4 sm:px-6`.
- Espacement vertical de contenu: `py-6 sm:py-8`.
- Mise en page interne basée sur des grilles simples en 1, 2 ou 3 colonnes selon la largeur.

### Header

Le header est blanc, bordé en bas, et contient quatre éléments stables.

- Logo à gauche.
- Titre produit `Déclic Mobilité 2026` avec code couleur violet + rose.
- Sous-titre descriptif sur desktop.
- Bloc de statut à droite avec point animé vert anis et date de mise à jour.
- Bascule `Collectif / Individuel` intégrée dans le header, en version compacte.

Le header doit rester visuellement léger, horizontal, et immédiatement identifiable comme zone de pilotage global.

### Navigation principale

- Barre secondaire blanche, collée sous le header.
- Navigation sticky en haut de fenêtre.
- Onglets présentés sous forme de pills arrondies avec icône Lucide.
- État actif: fond violet, texte blanc.
- État inactif: texte gris-violet avec hover lavande.
- Défilement horizontal autorisé si manque de place.

### Main content

- Les pages s'inscrivent dans le même conteneur centré que le header.
- Les motifs SVG des graphiques sont injectés une seule fois à ce niveau.
- Le fond général reste uniforme sur toute l'application.

### Footer

- Footer blanc bordé en haut.
- Deux lignes d'information en desktop, empilables sur mobile.
- Typographie discrète en gris-violet.

## 4. Langage des composants

### Cartes

Le dashboard repose presque entièrement sur des cartes blanches à bord fin.

- Fond blanc.
- Bordure légère.
- Rayon `xl`.
- Ombre absente ou très discrète.
- Padding interne confortable, souvent `p-5` ou `p-6`.

Les cartes servent aux KPI, graphiques, sections explicatives, modales et panneaux cartographiques.

### Boutons et contrôles

- Boutons de filtre presque toujours en pills.
- Boutons utilitaires secondaires sur fond lavande.
- États actifs pilotés par le violet principal.
- Les contrôles doivent rester bas en hauteur et rapides à scanner.

### Icônes

- Bibliothèque Lucide.
- Taille fréquente: `12` à `24` px.
- Usage systématique dans navigation, KPI, filtres, aides à la lecture et cartes.
- Les icônes sont souvent placées dans des pastilles colorées avec couleur à `15%` d'opacité en fond.

## 5. Composants récurrents à préserver

### `PageHeader`

Structure standard des pages thématiques.

- Titre à gauche en violet.
- Sous-titre gris-violet sous le titre.
- Contrôles à droite sur desktop.
- Contrôles empilés sur mobile.
- Peut accueillir un sélecteur de phase et un filtre jour/semaine/weekend.

### `PhaseSelector`

- Groupe de pills `Avant`, `Pendant`, `Après`.
- Variante optionnelle avec bouton `Suivi complet`.
- Actif en violet, inactif blanc bordé.

### `DayFilter`

- Trois pills avec icône: `Tous les jours`, `Semaine`, `Weekend`.
- Même logique visuelle que le sélecteur de phase.

### `ViewToggle`

Composant stratégique d'identité du dashboard.

- Segmented control `Collectif / Individuel` sur fond lavande.
- En mode individuel, affichage d'un champ de recherche compact.
- Sélections affichées sous forme de badges roses supprimables.
- Badge de compteur en violet si plusieurs identifiants sont sélectionnés.
- Message `Max.` discret une fois la limite atteinte.

### `StatCard`

- Carte KPI simple avec icône, valeur et libellé.
- Deux variantes: `inline` et `centered`.
- Couleur métier injectée via l'icône et sa pastille.

### `ChartCard`

Composant standard pour toute visualisation.

- Titre en haut à gauche.
- Sous-titre optionnel.
- Bouton de bascule `Graphique / Tableau` en haut à droite.
- Tableau accessible avec alternance de lignes sur fond beige clair.
- Toujours présenté dans une carte blanche homogène avec le reste du produit.

### `InfoButton`

- Bouton flottant circulaire violet en bas à droite.
- Ouvre une modale centrale blanche à coins très arrondis.
- Overlay noir translucide.
- Bouton de fermeture discret sur fond lavande.

## 6. Signature visuelle des pages

### Page d'accueil `À propos`

- Grand hero centré dans une carte blanche.
- Logo très grand.
- Titre violet fort et texte d'introduction centré.
- Trois cartes de phase avec un liseré coloré horizontal en haut.
- Bloc d'aide à la lecture en grille 2 colonnes sur desktop.
- Bandeau final lavande centré.

### Pages analytiques

Exemples: `Chiffres clés`, `CO₂`, `Espace`, `Santé`, `Rythme de vie`, `Indicateurs`.

- Ouverture par `PageHeader`.
- Série de KPI cards en tête.
- Corps de page structuré en grilles de `ChartCard`.
- Alternance de vues synthétiques et de vues comparatives par phase.
- Encarts de note méthodologique sur fond lavande en bas de certaines pages.

### Cartothèque

La cartothèque suit une logique différente mais reste cohérente avec le système général.

- Titre et sous-titre au-dessus de la carte.
- Layout en deux zones sur desktop: panneau couches à gauche, carte à droite.
- Sur mobile, panneau au-dessus de la carte.
- Carte dans un conteneur bordé à coins arrondis, hauteur fixe `560px`.
- Panneau des couches blanc, compact, repliable.
- Légende sous la carte, jamais superposée à la carte.
- Badge d'information lavande sous le panneau de couches.
- État de chargement avec spinner violet sur voile blanc translucide.

### Mini-cartes

- Intégrées dans des cartes blanches.
- Onglets internes en haut de carte.
- Hauteur fixe de `280px`.
- Légende inline sous la carte.

## 7. Graphiques et data visualisation

Le produit privilégie Recharts avec une grammaire visuelle constante.

### Principes de style

- Fonds de graphiques blancs, sans surcharge décorative.
- Grilles fines et sobres.
- Axes et labels en gris-violet.
- Couleurs cohérentes avec la palette produit.
- Arrondis sur les barres quand pertinent.
- Dots blancs cerclés sur les courbes principales.

### Accessibilité

- Utilisation de motifs SVG pour distinguer les séries sans dépendre uniquement de la couleur.
- Motifs figés:
  - voiture: violet à diagonales
  - TP: rose à pois
  - vélo: bleu à lignes horizontales
  - marche: vert anis hachuré croisé
  - autre: orange à lignes verticales
- Chaque graphique important peut être consulté en mode tableau.

### Lignes de référence

- Les comparaisons avec groupe témoin utilisent des pointillés.
- La présence d'une légende `ControlLegend` accompagne les cas concernés.

## 8. Cartographie

### Fond de plan

- Base raster OSM/CARTO claire type Positron.
- Commandes MapLibre standards en coin de carte.
- Palette de couches cohérente avec le reste de l'application.

### Popups

- Style `.dm-popup` dédié.
- Police Arial.
- Taille de texte compacte.
- Fond blanc, bordure claire, légère ombre.
- Apparence institutionnelle simple, sans cartouche massif.

### Légendes et couches

- Les couches actives sont indiquées par contraste d'opacité et couleur.
- Les états `loading`, `SITG`, `démo` sont signalés par petits badges.
- Les légendes utilisent micro-typographie et éléments géométriques simples.

## 9. Responsive design figé

- Le dashboard reste centré sur desktop avec largeur bornée.
- Les grilles passent en 1 colonne sur mobile.
- Le sous-titre du header est masqué sur petits écrans.
- Les dates et labels sont abrégés sur mobile.
- La navigation peut défiler horizontalement.
- Les panneaux cartographiques deviennent verticaux en dessous de `md`.
- Les contrôles de page passent d'un alignement horizontal à un empilement vertical.

## 10. Règles de cohérence à respecter pour toute évolution

- Ne pas remplacer le fond chaud `#f8f7f4` par un blanc pur ou un gris froid.
- Ne pas introduire de nouvelles couleurs dominantes sans besoin fonctionnel explicite.
- Conserver la logique cartes blanches + bord fin + rayon généreux.
- Conserver les pills comme langage principal de filtrage et navigation secondaire.
- Conserver la lecture `titre / sous-titre / contrôle` des pages analytiques.
- Préserver la bascule `Graphique / Tableau` sur les visualisations majeures.
- Préserver les motifs accessibles dans les graphiques.
- Éviter les overlays superposés à la carte lorsque l'information peut être déportée en dessous ou à côté.
- Maintenir une iconographie cohérente Lucide et des tailles discrètes.
- Garder une esthétique sobre, éditoriale et institutionnelle, non marketing.

## 11. Références code qui définissent ce design

- Structure globale: `src/app/components/Layout.tsx`
- Header, navigation, footer: `src/app/components/Layout.tsx`
- Tokens couleurs et base theme: `src/styles/theme.css`
- Imports styles globaux: `src/styles/index.css`
- Hero et sections d'introduction: `src/app/components/About.tsx`
- En-tête de page standard: `src/app/components/PageHeader.tsx`
- Vue collectif / individuel: `src/app/components/ViewToggle.tsx`
- Sélecteur de phase: `src/app/components/PhaseSelector.tsx`
- Filtre jour/semaine/weekend: `src/app/components/DayFilter.tsx`
- Cartes KPI: `src/app/components/StatCard.tsx`
- Cartes graphiques: `src/app/components/ChartCard.tsx`
- Motifs SVG: `src/app/components/HatchPattern.tsx`
- Bouton flottant et modale info: `src/app/components/InfoButton.tsx`
- Cartothèque: `src/app/components/CartothequePage.tsx`
- Mini-cartes: `src/app/components/MiniMap.tsx`

## 12. Statut du document

Ce document fige le design actuellement en production dans le dépôt. Il ne décrit pas un design cible ou spéculatif. Toute modification visuelle notable devrait être répercutée ici pour conserver une source de vérité simple à relire par produit, design et développement.
