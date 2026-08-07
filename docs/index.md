<div class="as-home-brand">
  <div class="as-home-brand__logos">
    <img class="as-home-brand__logo as-home-brand__logo--action-situee" src="assets/logo-action-situee.svg" alt="Logo Bureau Action Située">
    <img class="as-home-brand__logo as-home-brand__logo--modus" src="assets/logo-modus-2025.png" alt="Logo Fondation MODUS">
  </div>
  <div>
    <p class="as-home-brand__label">Des données GPS brutes aux données de mobilité</p>
    <p class="as-home-brand__meta">Par Marc-Edouard Schultheiss, chercheur et gérant du Bureau Action Située. Développement issu de thèse de doctorat, recherches postdoctorales et recherche appliquée soutenues financièrement respectivement par la faculté ENAC de l’EPFL, le SNSF et plus récemment par la Fondation MODUS.</p>
  </div>
</div>

<section class="as-hero">
  <div class="as-hero__heading">
    <h1>xyt_gps</h1>
    <p class="as-hero__subtitle">Package Python pour transformer des exports GPS en données de mobilité structurées, contrôlables et exploitables pour l’analyse.</p>
  </div>
  <div class="as-hero__actions">
    <a class="as-button as-button--primary" href="getting-started/installation/">Installer</a>
    <a class="as-button" href="project_workflow/">Voir le workflow</a>
  </div>
</section>

## Chaîne de transformation

`xyt_gps` est organisé comme une chaîne courte : préparer les fichiers d’entrée, transformer les exports GPS, contrôler la qualité, produire les indicateurs, puis exporter une base propre pour l’analyse ou un dashboard.

<div class="as-pipeline-grid">
  <div class="as-pipeline-card as-pipeline-card--landing">
    <p class="as-pipeline-card__label">1. Entrée</p>
    <h3>Préparer les fichiers</h3>
    <p>Homogénéiser les sources GPS, vérifier les colonnes, retirer les champs sensibles et produire une base landed.</p>
    <dl>
      <dt>Fonctions principales</dt>
      <dd><code>expected_gps_schema()</code>, <code>check_raw_import_columns()</code>, <code>load_gps_export()</code></dd>
      <dt>Sorties</dt>
      <dd><code>landing_dir</code>, manifest, <code>user_presence</code></dd>
    </dl>
  </div>
  <div class="as-pipeline-card as-pipeline-card--transform">
    <p class="as-pipeline-card__label">2. Transformation</p>
    <h3>Construire les tables mobilité</h3>
    <p>Parser les dates et géométries, appliquer les mappings, séparer legs et staypoints, puis relier trips, journeys et users.</p>
    <dl>
      <dt>Fonctions principales</dt>
      <dd><code>run_mobility_pipeline()</code>, <code>prepare_mobility_dataset()</code>, <code>mode_purpose_mapping()</code></dd>
      <dt>Sorties</dt>
      <dd><code>MobilityDataset</code>, <code>transformed_dir</code></dd>
    </dl>
  </div>
  <div class="as-pipeline-card as-pipeline-card--quality">
    <p class="as-pipeline-card__label">3. Qualité</p>
    <h3>Contrôler et filtrer</h3>
    <p>Mesurer la participation, les jours manquants, les valeurs extrêmes, le signal GPS et les critères de sélection utilisateurs.</p>
    <dl>
      <dt>Fonctions principales</dt>
      <dd><code>build_weekly_participation_grid()</code>, <code>build_user_selection_table()</code>, <code>add_signal_quality_flags()</code></dd>
      <dt>Sorties</dt>
      <dd>rapports qualité, filtres utilisateurs, zones et contrôles cartographiques</dd>
    </dl>
  </div>
  <div class="as-pipeline-card as-pipeline-card--indicators">
    <p class="as-pipeline-card__label">4. Indicateurs</p>
    <h3>Enrichir et résumer</h3>
    <p>Ajouter les enrichissements transversaux, calculer les indicateurs personne-jour, personne-période et population.</p>
    <dl>
      <dt>Fonctions principales</dt>
      <dd><code>compute_mobility_indicators()</code>, <code>build_person_day_indicators()</code>, <code>plot_indicator_bars()</code></dd>
      <dt>Sorties</dt>
      <dd><code>enriched_dir</code>, figures HTML, carte d’identité des indicateurs</dd>
    </dl>
  </div>
  <div class="as-pipeline-card as-pipeline-card--exports">
    <p class="as-pipeline-card__label">5. Exports</p>
    <h3>Préparer l’analyse</h3>
    <p>Consolider les tables propres et produire les formats interrogeables ou cartographiables pour notebooks, SQL ou dashboard.</p>
    <dl>
      <dt>Fonctions principales</dt>
      <dd><code>write_mobility_dataset()</code>, <code>build_spatial_analytics_tables()</code>, <code>write_duckdb_spatial_database()</code></dd>
      <dt>Sorties</dt>
      <dd><code>clean_dir</code>, CSV, Parquet, GeoParquet, H3, DuckDB</dd>
    </dl>
  </div>
</div>

## Utilisation en projet

Le package fournit les fonctions de transformation, de contrôle, d'indicateurs et d'export. Les notebooks de production comme `000_data_landing`, `010_data_loading` ou `030_enrich_indicators` vivent dans un repo projet et utilisent `xyt_gps` comme dépendance ; pour un premier exemple autonome inclus avec le package, ouvrir `examples/quickstart-analyse-gps.ipynb`.

## Parcours recommandé

1. [Installer l’environnement](getting-started/installation.md).
2. [Définir la structure de projet](getting-started/project-structure.md).
3. Lire le notebook court `examples/quickstart-analyse-gps.ipynb`.
4. Vérifier les données avec `examples/diagnostiquer-ses-donnees.ipynb`.
5. [Lire l’API recommandée](reference/recommended-api.md).

## Périmètre

Le package s’arrête à la production d’une base de mobilité analysable : tables structurées, contrôles qualité, enrichissements transversaux et indicateurs génériques. Les analyses thématiques détaillées se font ensuite dans des notebooks, rapports ou applications dédiées.

!!! info "Licence et citation"
    Le code est placé sous PolyForm Noncommercial License 1.0.0. La documentation est placée sous CC BY-NC 4.0. Les usages commerciaux nécessitent un accord écrit préalable de Bureau Action Située.
