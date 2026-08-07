"""Notebook-friendly indicator charts."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd

from .models import IndicatorResult
from .viz_demand import DEMAND_PROFILE_CSS, render_demand_profile_html


ALL_MODES_COLOR = "#e84b8a"
DEFAULT_MODE_ORDER = (
    "Voiture",
    "Carsharing",
    "Taxi",
    "2RM",
    "TP",
    "Train",
    "TPU",
    "Vélo",
    "Marche",
    "Autres",
)


def _as_notebook_html(html: str):
    try:
        from IPython.display import HTML
    except ImportError:
        return html
    return HTML(html)


def _indicator_frame(indicators: IndicatorResult | pd.DataFrame, table: str) -> pd.DataFrame:
    if isinstance(indicators, IndicatorResult):
        tables = indicators.tables()
        if table not in tables:
            raise KeyError(f"Unknown indicator table: {table}. Expected one of {sorted(tables)}.")
        return tables[table].copy()
    return indicators.copy()


def _empty_indicator_html(title: str, message: str, *, save_path: str | Path | None = None):
    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<style>
.xyt-indicators {{
  font-family: Arial, sans-serif;
  color: #2b2340;
  max-width: 900px;
}}
.xyt-indicators-title {{
  font-weight: 650;
  margin: 0 0 12px 0;
  color: #6b4c9a;
}}
.xyt-indicators-empty {{
  border: 1px solid #ddd6ea;
  border-radius: 8px;
  padding: 14px;
  background: #fbfafc;
  color: #7e7494;
  font-size: 13px;
  line-height: 1.45;
}}
</style>
</head>
<body>
<div class="xyt-indicators">
  <div class="xyt-indicators-title">{escape(title)}</div>
  <div class="xyt-indicators-empty">{escape(message)}</div>
</div>
</body>
</html>
"""
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
    return _as_notebook_html(html)


def _default_indicator_metrics(df: pd.DataFrame, group_col: str, facet_col: str | None) -> list[str]:
    preferred = [
        "distance_km_per_day",
        "travel_time_min_per_day",
        "trip_count_per_day",
        "leg_count_per_day",
        "co2_kg_per_day",
        "co2_direct_kg_per_day",
        "calories_kcal_per_day",
        "mets_per_day",
        "distance_km",
        "travel_time_min",
        "trip_count",
        "co2_kg",
        "co2_direct_kg",
        "calories_kcal",
        "mets",
    ]
    selected = [column for column in preferred if column in df.columns]
    if selected:
        return selected
    excluded = {group_col, "user_id", "n_users"}
    if facet_col is not None:
        excluded.add(facet_col)
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def _metric_label(metric: str) -> str:
    labels = {
        "distance_km_per_day": "Distance par jour (km)",
        "travel_time_min_per_day": "Temps de trajet par jour (min)",
        "trip_count_per_day": "Déplacements par jour",
        "leg_count_per_day": "Étapes par jour",
        "distance_km": "Distance (km)",
        "travel_time_min": "Temps de trajet (min)",
        "trip_count": "Déplacements",
        "leg_count": "Étapes",
        "co2_kg_per_day": "CO2 par jour (kg)",
        "co2_direct_kg_per_day": "CO2 direct par jour (kg)",
        "calories_kcal_per_day": "Calories par jour (kcal)",
        "mets_per_day": "METs moyens par jour",
        "co2_kg": "CO2 (kg)",
        "co2_direct_kg": "CO2 direct (kg)",
        "calories_kcal": "Calories (kcal)",
        "mets": "METs moyens",
    }
    return labels.get(metric, metric.replace("_", " "))


def _display_group_order(
    values: Iterable[object],
    *,
    group_order: Iterable[str] | None,
    all_modes_label: str,
) -> list[str]:
    observed = [str(value) for value in values if pd.notna(value)]
    observed_unique = list(dict.fromkeys(observed))
    if group_order is not None:
        preferred = [str(value) for value in group_order]
    elif any(value in DEFAULT_MODE_ORDER for value in observed_unique):
        preferred = list(DEFAULT_MODE_ORDER)
    else:
        preferred = []

    ordered = [value for value in preferred if value in observed_unique and value != all_modes_label]
    ordered.extend(
        value
        for value in observed_unique
        if value not in ordered and value != all_modes_label
    )
    return ordered


def _order_panel_rows(
    panel_df: pd.DataFrame,
    *,
    group_col: str,
    metric: str,
    group_order: list[str],
    sort_bars_by_value: bool,
) -> pd.DataFrame:
    if panel_df.empty:
        return panel_df
    out = panel_df.copy()
    if sort_bars_by_value:
        return out.sort_values(metric, ascending=False).reset_index(drop=True)
    order_lookup = {value: index for index, value in enumerate(group_order)}
    out["_xyt_group_order"] = out[group_col].astype(str).map(order_lookup).fillna(len(order_lookup))
    out = out.sort_values(["_xyt_group_order", group_col]).drop(columns="_xyt_group_order")
    return out.reset_index(drop=True)


def _occupancy_source_label(source: str) -> str:
    labels = {
        "computed": "recalculé",
        "observed": "observé",
        "computed_fallback": "recalculé faute de valeur observée",
    }
    return labels.get(str(source), str(source))


def _indicator_metadata(
    indicators: IndicatorResult | pd.DataFrame,
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    base = {}
    if isinstance(indicators, IndicatorResult):
        base.update(indicators.metadata or {})
    if metadata:
        base.update(dict(metadata))
    return base


def _format_metadata_value(
    key: str,
    value: object,
    *,
    value_format: str,
    metadata: Mapping[str, object] | None = None,
) -> str:
    metadata = metadata or {}
    if key == "n_users":
        source_count = metadata.get("source_user_count")
        if source_count is not None:
            try:
                source_int = int(source_count)
                value_int = int(value)
            except (TypeError, ValueError):
                source_int = value_int = None
            if source_int is not None and source_int > value_int:
                return f"{value_int}/{source_int}"
    if key == "filtered_dataset" and isinstance(value, bool):
        return "Sous-échantillon filtré" if value else "Données non filtrées"
    if key == "include_zero_days" and isinstance(value, bool):
        return "Toute la période suivie" if value else "Jours avec mobilité observée seulement"
    if key == "occupancy_sources" and isinstance(value, (list, tuple, set)):
        return ", ".join(_occupancy_source_label(str(item)) for item in value)
    if key == "mean_occupancy_rate_by_source" and isinstance(value, Mapping):
        return "; ".join(
            f"{_occupancy_source_label(str(source))}: {value_format.format(float(rate))}"
            for source, rate in value.items()
        )
    if isinstance(value, bool):
        return "Oui" if value else "Non"
    if isinstance(value, Mapping):
        return "; ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, float):
        return value_format.format(value)
    return str(value)


def _identity_card(
    metadata: Mapping[str, object],
    *,
    value_format: str,
    all_modes_label: str,
    all_modes_color: str,
    show_demand_profile: bool,
    demand_profile_max_modes: int,
) -> str:
    if not metadata:
        return ""

    n_users_label = "Utilisateurs"
    try:
        source_count = int(metadata.get("source_user_count", metadata.get("n_users", 0)))
        retained_count = int(metadata.get("n_users", 0))
    except (TypeError, ValueError):
        source_count = retained_count = 0
    if source_count > retained_count:
        n_users_label = "Utilisateurs retenus / inscrits"

    labels = [
        ("n_users", n_users_label),
        ("n_person_days", "Jours-utilisateurs"),
        ("n_legs", "Étapes"),
        ("n_trips", "Déplacements"),
        ("mean_leg_count_per_user_day", "Étapes moyennes / jour-utilisateur"),
        ("mean_trip_count_per_user_day", "Déplacements moyens / jour-utilisateur"),
        ("include_excursions", "Excursions incluses dans le calcul"),
        ("include_airplane", "Étapes avion incluses"),
        ("airplane_rows_excluded", "Étapes avion exclues"),
        ("excursions_available", "Variable excursion disponible"),
        ("filtered_dataset", "Périmètre de calcul"),
        ("filtered_user_count", "Utilisateurs conservés par filtre"),
        ("bad_user_count", "Utilisateurs exclus qualité"),
        ("tracking_quality_ok_count", "Utilisateurs avec tracking suffisant"),
        ("bad_signal_user_count", "Utilisateurs avec signal GPS dégradé"),
        ("include_zero_days", "Période prise pour moyenniser"),
        ("zero_mobility_days_count", "Jours-utilisateurs sans déplacement"),
        ("use_weights", "Pondération utilisée"),
        ("weight_col", "Colonne de pondération"),
        ("occupancy_sources", "Source du taux d'occupation"),
        ("occupancy_observed_available", "Colonne occupation disponible"),
        ("car_legs_with_declared_occupancy_share", "Étapes voiture avec occupation déclarée"),
        ("mean_occupancy_rate_by_source", "Taux moyen d'occupation"),
        ("mode_col", "Niveau de mode"),
        ("phases", "Périodes affichées"),
    ]
    items = []
    for key, label in labels:
        if key == "filtered_user_count" and source_count > retained_count:
            continue
        if key not in metadata:
            continue
        value = metadata[key]
        if value is None or (isinstance(value, str) and value == ""):
            continue
        items.append(
            '<div class="xyt-indicator-card-item">'
            f'<span class="xyt-indicator-card-label">{escape(label)}</span>'
            f'<span class="xyt-indicator-card-value">{escape(_format_metadata_value(key, value, value_format=value_format, metadata=metadata))}</span>'
            "</div>"
        )
    if not items:
        return ""
    demand_profile = (
        render_demand_profile_html(
            metadata,
            all_modes_label=all_modes_label,
            all_modes_color=all_modes_color,
            max_modes=demand_profile_max_modes,
        )
        if show_demand_profile
        else ""
    )
    return (
        '<section class="xyt-indicator-card">'
        "<h3>Carte d'identité des données</h3>"
        f'<div class="xyt-indicator-card-grid">{"".join(items)}</div>'
        f"{demand_profile}"
        "</section>"
    )


def plot_indicator_bars(
    indicators: IndicatorResult | pd.DataFrame,
    *,
    table: str = "population",
    metrics: Iterable[str] | None = None,
    group_col: str = "mode",
    facet_col: str | None = "phase",
    title: str = "Indicateurs de mobilité",
    max_bars: int | None = None,
    group_order: Iterable[str] | None = None,
    sort_bars_by_value: bool = False,
    value_format: str = "{:.1f}",
    bar_color: str = "#6b4c9a",
    include_all_modes: bool = True,
    all_modes_label: str = "Tous modes",
    all_modes_color: str = ALL_MODES_COLOR,
    facet_colors: Mapping[str, str] | None = None,
    show_demand_profile: bool = True,
    demand_profile_max_modes: int = 6,
    metadata: Mapping[str, object] | None = None,
    show_identity_card: bool = True,
    reference_scope: str = "group",
    save_path: str | Path | None = None,
) -> object:
    """Render mobility indicators as simple notebook bar charts.

    The default input is an `IndicatorResult` returned by
    `compute_mobility_indicators()`. The function reads its population table and
    plots the main per-day metrics by mode. It can also receive a DataFrame
    directly, for example `indicators.person_phase`.

    Args:
        indicators: `IndicatorResult` or an indicator DataFrame.
        table: Indicator table to use when `indicators` is an
            `IndicatorResult`: `person_day`, `person_phase` or `population`.
        metrics: Numeric columns to plot. When omitted, common indicator
            columns are inferred.
        group_col: Categorical column used for bars, usually `mode`.
        facet_col: Optional column used to create one panel per period/phase.
        title: Displayed title.
        max_bars: Optional maximum number of bars per panel.
        group_order: Optional explicit order for the categorical bars. When
            omitted, common mobility modes use a stable default order so that
            phases remain visually comparable.
        sort_bars_by_value: When true, sort bars by decreasing value inside
            each panel. This reproduces the older compact ranking behavior, but
            can make modes change position between phases.
        value_format: Python format string used for numeric labels.
        bar_color: CSS color for bars.
        include_all_modes: Whether to add a total row across all displayed
            modes in each phase/panel.
        all_modes_label: Label used for the total row.
        all_modes_color: CSS color used for the total row.
        facet_colors: Optional CSS colors by facet value. Useful to highlight
            a reference group such as a control experiment.
        show_demand_profile: Whether to display 5-minute daily demand curves
            when `IndicatorResult.metadata` contains them.
        demand_profile_max_modes: Maximum number of demand curves per phase,
            including the all-modes curve.
        metadata: Optional metadata displayed in the identity card. Values
            override `IndicatorResult.metadata` when the input is an
            `IndicatorResult`.
        show_identity_card: Whether to display calculation metadata before
            the bars.
        reference_scope: Bar scaling for detailed modes. `group` compares a
            mode with itself across phases; `metric` uses one common scale for
            all detailed modes.
        save_path: Optional HTML file path.

    Returns:
        An `IPython.display.HTML` object when IPython is available, otherwise
        the raw HTML string.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If no metric can be plotted.
    """

    df = _indicator_frame(indicators, table)
    if df.empty:
        return _empty_indicator_html(
            title,
            "Aucun indicateur disponible pour le périmètre filtré.",
            save_path=save_path,
        )
    if group_col not in df.columns:
        raise KeyError(f"indicator table must contain {group_col}.")
    if facet_col is not None and facet_col not in df.columns:
        facet_col = None

    metric_list = list(metrics) if metrics is not None else _default_indicator_metrics(df, group_col, facet_col)
    missing_metrics = [metric for metric in metric_list if metric not in df.columns]
    if missing_metrics:
        raise KeyError(f"indicator table missing metric column(s): {missing_metrics}")
    if not metric_list:
        raise ValueError("No numeric indicator metric found to plot.")
    if reference_scope not in {"group", "metric"}:
        raise ValueError("reference_scope must be 'group' or 'metric'.")

    plot_df = df.copy()
    plot_df[group_col] = plot_df[group_col].fillna("Autres").astype(str)
    if facet_col is not None:
        plot_df[facet_col] = plot_df[facet_col].fillna("All").astype(str)
        facet_values = plot_df[facet_col].drop_duplicates().tolist()
    else:
        facet_values = [None]
    display_group_order = _display_group_order(
        plot_df[group_col],
        group_order=group_order,
        all_modes_label=all_modes_label,
    )

    indicator_metadata = _indicator_metadata(indicators, metadata)
    facet_color_map = {str(key): str(value) for key, value in (facet_colors or {}).items()}
    sections = []
    for metric in metric_list:
        panel_tables = []
        for facet_value in facet_values:
            panel_df = plot_df if facet_value is None else plot_df.loc[plot_df[facet_col].eq(facet_value)]
            panel_df = panel_df[[group_col, metric]].copy()
            panel_df[metric] = pd.to_numeric(panel_df[metric], errors="coerce").fillna(0)
            panel_df = (
                panel_df.groupby(group_col, dropna=False)[metric]
                .sum()
                .reset_index()
            )
            panel_df = _order_panel_rows(
                panel_df,
                group_col=group_col,
                metric=metric,
                group_order=display_group_order,
                sort_bars_by_value=sort_bars_by_value,
            )
            panel_df["_xyt_is_all_modes"] = False
            if include_all_modes and not panel_df.empty:
                all_modes_row = pd.DataFrame(
                    [{group_col: all_modes_label, metric: float(panel_df[metric].sum()), "_xyt_is_all_modes": True}]
                )
                if max_bars is not None:
                    panel_df = panel_df.head(max_bars)
                panel_df = pd.concat([all_modes_row, panel_df], ignore_index=True)
            elif max_bars is not None:
                panel_df = panel_df.head(max_bars)
            panel_tables.append((facet_value, panel_df))

        metric_max = max(
            (
                float(panel_df.loc[~panel_df["_xyt_is_all_modes"], metric].max())
                for _, panel_df in panel_tables
                if not panel_df.empty and (~panel_df["_xyt_is_all_modes"]).any()
            ),
            default=0.0,
        )
        all_modes_metric_max = max(
            (
                float(panel_df.loc[panel_df["_xyt_is_all_modes"], metric].max())
                for _, panel_df in panel_tables
                if not panel_df.empty and panel_df["_xyt_is_all_modes"].any()
            ),
            default=0.0,
        )
        group_metric_max = {}
        for _, panel_df in panel_tables:
            if panel_df.empty:
                continue
            detailed = panel_df.loc[~panel_df["_xyt_is_all_modes"]]
            for _, row in detailed.iterrows():
                group_value = str(row[group_col])
                group_metric_max[group_value] = max(
                    group_metric_max.get(group_value, 0.0),
                    float(row[metric]),
                )
        metric_panels = []
        for facet_value, panel_df in panel_tables:
            facet_key = str(facet_value) if facet_value is not None else "All"
            facet_color = facet_color_map.get(facet_key)
            rows = []
            for _, panel_row in panel_df.iterrows():
                group = panel_row[group_col]
                value = float(panel_row[metric])
                is_all_modes = bool(panel_row.get("_xyt_is_all_modes", False))
                if is_all_modes:
                    reference_max = all_modes_metric_max
                elif reference_scope == "group":
                    reference_max = group_metric_max.get(str(group), metric_max)
                else:
                    reference_max = metric_max
                width = 0 if reference_max <= 0 else max(2, min(100, value / reference_max * 100))
                color = facet_color or (all_modes_color if is_all_modes else bar_color)
                label_class = " xyt-indicator-label--all-modes" if is_all_modes else ""
                label_style = f' style="color:{escape(facet_color)};"' if facet_color else ""
                rows.append(
                    '<div class="xyt-indicator-row">'
                    f'<span class="xyt-indicator-label{label_class}"{label_style}>{escape(str(group))}</span>'
                    '<span class="xyt-indicator-track">'
                    f'<span class="xyt-indicator-bar" style="width:{width:.2f}%; background:{color};"></span>'
                    "</span>"
                    f'<span class="xyt-indicator-value">{escape(value_format.format(value))}</span>'
                    "</div>"
                )

            panel_title = str(facet_value) if facet_value is not None else "All"
            panel_title_style = f' style="color:{escape(facet_color)}; font-weight:650;"' if facet_color else ""
            metric_panels.append(
                '<div class="xyt-indicator-panel">'
                f'<div class="xyt-indicator-panel-title"{panel_title_style}>{escape(panel_title)}</div>'
                f'{"".join(rows)}'
                "</div>"
            )
        sections.append(
            '<section class="xyt-indicator-section">'
            f'<h4>{escape(_metric_label(metric))}</h4>'
            f'<div class="xyt-indicator-panels">{"".join(metric_panels)}</div>'
            "</section>"
        )

    identity_card = (
        _identity_card(
            indicator_metadata,
            value_format=value_format,
            all_modes_label=all_modes_label,
            all_modes_color=all_modes_color,
            show_demand_profile=show_demand_profile,
            demand_profile_max_modes=demand_profile_max_modes,
        )
        if show_identity_card
        else ""
    )
    if reference_scope == "group":
        reference_note = (
            "Barre de référence : maximum observé par indicateur et par mode, "
            "pour comparer chaque mode entre périodes. « Tous modes » utilise sa propre échelle."
        )
    else:
        reference_note = (
            "Barre de référence : maximum observé pour chaque indicateur, avec une échelle séparée "
            "pour « Tous modes » et pour les modes détaillés."
        )
    footnotes = [
        "La catégorie « Autres » regroupe les modes non détaillés dans le niveau affiché, "
        "notamment les valeurs non renseignées ou classées Other/Airplane selon le mapping utilisé.",
        "Déplacements moyens / jour-utilisateur : moyenne journalière rapportée à la période de suivi "
        "retenue pour le calcul, avec ou sans jours sans déplacement selon le paramètre affiché.",
    ]
    try:
        source_count = int(indicator_metadata.get("source_user_count", indicator_metadata.get("n_users", 0)))
        retained_count = int(indicator_metadata.get("n_users", 0))
    except (TypeError, ValueError):
        source_count = retained_count = 0
    if source_count > retained_count:
        footnotes.append(
            "Le ratio utilisateurs indique les personnes conservées après filtre sur l'assiduité du tracking "
            "et la complétude / qualité des traces, rapportées aux personnes inscrites dans l'export GPS."
        )
    footnote_html = "".join(
        f'<div class="xyt-indicators-footnote">Note : {escape(note)}</div>'
        for note in footnotes
    )
    body = f"""
<div class="xyt-indicators">
  <div class="xyt-indicators-title">{escape(title)}</div>
  {identity_card}
  <div class="xyt-indicators-note">{escape(reference_note)}</div>
  {''.join(sections)}
  {footnote_html}
</div>
"""
    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<style>
.xyt-indicators {{
  font-family: Arial, sans-serif;
  color: #2b2340;
  max-width: 1280px;
}}
.xyt-indicators-title {{
  font-weight: 650;
  margin: 0 0 12px 0;
  color: #6b4c9a;
}}
.xyt-indicators-note {{
  color: #7e7494;
  font-size: 12px;
  margin: 0 0 14px 0;
}}
.xyt-indicator-card {{
  border: 1px solid #ddd6ea;
  border-radius: 8px;
  padding: 14px;
  margin: 0 0 14px 0;
  background: #fbfafc;
}}
.xyt-indicator-card h3 {{
  margin: 0 0 10px 0;
  font-size: 14px;
  color: #2b2340;
}}
.xyt-indicator-card-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 8px 14px;
}}
.xyt-indicator-card-item {{
  display: flex;
  flex-direction: column;
  gap: 2px;
}}
.xyt-indicator-card-label {{
  color: #7e7494;
  font-size: 11px;
}}
.xyt-indicator-card-value {{
  color: #2b2340;
  font-size: 13px;
  font-weight: 600;
}}
{DEMAND_PROFILE_CSS}
.xyt-indicator-section {{
  margin: 0 0 18px 0;
}}
.xyt-indicator-section h4 {{
  margin: 0 0 8px 0;
  font-size: 14px;
}}
.xyt-indicator-panels {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 14px;
}}
.xyt-indicator-panel-title {{
  color: #7e7494;
  font-size: 12px;
  margin-bottom: 6px;
}}
.xyt-indicator-row {{
  display: grid;
  grid-template-columns: minmax(70px, 110px) minmax(120px, 1fr) 64px;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}}
.xyt-indicator-label {{
  font-size: 12px;
  color: #7e7494;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.xyt-indicator-label--all-modes {{
  color: {all_modes_color};
  font-weight: 650;
}}
.xyt-indicator-track {{
  display: block;
  height: 12px;
  background: #f3f0f8;
  border-radius: 999px;
  overflow: hidden;
}}
.xyt-indicator-bar {{
  display: block;
  height: 12px;
  border-radius: 999px;
}}
.xyt-indicator-value {{
  color: #7e7494;
  font-variant-numeric: tabular-nums;
  font-size: 12px;
  text-align: right;
}}
.xyt-indicators-footnote {{
  margin-top: 12px;
  color: #7e7494;
  font-size: 11px;
  line-height: 1.4;
}}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
    return _as_notebook_html(html)
