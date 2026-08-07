"""HTML rendering helpers for daily demand profiles."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape


DEMAND_MODE_COLORS = (
    "#e84b8a",
    "#6b4c9a",
    "#3997a9",
    "#7a5fb5",
    "#d45f7d",
    "#8e8a9d",
    "#3f8f5f",
)

DEMAND_MODE_COLOR_BY_LABEL = {
    "Tous modes": "#e84b8a",
    "Voiture": "#e9413f",
    "Marche": "#f2c94c",
    "Vélo": "#3f8f5f",
    "TP": "#3997a9",
    "Train": "#3997a9",
    "TPU": "#3997a9",
    "Autres": "#8e8a9d",
}

DEMAND_HEATMAP_COLORS = (
    "#fff8c9",
    "#f8d27d",
    "#f4a261",
    "#e76f51",
    "#e84b8a",
    "#8b0036",
)

DEMAND_PROFILE_CSS = """
.xyt-demand-profile {
  margin-top: 14px;
  border-top: 1px solid #ece6f3;
  padding-top: 12px;
}
.xyt-demand-profile h4 {
  margin: 0 0 4px 0;
  color: #2b2340;
  font-size: 13px;
}
.xyt-demand-profile p {
  margin: 0 0 10px 0;
  color: #7e7494;
  font-size: 11px;
}
.xyt-demand-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}
.xyt-demand-card {
  background: #ffffff;
  border: 1px solid #ece6f3;
  border-radius: 8px;
  padding: 10px;
}
.xyt-demand-card-title {
  color: #2b2340;
  font-size: 12px;
  font-weight: 650;
}
.xyt-demand-card-subtitle {
  margin-top: 1px;
  color: #7e7494;
  font-size: 10px;
}
.xyt-demand-svg {
  width: 100%;
  height: auto;
  margin-top: 4px;
}
.xyt-demand-axis {
  stroke: #ddd6ea;
  stroke-width: 1;
}
.xyt-demand-tick {
  fill: #8e86a0;
  font-size: 9px;
}
.xyt-demand-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 10px;
  margin-top: 4px;
  color: #7e7494;
  font-size: 10px;
}
.xyt-demand-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.xyt-demand-legend-swatch {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 99px;
}
.xyt-demand-heatmap {
  margin-top: 14px;
  border-top: 1px solid #ece6f3;
  padding-top: 12px;
}
.xyt-demand-heatmap h4 {
  margin: 0 0 4px 0;
  color: #2b2340;
  font-size: 13px;
}
.xyt-demand-heatmap p {
  margin: 0 0 10px 0;
  color: #7e7494;
  font-size: 11px;
}
.xyt-demand-heatmap-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 12px;
}
.xyt-demand-heatmap-card {
  background: #ffffff;
  border: 1px solid #ece6f3;
  border-radius: 8px;
  padding: 10px;
}
.xyt-demand-heatmap-title {
  color: #2b2340;
  font-size: 12px;
  font-weight: 650;
  margin-bottom: 8px;
}
.xyt-demand-heatmap-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}
.xyt-demand-heatmap-control {
  display: inline-flex;
  flex-direction: column;
  gap: 2px;
  color: #7e7494;
  font-size: 10px;
}
.xyt-demand-heatmap-control select {
  border: 1px solid #ddd6ea;
  border-radius: 6px;
  background: #fbfafc;
  color: #2b2340;
  font: inherit;
  font-size: 11px;
  padding: 4px 20px 4px 6px;
}
.xyt-demand-heatmap-matrix {
  overflow-x: auto;
}
.xyt-demand-heatmap-row {
  display: grid;
  align-items: center;
  gap: 2px;
  margin-bottom: 2px;
}
.xyt-demand-heatmap-row-label {
  color: #7e7494;
  font-size: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.xyt-demand-heatmap-hour {
  color: #8e86a0;
  font-size: 8px;
  text-align: center;
}
.xyt-demand-heatmap-cell {
  display: block;
  min-width: 7px;
  height: 13px;
  border-radius: 2px;
}
.xyt-demand-heatmap-legend {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 7px;
  color: #7e7494;
  font-size: 10px;
}
.xyt-demand-heatmap-legend-ramp {
  display: inline-grid;
  grid-template-columns: repeat(6, 13px);
  gap: 2px;
}
"""


def _line_points(values: list[float], *, width: int, height: int, margin: int, max_value: float) -> str:
    if not values:
        return ""
    inner_width = width - 2 * margin
    inner_height = height - 2 * margin
    denominator = max(1, len(values) - 1)
    scale = max_value if max_value > 0 else 1.0
    points = []
    for index, value in enumerate(values):
        x = margin + (index / denominator) * inner_width
        y = height - margin - (float(value) / scale) * inner_height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _is_all_modes_label(mode: object, all_modes_label: str) -> bool:
    return str(mode) in {str(all_modes_label), "Tous modes"}


def _mode_color(mode: str, *, all_modes_label: str, all_modes_color: str, fallback_index: int) -> str:
    if _is_all_modes_label(mode, all_modes_label):
        return all_modes_color
    return DEMAND_MODE_COLOR_BY_LABEL.get(mode, DEMAND_MODE_COLORS[fallback_index % len(DEMAND_MODE_COLORS)])


def _heatmap_color(value: float, *, max_value: float) -> str:
    if value <= 0 or max_value <= 0:
        return DEMAND_HEATMAP_COLORS[0]
    ratio = min(1.0, max(0.0, float(value) / max_value))
    index = min(len(DEMAND_HEATMAP_COLORS) - 1, max(1, round(ratio * (len(DEMAND_HEATMAP_COLORS) - 1))))
    return DEMAND_HEATMAP_COLORS[index]


def _matrix_by_phase_group_mode(profile: Mapping[str, object]) -> dict[tuple[str, str, str], dict]:
    matrices = {}
    heatmaps = profile.get("heatmaps")
    if not isinstance(heatmaps, list):
        return matrices
    for group in heatmaps:
        if not isinstance(group, Mapping):
            continue
        group_key = str(group.get("key", ""))
        group_label = str(group.get("label", group_key))
        time_labels = group.get("time_labels")
        if not group_key or not isinstance(time_labels, list):
            continue
        modes = group.get("modes")
        if not isinstance(modes, list):
            continue
        for mode_payload in modes:
            if not isinstance(mode_payload, Mapping):
                continue
            mode = str(mode_payload.get("mode", ""))
            phases = mode_payload.get("phases")
            if not mode or not isinstance(phases, list):
                continue
            for phase_payload in phases:
                if not isinstance(phase_payload, Mapping):
                    continue
                phase = str(phase_payload.get("phase", ""))
                rows = phase_payload.get("rows")
                if phase and isinstance(rows, list) and rows:
                    matrices[(phase, group_key, mode)] = {
                        "group_label": group_label,
                        "time_labels": time_labels,
                        "rows": rows,
                    }
    return matrices


def _render_heatmap_matrix(
    matrix: Mapping[str, object],
    *,
    phase: str,
    group_key: str,
    mode: str,
) -> str:
    rows = matrix.get("rows")
    time_labels = matrix.get("time_labels")
    if not isinstance(rows, list) or not isinstance(time_labels, list):
        return ""
    max_value = max(
        (
            float(value)
            for row in rows
            if isinstance(row, Mapping)
            for value in row.get("values", [])
            if value is not None
        ),
        default=0.0,
    )
    grid_template = f"82px repeat({len(time_labels)}, minmax(7px, 1fr))"
    header_cells = ['<span class="xyt-demand-heatmap-hour"></span>']
    for index, label in enumerate(time_labels):
        text = str(label)[:2] if index % 3 == 0 else ""
        header_cells.append(f'<span class="xyt-demand-heatmap-hour">{escape(text)}</span>')
    row_html = [
        f'<div class="xyt-demand-heatmap-row" style="grid-template-columns:{grid_template};">'
        f'{"".join(header_cells)}</div>'
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        label = str(row.get("label", ""))
        values = [float(value or 0) for value in row.get("values", [])]
        cells = [f'<span class="xyt-demand-heatmap-row-label" title="{escape(label)}">{escape(label)}</span>']
        for time_label, value in zip(time_labels, values, strict=False):
            color = _heatmap_color(value, max_value=max_value)
            title = f"{phase} | {label} | {time_label} | {mode}: {value:.1f}"
            cells.append(
                f'<span class="xyt-demand-heatmap-cell" title="{escape(title)}" '
                f'style="background:{color};"></span>'
            )
        row_html.append(
            f'<div class="xyt-demand-heatmap-row" style="grid-template-columns:{grid_template};">'
            f'{"".join(cells)}</div>'
        )
    ramp = "".join(
        f'<span class="xyt-demand-heatmap-cell" style="background:{color};"></span>'
        for color in DEMAND_HEATMAP_COLORS
    )
    return (
        '<div class="xyt-demand-heatmap-matrix" '
        f'data-group="{escape(group_key)}" data-mode="{escape(mode)}">'
        f'{"".join(row_html)}'
        '<div class="xyt-demand-heatmap-legend">'
        "<span>Moins</span>"
        f'<span class="xyt-demand-heatmap-legend-ramp">{ramp}</span>'
        "<span>Plus</span>"
        "</div>"
        "</div>"
    )


def _heatmap_script() -> str:
    return """
<script>
(function () {
  document.querySelectorAll('.xyt-demand-heatmap-card').forEach(function (card) {
    const groupSelect = card.querySelector('.xyt-demand-heatmap-group-select');
    const modeSelect = card.querySelector('.xyt-demand-heatmap-mode-select');
    const matrices = Array.from(card.querySelectorAll('.xyt-demand-heatmap-matrix'));
    if (!groupSelect || !modeSelect || matrices.length === 0) return;
    function updateModeOptions() {
      const group = groupSelect.value;
      const available = new Set(matrices.filter(function (matrix) {
        return matrix.dataset.group === group;
      }).map(function (matrix) {
        return matrix.dataset.mode;
      }));
      let firstAvailable = null;
      Array.from(modeSelect.options).forEach(function (option) {
        const isAvailable = available.has(option.value);
        option.hidden = !isAvailable;
        option.disabled = !isAvailable;
        if (isAvailable && firstAvailable === null) firstAvailable = option.value;
      });
      if (!available.has(modeSelect.value) && firstAvailable !== null) {
        modeSelect.value = available.has('Tous modes') ? 'Tous modes' : firstAvailable;
      }
    }
    function update() {
      updateModeOptions();
      const group = groupSelect.value;
      const mode = modeSelect.value;
      matrices.forEach(function (matrix) {
        matrix.hidden = !(matrix.dataset.group === group && matrix.dataset.mode === mode);
      });
    }
    groupSelect.addEventListener('change', update);
    modeSelect.addEventListener('change', update);
    update();
  });
})();
</script>
"""


def render_demand_heatmap_html(
    profile: Mapping[str, object],
    *,
    all_modes_label: str,
) -> str:
    """Render hourly demand heatmaps with mode and row-group selectors."""

    matrices = _matrix_by_phase_group_mode(profile)
    if not matrices:
        return ""
    phases = sorted({phase for phase, _, _ in matrices})
    group_options = sorted(
        {(group_key, str(matrix["group_label"])) for (_, group_key, _), matrix in matrices.items()},
        key=lambda item: item[1],
    )
    cards = []
    for phase in phases:
        phase_matrices = {
            key: matrix
            for key, matrix in matrices.items()
            if key[0] == phase
        }
        modes = sorted({mode for _, _, mode in phase_matrices})
        if "Tous modes" in modes:
            modes = ["Tous modes"] + [mode for mode in modes if mode != "Tous modes"]
        elif all_modes_label in modes:
            modes = [all_modes_label] + [mode for mode in modes if mode != all_modes_label]
        group_select = "".join(
            f'<option value="{escape(group_key)}">{escape(label)}</option>'
            for group_key, label in group_options
            if any(key[1] == group_key for key in phase_matrices)
        )
        mode_select = "".join(
            f'<option value="{escape(mode)}">{escape(mode)}</option>'
            for mode in modes
        )
        matrix_html = "".join(
            _render_heatmap_matrix(matrix, phase=phase, group_key=group_key, mode=mode)
            for (phase_name, group_key, mode), matrix in phase_matrices.items()
            if phase_name == phase
        )
        cards.append(
            '<div class="xyt-demand-heatmap-card">'
            f'<div class="xyt-demand-heatmap-title">{escape(phase)}</div>'
            '<div class="xyt-demand-heatmap-controls">'
            '<label class="xyt-demand-heatmap-control">Lecture'
            f'<select class="xyt-demand-heatmap-group-select">{group_select}</select>'
            "</label>"
            '<label class="xyt-demand-heatmap-control">Mode'
            f'<select class="xyt-demand-heatmap-mode-select">{mode_select}</select>'
            "</label>"
            "</div>"
            f"{matrix_html}"
            "</div>"
        )
    if not cards:
        return ""
    bin_minutes = profile.get("heatmap_bin_minutes", 60)
    return (
        '<section class="xyt-demand-heatmap">'
        "<h4>Configurations temporelles des fréquentations</h4>"
        f"<p>Nombre moyen de personnes en déplacement par tranche de {escape(str(bin_minutes))} minutes.</p>"
        f'<div class="xyt-demand-heatmap-grid">{"".join(cards)}</div>'
        f"{_heatmap_script()}"
        "</section>"
    )


def render_demand_profile_html(
    metadata: Mapping[str, object],
    *,
    all_modes_label: str,
    all_modes_color: str,
    max_modes: int,
) -> str:
    """Render demand profiles stored in indicator metadata as inline SVG."""

    profile = metadata.get("demand_profile_5min")
    if not isinstance(profile, Mapping):
        return ""
    phases = profile.get("phases")
    labels = profile.get("time_labels")
    if not isinstance(phases, list) or not isinstance(labels, list) or not phases:
        return ""
    reference_phases = {
        str(value)
        for value in profile.get("reference_phases", [])
        if value is not None
    }
    reference_color = str(profile.get("reference_color", "#e9413f"))

    width = 300
    height = 138
    margin = 24
    phase_cards = []
    for phase_profile in phases:
        if not isinstance(phase_profile, Mapping):
            continue
        series = phase_profile.get("series")
        if not isinstance(series, list) or not series:
            continue
        phase = str(phase_profile.get("phase", "Période"))
        is_reference_phase = phase in reference_phases
        all_modes = [
            item
            for item in series
            if isinstance(item, Mapping) and _is_all_modes_label(item.get("mode"), all_modes_label)
        ]
        mode_series = [
            item
            for item in series
            if isinstance(item, Mapping) and not _is_all_modes_label(item.get("mode"), all_modes_label)
        ]
        mode_series = sorted(
            mode_series,
            key=lambda item: sum(float(value) for value in item.get("values", []) if value is not None),
            reverse=True,
        )
        displayed_series = (all_modes + mode_series)[:max_modes]
        max_value = max(
            (
                float(value)
                for item in displayed_series
                for value in item.get("values", [])
                if value is not None
            ),
            default=0.0,
        )
        lines = []
        legend_items = []
        for index, item in enumerate(displayed_series):
            mode = str(item.get("mode", ""))
            values = [float(value or 0) for value in item.get("values", [])]
            is_all_modes = _is_all_modes_label(mode, all_modes_label)
            color = _mode_color(
                mode,
                all_modes_label=all_modes_label,
                all_modes_color=all_modes_color,
                fallback_index=index,
            )
            if is_reference_phase and is_all_modes:
                color = reference_color
            line_width = 3.0 if is_reference_phase and is_all_modes else (2.4 if is_all_modes else 1.5)
            opacity = 0.95 if is_all_modes else 0.75
            points = _line_points(values, width=width, height=height, margin=margin, max_value=max_value)
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{escape(color)}" '
                f'stroke-width="{line_width}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"/>'
            )
            legend_items.append(
                '<span class="xyt-demand-legend-item">'
                f'<span class="xyt-demand-legend-swatch" style="background:{escape(color)}"></span>'
                f'{escape(mode)}'
                "</span>"
            )
        if not lines:
            continue
        n_days = phase_profile.get("n_days")
        max_label = f"{max_value:.1f}"
        title_style = (
            f' style="color:{escape(reference_color)}; font-weight:650;"'
            if is_reference_phase
            else ""
        )
        phase_cards.append(
            '<div class="xyt-demand-card">'
            f'<div class="xyt-demand-card-title"{title_style}>{escape(phase)}</div>'
            f'<div class="xyt-demand-card-subtitle">moyenne sur {escape(str(n_days))} jour(s)</div>'
            f'<svg class="xyt-demand-svg" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Courbe de demande {escape(phase)}">'
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" class="xyt-demand-axis"/>'
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" class="xyt-demand-axis"/>'
            f'<text x="{margin}" y="{height - 6}" class="xyt-demand-tick">00:00</text>'
            f'<text x="{width / 2}" y="{height - 6}" text-anchor="middle" class="xyt-demand-tick">12:00</text>'
            f'<text x="{width - margin}" y="{height - 6}" text-anchor="end" class="xyt-demand-tick">24:00</text>'
            f'<text x="{margin - 4}" y="{margin + 4}" text-anchor="end" class="xyt-demand-tick">{escape(max_label)}</text>'
            f'{"".join(lines)}'
            "</svg>"
            f'<div class="xyt-demand-legend">{"".join(legend_items)}</div>'
            "</div>"
        )
    if not phase_cards:
        return ""
    bin_minutes = profile.get("bin_minutes", 5)
    unit = profile.get("unit", "personnes en déplacement")
    heatmap_html = render_demand_heatmap_html(profile, all_modes_label=all_modes_label)
    return (
        '<section class="xyt-demand-profile">'
        "<h4>Demande journalière par heure</h4>"
        f'<p>Nombre moyen de {escape(str(unit))} par tranche de {escape(str(bin_minutes))} minutes.</p>'
        f'<div class="xyt-demand-grid">{"".join(phase_cards)}</div>'
        f"{heatmap_html}"
        "</section>"
    )
