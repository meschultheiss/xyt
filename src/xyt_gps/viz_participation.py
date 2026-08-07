"""Notebook-friendly participation heatmaps."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


GITHUB_GREEN_PALETTE = (
    "#ebedf0",
    "#9be9a8",
    "#40c463",
    "#30a14e",
    "#216e39",
)


def _as_notebook_html(html: str):
    try:
        from IPython.display import HTML
    except ImportError:
        return html
    return HTML(html)


def _score_color(score, *, max_score: int = 7, palette: tuple[str, ...] = GITHUB_GREEN_PALETTE) -> str:
    value = pd.to_numeric(pd.Series([score]), errors="coerce").fillna(0).iloc[0]
    if value <= 0 or max_score <= 0:
        return palette[0]
    level = int(min(len(palette) - 1, max(1, round((float(value) / max_score) * (len(palette) - 1)))))
    return palette[level]


def _is_displayable_summary_value(value: object) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, bool):
        return not missing
    return True


def _summary_items_from_input(
    summary: Mapping[str, object] | pd.DataFrame | None,
) -> list[tuple[str, object]]:
    if summary is None:
        return []
    if isinstance(summary, pd.DataFrame):
        if summary.empty:
            return []
        if {"metric", "value"}.issubset(summary.columns):
            return [
                (str(row.metric), row.value)
                for row in summary[["metric", "value"]].itertuples(index=False)
            ]
        if {"label", "value"}.issubset(summary.columns):
            return [
                (str(row.label), row.value)
                for row in summary[["label", "value"]].itertuples(index=False)
            ]
        return [
            (str(column), summary[column].iloc[0])
            for column in summary.columns
            if len(summary[column]) > 0
        ]
    return list(summary.items())


def _add_experiment_to_summary(
    summary_items: list[tuple[str, object]],
    experiment_name: str | None,
) -> list[tuple[str, object]]:
    if not experiment_name:
        return summary_items
    normalized_labels = {label.casefold() for label, _ in summary_items}
    if {"expérience", "experiment_name", "expérimentation"} & normalized_labels:
        return summary_items
    return [("Expérience", experiment_name), *summary_items]


def _write_participation_heatmap_png(
    *,
    path: str | Path,
    title: str,
    users: list[str],
    weeks: list[int],
    scores: pd.DataFrame,
    phase_by_week: Mapping[int, str],
    phase_break_weeks: set[int],
    max_score: int,
    score_label: str,
    summary_items: list[tuple[str, object]],
    notes: Iterable[str] | None,
    cell_width: int,
    cell_height: int,
    cell_gap: int,
    phase_separator_width: int,
    phase_separator_column_width: int,
    phase_separator_color: str,
    user_label_colors: Mapping[str, str] | None,
    png_dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Rectangle
    except ImportError as error:
        raise ImportError(
            "PNG export requires matplotlib. Install matplotlib or use `save_path` "
            "to export the HTML heatmap."
        ) from error

    label_width = 190
    title_height = 34
    header_height = 24
    legend_height = 30
    summary_width = 310 if summary_items or notes else 0
    summary_gap = 24 if summary_width else 0
    grid_width = label_width
    for week in weeks:
        if week in phase_break_weeks:
            grid_width += phase_separator_column_width + cell_gap
        grid_width += cell_width + cell_gap
    grid_height = title_height + header_height + max(1, len(users)) * (cell_height + cell_gap) + legend_height
    total_width = grid_width + summary_gap + summary_width
    total_height = max(grid_height, 210)
    fig_width = max(7.0, total_width / 100)
    fig_height = max(2.4, total_height / 100)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=png_dpi)
    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.axis("off")
    ax.text(0, 16, title, fontsize=13, fontweight="bold", color="#24292f", va="top")

    y0 = title_height + header_height
    week_y = title_height + 6
    row_y = y0
    x_positions: dict[int, float] = {}
    x = label_width
    for week in weeks:
        if week in phase_break_weeks:
            separator_x = x + phase_separator_column_width / 2 - phase_separator_width / 2
            ax.add_patch(
                Rectangle(
                    (separator_x, title_height + 2),
                    phase_separator_width,
                    max(cell_height + 4, (cell_height + cell_gap) * max(1, len(users))),
                    color=phase_separator_color,
                    linewidth=0,
                    alpha=0.9,
                )
            )
            x += phase_separator_column_width + cell_gap
        x_positions[week] = x
        ax.text(x + cell_width / 2, week_y, str(week), ha="center", va="center", fontsize=9, color="#6e7781")
        x += cell_width + cell_gap

    for user in users:
        label = user if len(user) <= 28 else f"{user[:25]}..."
        user_color = (user_label_colors or {}).get(str(user), "#57606a")
        ax.text(0, row_y + cell_height / 2, label, ha="left", va="center", fontsize=8.5, color=user_color)
        for week in weeks:
            score = scores.loc[user, week] if user in scores.index and week in scores.columns else 0
            score_int = int(pd.to_numeric(pd.Series([score]), errors="coerce").fillna(0).iloc[0])
            color = _score_color(score_int, max_score=max_score)
            ax.add_patch(
                Rectangle(
                    (x_positions[week], row_y),
                    cell_width,
                    cell_height,
                    facecolor=color,
                    edgecolor=(0.106, 0.122, 0.141, 0.08),
                    linewidth=0.5,
                )
            )
        row_y += cell_height + cell_gap

    legend_y = min(row_y + 14, total_height - 14)
    ax.text(label_width, legend_y + cell_height / 2, "Moins", ha="left", va="center", fontsize=8, color="#6e7781")
    legend_x = label_width + 42
    for color in GITHUB_GREEN_PALETTE:
        ax.add_patch(Rectangle((legend_x, legend_y), cell_width, cell_height, facecolor=color, linewidth=0))
        legend_x += cell_width + cell_gap
    ax.text(legend_x + 4, legend_y + cell_height / 2, "Plus", ha="left", va="center", fontsize=8, color="#6e7781")

    if summary_width:
        summary_x = grid_width + summary_gap
        box_height = min(total_height - 20, 52 + 20 * len(summary_items) + 30 * len(list(notes or [])))
        box = FancyBboxPatch(
            (summary_x, title_height),
            summary_width,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=8",
            facecolor="#ffffff",
            edgecolor="#d8dee4",
            linewidth=1,
        )
        ax.add_patch(box)
        ax.text(summary_x + 14, title_height + 20, "Lecture qualité", fontsize=9, fontweight="bold", color="#24292f")
        y = title_height + 42
        for label, value in summary_items:
            if not _is_displayable_summary_value(value):
                continue
            ax.text(summary_x + 14, y, str(label), fontsize=7.5, color="#57606a", va="center")
            ax.text(summary_x + summary_width - 14, y, str(value), fontsize=7.5, color="#24292f", ha="right", va="center", fontweight="bold")
            y += 18
        for note in notes or []:
            if not str(note).strip():
                continue
            ax.text(summary_x + 14, y + 6, str(note), fontsize=7, color="#6e7781", va="top", wrap=True)
            y += 30

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_participation_heatmap(
    participation_grid: pd.DataFrame,
    *,
    user_col: str = "user_id",
    week_col: str = "protocol_week_number",
    score_col: str = "tracked_days_count",
    phase_col: str = "phase",
    title: str = "Participation hebdomadaire",
    max_score: int = 7,
    max_users: int | None = None,
    cell_size: int = 13,
    cell_width: int | None = None,
    cell_height: int | None = None,
    cell_gap: int = 3,
    show_phase_separators: bool = True,
    phase_separator_color: str = "#e83b46",
    phase_separator_width: int = 3,
    phase_separator_column_width: int | None = None,
    user_label_colors: Mapping[str, str] | None = None,
    summary: Mapping[str, object] | pd.DataFrame | None = None,
    notes: Iterable[str] | None = None,
    experiment_name: str | None = None,
    save_path: str | Path | None = None,
    save_png_path: str | Path | None = None,
    png_dpi: int = 180,
) -> object:
    """Render a GitHub-style participation heatmap in a notebook.

    The input is the long table produced by `build_weekly_participation_grid`.
    Rows are users, columns are protocol weeks, and cell color intensity is
    based on `score_col`, usually the number of observed tracking days from 0 to 7.

    Args:
        participation_grid: Weekly participation table.
        user_col: User identifier column.
        week_col: Week number column.
        score_col: Participation score column.
        phase_col: Optional phase/period column used in cell tooltips.
        title: Displayed title.
        max_score: Maximum score used for the color scale.
        max_users: Optional maximum number of users displayed.
        cell_size: Backward-compatible square size in pixels, used when
            `cell_width` or `cell_height` are not provided.
        cell_width: Cell width in pixels. Set greater than `cell_height` for a
            flatter landscape grid.
        cell_height: Cell height in pixels.
        cell_gap: Gap between squares in pixels.
        show_phase_separators: Whether to add a red separator when the phase
            changes between two consecutive weeks.
        phase_separator_color: CSS color used for phase separators.
        phase_separator_width: Separator column width in pixels.
        phase_separator_column_width: Width of the separator slot. Keeping a
            slot wider than the red line avoids overlap with adjacent cells.
        user_label_colors: Optional CSS colors by displayed user label. This
            can highlight a reference group in a concatenated grid.
        summary: Optional key-value metadata displayed next to the heatmap,
            for example total users, retained users or bad-signal users.
        notes: Optional explanatory notes displayed below the summary.
        experiment_name: Optional experiment label added to the summary box.
        save_path: Optional HTML file path.
        save_png_path: Optional PNG file path. The PNG is rendered with
            matplotlib from the same grid values.
        png_dpi: Resolution used when `save_png_path` is provided.

    Returns:
        An `IPython.display.HTML` object when IPython is available, otherwise
        the raw HTML string.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If the participation grid is empty.
    """

    if score_col == "tracked_days_count" and score_col not in participation_grid.columns:
        score_col = "active_days_count"
    score_label = "jours trackés" if score_col == "tracked_days_count" else "jours actifs"
    cell_width = int(cell_width if cell_width is not None else cell_size)
    cell_height = int(cell_height if cell_height is not None else cell_size)
    phase_separator_column_width = int(
        phase_separator_column_width
        if phase_separator_column_width is not None
        else max(phase_separator_width + 2 * cell_gap, phase_separator_width + 4)
    )
    required = {user_col, week_col, score_col}
    missing = required - set(participation_grid.columns)
    if missing:
        raise KeyError(f"participation_grid missing required column(s): {sorted(missing)}")
    if participation_grid.empty:
        raise ValueError("participation_grid is empty.")

    grid = participation_grid.copy()
    grid["_user_label"] = grid[user_col].astype(str)
    grid["_week_sort"] = pd.to_numeric(grid[week_col], errors="coerce")
    grid = grid.dropna(subset=[user_col, "_week_sort"])
    if grid.empty:
        raise ValueError("participation_grid has no valid user/week rows.")
    grid["_week_sort"] = grid["_week_sort"].astype(int)

    users = grid["_user_label"].dropna().drop_duplicates().sort_values().tolist()
    if max_users is not None:
        users = users[:max_users]
        grid = grid.loc[grid["_user_label"].isin(users)].copy()
    weeks = sorted(grid["_week_sort"].drop_duplicates().tolist())

    scores = grid.pivot_table(
        index="_user_label",
        columns="_week_sort",
        values=score_col,
        aggfunc="max",
        fill_value=0,
    )
    phase_by_week = {}
    if phase_col in grid.columns:
        phase_by_week = (
            grid.sort_values("_week_sort")
            .dropna(subset=[phase_col])
            .drop_duplicates("_week_sort")
            .set_index("_week_sort")[phase_col]
            .astype(str)
            .to_dict()
        )
    phase_break_weeks = set()
    if show_phase_separators and phase_by_week:
        previous_phase = None
        for week in weeks:
            phase = phase_by_week.get(week)
            if previous_phase is not None and phase is not None and phase != previous_phase:
                phase_break_weeks.add(week)
            if phase is not None:
                previous_phase = phase

    grid_columns = []
    week_label_parts = []
    for week in weeks:
        if week in phase_break_weeks:
            grid_columns.append(f"{phase_separator_column_width}px")
            week_label_parts.append(
                '<span class="xyt-participation-phase-separator" '
                f'title="Changement de phase avant la semaine {escape(str(week))}"></span>'
            )
        grid_columns.append(f"{cell_width}px")
        week_label_parts.append(f'<span class="xyt-participation-week">{escape(str(week))}</span>')
    grid_template_columns = " ".join(grid_columns)
    week_labels = "".join(week_label_parts)
    rows = []
    for user in users:
        cells = []
        for week in weeks:
            if week in phase_break_weeks:
                cells.append('<span class="xyt-participation-phase-separator"></span>')
            score = scores.loc[user, week] if user in scores.index and week in scores.columns else 0
            score_int = int(pd.to_numeric(pd.Series([score]), errors="coerce").fillna(0).iloc[0])
            color = _score_color(score_int, max_score=max_score)
            phase = phase_by_week.get(week, "")
            title_text = f"{user} | semaine {week}"
            if phase:
                title_text += f" | {phase}"
            title_text += f" | {score_int}/{max_score} {score_label}"
            cells.append(
                f'<span class="xyt-participation-cell" title="{escape(title_text)}" '
                f'style="background:{color};"></span>'
            )
        user_label_color = (user_label_colors or {}).get(str(user))
        user_style = (
            f' style="color:{escape(str(user_label_color))}; font-weight:650;"'
            if user_label_color
            else ""
        )
        rows.append(
            '<div class="xyt-participation-row">'
            f'<span class="xyt-participation-user"{user_style}>{escape(str(user))}</span>'
            f'<span class="xyt-participation-cells">{"".join(cells)}</span>'
            "</div>"
        )

    legend = "".join(
        f'<span class="xyt-participation-cell" style="background:{color};"></span>'
        for color in GITHUB_GREEN_PALETTE
    )
    summary_items = _add_experiment_to_summary(
        _summary_items_from_input(summary),
        experiment_name,
    )
    summary_html = ""
    if summary_items or notes:
        rows_html = "".join(
            '<div class="xyt-participation-summary-row">'
            f'<span>{escape(str(label))}</span>'
            f'<strong>{escape(str(value))}</strong>'
            "</div>"
            for label, value in summary_items
            if _is_displayable_summary_value(value)
        )
        notes_html = ""
        if notes:
            notes_html = "".join(
                f'<p class="xyt-participation-note">{escape(str(note))}</p>'
                for note in notes
                if str(note).strip()
            )
        if rows_html or notes_html:
            summary_html = (
                '<aside class="xyt-participation-summary">'
                "<h3>Lecture qualité</h3>"
                f"{rows_html}"
                f"{notes_html}"
                "</aside>"
            )
    html = f"""
<style>
.xyt-participation {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #24292f;
  max-width: 100%;
  overflow-x: auto;
}}
.xyt-participation-layout {{
  display: flex;
  gap: 18px;
  align-items: flex-start;
}}
.xyt-participation-title {{
  font-weight: 650;
  margin: 0 0 8px 0;
}}
.xyt-participation-grid {{
  display: inline-block;
  min-width: max-content;
}}
.xyt-participation-header,
.xyt-participation-row {{
  display: grid;
  grid-template-columns: 130px auto;
  align-items: center;
  column-gap: 8px;
  margin-bottom: {cell_gap}px;
}}
.xyt-participation-weeks,
.xyt-participation-cells {{
  display: grid;
  grid-template-columns: {grid_template_columns};
  gap: {cell_gap}px;
  align-items: center;
}}
.xyt-participation-week {{
  width: {cell_width}px;
  font-size: 10px;
  color: #6e7781;
  text-align: center;
}}
.xyt-participation-user {{
  font-size: 12px;
  color: #57606a;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.xyt-participation-cell {{
  display: inline-block;
  width: {cell_width}px;
  height: {cell_height}px;
  border-radius: 2px;
  outline: 1px solid rgba(27,31,36,0.06);
}}
.xyt-participation-phase-separator {{
  display: flex;
  width: {phase_separator_column_width}px;
  height: {cell_height}px;
  align-items: center;
  justify-content: center;
}}
.xyt-participation-phase-separator::before {{
  content: "";
  display: block;
  width: {phase_separator_width}px;
  height: {cell_height + 8}px;
  background: {phase_separator_color};
  border-radius: 999px;
  opacity: 0.9;
}}
.xyt-participation-legend {{
  display: flex;
  gap: 6px;
  align-items: center;
  margin-top: 8px;
  color: #6e7781;
  font-size: 12px;
}}
.xyt-participation-legend-cells {{
  display: flex;
  gap: {cell_gap}px;
}}
.xyt-participation-summary {{
  min-width: 230px;
  max-width: 320px;
  border: 1px solid #d8dee4;
  border-radius: 8px;
  padding: 12px 14px;
  background: #ffffff;
  color: #57606a;
  font-size: 12px;
  line-height: 1.35;
}}
.xyt-participation-summary h3 {{
  color: #24292f;
  font-size: 13px;
  margin: 0 0 10px 0;
}}
.xyt-participation-summary-row {{
  display: flex;
  justify-content: space-between;
  gap: 14px;
  border-bottom: 1px solid #f0f2f4;
  padding: 4px 0;
}}
.xyt-participation-summary-row strong {{
  color: #24292f;
  font-weight: 650;
}}
.xyt-participation-note {{
  margin: 10px 0 0 0;
  color: #6e7781;
}}
@media (max-width: 840px) {{
  .xyt-participation-layout {{
    display: block;
  }}
  .xyt-participation-summary {{
    margin-top: 12px;
    max-width: none;
  }}
}}
</style>
<div class="xyt-participation">
  <div class="xyt-participation-title">{escape(title)}</div>
  <div class="xyt-participation-layout">
    <div>
      <div class="xyt-participation-grid">
        <div class="xyt-participation-header">
          <span></span>
          <span class="xyt-participation-weeks">{week_labels}</span>
        </div>
        {''.join(rows)}
      </div>
      <div class="xyt-participation-legend">
        <span>Moins</span>
        <span class="xyt-participation-legend-cells">{legend}</span>
        <span>Plus</span>
      </div>
    </div>
    {summary_html}
  </div>
</div>
"""
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
    if save_png_path is not None:
        _write_participation_heatmap_png(
            path=save_png_path,
            title=title,
            users=users,
            weeks=weeks,
            scores=scores,
            phase_by_week=phase_by_week,
            phase_break_weeks=phase_break_weeks,
            max_score=max_score,
            score_label=score_label,
            summary_items=summary_items,
            notes=notes,
            cell_width=cell_width,
            cell_height=cell_height,
            cell_gap=cell_gap,
            phase_separator_width=phase_separator_width,
            phase_separator_column_width=phase_separator_column_width,
            phase_separator_color=phase_separator_color,
            user_label_colors=user_label_colors,
            png_dpi=png_dpi,
        )
    return _as_notebook_html(html)
