"""Notebook-friendly mobility motif graph visualizations."""

from __future__ import annotations

import ast
from html import escape
from math import cos, pi, sin, sqrt
from pathlib import Path

import pandas as pd


def _as_notebook_html(html: str):
    try:
        from IPython.display import HTML
    except ImportError:
        return html
    return HTML(html)


def _is_missing_scalar(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _parse_signature(signature: object) -> tuple[int | None, list[int] | None]:
    if _is_missing_scalar(signature):
        return None, None
    text = str(signature)
    if not text.startswith("n=") or ";adj=" not in text:
        return None, None
    n_part, adjacency_part = text.split(";adj=", maxsplit=1)
    try:
        n_locations = int(n_part.removeprefix("n="))
    except ValueError:
        n_locations = None
    try:
        values = [int(value) for value in adjacency_part.split(",") if value != ""]
    except ValueError:
        values = None
    return n_locations, values


def _parse_adjacency_matrix(adjacency: object, signature: object = None) -> list[list[int]]:
    n_locations, signature_values = _parse_signature(signature)
    values = None

    if not _is_missing_scalar(adjacency):
        if isinstance(adjacency, str):
            try:
                parsed = ast.literal_eval(adjacency)
            except (SyntaxError, ValueError):
                parsed = [value.strip() for value in adjacency.split(",") if value.strip()]
            values = [int(value) for value in parsed]
        else:
            values = [int(value) for value in adjacency]

    if values is None:
        values = signature_values
    if values is None:
        raise ValueError("Cannot parse motif adjacency. Keep `adjacency_flat` or `motif_signature` in the motif table.")

    if n_locations is None:
        n_locations = int(sqrt(len(values)))
    if n_locations <= 0 or n_locations * n_locations != len(values):
        raise ValueError("Motif adjacency length must be a square number.")

    return [
        values[row_start : row_start + n_locations]
        for row_start in range(0, len(values), n_locations)
    ]


def _node_positions(n_nodes: int, *, width: int, height: int, radius: float = 58.0) -> dict[int, tuple[float, float]]:
    if n_nodes <= 1:
        return {0: (width / 2, height / 2)}
    center_x = width / 2
    center_y = height / 2
    return {
        node: (
            center_x + radius * cos(-pi / 2 + 2 * pi * node / n_nodes),
            center_y + radius * sin(-pi / 2 + 2 * pi * node / n_nodes),
        )
        for node in range(n_nodes)
    }


def _edge_path(start: tuple[float, float], end: tuple[float, float], *, node_radius: float) -> str:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    distance = max((dx**2 + dy**2) ** 0.5, 1.0)
    ux = dx / distance
    uy = dy / distance
    sx += ux * node_radius
    sy += uy * node_radius
    ex -= ux * (node_radius + 3)
    ey -= uy * (node_radius + 3)
    control_x = (sx + ex) / 2 - uy * 18
    control_y = (sy + ey) / 2 + ux * 18
    return f"M {sx:.1f} {sy:.1f} Q {control_x:.1f} {control_y:.1f} {ex:.1f} {ey:.1f}"


def _loop_path(position: tuple[float, float], *, node_radius: float) -> str:
    x, y = position
    return (
        f"M {x + node_radius * 0.6:.1f} {y - node_radius * 0.7:.1f} "
        f"C {x + 50:.1f} {y - 58:.1f}, {x + 58:.1f} {y + 28:.1f}, "
        f"{x + node_radius * 0.9:.1f} {y + node_radius * 0.35:.1f}"
    )


def _motif_svg(matrix: list[list[int]], *, motif_key: str, width: int = 190, height: int = 170) -> str:
    n_nodes = len(matrix)
    node_radius = 13.0
    positions = _node_positions(n_nodes, width=width, height=height)
    marker_id = f"arrow-{motif_key}"

    edge_parts = []
    for origin, row in enumerate(matrix):
        for destination, weight in enumerate(row):
            if weight <= 0:
                continue
            path = (
                _loop_path(positions[origin], node_radius=node_radius)
                if origin == destination
                else _edge_path(positions[origin], positions[destination], node_radius=node_radius)
            )
            edge_parts.append(
                "<path "
                f'd="{path}" '
                'fill="none" stroke="#3f3457" stroke-opacity="0.82" '
                f'stroke-width="{1.2 + min(weight, 3) * 0.55:.2f}" '
                f'marker-end="url(#{marker_id})" />'
            )

    node_parts = []
    for node, (x, y) in positions.items():
        fill = "#dbcc9c" if node == 0 else "#6b4c9a"
        stroke = "#3f3457" if node == 0 else "#4c3575"
        text_color = "#3f3457" if node == 0 else "#ffffff"
        node_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node_radius:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.2" />'
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
            f'font-size="11" font-family="Arial, sans-serif" fill="{text_color}">{node}</text>'
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Graphe du motif {escape(str(motif_key))}">
      <defs>
        <marker id="{marker_id}" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">
          <path d="M 0 0 L 7 3.5 L 0 7 z" fill="#3f3457" fill-opacity="0.82"></path>
        </marker>
      </defs>
      <rect x="0" y="0" width="{width}" height="{height}" rx="8" fill="#fbfaf7"></rect>
      {''.join(edge_parts)}
      {''.join(node_parts)}
    </svg>
    """


def plot_mobility_motif_graphs(
    motifs: pd.DataFrame,
    *,
    motif_id_col: str = "motif_id",
    adjacency_col: str = "adjacency_flat",
    signature_col: str = "motif_signature",
    top_n: int = 9,
    other_motif_id: int = 99,
    include_other: bool = False,
    title: str = "Graphes des motifs journaliers",
    save_path: str | Path | None = None,
) -> object:
    """Render frequent daily mobility motifs as directed graph cards.

    This is the lightweight package equivalent of the historical
    `GPStoGraph.plot_motif()` helper. It reads the motif table produced by
    `build_mobility_motifs()` and draws one directed graph per frequent motif.
    Node `0` is the first visited place in the canonical daily sequence; arrows
    represent observed transitions between canonical places.

    Args:
        motifs: Motif table returned by `build_mobility_motifs()`.
        motif_id_col: Column containing the motif id.
        adjacency_col: Column containing the flattened adjacency matrix.
        signature_col: Column containing the portable motif signature. Used as
            fallback when `adjacency_col` is absent.
        top_n: Maximum number of motif graphs to draw.
        other_motif_id: Id used for rare motifs.
        include_other: Whether to include the rare-motif bucket.
        title: Figure title.
        save_path: Optional HTML file path.

    Returns:
        An `IPython.display.HTML` object when IPython is available, otherwise
        the raw HTML string.
    """

    required = {motif_id_col}
    if adjacency_col not in motifs.columns and signature_col not in motifs.columns:
        required.add(adjacency_col)
    missing = required - set(motifs.columns)
    if missing:
        raise KeyError(f"motifs missing required column(s): {sorted(missing)}")
    if motifs.empty:
        raise ValueError("motifs is empty.")

    work = motifs.copy()
    if not include_other:
        work = work.loc[~work[motif_id_col].eq(other_motif_id)].copy()
    if work.empty:
        raise ValueError("No motif available to plot after filtering.")

    total_days = len(motifs)
    rows = []
    group_columns = [motif_id_col, signature_col] if signature_col in work.columns else [motif_id_col]
    for keys, group in work.groupby(group_columns, dropna=False, sort=True):
        motif_id = keys[0] if isinstance(keys, tuple) else keys
        signature = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else None
        representative = group.iloc[0]
        matrix = _parse_adjacency_matrix(
            representative[adjacency_col] if adjacency_col in representative.index else None,
            signature=signature,
        )
        rows.append(
            {
                "motif_id": motif_id,
                "signature": signature,
                "days_count": len(group),
                "days_share": len(group) / total_days,
                "n_locations": len(matrix),
                "matrix": matrix,
            }
        )

    rows = sorted(rows, key=lambda row: (-row["days_count"], row["motif_id"]))[:top_n]

    cards = []
    for index, row in enumerate(rows, start=1):
        motif_key = f"{index}-{escape(str(row['motif_id']))}".replace(" ", "-")
        share_pct = row["days_share"] * 100
        cards.append(
            f"""
            <article class="xyt-motif-card">
              <div class="xyt-motif-card-head">
                <div>
                  <h3>Motif {escape(str(row["motif_id"]))}</h3>
                  <p>{row["days_count"]} jours · {share_pct:.1f}%</p>
                </div>
                <span>{row["n_locations"]} lieux</span>
              </div>
              {_motif_svg(row["matrix"], motif_key=motif_key)}
            </article>
            """
        )

    html = f"""
    <section class="xyt-motif-graphs">
      <style>
        .xyt-motif-graphs {{
          font-family: Arial, sans-serif;
          color: #3f3457;
          background: #fffdf8;
          border: 1px solid #e8e0ef;
          border-radius: 12px;
          padding: 18px;
        }}
        .xyt-motif-graphs h2 {{
          margin: 0 0 6px;
          font-size: 20px;
          font-weight: 700;
        }}
        .xyt-motif-graphs .xyt-motif-note {{
          margin: 0 0 16px;
          color: #746a85;
          font-size: 13px;
        }}
        .xyt-motif-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 12px;
        }}
        .xyt-motif-card {{
          border: 1px solid #ded4e8;
          border-radius: 10px;
          background: #ffffff;
          padding: 10px;
        }}
        .xyt-motif-card-head {{
          display: flex;
          justify-content: space-between;
          gap: 8px;
          align-items: flex-start;
          margin-bottom: 6px;
        }}
        .xyt-motif-card h3 {{
          margin: 0;
          font-size: 15px;
          font-weight: 700;
        }}
        .xyt-motif-card p {{
          margin: 2px 0 0;
          color: #746a85;
          font-size: 12px;
        }}
        .xyt-motif-card span {{
          flex: 0 0 auto;
          color: #e84b8a;
          font-size: 12px;
          font-weight: 700;
        }}
        .xyt-motif-card svg {{
          width: 100%;
          height: auto;
          display: block;
        }}
      </style>
      <h2>{escape(title)}</h2>
      <p class="xyt-motif-note">Le nœud 0 correspond au premier lieu visité dans la séquence canonique. Les flèches indiquent les transitions entre lieux, indépendamment des coordonnées réelles.</p>
      <div class="xyt-motif-grid">
        {''.join(cards)}
      </div>
    </section>
    """

    if save_path is not None:
        Path(save_path).write_text(html, encoding="utf-8")
    return _as_notebook_html(html)
