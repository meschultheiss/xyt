"""Interactive map visualizations for GPS traces and H3 cells."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

from .models import MobilityDataset


DEFAULT_TRACE_COLORS = (
    "#6b4c9a",
    "#e84b8a",
    "#4a90a4",
    "#d4e64a",
    "#f5a623",
    "#847996",
    "#28B4A8",
    "#DFAB9A",
    "#1D3557",
    "#bab0ab",
)

MODE_TRACE_COLORS = {
    "Autres": "#bab0ab",
    "Marche": "#f5a623",
    "TP": "#4a90a4",
    "TPU": "#76b7b2",
    "Train": "#4a90a4",
    "Bus": "#4a90a4",
    "Tram/Métro": "#76b7b2",
    "Voiture": "#e84b8a",
    "Carsharing": "#6b4c9a",
    "Vélo": "#d4e64a",
    "Vélo conventionnel": "#d4e64a",
    "Vélo électrique": "#96C8A6",
    "2RM": "#847996",
    "Taxi": "#DFAB9A",
    "Bateau": "#28B4A8",
    "Airplane": "#1D3557",
}

MAX_LEGEND_ITEMS = 12


def _import_folium():
    try:
        import folium
        from folium.plugins import AntPath, Fullscreen
    except ImportError as error:
        raise ImportError(
            "Interactive maps require the optional visualization dependency. "
            "Install it with `python -m pip install -r requirements-viz.txt`."
        ) from error
    return folium, AntPath, Fullscreen


def _import_h3_for_viz():
    try:
        import h3
    except ImportError as error:
        raise ImportError(
            "H3 maps require the optional analytics dependency. "
            "Install it with `python -m pip install -r requirements-analytics.txt`."
        ) from error
    return h3


def _as_wgs84(gdf: pd.DataFrame, *, geometry_col: str = "geometry") -> gpd.GeoDataFrame:
    if geometry_col not in gdf.columns:
        raise KeyError(f"Table must contain {geometry_col}.")
    out = gdf.copy()
    if not isinstance(out, gpd.GeoDataFrame):
        out = gpd.GeoDataFrame(out, geometry=geometry_col, crs="EPSG:4326")
    if out.crs is not None and str(out.crs).upper() != "EPSG:4326":
        out = out.to_crs("EPSG:4326")
    return out


def _filter_users(df: pd.DataFrame, user_ids: Iterable | None) -> pd.DataFrame:
    if user_ids is None or "user_id" not in df.columns:
        return df
    return df[df["user_id"].isin(set(user_ids))]


def _sample(df: pd.DataFrame, sample_n: int | None, random_state: int | None) -> pd.DataFrame:
    if sample_n is None or len(df) <= sample_n:
        return df
    return df.sample(n=sample_n, random_state=random_state).sort_index()


def _map_center(legs: gpd.GeoDataFrame, staypoints: gpd.GeoDataFrame | None) -> list[float]:
    frames = [legs]
    if staypoints is not None and not staypoints.empty:
        frames.append(staypoints)
    bounds = pd.concat([frame.geometry.dropna().bounds for frame in frames if not frame.empty])
    if bounds.empty:
        return [46.8, 8.2]
    lat = float((bounds["miny"].min() + bounds["maxy"].max()) / 2)
    lon = float((bounds["minx"].min() + bounds["maxx"].max()) / 2)
    return [lat, lon]


def _map_center_from_geometries(gdf: gpd.GeoDataFrame) -> list[float]:
    bounds = gdf.geometry.dropna().bounds
    if bounds.empty:
        return [46.8, 8.2]
    lat = float((bounds["miny"].min() + bounds["maxy"].max()) / 2)
    lon = float((bounds["minx"].min() + bounds["maxx"].max()) / 2)
    return [lat, lon]


def _line_locations(geometry) -> list[list[float]]:
    if isinstance(geometry, LineString):
        return [[lat, lon] for lon, lat in geometry.coords]
    if isinstance(geometry, MultiLineString):
        locations = []
        for part in geometry.geoms:
            locations.extend(_line_locations(part))
        return locations
    return []


def _point_location(geometry) -> list[float] | None:
    if isinstance(geometry, Point):
        return [geometry.y, geometry.x]
    return None


def _popup_html(row: pd.Series, columns: tuple[str, ...]) -> str:
    lines = ['<div class="xyt-map-popup">']
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            lines.append(
                "<div class=\"xyt-map-popup-row\">"
                f"<span>{escape(str(column))}</span>"
                f"<strong>{escape(str(row[column]))}</strong>"
                "</div>"
            )
    lines.append("</div>")
    return "".join(lines)


def _color_map(values: Iterable, preferred_colors: dict | None = None) -> dict:
    unique_values = [value for value in pd.Series(list(values)).dropna().drop_duplicates()]
    preferred_colors = preferred_colors or {}
    return {
        value: preferred_colors.get(
            value,
            preferred_colors.get(str(value), DEFAULT_TRACE_COLORS[index % len(DEFAULT_TRACE_COLORS)]),
        )
        for index, value in enumerate(unique_values)
    }


def _add_xyt_map_styles(
    map_object,
    folium_module,
    *,
    title: str | None = None,
    subtitle: str | None = None,
) -> None:
    style = """
<style>
.leaflet-container {
  font-family: Arial, sans-serif;
  color: #2b2340;
  background: #f8f7f4;
}
.xyt-map-title {
  position: fixed;
  top: 18px;
  left: 52px;
  z-index: 9999;
  max-width: 360px;
  padding: 10px 12px;
  border: 1px solid rgba(107, 76, 154, 0.14);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 8px 22px rgba(43, 35, 64, 0.12);
}
.xyt-map-title strong {
  display: block;
  font-size: 14px;
  line-height: 1.25;
  color: #2b2340;
}
.xyt-map-title span {
  display: block;
  margin-top: 3px;
  font-size: 11px;
  line-height: 1.25;
  color: #7e7494;
}
.xyt-map-legend {
  position: fixed;
  bottom: 24px;
  left: 24px;
  z-index: 9999;
  max-width: 280px;
  padding: 10px 12px;
  border: 1px solid rgba(107, 76, 154, 0.14);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 8px 22px rgba(43, 35, 64, 0.12);
  font-size: 12px;
  color: #2b2340;
}
.xyt-map-legend-title {
  display: block;
  margin-bottom: 7px;
  font-weight: 700;
  color: #6b4c9a;
}
.xyt-map-legend-item {
  display: flex;
  align-items: center;
  gap: 7px;
  margin: 3px 0;
  white-space: nowrap;
}
.xyt-map-legend-swatch {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  border: 1px solid rgba(43, 35, 64, 0.18);
  flex: 0 0 auto;
}
.xyt-map-legend-note {
  display: block;
  margin-top: 6px;
  color: #7e7494;
  font-size: 11px;
}
.xyt-map-popup {
  min-width: 210px;
  font-family: Arial, sans-serif;
}
.xyt-map-popup-row {
  display: grid;
  grid-template-columns: 86px 1fr;
  gap: 8px;
  padding: 4px 0;
  border-bottom: 1px solid #f3f0f8;
}
.xyt-map-popup-row:last-child {
  border-bottom: 0;
}
.xyt-map-popup-row span {
  color: #7e7494;
  font-size: 11px;
}
.xyt-map-popup-row strong {
  color: #2b2340;
  font-size: 12px;
  font-weight: 600;
}
</style>
"""
    map_object.get_root().header.add_child(folium_module.Element(style))
    if title:
        subtitle_html = f"<span>{escape(subtitle)}</span>" if subtitle else ""
        title_html = (
            '<div class="xyt-map-title">'
            f"<strong>{escape(title)}</strong>"
            f"{subtitle_html}"
            "</div>"
        )
        map_object.get_root().html.add_child(folium_module.Element(title_html))


def _add_xyt_legend(map_object, folium_module, *, title: str, colors: dict) -> None:
    if not colors:
        return
    items = []
    for index, (value, color) in enumerate(colors.items()):
        if index >= MAX_LEGEND_ITEMS:
            remaining = len(colors) - MAX_LEGEND_ITEMS
            items.append(f'<span class="xyt-map-legend-note">+ {remaining} autres valeurs</span>')
            break
        items.append(
            '<span class="xyt-map-legend-item">'
            f'<span class="xyt-map-legend-swatch" style="background:{escape(str(color))}"></span>'
            f"<span>{escape(str(value))}</span>"
            "</span>"
        )
    legend = (
        '<div class="xyt-map-legend">'
        f'<span class="xyt-map-legend-title">{escape(str(title))}</span>'
        f"{''.join(items)}"
        "</div>"
    )
    map_object.get_root().html.add_child(folium_module.Element(legend))


def _table_to_geodataframe(
    table: pd.DataFrame,
    *,
    geo_columns: str | tuple[str, str] | list[str] | None = None,
    geometry_col: str = "geometry",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    out = pd.DataFrame(table).copy()
    if geo_columns is not None:
        if isinstance(geo_columns, str):
            geometry_col = geo_columns
        elif isinstance(geo_columns, (tuple, list)) and len(geo_columns) == 2:
            lat_col, lon_col = geo_columns
            missing = [column for column in (lat_col, lon_col) if column not in out.columns]
            if missing:
                raise KeyError(f"GPS table missing coordinate column(s): {missing}")
            out[geometry_col] = [
                Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
                for lat, lon in zip(out[lat_col], out[lon_col])
            ]
        else:
            raise ValueError("geo_columns must be a geometry column name or a pair `(lat_col, lon_col)`.")
    elif geometry_col not in out.columns:
        coordinate_pairs = [
            ("latitude", "longitude"),
            ("lat", "lon"),
            ("y", "x"),
        ]
        for lat_col, lon_col in coordinate_pairs:
            if lat_col in out.columns and lon_col in out.columns:
                out[geometry_col] = [
                    Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
                    for lat, lon in zip(out[lat_col], out[lon_col])
                ]
                break
        else:
            raise KeyError(
                "GPS table must contain a geometry column, `(latitude, longitude)`, "
                "`(lat, lon)`, or `geo_columns=(lat_col, lon_col)`."
            )

    if isinstance(out, gpd.GeoDataFrame):
        gdf = out.set_geometry(geometry_col)
    else:
        gdf = gpd.GeoDataFrame(out, geometry=geometry_col, crs=crs)
    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _scaled_color(value: float, breaks: list[float], palette: tuple[str, ...]) -> str:
    for index, threshold in enumerate(breaks):
        if value <= threshold:
            return palette[min(index, len(palette) - 1)]
    return palette[-1]


def _map_center_from_h3_cells(h3_cells: pd.Series) -> list[float]:
    h3 = _import_h3_for_viz()
    centers = []
    for cell in h3_cells.dropna().astype(str).head(1000):
        try:
            lat, lon = h3.cell_to_latlng(cell)
            centers.append((lat, lon))
        except Exception:
            continue
    if not centers:
        return [46.8, 8.2]
    return [
        float(sum(lat for lat, _ in centers) / len(centers)),
        float(sum(lon for _, lon in centers) / len(centers)),
    ]


def plot_h3_frequency_map(
    h3_frequency: pd.DataFrame,
    *,
    h3_col: str = "h3_cell",
    value_col: str = "point_count",
    tooltip_cols: Iterable[str] | None = None,
    aggregate_cells: bool = True,
    max_cells: int | None = 2500,
    palette: tuple[str, ...] = ("#f4f1f8", "#d9cce9", "#b79bd5", "#8f67bd", "#6b4c9a", "#3d275f"),
    fill_opacity: float = 0.62,
    line_opacity: float = 0.25,
    tiles: str = "cartodbpositron",
    zoom_start: int = 11,
    map_center: tuple[float, float] | None = (46.2074, 6.1566),
    save_path: str | Path | None = None,
) -> object:
    """Plot H3 fréquentation cells on an interactive Folium map.

    Args:
        h3_frequency: Table produced by `aggregate_h3_frequencies()` or any
            table containing an H3 cell column and a numeric count.
        h3_col: Column containing H3 cell identifiers.
        value_col: Numeric column used to color cells.
        tooltip_cols: Columns shown in the popup. When omitted, common count
            columns are used when present.
        aggregate_cells: If true, aggregate rows by H3 cell before plotting.
            This is useful when the input is split by mode, phase or project.
        max_cells: Optional maximum number of cells to draw. The most frequent
            cells are kept. Set to `None` to draw all cells, which can be slow.
        palette: Sequential color palette from low to high values.
        fill_opacity: Polygon fill opacity.
        line_opacity: Polygon border opacity.
        tiles: Folium base map.
        zoom_start: Initial zoom level.
        map_center: Optional `(lat, lon)` map center. If omitted, it is
            estimated from the H3 cells.
        save_path: Optional HTML output path.

    Returns:
        A `folium.Map` object.

    Raises:
        ImportError: If `folium` or `h3` is not installed.
        KeyError: If required columns are missing.
        ValueError: If no valid H3 cell can be plotted.
    """

    folium, _, Fullscreen = _import_folium()
    h3 = _import_h3_for_viz()

    required = {h3_col, value_col}
    missing = required - set(h3_frequency.columns)
    if missing:
        raise KeyError(f"h3_frequency missing required column(s): {sorted(missing)}")

    table = h3_frequency.copy()
    table[value_col] = pd.to_numeric(table[value_col], errors="coerce").fillna(0)
    table = table.loc[table[h3_col].notna() & table[value_col].gt(0)].copy()
    if table.empty:
        raise ValueError("h3_frequency has no positive value to plot.")

    if aggregate_cells:
        aggregations: dict[str, tuple[str, str]] = {value_col: (value_col, "sum")}
        if "user_count" in table.columns and "user_count" != value_col:
            aggregations["user_count"] = ("user_count", "max")
        for column in ("leg_count", "trip_count"):
            if column in table.columns and column != value_col:
                aggregations[column] = (column, "sum")
        table = table.groupby(h3_col, dropna=False).agg(**aggregations).reset_index()

    table = table.sort_values(value_col, ascending=False)
    if max_cells is not None and len(table) > max_cells:
        table = table.head(int(max_cells)).copy()

    values = table[value_col].astype(float)
    quantiles = values.quantile([0.2, 0.4, 0.6, 0.8]).drop_duplicates().tolist()
    breaks = quantiles if quantiles else [float(values.max())]

    center = list(map_center) if map_center is not None else _map_center_from_h3_cells(table[h3_col])
    h3_map = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=tiles,
        control_scale=True,
        prefer_canvas=True,
    )
    _add_xyt_map_styles(h3_map, folium)

    default_tooltip_cols = [
        column
        for column in (h3_col, value_col, "user_count", "leg_count", "trip_count")
        if column in table.columns
    ]
    popup_cols = list(tooltip_cols) if tooltip_cols is not None else default_tooltip_cols

    for _, row in table.iterrows():
        cell = str(row[h3_col])
        try:
            boundary = h3.cell_to_boundary(cell)
        except Exception:
            continue
        locations = [[lat, lon] for lat, lon in boundary]
        value = float(row[value_col])
        color = _scaled_color(value, breaks, palette)
        popup = folium.Popup(_popup_html(row, tuple(popup_cols)), max_width=340)
        folium.Polygon(
            locations=locations,
            color=color,
            weight=1,
            opacity=line_opacity,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            popup=popup,
            tooltip=f"{value_col}: {value:,.0f}".replace(",", " "),
        ).add_to(h3_map)

    legend_cells = "".join(
        f'<span style="display:inline-block;width:18px;height:10px;background:{color};'
        'margin-right:3px;border:1px solid rgba(0,0,0,0.08);"></span>'
        for color in palette
    )
    legend = f"""
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
background: white; border: 1px solid #ddd; border-radius: 6px; padding: 10px;
font-size: 12px; color: #3d275f; box-shadow: 0 2px 8px rgba(0,0,0,0.12);">
  <b>Fréquentation H3</b><br>
  <span style="color:#6b5f7a;">{value_col}</span><br>
  <span>Moins</span> {legend_cells} <span>Plus</span><br>
  <span style="color:#6b5f7a;">Cellules affichées : {len(table)}</span>
</div>
"""
    h3_map.get_root().html.add_child(folium.Element(legend))
    Fullscreen().add_to(h3_map)
    folium.LayerControl().add_to(h3_map)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        h3_map.save(output_path)

    return h3_map


def plot_gps_on_map(
    table: pd.DataFrame | gpd.GeoDataFrame,
    *,
    trace_type: str | Iterable[str] | None = None,
    type_col: str = "type",
    user_col: str = "user_id",
    home_col: str | None = None,
    work_col: str | None = None,
    geo_columns: str | tuple[str, str] | list[str] | None = None,
    geometry_col: str = "geometry",
    color_by: str | None = "user_id",
    popup_cols: Iterable[str] | None = None,
    sample_n: int | None = None,
    random_state: int | None = 42,
    tiles: str = "cartodbpositron",
    zoom_start: int = 12,
    map_center: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> object:
    """Plot a generic GPS table on a Folium map.

    This function is the `xyt_gps` adaptation of the historical
    `xyt_plot.plot_gps_on_map()` helper. It is intentionally permissive for
    exploratory checks: the input can contain points, lines, a `geometry`
    column, or explicit latitude/longitude columns. For structured mobility
    datasets, prefer `plot_gps_traces()`.

    Args:
        table: Table containing GPS observations or mobility objects.
        trace_type: Optional type or list of types to keep, using `type_col`.
        type_col: Column containing values such as `Staypoint`, `Track`, `Leg`
            or `Waypoint`.
        user_col: User identifier column, used for default coloring and
            home/work markers.
        home_col: Optional column containing home `Point` geometries.
        work_col: Optional column containing work `Point` geometries.
        geo_columns: Geometry source. Pass a geometry column name, or
            `(lat_col, lon_col)` when coordinates are stored in two columns.
        geometry_col: Name of the geometry column to read or create.
        color_by: Column used to color objects. Defaults to `user_id`.
        popup_cols: Columns shown in popups. Common GPS columns are used when
            omitted.
        sample_n: Optional sample size for large tables.
        random_state: Random seed used for sampling.
        tiles: Folium base map.
        zoom_start: Initial zoom level.
        map_center: Optional `(lat, lon)` center. If omitted, the center is
            computed from geometries.
        save_path: Optional HTML output path.

    Returns:
        A `folium.Map` object.
    """

    folium, _, Fullscreen = _import_folium()
    gdf = _table_to_geodataframe(table, geo_columns=geo_columns, geometry_col=geometry_col)

    if trace_type is not None:
        if type_col not in gdf.columns:
            raise KeyError(f"GPS table must contain {type_col} when trace_type is used.")
        kept_types = {trace_type} if isinstance(trace_type, str) else set(trace_type)
        gdf = gdf.loc[gdf[type_col].isin(kept_types)].copy()

    gdf = _sample(gdf, sample_n, random_state)
    gdf = gdf.loc[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise ValueError("No GPS geometry available for plotting.")

    center = list(map_center) if map_center is not None else _map_center_from_geometries(gdf)
    gps_map = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=tiles,
        control_scale=True,
        prefer_canvas=True,
    )
    _add_xyt_map_styles(gps_map, folium)

    colors = _color_map(gdf[color_by]) if color_by and color_by in gdf.columns else {}
    default_color = "#6b4c9a"
    default_popup_cols = tuple(
        column
        for column in (
            type_col,
            user_col,
            "started_at",
            "finished_at",
            "mode_niv1",
            "purpose_niv1",
            "leg_id",
            "activity_id",
            "trip_id",
        )
        if column in gdf.columns
    )
    popup_columns = tuple(popup_cols) if popup_cols is not None else default_popup_cols

    for _, row in gdf.iterrows():
        geometry = row.geometry
        color = colors.get(row.get(color_by), default_color) if color_by else default_color
        popup = folium.Popup(_popup_html(row, popup_columns), max_width=360)
        tooltip = str(row.get(type_col, "GPS")) if type_col in row.index else "GPS"
        point_location = _point_location(geometry)
        line_locations = _line_locations(geometry)

        if point_location is not None:
            is_staypoint = str(row.get(type_col, "")).lower() in {"stay", "staypoint", "activity"}
            folium.CircleMarker(
                location=point_location,
                radius=5 if is_staypoint else 3,
                color="#e84b8a" if is_staypoint else color,
                fill=True,
                fill_color="#e84b8a" if is_staypoint else color,
                fill_opacity=0.8,
                popup=popup,
                tooltip=tooltip,
            ).add_to(gps_map)
        elif line_locations:
            folium.PolyLine(
                locations=line_locations,
                color=color,
                weight=2.5,
                opacity=0.78,
                popup=popup,
                tooltip=tooltip,
            ).add_to(gps_map)

    plotted_home: set[object] = set()
    plotted_work: set[object] = set()
    if user_col in gdf.columns:
        for _, row in gdf.iterrows():
            user_id = row[user_col]
            if home_col and home_col in row.index and user_id not in plotted_home:
                home_location = _point_location(row[home_col])
                if home_location is not None:
                    folium.Marker(
                        location=home_location,
                        icon=folium.Icon(color="red", icon="home", prefix="fa"),
                        tooltip=f"{user_id} · domicile",
                    ).add_to(gps_map)
                    plotted_home.add(user_id)
            if work_col and work_col in row.index and user_id not in plotted_work:
                work_location = _point_location(row[work_col])
                if work_location is not None:
                    folium.Marker(
                        location=work_location,
                        icon=folium.Icon(color="blue", icon="briefcase", prefix="fa"),
                        tooltip=f"{user_id} · travail",
                    ).add_to(gps_map)
                    plotted_work.add(user_id)

    if colors:
        _add_xyt_legend(gps_map, folium, title=str(color_by), colors=colors)

    Fullscreen().add_to(gps_map)
    folium.LayerControl().add_to(gps_map)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gps_map.save(output_path)

    return gps_map


def plot_gps_traces(
    dataset_or_legs: MobilityDataset | gpd.GeoDataFrame,
    *,
    staypoints: gpd.GeoDataFrame | None = None,
    user_ids: Iterable | None = None,
    sample_n: int | None = None,
    random_state: int | None = 42,
    color_by: str | None = "mode_niv1",
    geometry_col: str = "geometry",
    show_staypoints: bool = True,
    use_antpath: bool = False,
    tiles: str = "cartodbpositron",
    zoom_start: int = 9,
    map_center: tuple[float, float] | None = (46.2074, 6.1566),
    save_path: str | Path | None = None,
) -> object:
    """Plot GPS legs and optional staypoints on an interactive Folium map.

    The function is designed for notebook checks. It keeps the API small:
    pass a `MobilityDataset` after transformation, or pass a leg
    GeoDataFrame directly.

    Args:
        dataset_or_legs: `MobilityDataset` or leg GeoDataFrame.
        staypoints: Optional staypoint GeoDataFrame when passing legs
            directly.
        user_ids: Optional users to display.
        sample_n: Optional number of legs to sample before plotting.
        random_state: Random seed used for leg sampling.
        color_by: Column used to color legs. Set to `None` for one color.
        geometry_col: Geometry column name.
        show_staypoints: Whether to draw staypoints as circles.
        use_antpath: Whether to animate legs with Folium `AntPath`.
        tiles: Folium base map.
        zoom_start: Initial zoom level.
        map_center: Optional `(lat, lon)` center. Defaults to Geneva's
            Jet d'Eau.
        save_path: Optional HTML output path.

    Returns:
        A `folium.Map` object.

    Raises:
        ImportError: If `folium` is not installed.
        KeyError: If the geometry column is missing.
        ValueError: If no leg geometry is available for plotting.
    """

    folium, AntPath, Fullscreen = _import_folium()

    if isinstance(dataset_or_legs, MobilityDataset):
        legs = dataset_or_legs.legs
        staypoints = dataset_or_legs.staypoints if staypoints is None else staypoints
    else:
        legs = dataset_or_legs

    plot_legs = _as_wgs84(legs, geometry_col=geometry_col)
    plot_legs = _filter_users(plot_legs, user_ids)
    plot_legs = _sample(plot_legs, sample_n, random_state)
    plot_legs = plot_legs.loc[plot_legs[geometry_col].notna()].copy()
    if plot_legs.empty:
        raise ValueError("No leg geometry available for plotting.")

    plot_staypoints = None
    if show_staypoints and staypoints is not None:
        plot_staypoints = _as_wgs84(staypoints, geometry_col=geometry_col)
        plot_staypoints = _filter_users(plot_staypoints, user_ids)
        plot_staypoints = plot_staypoints.loc[plot_staypoints[geometry_col].notna()].copy()

    center = list(map_center) if map_center is not None else _map_center(plot_legs, plot_staypoints)
    mobility_map = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=tiles,
        control_scale=True,
        prefer_canvas=True,
    )
    users_count = plot_legs["user_id"].nunique() if "user_id" in plot_legs.columns else "n/a"
    legs_count = f"{len(plot_legs):,}".replace(",", " ")
    _add_xyt_map_styles(
        mobility_map,
        folium,
        title="Traces GPS",
        subtitle=f"{legs_count} étapes · {users_count} utilisateurs",
    )

    colors = _color_map(plot_legs[color_by], MODE_TRACE_COLORS) if color_by and color_by in plot_legs.columns else {}
    default_color = "#6b4c9a"
    leg_popup_columns = (
        "leg_id",
        "user_id",
        "started_at",
        "finished_at",
        "mode",
        "mode_niv1",
        "relative_signal_loss",
        "low_quality_signal",
    )
    for _, row in plot_legs.iterrows():
        locations = _line_locations(row[geometry_col])
        if len(locations) < 2:
            continue
        color = colors.get(row.get(color_by), default_color) if color_by else default_color
        popup = folium.Popup(_popup_html(row, leg_popup_columns), max_width=360)
        tooltip = str(row.get(color_by, "leg")) if color_by else "leg"
        if use_antpath:
            AntPath(
                locations=locations,
                color=color,
                pulse_color=color,
                weight=2.4,
                opacity=0.68,
                delay=1500,
            ).add_to(mobility_map)
            folium.PolyLine(locations=locations, color=color, weight=2.2, opacity=0.18, popup=popup).add_to(
                mobility_map
            )
        else:
            folium.PolyLine(
                locations=locations,
                color=color,
                weight=2.4,
                opacity=0.72,
                popup=popup,
                tooltip=tooltip,
            ).add_to(mobility_map)

    if plot_staypoints is not None and not plot_staypoints.empty:
        stay_popup_columns = ("activity_id", "user_id", "started_at", "finished_at", "purpose_niv1", "duration")
        for _, row in plot_staypoints.iterrows():
            geometry = row[geometry_col]
            if not isinstance(geometry, Point):
                continue
            folium.CircleMarker(
                location=[geometry.y, geometry.x],
                radius=3.8,
                color="#e84b8a",
                weight=1.4,
                fill=True,
                fill_color="#ffffff",
                fill_opacity=0.92,
                popup=folium.Popup(_popup_html(row, stay_popup_columns), max_width=360),
                tooltip=str(row.get("purpose_niv1", "staypoint")),
            ).add_to(mobility_map)

    if colors:
        _add_xyt_legend(mobility_map, folium, title=str(color_by), colors=colors)

    Fullscreen().add_to(mobility_map)
    folium.LayerControl().add_to(mobility_map)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mobility_map.save(output_path)

    return mobility_map
