"""Spatial labels, area relations and excursion flags."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from .spatial_geometry import _area_from_inputs, _as_geodataframe, _line_endpoint, _read_zones

ZONE_ID_COL_CANDIDATES = (
    "zone_id",
    "id_zone",
    "ID_Zone",
    "home_zone_id",
    "NUMERO_COM",
    "OBJECTID",
    "id",
    "ID",
)
ZONE_NAME_COL_CANDIDATES = (
    "zone_name",
    "name",
    "nom",
    "NOM",
    "COMMUNE",
    "local_admin_unit_id",
    "LIBELLE",
)


def _labels_for_points(
    points: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    *,
    zone_label_col: str,
    fill_value: str,
    predicate: str = "within",
) -> pd.Series:
    if points.empty:
        return pd.Series(dtype=object)
    if zone_label_col not in zones.columns:
        raise KeyError(f"zones must contain {zone_label_col}")
    joined = gpd.sjoin(
        points,
        zones[[zone_label_col, "geometry"]],
        how="left",
        predicate=predicate,
    )
    labels = (
        joined.sort_values("_xyt_row_id")
        .drop_duplicates(subset="_xyt_row_id")
        .set_index("_xyt_row_id")[zone_label_col]
    )
    return labels.fillna(fill_value)


def add_excursion_flags(
    table: pd.DataFrame,
    *,
    area: BaseGeometry | gpd.GeoDataFrame | None = None,
    area_path: str | None = None,
    center: tuple[float, float] | None = None,
    radius_m: float | None = None,
    geometry_col: str = "geometry",
    flag_col: str = "excursion",
    mode: str = "outside",
    target_crs: str = "EPSG:4326",
    operations_crs: str = "EPSG:2056",
) -> gpd.GeoDataFrame:
    """Flag geographic excursions on a geometry table."""

    if mode not in {"outside", "inside"}:
        raise ValueError("mode must be 'outside' or 'inside'.")

    out = _as_geodataframe(table, geometry_col=geometry_col, default_crs=target_crs)
    reference_area = _area_from_inputs(
        area=area,
        area_path=area_path,
        center=center,
        radius_m=radius_m,
        target_crs=target_crs,
        operations_crs=operations_crs,
    )
    metric = out.to_crs(operations_crs)
    valid_geometry = metric[geometry_col].notna()
    if mode == "outside":
        flags = valid_geometry & ~metric[geometry_col].within(reference_area)
    else:
        flags = valid_geometry & metric[geometry_col].intersects(reference_area)
    out[flag_col] = flags.fillna(False).astype(int).to_numpy()
    return out


def add_spatial_zone_labels(
    table: pd.DataFrame,
    zones: str | Path | gpd.GeoDataFrame,
    *,
    zone_label_col: str,
    output_col: str,
    geometry_col: str = "geometry",
    predicate: str = "within",
    fill_value: str = "Unknown",
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Add a zone label to a geometry table by spatial join."""

    out = _as_geodataframe(table, geometry_col=geometry_col, default_crs=target_crs)
    zones_gdf = _read_zones(zones, default_crs=target_crs).to_crs(out.crs)
    temp = out[[geometry_col]].copy()
    temp["_xyt_row_id"] = range(len(temp))
    temp = temp.set_geometry(geometry_col)
    labels = _labels_for_points(
        temp,
        zones_gdf,
        zone_label_col=zone_label_col,
        fill_value=fill_value,
        predicate=predicate,
    )
    out[output_col] = pd.Series(range(len(out)), index=out.index).map(labels).fillna(fill_value)
    return out


def add_leg_origin_destination_zones(
    legs: pd.DataFrame,
    zones: str | Path | gpd.GeoDataFrame,
    *,
    zone_label_col: str,
    origin_col: str = "origin_zone",
    destination_col: str = "destination_zone",
    geometry_col: str = "geometry",
    fill_value: str = "Outside",
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Label leg origins and destinations from a zone layer."""

    out = _as_geodataframe(legs, geometry_col=geometry_col, default_crs=target_crs)
    zones_gdf = _read_zones(zones, default_crs=target_crs).to_crs(out.crs)

    base = pd.DataFrame({"_xyt_row_id": range(len(out))}, index=out.index)
    origin_points = gpd.GeoDataFrame(
        base.copy(),
        geometry=out[geometry_col].apply(lambda geom: _line_endpoint(geom, position="origin")),
        crs=out.crs,
    ).dropna(subset=["geometry"])
    destination_points = gpd.GeoDataFrame(
        base.copy(),
        geometry=out[geometry_col].apply(lambda geom: _line_endpoint(geom, position="destination")),
        crs=out.crs,
    ).dropna(subset=["geometry"])

    origin_labels = _labels_for_points(
        origin_points,
        zones_gdf,
        zone_label_col=zone_label_col,
        fill_value=fill_value,
    )
    destination_labels = _labels_for_points(
        destination_points,
        zones_gdf,
        zone_label_col=zone_label_col,
        fill_value=fill_value,
    )
    row_ids = pd.Series(range(len(out)), index=out.index)
    out[origin_col] = row_ids.map(origin_labels).fillna(fill_value)
    out[destination_col] = row_ids.map(destination_labels).fillna(fill_value)
    return out


def _safe_zone_layer_key(layer_key: str) -> str:
    safe_key = re.sub(r"[^0-9A-Za-z]+", "_", str(layer_key).strip().lower()).strip("_")
    if not safe_key:
        raise ValueError("Zone layer keys must contain at least one alphanumeric character.")
    return safe_key


def _first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _read_zone_layer_from_spec(
    layer_key: str,
    layer_spec: str | Path | gpd.GeoDataFrame | Mapping[str, Any],
    *,
    target_crs: str,
) -> tuple[gpd.GeoDataFrame, str | None, str | None, str]:
    if isinstance(layer_spec, Mapping):
        source = layer_spec.get("zones", layer_spec.get("path", layer_spec.get("zones_path")))
        if source is None:
            raise ValueError(f"Zone layer {layer_key!r} must define 'path', 'zones_path' or 'zones'.")
        layer_name = layer_spec.get("layer")
        zone_id_col = layer_spec.get("zone_id_col")
        zone_name_col = layer_spec.get("zone_name_col")
        output_prefix = str(layer_spec.get("output_prefix", layer_key))
    else:
        source = layer_spec
        layer_name = None
        zone_id_col = None
        zone_name_col = None
        output_prefix = layer_key

    if isinstance(source, gpd.GeoDataFrame):
        zones = source.copy()
    elif isinstance(source, (str, Path)):
        path = Path(source)
        read_kwargs = {"layer": layer_name} if layer_name else {}
        zones = gpd.read_file(path, **read_kwargs)
    else:
        raise TypeError(f"Unsupported zone source for {layer_key!r}: {type(source).__name__}.")

    if zones.crs is None:
        zones = zones.set_crs(target_crs)
    return zones, zone_id_col, zone_name_col, output_prefix


def _prepare_zone_layer(
    layer_key: str,
    layer_spec: str | Path | gpd.GeoDataFrame | Mapping[str, Any],
    *,
    target_crs: str,
    id_candidates: Sequence[str],
    name_candidates: Sequence[str],
) -> tuple[gpd.GeoDataFrame, str]:
    zones, configured_id_col, configured_name_col, output_prefix = _read_zone_layer_from_spec(
        layer_key,
        layer_spec,
        target_crs=target_crs,
    )
    zone_key = _safe_zone_layer_key(output_prefix)
    id_col = configured_id_col or _first_existing_column(zones.columns, id_candidates)
    name_col = configured_name_col or _first_existing_column(zones.columns, name_candidates)

    if id_col is None or id_col not in zones.columns:
        id_col = "_xyt_generated_zone_id"
        warnings.warn(
            f"Zone layer {layer_key!r} does not define a usable zone id column; "
            f"generated {id_col}.",
            UserWarning,
            stacklevel=3,
        )
        zones[id_col] = [f"{zone_key}_{index + 1:06d}" for index in range(len(zones))]
    elif zones[id_col].isna().any():
        warnings.warn(
            f"Zone layer {layer_key!r} contains missing values in {id_col}; "
            "generated ids are used for missing rows.",
            UserWarning,
            stacklevel=3,
        )
        generated = pd.Series(
            [f"{zone_key}_{index + 1:06d}" for index in range(len(zones))],
            index=zones.index,
            dtype="string",
        )
        zones[id_col] = zones[id_col].astype("string").fillna(generated)

    if name_col is None or name_col not in zones.columns:
        name_col = "_xyt_generated_zone_name"
        warnings.warn(
            f"Zone layer {layer_key!r} does not define a usable zone name column; "
            f"generated {name_col}.",
            UserWarning,
            stacklevel=3,
        )
        zones[name_col] = zones[id_col].astype("string")
    elif zones[name_col].isna().any():
        warnings.warn(
            f"Zone layer {layer_key!r} contains missing values in {name_col}; "
            "zone ids are used for missing names.",
            UserWarning,
            stacklevel=3,
        )
        zones[name_col] = zones[name_col].astype("string").fillna(zones[id_col].astype("string"))

    prepared = zones[[id_col, name_col, "geometry"]].copy()
    prepared = prepared.rename(columns={id_col: "_xyt_zone_id", name_col: "_xyt_zone_name"})
    prepared["_xyt_zone_id"] = prepared["_xyt_zone_id"].astype("string")
    prepared["_xyt_zone_name"] = prepared["_xyt_zone_name"].astype("string")
    prepared = prepared.dropna(subset=["geometry"])
    if not prepared.empty:
        prepared["_xyt_geometry_wkb"] = prepared.geometry.to_wkb(hex=True)
        prepared = prepared.drop_duplicates(subset=["_xyt_zone_id", "_xyt_geometry_wkb"])
        prepared = prepared.drop(columns="_xyt_geometry_wkb")
    return prepared, zone_key


def build_origin_destination_zone_correspondence(
    legs: pd.DataFrame,
    zone_layers: Mapping[str, str | Path | gpd.GeoDataFrame | Mapping[str, Any]],
    *,
    map_track_trip_journey: pd.DataFrame | None = None,
    leg_id_col: str = "leg_id",
    trip_id_col: str = "trip_id",
    journey_id_col: str = "journey_id",
    user_id_col: str = "user_id",
    experiment_col: str = "experiment_name",
    geometry_col: str = "geometry",
    fill_value: str = "Outside",
    target_crs: str = "EPSG:4326",
    zone_id_candidates: Sequence[str] = ZONE_ID_COL_CANDIDATES,
    zone_name_candidates: Sequence[str] = ZONE_NAME_COL_CANDIDATES,
    include_trip_rows: bool = True,
) -> pd.DataFrame:
    """Build leg and trip OD-zone ids for one or more zoning layers.

    `zone_layers` is a mapping whose keys become column prefixes. Each value can
    be a path, a GeoDataFrame or a dict with `path`/`zones`, optional `layer`,
    `zone_id_col`, `zone_name_col`, `output_prefix` and `fill_value`.
    """

    if not zone_layers:
        return pd.DataFrame(columns=["object_type", "object_id", leg_id_col, trip_id_col])
    if leg_id_col not in legs.columns:
        raise KeyError(f"legs missing column: {leg_id_col}")

    labelled = _as_geodataframe(legs, geometry_col=geometry_col, default_crs=target_crs)
    zone_columns: list[str] = []
    for layer_key, layer_spec in zone_layers.items():
        layer_fill_value = fill_value
        if isinstance(layer_spec, Mapping):
            layer_fill_value = str(layer_spec.get("fill_value", fill_value))
        zones, zone_key = _prepare_zone_layer(
            layer_key,
            layer_spec,
            target_crs=target_crs,
            id_candidates=zone_id_candidates,
            name_candidates=zone_name_candidates,
        )
        origin_col = f"{zone_key}_zone_o"
        destination_col = f"{zone_key}_zone_d"
        labelled = add_leg_origin_destination_zones(
            labelled,
            zones,
            zone_label_col="_xyt_zone_id",
            origin_col=origin_col,
            destination_col=destination_col,
            geometry_col=geometry_col,
            fill_value=layer_fill_value,
            target_crs=target_crs,
        )
        zone_columns.extend([origin_col, destination_col])

    base_columns = [leg_id_col, *zone_columns]
    for column in [user_id_col, experiment_col, trip_id_col, journey_id_col]:
        if column in labelled.columns and column not in base_columns:
            base_columns.append(column)
    base = labelled[base_columns].copy()
    base["_xyt_order"] = range(len(base))

    if map_track_trip_journey is not None and leg_id_col in map_track_trip_journey.columns:
        mapping_columns = [
            column
            for column in [leg_id_col, trip_id_col, journey_id_col]
            if column in map_track_trip_journey.columns
        ]
        mapping = map_track_trip_journey[mapping_columns].drop_duplicates(subset=[leg_id_col])
        base = base.drop(columns=[trip_id_col, journey_id_col], errors="ignore")
        base = pd.merge(base, mapping, on=leg_id_col, how="left")

    context_columns = [
        column
        for column in [leg_id_col, trip_id_col, journey_id_col, user_id_col, experiment_col]
        if column in base.columns
    ]
    leg_rows = base[context_columns + zone_columns].copy()
    leg_rows.insert(0, "object_id", leg_rows[leg_id_col])
    leg_rows.insert(0, "object_type", "leg")

    rows = [leg_rows]
    if include_trip_rows and trip_id_col in base.columns and base[trip_id_col].notna().any():
        trip_source = base.dropna(subset=[trip_id_col]).sort_values("_xyt_order")
        aggregation: dict[str, tuple[str, str]] = {}
        for column in [journey_id_col, user_id_col, experiment_col]:
            if column in trip_source.columns:
                aggregation[column] = (column, "first")
        for origin_col, destination_col in zip(zone_columns[0::2], zone_columns[1::2]):
            aggregation[origin_col] = (origin_col, "first")
            aggregation[destination_col] = (destination_col, "last")
        trip_rows = trip_source.groupby(trip_id_col, sort=False).agg(**aggregation).reset_index()
        trip_rows.insert(0, leg_id_col, pd.NA)
        trip_rows.insert(0, "object_id", trip_rows[trip_id_col])
        trip_rows.insert(0, "object_type", "trip")
        rows.append(trip_rows)

    output = pd.concat(rows, ignore_index=True, sort=False)
    output_columns = [
        "object_type",
        "object_id",
        leg_id_col,
        trip_id_col,
        journey_id_col,
        user_id_col,
        experiment_col,
        *zone_columns,
    ]
    return output.reindex(columns=[column for column in output_columns if column in output.columns])


def classify_leg_relation_to_area(
    legs: pd.DataFrame,
    *,
    area: BaseGeometry | gpd.GeoDataFrame | None = None,
    area_path: str | None = None,
    center: tuple[float, float] | None = None,
    radius_m: float | None = None,
    geometry_col: str = "geometry",
    relation_col: str = "area_relation",
    code_col: str | None = None,
    target_crs: str = "EPSG:4326",
    operations_crs: str = "EPSG:2056",
) -> gpd.GeoDataFrame:
    """Classify each leg as intra, extra or exchange relative to an area."""

    out = _as_geodataframe(legs, geometry_col=geometry_col, default_crs=target_crs)
    reference_area = _area_from_inputs(
        area=area,
        area_path=area_path,
        center=center,
        radius_m=radius_m,
        target_crs=target_crs,
        operations_crs=operations_crs,
    )
    metric = out.to_crs(operations_crs)

    origins = gpd.GeoSeries(
        metric[geometry_col].apply(lambda geom: _line_endpoint(geom, position="origin")),
        index=metric.index,
        crs=metric.crs,
    )
    destinations = gpd.GeoSeries(
        metric[geometry_col].apply(lambda geom: _line_endpoint(geom, position="destination")),
        index=metric.index,
        crs=metric.crs,
    )
    valid_origin = origins.notna()
    valid_destination = destinations.notna()
    known = valid_origin & valid_destination

    origin_inside = pd.Series(False, index=metric.index)
    destination_inside = pd.Series(False, index=metric.index)
    if valid_origin.any():
        origin_inside.loc[valid_origin] = origins.loc[valid_origin].within(reference_area).to_numpy()
    if valid_destination.any():
        destination_inside.loc[valid_destination] = (
            destinations.loc[valid_destination].within(reference_area).to_numpy()
        )

    relations = pd.Series("unknown", index=metric.index, dtype=object)
    intra = known & origin_inside & destination_inside
    extra = known & ~origin_inside & ~destination_inside
    exchange = known & ~(intra | extra)
    relations.loc[intra] = "intra"
    relations.loc[extra] = "extra"
    relations.loc[exchange] = "exchange"

    out[relation_col] = relations.to_numpy()
    if code_col is not None:
        codes = pd.Series(pd.NA, index=metric.index, dtype="Float64")
        codes.loc[intra] = 1.0
        codes.loc[extra] = 0.0
        codes.loc[exchange] = 0.5
        out[code_col] = codes.to_numpy()
    return out


def add_trip_origin_destination_from_legs(
    trips: pd.DataFrame,
    legs: pd.DataFrame,
    map_track_trip_journey: pd.DataFrame,
    *,
    leg_origin_col: str = "origin_zone",
    leg_destination_col: str = "destination_zone",
    output_origin_col: str = "trip_origin_zone",
    output_destination_col: str = "trip_destination_zone",
    fill_value: str = "Unknown",
) -> pd.DataFrame:
    """Propagate first-origin and last-destination labels from legs to trips."""

    required_mapping = {"leg_id", "trip_id"}
    missing_mapping = required_mapping - set(map_track_trip_journey.columns)
    if missing_mapping:
        raise KeyError(f"map_track_trip_journey missing column(s): {sorted(missing_mapping)}")
    required_legs = {"leg_id", leg_origin_col, leg_destination_col}
    missing_legs = required_legs - set(legs.columns)
    if missing_legs:
        raise KeyError(f"legs missing column(s): {sorted(missing_legs)}")

    mapped = pd.merge(
        map_track_trip_journey[["leg_id", "trip_id"]].dropna(subset=["trip_id"]),
        legs[["leg_id", leg_origin_col, leg_destination_col]],
        on="leg_id",
        how="left",
    )
    trip_labels = (
        mapped.groupby("trip_id", sort=False)
        .agg(
            **{
                output_origin_col: (leg_origin_col, "first"),
                output_destination_col: (leg_destination_col, "last"),
            }
        )
        .reset_index()
    )
    out = trips.drop(columns=[output_origin_col, output_destination_col], errors="ignore").copy()
    out = pd.merge(out, trip_labels, on="trip_id", how="left")
    out[output_origin_col] = out[output_origin_col].fillna(fill_value)
    out[output_destination_col] = out[output_destination_col].fillna(fill_value)
    return out


def add_journey_origin_destination_from_trips(
    journeys: pd.DataFrame,
    trips: pd.DataFrame,
    map_track_trip_journey: pd.DataFrame,
    *,
    trip_origin_col: str = "trip_origin_zone",
    trip_destination_col: str = "trip_destination_zone",
    output_origin_col: str = "journey_origin_zone",
    output_destination_col: str = "journey_destination_zone",
    fill_value: str = "Unknown",
) -> pd.DataFrame:
    """Propagate first-origin and last-destination labels from trips to journeys."""

    required_mapping = {"trip_id", "journey_id"}
    missing_mapping = required_mapping - set(map_track_trip_journey.columns)
    if missing_mapping:
        raise KeyError(f"map_track_trip_journey missing column(s): {sorted(missing_mapping)}")
    required_trips = {"trip_id", trip_origin_col, trip_destination_col}
    missing_trips = required_trips - set(trips.columns)
    if missing_trips:
        raise KeyError(f"trips missing column(s): {sorted(missing_trips)}")

    mapped = pd.merge(
        map_track_trip_journey[["trip_id", "journey_id"]]
        .dropna(subset=["trip_id", "journey_id"])
        .drop_duplicates(),
        trips[["trip_id", trip_origin_col, trip_destination_col]],
        on="trip_id",
        how="left",
    )
    journey_labels = (
        mapped.groupby("journey_id", sort=False)
        .agg(
            **{
                output_origin_col: (trip_origin_col, "first"),
                output_destination_col: (trip_destination_col, "last"),
            }
        )
        .reset_index()
    )
    out = journeys.drop(columns=[output_origin_col, output_destination_col], errors="ignore").copy()
    out = pd.merge(out, journey_labels, on="journey_id", how="left")
    out[output_origin_col] = out[output_origin_col].fillna(fill_value)
    out[output_destination_col] = out[output_destination_col].fillna(fill_value)
    return out


__all__ = [
    "add_excursion_flags",
    "add_spatial_zone_labels",
    "add_leg_origin_destination_zones",
    "build_origin_destination_zone_correspondence",
    "classify_leg_relation_to_area",
    "add_trip_origin_destination_from_legs",
    "add_journey_origin_destination_from_trips",
]
