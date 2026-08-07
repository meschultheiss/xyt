import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from xyt_gps import (
    TimeSlice,
    add_time_slices,
    aggregate_h3_frequencies,
    build_h3_count_matrix,
    legs_to_h3_points,
    write_spatial_analytics_tables,
    write_duckdb_spatial_database,
    write_spatial_analytics_exports,
    plot_h3_frequency_map,
    plot_gps_on_map,
)


def _legs():
    return gpd.GeoDataFrame(
        [
            {
                "leg_id": "leg-1",
                "user_id": "u1",
                "trip_id": "trip-1",
                "mode_niv1": "Marche",
                "purpose_mrmt": "Travail",
                "started_at": pd.Timestamp("2026-04-01T08:00:00Z"),
                "geometry": LineString([(6.60, 46.52), (6.61, 46.525), (6.62, 46.53)]),
            },
            {
                "leg_id": "leg-2",
                "user_id": "u2",
                "trip_id": "trip-2",
                "mode_niv1": "Velo",
                "purpose_mrmt": "Loisir",
                "started_at": pd.Timestamp("2026-04-01T09:00:00Z"),
                "geometry": LineString([(6.60, 46.52), (6.615, 46.526)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def test_aggregate_h3_frequencies_counts_points_users_and_legs():
    points = pd.DataFrame(
        [
            {"h3_cell": "cell-a", "mode_niv1": "walk", "user_id": "u1", "leg_id": "leg-1", "trip_id": "trip-1"},
            {"h3_cell": "cell-a", "mode_niv1": "walk", "user_id": "u1", "leg_id": "leg-1", "trip_id": "trip-1"},
            {"h3_cell": "cell-a", "mode_niv1": "bike", "user_id": "u2", "leg_id": "leg-2", "trip_id": "trip-2"},
            {"h3_cell": "cell-b", "mode_niv1": "bike", "user_id": "u2", "leg_id": "leg-2", "trip_id": "trip-2"},
        ]
    )

    frequency = aggregate_h3_frequencies(points)
    frequency_by_mode = aggregate_h3_frequencies(points, group_cols=["mode_niv1"])

    first = frequency.loc[frequency["h3_cell"].eq("cell-a")].iloc[0]
    assert first["point_count"] == 3
    assert first["user_count"] == 2
    assert first["leg_count"] == 2
    assert first["trip_count"] == 2
    assert {"h3_cell", "mode_niv1"}.issubset(frequency_by_mode.columns)


def test_add_time_slices_uses_configured_peak_periods():
    table = pd.DataFrame(
        {
            "started_at": [
                "2026-04-01T06:00:00Z",
                "2026-04-01T08:00:00Z",
                "2026-04-01T18:00:00Z",
            ]
        }
    )

    out = add_time_slices(
        table,
        time_slices=(TimeSlice("HPM", "07:10", "09:00"), TimeSlice("HPS", "17:30", "20:00")),
        timezone="UTC",
    )

    assert out["time_slice"].tolist() == ["HC", "HPM", "HPS"]


def test_legs_to_h3_points_indexes_line_vertices():
    pytest.importorskip("h3")

    points = legs_to_h3_points(_legs(), h3_resolution=(8, 9))

    assert len(points) == 10
    assert {"leg_id", "user_id", "lon", "lat", "h3_cell", "point_sequence"}.issubset(points.columns)
    assert set(points["h3_resolution"]) == {8, 9}
    assert points.crs.to_string() == "EPSG:4326"


def test_legs_to_h3_points_parallel_matches_sequential():
    pytest.importorskip("h3")

    sort_cols = ["leg_id", "h3_resolution", "point_sequence", "h3_cell"]
    sequential = legs_to_h3_points(_legs(), h3_resolution=(8, 9), parallel=False)
    parallel = legs_to_h3_points(
        _legs(),
        h3_resolution=(8, 9),
        parallel=True,
        max_workers=2,
        chunk_size=1,
    )

    pd.testing.assert_frame_equal(
        sequential.drop(columns="geometry").sort_values(sort_cols).reset_index(drop=True),
        parallel.drop(columns="geometry").sort_values(sort_cols).reset_index(drop=True),
    )


def test_build_h3_count_matrix_creates_wide_columns():
    points = pd.DataFrame(
        [
            {
                "h3_resolution": 8,
                "h3_cell": "cell-a",
                "mode_niv1": "Marche",
                "purpose_mrmt": "Travail",
                "time_slice": "HPM",
                "user_id": "u1",
                "leg_id": "leg-1",
                "trip_id": "trip-1",
            },
            {
                "h3_resolution": 8,
                "h3_cell": "cell-a",
                "mode_niv1": "Marche",
                "purpose_mrmt": "Travail",
                "time_slice": "HPM",
                "user_id": "u1",
                "leg_id": "leg-1",
                "trip_id": "trip-1",
            },
            {
                "h3_resolution": 8,
                "h3_cell": "cell-a",
                "mode_niv1": "Velo",
                "purpose_mrmt": "Loisir",
                "time_slice": "HC",
                "user_id": "u2",
                "leg_id": "leg-2",
                "trip_id": "trip-2",
            },
        ]
    )

    matrix = build_h3_count_matrix(
        points,
        dimension_sets=(("mode_niv1",), ("time_slice",), ("mode_niv1", "time_slice")),
        metrics=("point_count", "trip_count"),
    )

    assert "point_count__total" in matrix.columns
    assert "point_count__mode_niv1__marche" in matrix.columns
    assert "trip_count__time_slice__hpm" in matrix.columns
    assert "point_count__mode_niv1__marche__time_slice__hpm" in matrix.columns
    assert matrix.loc[0, "point_count__total"] == 3
    assert matrix.loc[0, "trip_count__time_slice__hpm"] == 1


def test_write_spatial_analytics_exports_writes_manifest(tmp_path):
    pytest.importorskip("h3")
    pytest.importorskip("pyarrow")

    manifest = write_spatial_analytics_exports(
        _legs(),
        tmp_path,
        h3_resolution=(8, 9),
    )

    assert set(manifest["table"]) == {"leg_points_h3", "h3_frequency", "h3_count_matrix"}
    assert set(manifest["format"]) == {"parquet", "csv"}
    assert (tmp_path / "leg_points_h3.parquet").exists()
    assert (tmp_path / "leg_points_h3.csv").exists()
    assert (tmp_path / "h3_frequency.parquet").exists()
    assert (tmp_path / "h3_frequency.csv").exists()
    assert (tmp_path / "h3_count_matrix.parquet").exists()
    assert (tmp_path / "h3_count_matrix.csv").exists()
    assert (tmp_path / "spatial_analytics_manifest.json").exists()


def test_write_spatial_analytics_tables_accepts_formats_by_table(tmp_path):
    pytest.importorskip("pyarrow")

    manifest = write_spatial_analytics_tables(
        {
            "leg_points_h3": pd.DataFrame({"h3_cell": ["abc"], "point_count": [1]}),
            "h3_frequency": pd.DataFrame({"h3_cell": ["abc"], "point_count": [1]}),
        },
        tmp_path,
        formats={"__default__": ("parquet", "csv"), "leg_points_h3": ("parquet",)},
    )

    assert (tmp_path / "leg_points_h3.parquet").exists()
    assert not (tmp_path / "leg_points_h3.csv").exists()
    assert (tmp_path / "h3_frequency.parquet").exists()
    assert (tmp_path / "h3_frequency.csv").exists()
    assert set(manifest["format"]) == {"parquet", "csv"}


def test_write_duckdb_spatial_database_accepts_geodataframe(tmp_path):
    pytest.importorskip("duckdb")

    manifest = write_duckdb_spatial_database(
        {"legs": _legs()},
        tmp_path / "mobility.duckdb",
        load_spatial=False,
    )

    assert manifest.loc[0, "table"] == "legs"
    assert manifest.loc[0, "rows"] == 2
    assert not bool(manifest.loc[0, "spatial_extension_loaded"])
    assert (tmp_path / "mobility.duckdb").exists()


def test_write_duckdb_spatial_database_accepts_mixed_boolean_integer_flags(tmp_path):
    pytest.importorskip("duckdb")

    manifest = write_duckdb_spatial_database(
        {
            "flags": pd.DataFrame(
                {
                    "user_id": ["u1", "u2", "u3"],
                    "analysis_user_ok": [True, 0, 1],
                }
            )
        },
        tmp_path / "mobility.duckdb",
        load_spatial=False,
    )

    assert manifest.loc[0, "table"] == "flags"
    assert manifest.loc[0, "rows"] == 3
    assert (tmp_path / "mobility.duckdb").exists()


def test_plot_h3_frequency_map_writes_html(tmp_path):
    h3 = pytest.importorskip("h3")
    pytest.importorskip("folium")

    cell = h3.latlng_to_cell(46.52, 6.60, 9)
    html_path = tmp_path / "h3_map.html"

    h3_map = plot_h3_frequency_map(
        pd.DataFrame([{"h3_cell": cell, "point_count": 10, "user_count": 2}]),
        save_path=html_path,
    )

    assert html_path.exists()
    assert h3_map.location


def test_plot_gps_on_map_accepts_lat_lon_columns(tmp_path):
    pytest.importorskip("folium")
    html_path = tmp_path / "gps_map.html"

    gps_map = plot_gps_on_map(
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "type": "Waypoint",
                    "latitude": 46.52,
                    "longitude": 6.60,
                    "started_at": "2026-04-01T08:00:00Z",
                },
                {
                    "user_id": "u2",
                    "type": "Staypoint",
                    "latitude": 46.53,
                    "longitude": 6.62,
                    "started_at": "2026-04-01T09:00:00Z",
                },
            ]
        ),
        geo_columns=("latitude", "longitude"),
        save_path=html_path,
    )

    assert html_path.exists()
    assert gps_map.location
