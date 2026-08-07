from __future__ import annotations

import binascii

import pandas as pd
import pytest
from shapely import wkb
from shapely.geometry import LineString, Point

import xyt_gps as xyt


pytest.importorskip("sklearn")


def _hex(geometry):
    return geometry.wkb_hex


def _parse(hex_geometry):
    return wkb.loads(binascii.unhexlify(str(hex_geometry)))


def test_geofence_anonymization_trims_track_and_pseudonymizes_users():
    storyline = pd.DataFrame(
        [
            {
                "storyline_id": "s1",
                "user_id": "real_user_1",
                "type": "Stay",
                "started_at": "2026-04-01T06:00:00+02:00",
                "geometry": _hex(Point(6.1400, 46.2000)),
                "purpose": "home",
            },
            {
                "storyline_id": "s2",
                "user_id": "real_user_1",
                "type": "Stay",
                "started_at": "2026-04-01T07:00:00+02:00",
                "geometry": _hex(Point(6.1402, 46.2001)),
                "purpose": "home",
            },
            {
                "storyline_id": "s3",
                "user_id": "real_user_1",
                "type": "Stay",
                "started_at": "2026-04-01T08:00:00+02:00",
                "geometry": _hex(Point(6.1399, 46.1999)),
                "purpose": "home",
            },
            {
                "storyline_id": "s4",
                "user_id": "real_user_1",
                "type": "Track",
                "started_at": "2026-04-01T08:30:00+02:00",
                "geometry": _hex(LineString([(6.1400, 46.2000), (6.1500, 46.2040), (6.1700, 46.2100)])),
                "purpose": "work",
            },
            {
                "storyline_id": "s5",
                "user_id": "real_user_1",
                "type": "Track",
                "started_at": "2026-04-01T10:30:00+02:00",
                "geometry": _hex(LineString([(6.2000, 46.2500), (6.2010, 46.2510)])),
                "purpose": "shopping",
            },
        ]
    )
    tables = {
        "storyline": storyline,
        "trips": pd.DataFrame(
            {"trip_id": ["t1", "t2"], "user_id": ["real_user_1", "real_user_1"], "purpose": ["work", "shopping"]}
        ),
        "journeys": pd.DataFrame({"journey_id": ["j1"], "user_id": ["real_user_1"]}),
        "user_statistics": pd.DataFrame(
            {"user_statistics_id": ["real_user_1"], "user_id": ["real_user_1"], "email": ["a@example.test"]}
        ),
    }

    config = xyt.GeofenceAnonymizationConfig(
        dbscan_eps_m=120,
        dbscan_min_samples=2,
        geofence_min_radius_m=160,
        trim_buffer_min_m=20,
        trim_buffer_max_m=20,
        min_remaining_leg_length_m=100,
        sample_user_count=None,
        swap_sensitive_purpose_values=True,
        random_state=42,
    )
    result = xyt.anonymize_landed_gps_tables(tables, config)

    private_storyline = result.tables["storyline"]
    private_track = private_storyline.loc[private_storyline["type"].eq("Track")].iloc[0]
    private_line = _parse(private_track["geometry"])
    original_line = _parse(storyline.loc[storyline["type"].eq("Track"), "geometry"].iloc[0])

    assert not result.geofences.empty
    assert bool(private_track["privacy_geofenced"]) is True
    assert private_track["privacy_geometry_status"] == "trimmed_by_geofence"
    assert private_line.length < original_line.length
    assert private_line.coords[0] != original_line.coords[0]
    assert set(result.tables["storyline"]["user_id"]) == {"demo_user_0001"}
    assert set(result.geofences["user_id"]) == {"demo_user_0001"}
    assert set(result.report["user_id"]) == {"demo_user_0001"}
    assert not result.geofences["geofence_id"].astype(str).str.contains("real_user").any()
    assert "shopping" in set(result.tables["storyline"]["purpose"])
    assert not result.tables["storyline"].loc[result.tables["storyline"]["storyline_id"].eq("s1"), "purpose"].eq("home").any()
    assert "email" not in result.tables["user_statistics"].columns
    assert result.metadata["private_user_id_mapping_exported"] is False


def test_geofence_anonymization_flags_short_legs_with_configured_policy():
    storyline = pd.DataFrame(
        [
            {
                "storyline_id": "s1",
                "user_id": "real_user_1",
                "type": "Stay",
                "started_at": "2026-04-01T06:00:00+02:00",
                "geometry": _hex(Point(6.1400, 46.2000)),
                "purpose": "home",
            },
            {
                "storyline_id": "s2",
                "user_id": "real_user_1",
                "type": "Stay",
                "started_at": "2026-04-01T07:00:00+02:00",
                "geometry": _hex(Point(6.1402, 46.2001)),
                "purpose": "home",
            },
            {
                "storyline_id": "s3",
                "user_id": "real_user_1",
                "type": "Track",
                "started_at": "2026-04-01T08:30:00+02:00",
                "geometry": _hex(LineString([(6.1400, 46.2000), (6.1405, 46.2002)])),
                "purpose": "work",
            },
        ]
    )

    config = xyt.GeofenceAnonymizationConfig(
        dbscan_eps_m=120,
        dbscan_min_samples=2,
        geofence_min_radius_m=200,
        trim_buffer_min_m=20,
        trim_buffer_max_m=20,
        min_remaining_leg_length_m=100,
        short_leg_policy="keep_original_flagged",
        swap_sensitive_purpose_values=False,
    )
    result = xyt.anonymize_landed_gps_tables({"storyline": storyline}, config)
    private_track = result.tables["storyline"].loc[result.tables["storyline"]["type"].eq("Track")].iloc[0]

    assert private_track["privacy_geometry_status"] == "short_leg_kept_original_after_geofence"
    assert _parse(private_track["geometry"]).equals_exact(_parse(storyline.loc[2, "geometry"]), tolerance=1e-6)
    assert result.metadata["short_leg_policy"] == "keep_original_flagged"
