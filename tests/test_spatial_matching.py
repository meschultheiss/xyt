from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from urllib.parse import parse_qs, urlparse
import warnings
from xml.etree import ElementTree

from xyt_gps import map_match_legs


def _legs(rows):
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _encode_polyline(coordinates):
    output = []
    previous_latitude = 0
    previous_longitude = 0
    for longitude, latitude in coordinates:
        latitude_value = round(latitude * 100_000)
        longitude_value = round(longitude * 100_000)
        for delta in (
            latitude_value - previous_latitude,
            longitude_value - previous_longitude,
        ):
            value = ~(delta << 1) if delta < 0 else delta << 1
            while value >= 0x20:
                output.append(chr((0x20 | (value & 0x1F)) + 63))
                value >>= 5
            output.append(chr(value + 63))
        previous_latitude = latitude_value
        previous_longitude = longitude_value
    return "".join(output)


def test_osrm_uses_all_observed_points_and_replaces_clean_geometry():
    original = LineString([(6.10, 46.20), (6.11, 46.21), (6.12, 46.20)])
    matched = [(6.10, 46.20), (6.105, 46.205), (6.11, 46.21), (6.12, 46.20)]
    calls = []

    def transport(url, **kwargs):
        calls.append((url, kwargs))
        return {
            "code": "Ok",
            "matchings": [
                {
                    "confidence": 0.92,
                    "geometry": {"type": "LineString", "coordinates": matched},
                }
            ],
        }

    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "leg-1",
                    "user_id": "u1",
                    "mode": "Mode::Walk",
                    "low_quality_signal": True,
                    "length": 9999.0,
                    "length_leg": 9999.0,
                    "geometry": original,
                }
            ]
        ),
        transport=transport,
    )

    assert len(calls) == 1
    assert calls[0][0].count(";") == 2
    assert result.legs.loc[0, "map_match_source"] == "osrm"
    assert result.legs.loc[0, "map_match_status"] == "matched"
    assert result.legs.loc[0, "map_match_confidence"] == 0.92
    assert result.legs.loc[0, "map_match_waypoints_count"] == 3
    assert list(result.legs.loc[0, "geometry"].coords) == matched
    assert result.legs.loc[0, "length"] != 9999.0
    assert result.geometry_correspondence["geometry_version"].tolist() == ["original", "clean"]
    assert result.geometry_correspondence.iloc[0].geometry.equals(original)


def test_google_fallback_keeps_ordered_waypoints_for_a_loop():
    loop = [
        (6.1000, 46.2000),
        (6.1100, 46.2000),
        (6.1150, 46.2050),
        (6.1100, 46.2100),
        (6.1000, 46.2100),
        (6.0950, 46.2050),
        (6.1000, 46.2000),
    ]
    google_payloads = []

    def transport(url, **kwargs):
        if kwargs["method"] == "GET":
            return {"code": "NoMatch", "message": "no local match"}
        google_payloads.append(kwargs["payload"])
        return {
            "routes": [
                {
                    "distanceMeters": 4000,
                    "polyline": {"encodedPolyline": _encode_polyline(loop)},
                }
            ]
        }

    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "loop-1",
                    "user_id": "u1",
                    "mode": "Mode::Walk",
                    "low_quality_signal": True,
                    "length": 4000.0,
                    "geometry": LineString(loop),
                }
            ]
        ),
        google_fallback_enabled=True,
        google_api_key="test-key",
        google_max_intermediate_waypoints=4,
        transport=transport,
    )

    assert result.legs.loc[0, "map_match_source"] == "google"
    assert result.legs.loc[0, "map_match_status"] == "matched"
    assert google_payloads[0]["optimizeWaypointOrder"] is False
    assert google_payloads[0]["origin"] == google_payloads[0]["destination"]
    assert 1 <= len(google_payloads[0]["intermediates"]) <= 4
    assert all(waypoint["via"] is True for waypoint in google_payloads[0]["intermediates"])
    assert result.legs.loc[0, "map_match_original_length_m"] > 0
    assert result.legs.loc[0, "map_match_clean_length_m"] > 0
    assert result.legs.loc[0, "map_match_provider_distance_m"] == 4000
    assert result.geometry_correspondence.loc[
        result.geometry_correspondence["geometry_version"].eq("clean"),
        "provider_distance_m",
    ].iloc[0] == 4000
    assert result.legs.loc[0, "geometry"].is_ring


def test_collapsed_google_loop_is_rejected():
    loop = LineString(
        [
            (6.1000, 46.2000),
            (6.1200, 46.2000),
            (6.1200, 46.2200),
            (6.1000, 46.2200),
            (6.1000, 46.2000),
        ]
    )
    collapsed = [(6.1000, 46.2000), (6.1001, 46.2000)]

    def transport(url, **kwargs):
        if kwargs["method"] == "GET":
            return {"code": "NoMatch"}
        return {
            "routes": [
                {"polyline": {"encodedPolyline": _encode_polyline(collapsed)}}
            ]
        }

    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "loop-1",
                    "mode": "Mode::Walk",
                    "low_quality_signal": True,
                    "length": 1000.0,
                    "geometry": loop,
                }
            ]
        ),
        google_fallback_enabled=True,
        google_api_key="test-key",
        transport=transport,
    )

    assert result.legs.loc[0, "map_match_source"] == "original"
    assert result.legs.loc[0, "map_match_status"] == "kept_original"
    assert "google_rejected_length_ratio" in result.legs.loc[0, "map_match_reason"]
    assert result.legs.loc[0, "geometry"].equals(loop)


def test_unavailable_osrm_is_cached_and_does_not_block_without_google_key():
    calls = []

    def transport(url, **kwargs):
        calls.append(url)
        raise ConnectionError("connection refused")

    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "leg-1",
                    "mode": "Mode::Walk",
                    "low_quality_signal": True,
                    "geometry": LineString([(6.10, 46.20), (6.11, 46.21)]),
                },
                {
                    "leg_id": "leg-2",
                    "mode": "Mode::Walk",
                    "low_quality_signal": True,
                    "geometry": LineString([(6.12, 46.22), (6.13, 46.23)]),
                },
                {
                    "leg_id": "leg-3",
                    "mode": "Mode::Walk",
                    "low_quality_signal": False,
                    "geometry": LineString([(6.14, 46.24), (6.15, 46.25)]),
                },
            ]
        ),
        transport=transport,
    )

    assert len(calls) == 1
    assert result.legs["map_match_status"].tolist() == [
        "kept_original",
        "kept_original",
        "not_requested",
    ]
    assert len(result.geometry_correspondence) == 4


def test_google_key_alone_does_not_enable_paid_fallback():
    methods = []

    def transport(url, **kwargs):
        methods.append(kwargs["method"])
        return {"code": "NoMatch"}

    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "leg-1",
                    "mode": "Mode::Walk",
                    "low_quality_signal": True,
                    "geometry": LineString([(6.10, 46.20), (6.11, 46.21)]),
                }
            ]
        ),
        google_api_key="accidentally-present-key",
        transport=transport,
    )

    assert methods == ["GET"]
    assert result.legs.loc[0, "map_match_source"] == "original"
    assert "google_fallback_disabled" in result.legs.loc[0, "map_match_reason"]


def test_google_fallback_respects_request_cap():
    methods = []

    def transport(url, **kwargs):
        methods.append(kwargs["method"])
        if kwargs["method"] == "GET":
            return {"code": "NoMatch"}
        payload = kwargs["payload"]
        origin = payload["origin"]["location"]["latLng"]
        destination = payload["destination"]["location"]["latLng"]
        coordinates = [
            (origin["longitude"], origin["latitude"]),
            (destination["longitude"], destination["latitude"]),
        ]
        return {
            "routes": [
                {"polyline": {"encodedPolyline": _encode_polyline(coordinates)}}
            ]
        }

    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "leg-1",
                    "mode": "Mode::Walk",
                    "low_quality_signal": True,
                    "geometry": LineString([(6.10, 46.20), (6.11, 46.21)]),
                },
                {
                    "leg_id": "leg-2",
                    "mode": "Mode::Walk",
                    "low_quality_signal": True,
                    "geometry": LineString([(6.12, 46.22), (6.13, 46.23)]),
                },
            ]
        ),
        google_fallback_enabled=True,
        google_api_key="test-key",
        google_max_requests=1,
        transport=transport,
    )

    assert methods.count("POST") == 1
    assert result.legs["map_match_status"].tolist() == ["matched", "kept_original"]
    assert "google_request_limit_reached" in result.legs.loc[1, "map_match_reason"]


def test_google_roads_snap_chunks_dense_car_trace_and_rotates_keys():
    calls = []

    def transport(url, **kwargs):
        query = parse_qs(urlparse(url).query)
        calls.append({"method": kwargs["method"], "key": query["key"][0]})
        coordinates = [
            tuple(float(value) for value in pair.split(","))
            for pair in query["path"][0].split("|")
        ]
        return {
            "snappedPoints": [
                {"location": {"latitude": lat, "longitude": lon}}
                for lat, lon in coordinates
            ]
        }

    coordinates = [
        (6.1000, 46.2000),
        (6.1002, 46.2001),
        (6.1004, 46.2002),
        (6.1006, 46.2003),
        (6.1008, 46.2004),
    ]
    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "car-1",
                    "mode": "Mode::Car",
                    "geometry": LineString(coordinates),
                }
            ]
        ),
        match_all_supported_modes=True,
        osrm_enabled=False,
        google_fallback_enabled=True,
        google_api_keys=["key-a", "key-b"],
        google_roads_max_points_per_request=3,
        google_max_requests=3,
        transport=transport,
    )

    assert [call["method"] for call in calls] == ["GET", "GET", "GET"]
    assert [call["key"] for call in calls] == ["key-a", "key-b", "key-a"]
    assert result.legs.loc[0, "map_match_source"] == "google_roads"
    assert result.legs.loc[0, "map_match_method"] == "google_roads_snap"
    assert result.legs.loc[0, "map_match_request_count"] == 3
    assert result.legs.loc[0, "map_match_status"] == "matched"


def _gpx_coordinates(body):
    root = ElementTree.fromstring(body)
    namespace = {"gpx": "http://www.topografix.com/GPX/1/1"}
    return [
        (float(point.attrib["lon"]), float(point.attrib["lat"]))
        for point in root.findall(".//gpx:trkpt", namespace)
    ]


def test_graphhopper_matches_all_car_bike_and_foot_legs_with_gpx():
    calls = []

    def transport(url, **kwargs):
        coordinates = _gpx_coordinates(kwargs["body"])
        calls.append(
            {
                "query": parse_qs(urlparse(url).query),
                "coordinates": coordinates,
                "headers": kwargs["headers"],
            }
        )
        return {"paths": [{"points": {"coordinates": coordinates}}]}

    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "car-1",
                    "mode": "Mode::Car",
                    "geometry": LineString([(6.10, 46.20), (6.101, 46.201)]),
                },
                {
                    "leg_id": "bike-1",
                    "mode": "Mode::Bicycle",
                    "geometry": LineString([(6.11, 46.21), (6.111, 46.211)]),
                },
                {
                    "leg_id": "foot-1",
                    "mode": "Mode::Walk",
                    "geometry": LineString([(6.12, 46.22), (6.121, 46.221)]),
                },
                {
                    "leg_id": "rail-1",
                    "mode": "Mode::Train",
                    "geometry": LineString([(6.13, 46.23), (6.131, 46.231)]),
                },
            ]
        ),
        match_all_supported_modes=True,
        graphhopper_enabled=True,
        graphhopper_api_key="test-key",
        graphhopper_max_requests=3,
        osrm_enabled=False,
        transport=transport,
    )

    assert [call["query"]["profile"][0] for call in calls] == ["car", "bike", "foot"]
    assert all(call["query"]["key"] == ["test-key"] for call in calls)
    assert all(call["headers"]["Content-Type"] == "application/gpx+xml" for call in calls)
    assert result.legs["map_match_status"].tolist() == [
        "matched",
        "matched",
        "matched",
        "not_requested",
    ]
    assert result.legs.loc[3, "map_match_reason"] == "unsupported_mode"
    assert result.legs.loc[0, "map_match_method"] == "graphhopper_snap"


def test_graphhopper_repairs_multilinestring_before_it_can_be_dropped():
    original = MultiLineString(
        [
            [(6.1000, 46.2000), (6.1010, 46.2010)],
            [(6.1040, 46.2040), (6.1050, 46.2050)],
        ]
    )

    def transport(url, **kwargs):
        coordinates = _gpx_coordinates(kwargs["body"])
        repaired = [coordinates[0], coordinates[1], (6.1025, 46.2025), *coordinates[2:]]
        return {"paths": [{"points": {"coordinates": repaired}}]}

    result = map_match_legs(
        _legs([{"leg_id": "leg-gap", "mode": "Mode::Walk", "geometry": original}]),
        match_all_supported_modes=True,
        graphhopper_enabled=True,
        graphhopper_api_key="test-key",
        osrm_enabled=False,
        transport=transport,
    )

    assert result.legs.loc[0, "map_match_status"] == "matched"
    assert result.legs.loc[0, "map_match_method"] == "graphhopper_repair"
    assert isinstance(result.legs.loc[0, "geometry"], LineString)
    assert isinstance(result.geometry_correspondence.iloc[0].geometry, MultiLineString)


def test_graphhopper_request_cap_is_checked_before_a_chunked_leg_is_sent():
    calls = []

    def transport(url, **kwargs):
        calls.append(url)
        raise AssertionError("No paid request should be sent")

    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "long-leg",
                    "mode": "Mode::Walk",
                    "geometry": LineString(
                        [(6.100 + step * 0.001, 46.200 + step * 0.001) for step in range(5)]
                    ),
                }
            ]
        ),
        match_all_supported_modes=True,
        graphhopper_enabled=True,
        graphhopper_api_key="test-key",
        graphhopper_max_points_per_request=3,
        graphhopper_max_requests=2,
        osrm_enabled=False,
        transport=transport,
    )

    assert calls == []
    assert result.legs.loc[0, "map_match_status"] == "kept_original"
    assert "graphhopper_request_limit_reached" in result.legs.loc[0, "map_match_reason"]


def test_graphhopper_subscription_unavailable_stops_following_requests():
    calls = []

    def transport(url, **kwargs):
        calls.append(url)
        raise RuntimeError(
            'HTTP 400: {"message":"Too many points for Map Matching API: 24, '
            'allowed: 0. Upgrade your subscription or contact us."}'
        )

    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "leg-1",
                    "mode": "Mode::Walk",
                    "geometry": LineString([(6.100, 46.200), (6.101, 46.201)]),
                },
                {
                    "leg_id": "leg-2",
                    "mode": "Mode::Walk",
                    "geometry": LineString([(6.110, 46.210), (6.111, 46.211)]),
                },
            ]
        ),
        match_all_supported_modes=True,
        graphhopper_enabled=True,
        graphhopper_api_key="test-key",
        osrm_enabled=False,
        transport=transport,
    )

    assert len(calls) == 1
    assert result.legs["map_match_status"].tolist() == ["kept_original", "kept_original"]
    assert result.legs["map_match_request_count"].tolist() == [1, 0]
    assert result.legs["map_match_reason"].str.contains(
        "graphhopper_map_matching_subscription_unavailable"
    ).all()


def test_graphhopper_key_alone_does_not_enable_paid_requests():
    calls = []

    def transport(url, **kwargs):
        calls.append(url)
        raise AssertionError("No paid request should be sent")

    result = map_match_legs(
        _legs(
            [
                {
                    "leg_id": "leg-1",
                    "mode": "Mode::Walk",
                    "low_quality_signal": True,
                    "geometry": LineString([(6.10, 46.20), (6.11, 46.21)]),
                }
            ]
        ),
        graphhopper_api_key="accidentally-present-key",
        osrm_enabled=False,
        transport=transport,
    )

    assert calls == []
    assert result.legs.loc[0, "map_match_source"] == "original"


def test_graphhopper_rejects_a_route_far_from_observed_points():
    original = LineString([(6.100, 46.200), (6.101, 46.201), (6.102, 46.202)])

    def transport(url, **kwargs):
        return {
            "paths": [
                {
                    "points": {
                        "coordinates": [(6.200, 46.300), (6.201, 46.301), (6.202, 46.302)]
                    }
                }
            ]
        }

    result = map_match_legs(
        _legs([{"leg_id": "leg-1", "mode": "Mode::Walk", "geometry": original}]),
        match_all_supported_modes=True,
        graphhopper_enabled=True,
        graphhopper_api_key="test-key",
        osrm_enabled=False,
        transport=transport,
    )

    assert result.legs.loc[0, "map_match_status"] == "kept_original"
    assert result.legs.loc[0, "geometry"].equals(original)
    assert "origin_shift" in result.legs.loc[0, "map_match_reason"]
    assert "point_coverage" in result.legs.loc[0, "map_match_reason"]


def test_non_finite_projected_provider_geometry_is_rejected_without_warning():
    original = LineString([(6.100, 46.200), (6.101, 46.201)])

    def transport(url, **kwargs):
        return {
            "paths": [
                {
                    "points": {
                        "coordinates": [(999.0, 999.0), (1000.0, 1000.0)]
                    }
                }
            ]
        }

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = map_match_legs(
            _legs([{"leg_id": "leg-1", "mode": "Mode::Car", "geometry": original}]),
            match_all_supported_modes=True,
            graphhopper_enabled=True,
            graphhopper_api_key="test-key",
            osrm_enabled=False,
            google_fallback_enabled=False,
            transport=transport,
        )

    assert captured == []
    assert result.legs.loc[0, "map_match_status"] == "kept_original"
    assert result.legs.loc[0, "geometry"].equals(original)
    assert "non_finite_metric" in result.legs.loc[0, "map_match_reason"]
