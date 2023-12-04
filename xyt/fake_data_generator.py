import pandas as pd
from faker import Faker
import random
from shapely.geometry import Point, LineString
import pytz
import geopandas as gpd
import pyproj
from geopy.geocoders import Nominatim
import folium
from shapely.wkt import loads
from shapely.geometry import Point
from IPython.display import display
from datetime import timedelta
from geopy.distance import geodesic
import numpy as np
from datetime import datetime, timedelta
import geopy.distance


class FakeDataGenerator:
    """
    A class for generating fake location-based data for testing and demonstration purposes.

    Attributes:
        - projection_epsg (int): The EPSG code for the desired projection (default is 4326).
        - location_name (str): The name of a location for which to generate data. If provided, the generated data will be constrained within the bounding box of this location.
        - home_radius_km (int): The radius in kilometers within which home locations for different groups should be generated.

    Example usage:
        .. code-block:: python

            generator = FakeDataGenerator(location_name="Switzerland", num_users=15)
            df_legs = generator.generate_legs(num_rows=10)
            df_staypoints = generator.generate_staypoints(num_rows=10)
            df_waypoints = generator.generate_waypoints(num_rows=10)


    """

    def __init__(
        self, projection_epsg=4326, location_name=None, home_radius_km=20, num_users=15
    ):
        self.fake = Faker()
        self.timezone = pytz.timezone("Europe/Zurich")
        self.projection_epsg = projection_epsg
        self.location_name = location_name
        self.projection = pyproj.Transformer.from_crs(
            "epsg:4326", f"epsg:{projection_epsg}", always_xy=True
        ).transform
        self.location_bbox = None
        self.home_radius_km = home_radius_km
        self.num_users = num_users

        if location_name:
            self.location_bbox = self.get_bounding_box(location_name)

    def get_bounding_box(self, location_name):
        """
        Retrieve the bounding box coordinates for a specified location.

        Args:
            - location_name (str): The name of the location for which to retrieve the bounding box.

        Returns:
            - list: A list of bounding box coordinates [south, north, west, east].
        """
        geolocator = Nominatim(user_agent="geo_data_generator")
        location = geolocator.geocode(location_name, exactly_one=True)

        if location is None:
            raise ValueError(f"Location '{location_name}' not found")

        return location.raw["boundingbox"]

    def generate_home_location(self):
        """
        Generate a random home location point.

        Returns:
            - Point: A Shapely Point object representing the home location.
        """
        if self.location_bbox:
            south, north, west, east = map(float, self.location_bbox)
            home_lon = random.uniform(float(west), float(east))
            home_lat = random.uniform(float(south), float(north))
            return Point(home_lon, home_lat)
        else:
            home_lon, home_lat = self.fake.longitude(), self.fake.latitude()
            return Point(home_lon, home_lat)

    def generate_random_point_in_location(self, home_location=None):
        """
        Generate a random point within the specified location.

        Args:
            - home_location (Point, optional): The home location within which to generate the random point.

        Returns:
            - Point: A Shapely Point object representing the random point within the location.
        """
        if home_location is None:
            home_location = self.generate_home_location()

        lat_offset, lon_offset = np.random.uniform(-1, 1, 2) * (
            self.home_radius_km / 111
        )

        random_lat = home_location.y + lat_offset
        random_lon = home_location.x + lon_offset

        return Point(random_lon, random_lat)

    def generate_intermediate_points(
        self, point_start, point_end, num_points, noise_range=(0.0001, 0.005)
    ):
        """
        Generate intermediate points between two given points.

        Args:
            - point_start (Point): The starting point.
            - point_end (Point): The ending point.
            - num_points (int): The number of intermediate points to generate.
            - noise_range (tuple, optional): Range for generating random noise (default is (0.0001, 0.005)).

        Returns:
            - list: A list of Shapely Point objects representing intermediate points.
        """
        x_start, y_start = point_start.x, point_start.y
        x_end, y_end = point_end.x, point_end.y

        intermediate_points = []
        for i in range(num_points):
            fraction = (i + 1) / (num_points + 1)
            lon = x_start + (x_end - x_start) * fraction
            lat = y_start + (y_end - y_start) * fraction

            # Generate random noise within the specified range
            noise = random.uniform(noise_range[0], noise_range[1])
            lon += random.uniform(-noise, noise)
            lat += random.uniform(-noise, noise)

            intermediate_points.append(Point(lon, lat))
        return intermediate_points

    def generate_dataframe(self, num_rows, trace_type):
        """
        Generate a GeoDataFrame with synthetic location-based data for multiple groups.

        Args:
            - num_rows (int): The number of data rows to generate for each group.
            - trace_type (str): The type of data to generate ('Staypoint' or 'Leg').

        Returns:
            - GeoDataFrame: A GeoDataFrame containing the generated location-based data.
        """
        data = []
        home_locations = []

        for group_id in range(self.num_users):
            id = self.fake.uuid4()
            home_location = self.generate_home_location()
            home_locations.extend([home_location] * num_rows)
            start_time = self.fake.date_time_this_decade()
            finished_at = start_time + timedelta(minutes=1)

            for _ in range(num_rows):
                if trace_type == "Staypoint":
                    start_time = finished_at + timedelta(minutes=random.randint(3, 60))
                    finished_at = start_time + timedelta(
                        minutes=random.randint(60, 600)
                    )
                elif trace_type == "Leg":
                    start_time = finished_at + timedelta(
                        minutes=random.randint(60, 600)
                    )
                    finished_at = start_time + timedelta(minutes=random.randint(3, 60))
                detected_mode = (
                    f"Mode::{random.choice(['Car', 'Transit', 'Walk', 'Bicycle'])}"
                )
                purpose = random.choice(["home", "work", "leisure", "other"])

                if trace_type == "Staypoint":
                    geometry = self.generate_random_point_in_location(home_location)
                elif trace_type == "Leg":
                    point_start = self.generate_random_point_in_location(home_location)
                    point_end = self.generate_random_point_in_location(home_location)

                    num_intermediate_points = random.randint(50, 200)
                    intermediate_points = self.generate_intermediate_points(
                        point_start, point_end, num_intermediate_points
                    )

                    geometry = LineString(
                        [point_start, *intermediate_points, point_end]
                    )

                    if point_start.x > 180 or point_start.y > 90:
                        estimated_length = geodesic(
                            (point_start.y, point_start.x), (point_end.y, point_end.x)
                        ).m
                    else:
                        estimated_length = point_start.distance(point_end)

                row = {
                    "user_id": id,
                    "type": trace_type,
                    "started_at": start_time,
                    "finished_at": finished_at,
                    "timezone": "Europe/Zurich",
                    "length_meters": None,
                    "detected_mode": detected_mode if trace_type == "Leg" else None,
                    "validated": bool(random.getrandbits(1)),
                    "purpose": purpose,
                    "geometry": geometry,
                    "started_on": start_time.date(),
                }
                data.append(row)

        df = pd.DataFrame(data)
        df["home_location"] = home_locations
        gdf = gpd.GeoDataFrame(df, crs=self.projection_epsg, geometry=df.geometry)
        if trace_type == "Leg":
            gdf["length_meters"] = round(gdf.geometry.to_crs("EPSG:2056").length, 1)
        else:
            del gdf["length_meters"]
        return gdf

    def generate_legs(self, num_rows=15):
        """
        Generate synthetic location-based data representing movement between locations (legs) for multiple users.

        Args:
            - num_rows (int, optional): The number of data rows to generate for each user (default is 15).

        Returns:
            - GeoDataFrame: A GeoDataFrame containing the generated legs data.
        """
        return self.generate_dataframe(num_rows, trace_type="Leg")

    def generate_staypoints(self, num_rows=15):
        """
        Generate synthetic location-based data representing user staypoints at various locations.

        Args:
            - num_rows (int, optional): The number of data rows to generate for each user (default is 15).

        Returns:
            - GeoDataFrame: A GeoDataFrame containing the generated staypoints data.
        """
        return self.generate_dataframe(num_rows, trace_type="Staypoint")

    def generate_waypoints(
        self, num_rows=15, num_extra_od_points=10, max_displacement_meters=20
    ):
        """
        Generate synthetic location-based data representing waypoints with timestamps and accuracy for multiple users.

        Args:
            - num_rows (int, optional): The number of data rows to generate for each user (default is 15).
            - num_extra_od_points (int, optional): The number of extra points to add at the beginning and end of each linestring (default is 2).
            - max_displacement_meters (float, optional): The maximum displacement in meters for the extra points (default is 10 meters).

        Returns:
            - GeoDataFrame: A GeoDataFrame containing the generated waypoints data.
        """
        df_tracks = self.generate_dataframe(num_rows, trace_type="Leg")
        waypoint_data = []

        for _, row in df_tracks.iterrows():
            line_string = row["geometry"]
            start_time = row["started_at"]
            end_time = row["finished_at"]
            point_count = len(list(line_string.coords))
            time_differences = np.linspace(0, 1, point_count)
            accuracies = np.random.poisson(100, point_count) / np.random.uniform(1, 100)

            for i in range(num_extra_od_points):
                lat, lon = line_string.coords[0]
                original_point = (lat, lon)
                displacement = geopy.distance.distance(
                    meters=random.uniform(0, max_displacement_meters)
                ).destination(original_point, random.uniform(0, 360))
                lat, lon = displacement[0], displacement[1]
                timestamp = start_time - timedelta(seconds=(num_extra_od_points - i))
                accuracy = accuracies[0] + accuracies[0] * (np.random.uniform(0, 0.50))
                waypoint_data.append(
                    [row["user_id"], "Waypoint", timestamp, lat, lon, accuracy]
                )

            for i, point in enumerate(line_string.coords):
                lat, lon = point
                timestamp = end_time + time_differences[i] * (end_time - start_time)
                accuracy = accuracies[i]
                waypoint_data.append(
                    [row["user_id"], "Waypoint", timestamp, lat, lon, accuracy]
                )

            for i in range(1, num_extra_od_points + 1):
                lat, lon = line_string.coords[-1]
                original_point = (lat, lon)
                displacement = geopy.distance.distance(
                    meters=random.uniform(0, max_displacement_meters)
                ).destination(original_point, random.uniform(0, 360))
                lat, lon = displacement[0], displacement[1]
                timestamp = end_time + timedelta(seconds=i)
                accuracy = accuracies[-1] + accuracies[-1] * (
                    np.random.uniform(0, 0.50)
                )
                waypoint_data.append(
                    [row["user_id"], "Waypoint", timestamp, lat, lon, accuracy]
                )

        waypoint_columns = [
            "user_id",
            "type",
            "tracked_at",
            "latitude",
            "longitude",
            "accuracy",
        ]
        df_waypoints = pd.DataFrame(waypoint_data, columns=waypoint_columns)

        return df_waypoints
