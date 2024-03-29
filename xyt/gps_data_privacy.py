import numpy as np
from haversine import haversine_vector, Unit
from datetime import datetime, timedelta
import pandas as pd


class GPSDataPrivacy:
    """
    A class for obfuscating and aggregating GPS data to enhance privacy.

    This class provides methods to obfuscate GPS data by adding noise, shifting coordinates,
    and aggregating data within defined cell centers. It also calculates utility metrics for
    evaluating the impact of obfuscation on the data.
    """

    def __init__(self) -> None:
        pd.options.mode.chained_assignment = None

    def _add_noise(self, point, radius=100, offset=30) -> (float, float):
        """
        Adds random uniform noise to the given point, in a radius = radius - offset.

        Args:
            - point (tuple): Points as a tuple (latitude, longitude) in degrees.
            - radius (int, optional): Radius of the point neighborhood. Defaults to 100.
            - offset (int, optional): Offset from the perimeter of the neighborhood circle,
            where the shifted point can't be located. Defaults to 30.

        Returns:
            - tuple: Shifted point expressed as (latitude, longitude)
        """

        adjusted_radius = radius - offset

        assert adjusted_radius > 0, "Offset must be bigger than radius."

        latitude_shift_km = np.random.uniform(-radius, radius) / 1000
        longitude_shift_km = np.random.uniform(-radius, radius) / 1000

        while (
            np.sqrt(latitude_shift_km**2 + longitude_shift_km**2)
            > (radius - offset) / 1000
        ):
            latitude_shift_km = np.random.uniform(-radius, radius) / 1000
            longitude_shift_km = np.random.uniform(-radius, radius) / 1000

        return self._shift_point(point, latitude_shift_km, longitude_shift_km)

    def _shift_point(
        self, point, latitude_shift_km, longitude_shift_km
    ) -> (float, float):
        """
        Shifts the coordinates of a given point.

        Args:
            - point (tuple): Starting point as (latitude, longitude), in degrees.
            - latitude_shift_km (float): Latitude offset in kilometers.
            - longitude_shift_km (float): Longitude offset in kilometers.

        Returns:
            - tuple: Shifted point expressed as (latitude, longitude)
        """

        lat = point[0]
        lon = point[1]

        lat_km = self._lat_to_km(lat) + latitude_shift_km
        lon_km = self._lon_to_km(lat, lon) + longitude_shift_km

        return (self._km_to_lat(lat_km), self._km_to_lon(lon_km, lat))

    @staticmethod
    def _lon_to_km(latitude, longitude) -> float:
        """
        Expresses given longitude in kilometers to the east.

        Args:
            - latitude (float): Latitude expressed in degrees.
            - longitude (float): Longitude expressed in degrees.

        Returns:
            - float: Longitude as kilometers to the east.
        """

        km_east = 111.320 * np.cos(np.deg2rad(latitude)) * longitude
        return km_east

    @staticmethod
    def _lat_to_km(latitude) -> float:
        """
        Expresses given latitude in kilometers to the north.

        Args:
            - latitude (float): Latitude in degrees.

        Returns:
            - float: Latitude expressed in kilometers to the north.
        """

        km_north = 110.574 * latitude
        return km_north

    @staticmethod
    def _km_to_lon(km_east, latitude) -> float:
        """
        Expresses given kilometers to the east in longitude degrees.

        Args:
            - km_east (float): Longitude expressed in km going east from longitude 0,
            at the given latitude.
            - latitude (float): Latitude in degrees.

        Returns:
            - float: Longitude in degrees.
        """

        lon = km_east / (111.320 * np.cos(np.deg2rad(latitude)))
        return lon

    @staticmethod
    def _km_to_lat(km_north) -> float:
        """
        Expresses latitude given in kilometers to the east in degrees.

        Args:
            - km_north (float): Latitude expressed in kilometers to the north.

        Returns:
            - float: Latitude in degrees.
        """

        return km_north / 110.574

    def _assign_cell_center(self, latitude, longitude, cell_size) -> (float, float):
        """
        Returns the closest cell center for the given coordinates,
        when the map is divided into a lattice with the supplied cell_size.

        Args:
            - latitude (float): Latitude as degrees.
            - longitude (float): Longitude as degrees.
            - cell_size (float): Cell size in kilometers.

        Returns:
            - tuple(float, float): Coordinates of the closest cell center on the map.
        """

        km_east = self._lon_to_km(latitude, longitude)
        km_north = self._lat_to_km(latitude)

        km_east = km_east - km_east % cell_size + cell_size / 2
        km_north = km_north - km_north % cell_size + cell_size / 2

        lat = self._km_to_lat(km_north)
        lon = self._km_to_lon(km_east, lat)

        return (lat, lon)

    def obfuscate(
        self, df, locations, radius=100, offset=30, mode="remove"
    ) -> (pd.DataFrame, list[tuple[float, float]]):
        """
        Obfuscates the regions of points given in 'locations' parameter
        by either removing all the points in their proximity or changing the location
        of these points to one noisy location in the proximity circle.

        Args:
            - df (pandas.DataFrame): DataFrame with 'latitude' and 'longitude' columns.
            - locations (list): List of locations given as (latitude, longitude) tuples.
            - radius (int, optional): Radius of the obfuscation circle. Defaults to 100.
            - offset (int, optional): Smallest distance from the perimeter of the obfuscation circle at which the location of interest must be located. Defaults to 30.
            - mode (str, optional): Obfuscation mode, can be either 'remove' or 'assign'. Defaults to 'remove'.

        Returns:
            - pandas.DataFrame: DataFrame with obfuscated regions.
        """

        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("radius must be a positive integer or float.")

        if not isinstance(offset, int) or offset <= 0:
            raise ValueError("offset must be a positive integer.")

        if not isinstance(locations, list):
            raise ValueError(
                "locations must be a list of (latitude, longitude) tuples."
            )

        if mode not in ["remove", "assign"]:
            raise ValueError("mode must be either 'remove' or 'assign'")

        # shift the home location
        shifted_locations = []
        output_df = df.copy()

        for location in locations:
            if location is not None:
                shifted_location = self._add_noise(location, radius, offset)
                shifted_locations.append(shifted_location)

                output_df["location_dist"] = haversine_vector(
                    np.column_stack(
                        (output_df.latitude.values, output_df.longitude.values)
                    ),
                    np.tile(np.array(shifted_location), (output_df.shape[0], 1)),
                    Unit.METERS,
                )

                if mode == "remove":
                    output_df = output_df.loc[output_df["location_dist"] > radius]
                elif mode == "assign":
                    output_df.loc[
                        output_df.location_dist <= radius, "latitude"
                    ] = shifted_location[0]
                    output_df.loc[
                        output_df.location_dist <= radius, "longitude"
                    ] = shifted_location[1]
            else:
                shifted_locations.append(None)

        return (
            output_df.drop(columns=["location_dist"]),
            shifted_locations,
        )

    def get_obfuscation_utility(self, w_prepared, w_obfuscated, legs) -> float:
        """
        Calculates the ratio of legs affected by obfuscation to total legs.

        Args:
            - w_prepared (pandas.DataFrame): Smoothed and cleaned waypoints DataFrame.
            - w_obfuscated (pandas.DataFrame): Smoothed, cleaned, and obfuscated waypoints DataFrame.
            - legs (pandas.DataFrame): DataFrame that contains assembled legs.

        Returns:
            - [float]: Ratio of legs affected by obfuscation to total legs.
        """

        # Check if required columns are present in the input DataFrames
        required_columns_w_prepared = ["tracked_at", "latitude", "longitude"]
        if not all(
            column in w_prepared.columns for column in required_columns_w_prepared
        ):
            raise ValueError(
                "w_prepared DataFrame must have columns: 'tracked_at', 'latitude', 'longitude'."
            )

        required_columns_w_obfuscated = ["tracked_at", "latitude", "longitude"]
        if not all(
            column in w_obfuscated.columns for column in required_columns_w_obfuscated
        ):
            raise ValueError(
                "w_obfuscated DataFrame must have columns: 'tracked_at', 'latitude', 'longitude'."
            )

        required_columns_legs = ["started_at", "finished_at"]
        if not all(column in legs.columns for column in required_columns_legs):
            raise ValueError(
                "legs DataFrame must have columns: 'started_at', 'finished_at'."
            )

        if w_obfuscated.shape[0] != w_prepared.shape[0]:  # assume remove mode
            timestamps = w_prepared[
                ~w_prepared.tracked_at.isin(w_obfuscated.tracked_at)
            ].tracked_at
        else:  # assume assign mode
            timestamps = w_prepared[
                (w_prepared.latitude != w_obfuscated.latitude)
                | (w_prepared.longitude != w_obfuscated.longitude)
            ].tracked_at

        count = 0

        timestamps = timestamps.values

        starts = legs.started_at.values
        ends = legs.finished_at.values

        legs_size = legs.shape[0]

        for i in range(legs_size):
            points_affected = timestamps[
                (timestamps >= starts[i]) & (timestamps <= ends[i])
            ].shape[0]
            if points_affected > 0:
                count += 1

        return 1 - count / legs.shape[0]

    @staticmethod
    def _dt_floor(dt, delta) -> datetime:
        """
        Performs the floor operation on datetime values,
        with given timedelta as the base, e.g. 2021-01-01 12:21:47 with timedelta of 15s
        returns 2021-01-01 12:21:45.

        Args:
            - dt (datetime.datetime): Datetime value to be floored.
            - delta (datetime.timedelta): Timedelta used for the floor operation.

        Returns:
            - datetime.datetime: Floored datetime.
        """

        # dt needs to be python datetime
        dt = pd.Timestamp(dt).to_pydatetime()
        seconds = int((dt - datetime.min.replace(tzinfo=dt.tzinfo)).total_seconds())
        remainder = timedelta(
            seconds=seconds % delta.total_seconds(),
            microseconds=dt.microsecond,
        )
        return dt - remainder

    def aggregate(
        self, waypoints_df, cell_size=0.2, delta=timedelta(minutes=15)
    ) -> pd.DataFrame:
        """
        Aggregates users in timedeltas and cells on the map.
        Returns a DataFrame with the count of users in a given timedelta and cell.

        Args:
            - waypoints_df (pandas.DataFrame): DataFrame with 'latitude', 'longitude', 'user_id' and 'tracked_at' columns.
            - cell_size (float): Size of the square cells on the map, in kilometers.
            - delta (datetime.timedelta): Frequency for the time aggregation, e.g. 15 minutes.

        Returns:
            - pandas.DataFrame: DataFrame with 'tracked_at', 'cell_latitude', 'cell_longitude' and 'count' columns.
                The 'cell_latitude' and 'cell_longitude' columns give coordinates of the centers of cells on the map.
        """

        # Check if required columns are present in the input DataFrame
        required_columns = ["latitude", "longitude", "tracked_at", "user_id"]
        if not all(column in waypoints_df.columns for column in required_columns):
            raise ValueError(
                "Input DataFrame must have columns: 'latitude', 'longitude', 'tracked_at', 'user_id'."
            )

        # Check if cell_size is a positive float
        if not isinstance(cell_size, (float, int)) or cell_size <= 0:
            raise ValueError("cell_size must be a positive float or integer.")

        # Check if delta is a timedelta
        if not isinstance(delta, timedelta):
            raise ValueError("delta must be a timedelta.")

        df = waypoints_df[["latitude", "longitude", "tracked_at", "user_id"]]

        cell_centers = [
            self._assign_cell_center(point[0], point[1], cell_size)
            for point in np.column_stack((df.latitude.values, df.longitude.values))
        ]

        cell_latitudes = [x[0] for x in cell_centers]
        cell_longitudes = [x[1] for x in cell_centers]

        df["cell_latitude"] = cell_latitudes
        df["cell_longitude"] = cell_longitudes

        tracked_at = pd.to_datetime(df.tracked_at).values
        tzinfo = df.tracked_at.iloc[0].tzinfo

        tracked_at = [self._dt_floor(dt, delta) for dt in tracked_at]

        df["tracked_at"] = tracked_at
        df.tracked_at = df.tracked_at.dt.tz_localize(tzinfo)
        df = df[["user_id", "tracked_at", "cell_latitude", "cell_longitude"]]
        df = df.drop_duplicates()

        return (
            df.groupby(["tracked_at", "cell_latitude", "cell_longitude"])
            .agg("count")
            .rename(columns={"user_id": "count"})
        )
