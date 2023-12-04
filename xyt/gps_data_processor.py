import pandas as pd
import numpy as np
from shapely.geometry import LineString
import geopy.distance
from sklearn.cluster import DBSCAN
import more_itertools as mit
import multiprocessing as mp
from numba import njit
import skfuzzy as fuzz
from pandas.api.types import CategoricalDtype
import datetime
from haversine import haversine, Unit
from dateutil.parser import isoparse
import folium
from folium import plugins
import uuid
import random
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint
import math
from math import radians, sin, cos, sqrt, atan2


class GPSDataProcessor:
    """
    GPSDataProcessor is a class designed for processing GPS data, including mode detection and leg unification.

    This class provides methods for detecting user activities and modes of transportation in GPS data, as well as
    unifying individual legs and activities into meaningful segments.

    Attributes:
        - speed_th (float): A threshold for detecting walks based on speed.
        - acceleration_th (float): A threshold for detecting walks based on acceleration.
        - minimal_walking_duration (float): Minimum duration required to classify a segment as a walk.
        - minimal_trip_duration (float): Minimum duration required for a trip.
        - use_multiprocessing (bool): Flag indicating whether to use multiple processes for parallel processing.

    Example Usage:
        .. code-block:: python

            # Initialize GPSDataProcessor with custom Args
            processor = GPSDataProcessor(speed_th=1.5, acceleration_th=0.5, minimal_walking_duration=300, minimal_trip_duration=600, use_multiprocessing=True)

            poi_waypoints = data_processor.guess_home_work(waypoints, cell_size=0.3)
            smoothed_df = data_processor.smooth(poi_waypoints, sigma=100)
            segmented_df = data_processor.segment(smoothed_df)
            mode_df = data_processor.mode_detection(segmented_df)
            legs = data_processor.get_legs(df = mode_df)
    """

    def __init__(
        self,
        radius=0.5,
        min_samples=5,
        time_gap=850,
        speed_th=2.78,
        acceleration_th=0.5,
        minimal_walking_duration=100,
        minimal_trip_duration=120,
        use_multiprocessing=False,
    ):
        self.radius = radius
        #
        self.min_samples = min_samples
        self.time_gap = time_gap
        self.use_multiprocessing = use_multiprocessing
        #
        self.speed_th = speed_th
        self.acceleration_th = acceleration_th
        self.minimal_walking_duration = minimal_walking_duration
        self.minimal_trip_duration = minimal_trip_duration

    ########HOME WORK DETECTION
    def assign_cell(self, latitude, longitude, cell_size=0.2):
        """
        Assign a cell number to a GPS point based on latitude and longitude.

        Args:
            - latitude (float): Latitude of the GPS point.
            - longitude (float): Longitude of the GPS point.
            - cell_size (float): Size of the grid cell.

        Returns:
            - int: Cell number for the GPS point.
        """
        km_east = 111.320 * np.cos(np.deg2rad(latitude)) * longitude
        km_north = 110.574 * latitude
        x_index = km_east // cell_size
        y_index = km_north // cell_size
        cell_number = (1 / 2) * (y_index + x_index) * (y_index + x_index + 1) + x_index
        return cell_number

    def process_gps_data(self, waypoints, cell_size=0.2):
        """
        Process GPS data to detect home and work locations for users.

        Args:
            - waypoints (pandas.DataFrame): DataFrame with GPS waypoints.
            - cell_size (float): Size of the grid cell.

        Returns:
            - dict: A dictionary containing user IDs as keys and their detected home and work locations as values.
        """
        home_work_locations = {}

        for user_id in waypoints["user_id"].unique():
            user_waypoints = waypoints[waypoints["user_id"] == user_id]

            if user_waypoints.shape[0] != 0:
                user_waypoints_ = user_waypoints[
                    ["tracked_at", "latitude", "longitude"]
                ]

                if user_waypoints_.shape[0] != 0:
                    user_waypoints_["cell_number"] = self.assign_cell(
                        user_waypoints_["latitude"].values,
                        user_waypoints_["longitude"].values,
                        cell_size,
                    )
                    user_waypoints_["cell_number"] = user_waypoints_[
                        "cell_number"
                    ].astype("int64")

                    # Filter waypoints for workdays between 7:00 and 19:00
                    user_workdates = user_waypoints_[
                        (user_waypoints_["tracked_at"].dt.hour >= 7)
                        & (user_waypoints_["tracked_at"].dt.hour <= 19)
                        & (user_waypoints_["tracked_at"].dt.weekday < 5)
                    ]

                    # Calculate the most visited cells during workdays and overall for this user
                    most_visited_cells_workdates = (
                        user_workdates[["cell_number", "latitude", "longitude"]]
                        .groupby("cell_number")
                        .agg(["count", "mean"])
                    )
                    most_visited_cells_workdates.columns = (
                        most_visited_cells_workdates.columns.droplevel()
                    )
                    most_visited_cells_workdates.columns = [
                        "count",
                        "mean_lat",
                        "count_lon",
                        "mean_lon",
                    ]
                    del most_visited_cells_workdates["count_lon"]
                    most_visited_cells_workdates = (
                        most_visited_cells_workdates.reset_index()
                    )
                    most_visited_cells_workdates = most_visited_cells_workdates.rename(
                        columns={"mean_lat": "latitude", "mean_lon": "longitude"}
                    ).sort_values(by="count", ascending=False)

                    home_user = (
                        most_visited_cells_workdates.iloc[0]["latitude"],
                        most_visited_cells_workdates.iloc[0]["longitude"],
                    )
                    first_workdates = (
                        most_visited_cells_workdates.iloc[0]["latitude"],
                        most_visited_cells_workdates.iloc[0]["longitude"],
                    )

                    if most_visited_cells_workdates.shape[0] > 1:
                        second_workdates = (
                            most_visited_cells_workdates.iloc[1]["latitude"],
                            most_visited_cells_workdates.iloc[1]["longitude"],
                        )
                    else:
                        second_workdates = None

                    work_user = (
                        first_workdates
                        if (first_workdates != home_user)
                        else second_workdates
                    )

                    home_work_locations[user_id] = {
                        "Home_location": home_user,
                        "Work_location": work_user,
                    }

        return home_work_locations

    def guess_home_work(self, waypoints, cell_size=0.2):
        """
        Guess and assign home and work locations to waypoints based on processed GPS data.

        Args:
            - waypoints (pandas.DataFrame): DataFrame with GPS waypoints.
            - cell_size (float): Size of the grid cell.

        Returns:
            - pandas.DataFrame: DataFrame with 'home_loc' and 'work_loc' columns added.
        """
        home_work_data = self.process_gps_data(waypoints, cell_size)
        waypoints_ = waypoints.copy()
        waypoints_["home_loc"] = waypoints_["user_id"].map(
            lambda x: Point(home_work_data.get(x, {}).get("Home_location"))
        )
        waypoints_["work_loc"] = waypoints_["user_id"].map(
            lambda x: Point(home_work_data.get(x, {}).get("Work_location"))
        )
        return waypoints_

    ########SMOOTHING SECTION

    @staticmethod
    def smooth(
        df, accuracy_threshold=100, sigma=10, smoothing_method="time"
    ) -> pd.DataFrame:
        """
        Clean and preprocess a raw GPS points DataFrame using smoothing techniques.

        Args:
            - df (pandas.DataFrame): DataFrame with GPS points to be preprocessed.
            - accuracy_threshold (int, optional): Accuracy threshold for filtering. Defaults to 100.
            - sigma (int, optional): Sigma for Gaussian smoothing, defines the size of the smoothing window. Defaults to 10.
            - smoothing_method (str, optional): Smoothing method ('time' or 'space'). Defaults to 'time'.

        Returns:
            - pandas.DataFrame: Cleaned and preprocessed DataFrame.
        """
        processor = GPSDataProcessor()
        return processor._prepare(df, accuracy_threshold, sigma, smoothing_method)

    def _prepare(
        self, df, accuracy_threshold=100, sigma=10, smoothing_method="time"
    ) -> pd.DataFrame:
        """
        Internal method to prepare the DataFrame based on accuracy and smoothing method.

        Args:
            - df (pandas.DataFrame): The raw GPS points DataFrame.
            - accuracy_threshold (int, optional): Accuracy threshold for filtering. Defaults to 100.
            - sigma (int, optional): Sigma for Gaussian smoothing, defines the size of the smoothing window. Defaults to 10.
            - smoothing_method (str, optional): Smoothing method ('time' or 'space'). Defaults to 'time'.

        Returns:
            - pandas.DataFrame: Cleaned and preprocessed DataFrame.
        """
        unique_user_ids = df["user_id"].unique()
        cleaned_and_smoothed_df = pd.DataFrame()

        for user_id in unique_user_ids:
            user_df = df[df["user_id"] == user_id]  # Subsample by user_id
            if "accuracy" in user_df.columns:
                user_df = user_df[user_df.accuracy < accuracy_threshold]

            if smoothing_method == "time":
                user_df = self._time_based_smoothing(user_df, sigma)
            elif smoothing_method == "space":
                user_df = self._space_based_smoothing(user_df, sigma)

            cleaned_and_smoothed_df = pd.concat([cleaned_and_smoothed_df, user_df])

        cleaned_and_smoothed_df.reset_index(drop=True, inplace=True)
        return cleaned_and_smoothed_df

    @staticmethod
    def _time_based_smoothing(df, sigma=10) -> pd.DataFrame:
        """
        Apply time-based Gaussian smoothing on a DataFrame of GPS waypoints.

        Args:
            - df (pandas.DataFrame): DataFrame with points to be smoothed.
            - sigma (float): Smoothing parameter.

        Returns:
            - pandas.DataFrame: The smoothed DataFrame.
        """
        if df.shape[0] == 0:
            return df  # Return the input DataFrame if it's empty

        size = df.shape[0]
        output = df.reset_index(drop=True)
        output = output.sort_values("tracked_at", ascending=True)

        # Check if the data size is less than the smoothing window size
        if size <= 6:
            return output  # Return the input DataFrame without smoothing

        output["latitude"] = pd.to_numeric(output["latitude"])
        output["longitude"] = pd.to_numeric(output["longitude"])

        timestamps = (
            pd.to_datetime(output["tracked_at"]).astype("int64") / 10**9
        ).astype("int64")
        timestamps = timestamps.values
        longitudes = output["longitude"].values
        latitudes = output["latitude"].values

        output_longitudes = []
        output_latitudes = []

        sigma_squared = sigma**2

        for i in range(size):
            start = i - 5 if (i - 5 > 0) else 0
            end = i + 6 if (i + 6 < size) else size

            timestamp_df = timestamps[start:end]
            window_longitudes = longitudes[start:end]
            window_latitudes = latitudes[start:end]

            center_timestamp = timestamps[i]

            weights = timestamp_df - center_timestamp
            weights = weights**2
            weights = weights / sigma_squared
            weights = np.array(list(map(lambda x: math.exp(-x), weights)))
            sum_weights = np.sum(weights)

            new_longitude = np.sum(weights * window_longitudes) / sum_weights
            new_latitude = np.sum(weights * window_latitudes) / sum_weights

            output_longitudes.append(new_longitude)
            output_latitudes.append(new_latitude)

        output_longitudes = pd.Series(output_longitudes)
        output_latitudes = pd.Series(output_latitudes)

        output.longitude = output_longitudes
        output.latitude = output_latitudes

        return output

    def _space_based_smoothing(self, df, sigma=10) -> pd.DataFrame:
        """
        Apply space-based Gaussian smoothing on a DataFrame of GPS waypoints.

        Args:
            - df (pandas.DataFrame): DataFrame with points to be smoothed.
            - sigma (float): Smoothing parameter.

        Returns:
            - pandas.DataFrame: The smoothed DataFrame.
        """

        def extract_coordinates(line):
            return list(line.coords)

        def gaussian_smoothing(coords):
            if len(coords) < 3:
                return coords

            smoothed_coords = []
            size = len(coords)

            for i in range(size):
                start = i - 1 if i > 0 else 0
                end = i + 2 if i < size - 2 else size

                window = coords[start:end]

                if len(window) == 3:
                    p1, p2, p3 = window
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3

                    new_x = (x1 + 4 * x2 + x3) / 6
                    new_y = (y1 + 4 * y2 + y3) / 6

                    smoothed_coords.append((new_x, new_y))
                else:
                    smoothed_coords.append(coords[i])

            return LineString(smoothed_coords)

    ########SEGMENTING SECTION

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two GPS coordinates.

        Args:
            - lat1 (float): Latitude of the first point.
            - lon1 (float): Longitude of the first point.
            - lat2 (float): Latitude of the second point.
            - lon2 (float): Longitude of the second point.

        Returns:
            - float: Haversine distance in meters.
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Radius of the Earth in meters (approximately)
        radius_ = 6371000  # meters

        # Calculate the distance in meters
        distance = radius_ * c
        return distance

    def create_route(self, df) -> pd.DataFrame:
        """
        Calculate additional statistics such as distance, time delta, speed, and acceleration for GPS waypoints.

        Args:
            - df (pandas.DataFrame): The waypoints DataFrame to be processed.

        Returns:
            - pandas.DataFrame: Waypoints DataFrame with additional statistics.
        """
        if df.shape[0] == 0:
            return pd.DataFrame(
                columns=[
                    "tracked_at_start",
                    "latitude_start",
                    "longitude_start",
                    "tracked_at_end",
                    "latitude_end",
                    "longitude_end",
                    "distance_meters",
                    "time_delta",
                    "speed",
                    "acceleration",
                    "detection",
                ]
            )

        res = df.reset_index().merge(
            df.shift(-1).dropna().reset_index(),
            on="index",
            suffixes=("_start", "_end"),
        )
        res = res.drop(
            columns=[
                "index",
                "user_id_end",
                "accuracy_start",
                "accuracy_end",
                "type_end",
            ],
            errors="ignore",
        )
        res.rename(
            columns={"user_id_start": "user_id", "type_start": "type"}, inplace=True
        )

        res["distance_meters"] = res.apply(
            lambda row: GPSDataProcessor.haversine_distance(
                row["latitude_start"],
                row["longitude_start"],
                row["latitude_end"],
                row["longitude_end"],
            ),
            axis=1,
        )

        res["time_delta"] = res.tracked_at_end - res.tracked_at_start
        res["time_delta"] = res["time_delta"].apply(
            lambda x: x.total_seconds()
        )  # in seconds
        res["speed"] = res["distance_meters"] / res["time_delta"]
        # This will be necessary for mode detection
        res["acceleration"] = res["speed"] / res["time_delta"]
        return res

    def prepare_for_detection(self, df) -> pd.DataFrame:
        """
        Flag detections as trips by default.

        Args:
            - df (pandas.DataFrame): Waypoints DataFrame to be flagged.

        Returns:
            - pandas.DataFrame: DataFrame with 'detection' column set to 'trip'.
        """
        df["detection"] = "trip"
        return df

    def activities_density(self, args):
        """
        Detect activities by density.

        Args:
            - args (tuple): A tuple containing the waypoints DataFrame and a clusterer object.

        Returns:
            - pandas.DataFrame: DataFrame with updated 'detection' column.
        """
        df, clusterer = args
        clusters_start = clusterer.fit_predict(
            np.radians(df[["latitude_start", "longitude_start"]])
        )

        df = df.copy()  # Create a copy of the DataFrame

        df.loc[:, "cluster_start"] = clusters_start.copy()
        df["cluster_end"] = df.cluster_start.shift(-1).fillna(0).astype("int")

        # update the detection column: Whenever the clusters column are different, we put trip in detection. Otherwise, we put activity.
        df.loc[
            (df["detection"] == "trip")
            & (df["cluster_start"] == df["cluster_end"])
            & (df["cluster_start"] != -1),
            ["detection"],
        ] = "activity"

        return df

    def correct_clusters(self, df):
        """
        Correct detected clusters by merging them based on time and speed.

        Args:
            - df (pandas.DataFrame): DataFrame with 'detection', 'speed', and 'time_delta' columns.
        """
        activity_indexes = df.loc[df.detection == "activity"].index.values
        trip_indexes = df.loc[df.detection == "trip"].index.values

        time_delta = df["time_delta"].values
        speed = df["speed"].values

        detection = df["detection"].values

        for group in mit.consecutive_groups(activity_indexes):
            indexes = list(group)
            window_time = np.sum(time_delta[indexes])
            if window_time < 120:
                detection[indexes] = "trip"

        for group in mit.consecutive_groups(trip_indexes):
            indexes = list(group)
            window_time = np.sum(time_delta[indexes])
            mean_speed = np.mean(speed[indexes])
            if window_time < 120 or mean_speed < 1:
                detection[indexes] = "activity"

        df["detection"] = detection

    def segment_per_user(self, df, user_id) -> pd.DataFrame:
        """
        Find clusters of waypoints for a specific user.

        Args:
            - df (pandas.DataFrame): Waypoints DataFrame to be processed.
            - user_id (int or str): The user_id for which to perform segmentation.

        Returns:
            - pandas.DataFrame: DataFrame with the segment starts and ends for the specified user.
        """
        # Filter the DataFrame for the specific user_id
        user_df = df[df["user_id"] == user_id]

        route_clusters_detected = pd.DataFrame()  # Initialize an empty DataFrame
        route_user = self.create_route(user_df)
        df = self.prepare_for_detection(route_user)

        df["day"] = df.tracked_at_start.apply(lambda x: x.day)
        df["month"] = df.tracked_at_start.apply(lambda x: x.month)
        df["year"] = df.tracked_at_start.apply(lambda x: x.year)

        db = DBSCAN(
            eps=self.radius / 6371.0,
            min_samples=self.min_samples,
            algorithm="ball_tree",
            metric="haversine",
        )

        route_clusters_detected = pd.DataFrame(
            columns=list(df.columns) + list(["cluster_start", "cluster_end"])
        )

        arguments = list(df.groupby(["day", "month", "year"]).groups.items())
        arguments = list(map(lambda x: (df.iloc[list(x[1])], db), arguments))

        if self.use_multiprocessing:
            pool = mp.Pool(processes=(mp.cpu_count() - 1))
            results = pool.map_async(self.activities_density, arguments)
            pool.close()
            pool.join()

            for res in results.get(timeout=1):
                route_clusters_detected = pd.concat(
                    [route_clusters_detected, res], ignore_index=True
                )
        else:
            for argument_list in arguments:
                res = self.activities_density(argument_list)
                route_clusters_detected = pd.concat(
                    [route_clusters_detected, res], ignore_index=True
                )

        route_clusters_detected = route_clusters_detected.sort_values(
            by="tracked_at_start", ascending=True
        )
        route_clusters_detected = route_clusters_detected.drop(
            columns=["day", "month", "year"]
        )
        route_clusters_detected = route_clusters_detected.reset_index().drop(
            columns="index"
        )

        self.correct_clusters(route_clusters_detected)

        route_clusters_detected = (
            route_clusters_detected.drop(
                route_clusters_detected[
                    route_clusters_detected.time_delta > self.time_gap
                ].index
            )
            .reset_index()
            .drop(columns="index")
        )

        route_clusters_detected = route_clusters_detected.drop(
            columns=[
                "cluster_start",
                "cluster_end",
                "type_end",
                "home_loc_end",
                "work_loc_end",
            ],
            errors="ignore",
        )

        route_clusters_detected.rename(
            columns={
                "user_id_start": "user_id",
                "type_start": "type",
                "home_loc_start": "home_loc",
                "work_loc_start": "work_loc",
            },
            inplace=True,
            errors="ignore",
        )

        return route_clusters_detected

    def segment(self, df) -> pd.DataFrame:
        """
        Segment waypoints for all unique user_ids in the DataFrame.

        Args:
            - df (pandas.DataFrame): Waypoints DataFrame with user_ids.

        Returns:
            - pandas.DataFrame: DataFrame with the segment starts and ends for all users.
        """
        unique_user_ids = df["user_id"].unique()
        result_df = pd.DataFrame()

        for user_id in unique_user_ids:
            user_segmentation = self.segment_per_user(df, user_id)
            result_df = pd.concat([result_df, user_segmentation], ignore_index=True)

        return result_df

    ##########MODE DETECTION SECTION

    def _detect_walks(self, args):
        """
        Tags the DataFrame at given indexes with 'walk' in the 'detection' column for the points corresponding to walks.

        Args:
            - args (tuple): A tuple containing the DataFrame `df`, `walk_speed_th`, `walk_acceleration_th`, `minimal_walk_duration`, and `minimal_trip_duration`.

        Returns:
            - pandas.DataFrame: DataFrame with 'walk' tags in the 'detection' column.
        """
        (
            df,
            walk_speed_th,
            walk_acceleration_th,
            minimal_walk_duration,
            minimal_trip_duration,
        ) = args
        i = 0
        window = []
        window_time = 0
        window_speed_median = 0
        window_acceleration_median = 0
        while i < len(df):
            condition_on_current_elem = (
                df.iloc[i]["speed"] <= walk_speed_th
                and df.iloc[i]["acceleration"] <= walk_acceleration_th
            )
            condition_on_existing_window = (
                window_speed_median <= walk_speed_th
                and window_acceleration_median <= walk_acceleration_th
            )

            if not condition_on_current_elem and not condition_on_existing_window:
                if len(window) and window_time >= minimal_walk_duration:
                    # Update walk status
                    trip_before_time = df.iloc[0 : window[0]]["time_delta"].sum()
                    trip_after_time = df.iloc[window[-1] + 1 :]["time_delta"].sum()

                    elements_before = window[0]
                    elements_after = len(df) - window[-1] + 1

                    if elements_before == 0 and elements_after == 0:
                        df.iloc[window, df.columns.get_loc("detection")] = "walk"
                    elif (
                        elements_before == 0
                        and trip_after_time >= minimal_trip_duration
                    ):
                        df.iloc[window, df.columns.get_loc("detection")] = "walk"
                    elif (
                        elements_after == 0
                        and trip_before_time >= minimal_trip_duration
                    ):
                        df.iloc[window, df.columns.get_loc("detection")] = "walk"
                    elif (
                        trip_before_time >= minimal_trip_duration
                        and trip_after_time >= minimal_trip_duration
                    ):
                        df.iloc[window, df.columns.get_loc("detection")] = "walk"

                    i = window[-1] + 1
                    window = []
                    window_time = 0
                    window_speed_median = 0
                    window_acceleration_median = 0
                else:
                    i += 1
                    window = []
                    window_time = 0
                    window_speed_median = 0
                    window_acceleration_median = 0
            else:
                window.append(i)
                window_time += df.iloc[i]["time_delta"]
                window_speed_median = df.iloc[window].speed.median()
                window_acceleration_median = df.iloc[window].acceleration.median()
                i += 1
        return df

    def _detect_modes(self, df):
        """
        Detects all modes except for walks using the fuzzy engine.

        Args:
            - df (pandas.DataFrame): DataFrame to be processed.

        Returns:
            - pandas.DataFrame: The modified DataFrame with estimated modes in the 'estimated_mode' column.
        """

        med_speed_verylow = [0, 0, 1.5, 2]
        med_speed_low = [1.5, 2, 4, 6]
        med_speed_medium = [5, 7, 11, 15]
        med_speed_high = [12, 15, 1000, 1000]

        acc95_low = [0, 0, 0.5, 0.6]
        acc95_medium = [0.5, 0.7, 1, 1.2]
        acc95_high = [1, 1.5, 1000, 1000]

        speed95_low = [0, 0, 6, 8]
        speed95_medium = [7.5, 9.5, 15, 18]
        speed95_high = [15, 20, 1000, 1000]

        possible_modes = []
        median_speed = np.median(df["speed"].values)
        acc_95per, speed_95per = (
            np.percentile(df["acceleration"].values, 95),
            np.percentile(df["speed"].values, 95),
        )

        medspeed_verylow_bool = (
            fuzz.trapmf(np.array([median_speed]), med_speed_verylow)[0] > 0
        )
        medspeed_low_bool = fuzz.trapmf(np.array([median_speed]), med_speed_low)[0] > 0
        medspeed_medium_bool = (
            fuzz.trapmf(np.array([median_speed]), med_speed_medium)[0] > 0
        )
        medspeed_high_bool = (
            fuzz.trapmf(np.array([median_speed]), med_speed_high)[0] > 0
        )

        acc95_low_bool = fuzz.trapmf(np.array([acc_95per]), acc95_low)[0] > 0
        acc95_medium_bool = fuzz.trapmf(np.array([acc_95per]), acc95_medium)[0] > 0
        acc95_high_bool = fuzz.trapmf(np.array([acc_95per]), acc95_high)[0] > 0

        speed95_low_bool = fuzz.trapmf(np.array([speed_95per]), speed95_low)[0] > 0
        speed95_medium_bool = (
            fuzz.trapmf(np.array([speed_95per]), speed95_medium)[0] > 0
        )
        speed95_high_bool = fuzz.trapmf(np.array([speed_95per]), speed95_high)[0] > 0

        # The paper suggests to take the minimum between the membership values.
        # We need to treat all cases where we have a non empty intersection
        if medspeed_verylow_bool and medspeed_low_bool:
            if (
                fuzz.trapmf(np.array([median_speed]), med_speed_verylow)[0]
                <= fuzz.trapmf(np.array([median_speed]), med_speed_low)[0]
            ):
                medspeed_low_bool = False
            else:
                medspeed_verylow_bool = False

        if medspeed_low_bool and medspeed_medium_bool:
            if (
                fuzz.trapmf(np.array([median_speed]), med_speed_low)[0]
                <= fuzz.trapmf(np.array([median_speed]), med_speed_medium)[0]
            ):
                medspeed_medium_bool = False
            else:
                medspeed_low_bool = False

        if medspeed_medium_bool and medspeed_high_bool:
            if (
                fuzz.trapmf(np.array([median_speed]), med_speed_medium)[0]
                <= fuzz.trapmf(np.array([median_speed]), med_speed_high)[0]
            ):
                medspeed_high_bool = False
            else:
                medspeed_medium_bool = False

        if acc95_low_bool and acc95_medium_bool:
            if (
                fuzz.trapmf(np.array([acc_95per]), acc95_low)[0]
                <= fuzz.trapmf(np.array([acc_95per]), acc95_medium)[0]
            ):
                acc95_medium_bool = False
            else:
                acc95_low_bool = False

        if acc95_medium_bool and acc95_high_bool:
            if (
                fuzz.trapmf(np.array([acc_95per]), acc95_medium)[0]
                <= fuzz.trapmf(np.array([acc_95per]), acc95_high)[0]
            ):
                acc95_high_bool = False
            else:
                acc95_medium_bool = False

        if speed95_low_bool and speed95_medium_bool:
            if (
                fuzz.trapmf(np.array([speed_95per]), speed95_low)[0]
                <= fuzz.trapmf(np.array([speed_95per]), speed95_medium)[0]
            ):
                speed95_medium_bool = False
            else:
                speed95_low_bool = False

        if speed95_medium_bool and speed95_high_bool:
            if (
                fuzz.trapmf(np.array([speed_95per]), speed95_medium)[0]
                <= fuzz.trapmf(np.array([speed_95per]), speed95_high)[0]
            ):
                speed95_high_bool = False
            else:
                speed95_medium_bool = False

        if medspeed_verylow_bool and acc95_low_bool:
            possible_modes.append("Walk")
        if medspeed_verylow_bool and acc95_medium_bool:
            possible_modes.append("Cycle")
        if medspeed_verylow_bool and acc95_high_bool:
            possible_modes.append("Cycle")

        if medspeed_low_bool and acc95_low_bool and speed95_low_bool:
            possible_modes.append("Cycle")
        if medspeed_low_bool and acc95_low_bool and speed95_medium_bool:
            possible_modes.append("Urban")
        if medspeed_low_bool and acc95_low_bool and speed95_high_bool:
            possible_modes.append("Car")
        if medspeed_low_bool and acc95_medium_bool:
            possible_modes.append("Urban")
        if medspeed_low_bool and acc95_high_bool and speed95_low_bool:
            possible_modes.append("Urban")
        if medspeed_low_bool and acc95_high_bool and speed95_medium_bool:
            possible_modes.append("Car")
        if medspeed_low_bool and acc95_high_bool and speed95_high_bool:
            possible_modes.append("Car")

        if medspeed_medium_bool and acc95_low_bool:
            possible_modes.append("Urban")
        if medspeed_medium_bool and acc95_medium_bool:
            possible_modes.append("Car")
        if medspeed_medium_bool and acc95_high_bool:
            possible_modes.append("Car")

        if medspeed_high_bool and acc95_low_bool:
            possible_modes.append("Rail")
        if medspeed_high_bool and acc95_medium_bool:
            possible_modes.append("Car")
        if medspeed_high_bool and acc95_high_bool:
            possible_modes.append("Car")

        # df['estimated_mode'] = ",".join(possible_modes)
        df.loc[:, "estimated_mode"] = ",".join(possible_modes)
        return df

    def mode_detection(self, df):
        """
        Perform mode detection on a DataFrame, including walk detection and other modes using fuzzy logic.

        Args:
            - df (pandas.DataFrame): DataFrame with GPS data.

        Returns:
            - pandas.DataFrame: DataFrame with modes and walk tags in the 'estimated_mode' and 'detection' columns.
        """
        df["estimated_mode"] = np.nan
        user_trips = df[df.detection == "trip"].index.values

        arguments = [list(x) for x in mit.consecutive_groups(user_trips)]
        arguments = list(
            map(
                lambda x: (
                    df.iloc[x],
                    self.speed_th,
                    self.acceleration_th,
                    self.minimal_walking_duration,
                    self.minimal_trip_duration,
                ),
                arguments,
            )
        )

        if self.use_multiprocessing:
            pool = mp.Pool(processes=(mp.cpu_count() - 1))
            results = pool.map_async(self._detect_walks, arguments)
            pool.close()
            pool.join()

            for res in results.get(timeout=1):
                df.loc[res.index] = res
        else:
            for argument_list in arguments:
                res = self._detect_walks(argument_list)
                df.loc[res.index] = res

        user_trips = df[df.detection == "trip"].index.values

        arguments = [list(x) for x in mit.consecutive_groups(user_trips)]
        arguments = list(map(lambda x: (df.iloc[x]), arguments))

        if self.use_multiprocessing:
            pool = mp.Pool(processes=(mp.cpu_count() - 1))
            results = pool.map_async(self._detect_modes, arguments)
            pool.close()
            pool.join()

            for res in results.get(timeout=1):
                df.loc[res.index] = res
        else:
            for argument_list in arguments:
                res = self._detect_modes(argument_list)
                df.loc[res.index] = res

        df.loc[df.detection == "walk", "estimated_mode"] = "Walk"

        return df

    ##########LEG UNIFICATION SECTION

    def get_legs(self, df):
        """
        Extract legs and activities from a DataFrame and aggregate the data for each leg or activity.

        Args:
            - df (pandas.DataFrame): DataFrame with GPS data, including 'detection', 'user_id', and 'type' columns.

        Returns:
            - pandas.DataFrame: DataFrame with aggregated legs and activities.
        """

        # Create a new grouping key based on consecutive values in the 'detection' column, 'user_id', and 'type'
        group_key = (
            (df["detection"] != df["detection"].shift())
            | (df["user_id"] != df["user_id"].shift())
            | (df["type"] != df["type"].shift())
        ).cumsum()

        # Define custom aggregation functions for activity and trip
        def aggregate_activity(group):
            first_start = group["tracked_at_start"].iloc[0]
            last_end = group["tracked_at_end"].iloc[-1]

            # Calculate the centroid of the activity points
            lon = group["longitude_start"].mean()
            lat = group["latitude_start"].mean()

            # Calculate the duration of the activity
            duration = pd.to_datetime(last_end) - pd.to_datetime(first_start)

            return pd.Series(
                {
                    "user_id": group["user_id"].iloc[0],
                    "type": group["type"].iloc[0],
                    "tracked_at_start": first_start,
                    "tracked_at_end": last_end,
                    "duration": duration,
                    "detection": group["detection"].iloc[0],
                    "longitude_start": lon,
                    "latitude_start": lat,
                    "longitude_end": lon,
                    "latitude_end": lat,
                    "distance": group["distance_meters"].sum(),
                    "time_delta": group["time_delta"].sum(),
                    "speed": group["speed"].mean(),
                    "acceleration": group["acceleration"].mean(),
                    "estimated_mode": group["estimated_mode"].iloc[0],
                    "geometry": Point(lat, lon),
                }
            )

        def aggregate_trip(group):
            first_start = group["tracked_at_start"].iloc[0]
            last_end = group["tracked_at_end"].iloc[-1]

            # Check if there are at least two points in the group
            if len(group) < 2:
                # Handle the case with only one point
                lon = group["longitude_start"].iloc[0]
                lat = group["latitude_start"].iloc[0]
            else:
                # Create a LineString from the trip points
                trip_line = LineString(
                    zip(group["latitude_start"], group["longitude_start"])
                )
                lon = group["longitude_start"].iloc[0]
                lat = group["latitude_start"].iloc[0]

            # Calculate the duration of the trip
            duration = pd.to_datetime(last_end) - pd.to_datetime(first_start)

            return pd.Series(
                {
                    "user_id": group["user_id"].iloc[0],
                    "type": group["type"].iloc[0],
                    "tracked_at_start": first_start,
                    "tracked_at_end": last_end,
                    "duration": duration,
                    "detection": group["detection"].iloc[0],
                    "longitude_start": lon,
                    "latitude_start": lat,
                    "longitude_end": group["longitude_end"].iloc[-1],
                    "latitude_end": group["latitude_end"].iloc[-1],
                    "distance": group["distance_meters"].sum(),
                    "time_delta": group["time_delta"].sum(),
                    "speed": group["speed"].mean(),
                    "acceleration": group["acceleration"].mean(),
                    "estimated_mode": group["estimated_mode"].iloc[0],
                    "geometry": trip_line if len(group) >= 2 else Point(lat, lon),
                }
            )

        # Group by the new key and apply the custom aggregation functions
        self.result = df.groupby([group_key, "detection", "user_id", "type"]).apply(
            lambda x: aggregate_activity(x)
            if x["detection"].iloc[0] == "activity"
            else aggregate_trip(x)
        )

        # Reset the index and drop the grouping key
        self.result = self.result.reset_index(level=[0, 1, 2, 3], drop=True)

        # Drop the 'longitude_start' and 'latitude_start' columns
        self.result = self.result.drop(
            ["longitude_start", "latitude_start", "longitude_end", "latitude_end"],
            axis=1,
        )

        self.result.loc[self.result.detection == "activity", "type"] = "Staypoint"
        self.result.loc[self.result.detection == "trip", "type"] = "Leg"

        # Add the home and work locations if available in df
        if "home_loc" in df and "work_loc" in df:
            # Both 'home_loc' and 'work_loc' columns are present
            self.result = pd.merge(
                self.result,
                df[["home_loc", "work_loc", "user_id"]].drop_duplicates(),
                on="user_id",
                how="left",
            )
        elif "home_loc" in df:
            # Only 'home_loc' is present
            self.result = pd.merge(
                self.result,
                df[["home_loc", "user_id"]].drop_duplicates(),
                on="user_id",
                how="left",
            )
        elif "work_loc" in df:
            # Only 'work_loc' is present
            self.result = pd.merge(
                self.result,
                df[["work_loc", "user_id"]].drop_duplicates(),
                on="user_id",
                how="left",
            )
        else:
            # Neither 'home_loc' nor 'work_loc' are present in df
            pass  # You can choose to ignore or perform another action

        return self.result
