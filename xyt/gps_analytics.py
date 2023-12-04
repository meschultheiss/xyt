import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric
from sklearn.cluster import DBSCAN
import datetime
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import math


class GPSAnalytics:
    """
    Performs analytics on GPS data, including splitting activities over midnight,
    spatial clustering, deriving metrics, and computing daily statistics.

    This class offers methods to manipulate GPS data stored in DataFrames. It provides
    functionalities to split activities that span midnight into two, perform spatial
    clustering using DBSCAN, derive various metrics related to trips and activities,
    and compute daily descriptive statistics.
    """

    def __init__(self) -> None:
        pd.set_option("display.max_columns", 999)
        self.expect_col_spt = [
            "activity_id",
            "started_at",
            "finished_at",
            "purpose",
            "user_id",
            "lon",
            "lat",
        ]
        self.expect_col_leg = [
            "leg_id",
            "started_at",
            "finished_at",
            "detected_mode",
            "mode",
            "user_id",
            "geometry",
            "next_activity_id",
            "length",
            "duration",
        ]
        self.expect_col_ext_stp = [
            # "leg_id",
            "started_at",
            "finished_at",
            # "detected_mode",
            # "mode",
            "user_id",
            # "geometry",
            # "next_activity_id",
            # "length",
            "duration",
            "cluster",
            "cluster_size",
            "cluster_info",
            "location_id",
            "peak",
            "first_dep",
            "last_arr",
            "home_loop",
            "daily_trip_dist",
            "num_trip",
            "max_dist",
            "min_dist",
            "max_dist_from_home",
            "dist_from_home",
            "home_location_id",
            "weekday",
        ]

    @staticmethod
    def verify_columns(df: pd.DataFrame, expected_columns: list[str]) -> None:
        """
        Verifies if expected columns are present in the DataFrame.

        Args:
            - df: DataFrame to check columns in.
            - expected_columns: List of expected column names.
        """

        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Columns missing in DataFrame: {', '.join(missing_columns)}"
            )

    def split_overnight(
        self, staypoint: pd.DataFrame, time_columns=["started_at", "finished_at"]
    ) -> pd.DataFrame:
        """
        Splits activities going over midnight into two activities.

        Description:
            1. Split activities that go over midnight into two activities
            2. Allocate the same geolocation and activity purpose to the splitted activity
            3. Compute the duration of the splitted activities

        Args:
            - staypoint: DataFrame containing activities to split. Columns: 'activity_id', 'started_at', 'finished_at', 'purpose', 'user_id', 'lon', 'lat'
            - time_columns: Columns for activity start and end times.

        Returns:
            - DataFrame with split activities and new 'duration' column.
        """

        self.verify_columns(df=staypoint, expected_columns=self.expect_col_spt)

        def split_activity(row):
            # Check if the activity spans midnight
            if row[time_columns[0]].date() != row[time_columns[1]].date():
                # Split the activity into two parts
                part1 = row.copy()
                part2 = row.copy()

                part1[time_columns[1]] = pd.to_datetime(
                    part1[time_columns[0]].date().strftime("%Y-%m-%d") + " 23:59:59"
                )
                part2[time_columns[0]] = pd.to_datetime(
                    part2[time_columns[1]].date().strftime("%Y-%m-%d") + " 00:00:01"
                )

                return pd.DataFrame([part1, part2])

            return pd.DataFrame([row])

        # Apply the split_activity function to each row and concatenate the result
        split_activities = pd.concat(
            staypoint.apply(split_activity, axis=1).tolist(), ignore_index=True
        )

        # Compute the duration in seconds
        split_activities["duration"] = (
            split_activities[time_columns[1]] - split_activities[time_columns[0]]
        ).dt.total_seconds() / 60

        return split_activities

    @staticmethod
    def _spatial_clustering(
        gdf,
        eps=300,
        minpts=2,
        lon_lat_columns=["lon", "lat"],
        user_id_col="user_id",
        purpose_col="imputed_purpose",
    ) -> pd.DataFrame:
        """
        Performs Density-based spatial clustering (DBSCAN) on GeoDataFrame.

        Desription:
            1. Use the Density-based spatial clustering of applications with noise (DBSCAN) method to aggregate neighboring nodes
            2. Label the clusters with -1 being noise, and 0, 1, ..., n the number of the cluster. NB. 0 is not necessarily the denser cluster
            3. Aggregate the lon/lat to the mean of all nodes in a same cluster
            4. NB. clustering done for a given user_id and a given imputed_purpose

        Args:
            - gdf: GeoDataFrame with Lon/Lat nodes. Columns: Lon/Lat columns defined by 'lon_lat_columns', 'user_id', 'imputed_purpose'
            - eps: Maximum distance between samples.
            - minpts: Number of nodes in a neighborhood.
            - lon_lat_columns: Columns for longitude and latitude.
            - user_id_col: Column name for user IDs.
            - purpose_col: Column name for activity purpose.

        Returns:
            - GeoDataFrame with clustered lon/lat and labels.
        """

        # parameterize DBSCAN
        eps_rad = eps / 3671000.0  # meters to radians
        db = DBSCAN(
            eps=eps_rad, min_samples=minpts, metric="haversine", algorithm="ball_tree"
        )
        # add a column for cluster labelling
        gdf["cluster"] = np.nan
        gdf["cluster_size"] = np.nan
        # initialize the output DF
        output = pd.DataFrame()

        # NB run DBSCAN per user_id and per activity purpose
        for user_id in gdf[user_id_col].unique():
            for purpose in gdf.loc[gdf[user_id_col] == user_id][purpose_col].unique():
                # compute DBSCAN using straight-line haversine distances
                sub_gdf = []
                sub_gdf = gdf.loc[
                    (gdf[user_id_col] == user_id) & (gdf[purpose_col] == purpose)
                ]
                sub_gdf = sub_gdf.copy(deep=True)
                sub_gdf.reset_index(inplace=True)

                # Perform DBSCAN clustering from features, and return cluster labels.
                cl = db.fit_predict(
                    np.deg2rad(sub_gdf[[lon_lat_columns[0], lon_lat_columns[1]]])
                )

                max_size = 0
                cl_size = 0
                for cluster in np.unique(cl):
                    sub_gdf.loc[(cl == cluster).tolist(), "cluster"] = cluster
                    if cluster != -1:
                        sub_gdf.loc[(cl == cluster), "cluster_size"] = len(
                            sub_gdf.loc[(cl == cluster)]
                        )
                    else:
                        sub_gdf.loc[(cl == cluster), "cluster_size"] = 1

                    if cluster != -1:
                        sub_gdf.loc[
                            sub_gdf.cluster == cluster, lon_lat_columns[0]
                        ] = sub_gdf[lon_lat_columns[0]][
                            sub_gdf.cluster == cluster
                        ].mean()
                        sub_gdf.loc[
                            sub_gdf.cluster == cluster, lon_lat_columns[1]
                        ] = sub_gdf[lon_lat_columns[1]][
                            sub_gdf.cluster == cluster
                        ].mean()

                output = pd.concat([output, sub_gdf], ignore_index=True)

        output.sort_values(by=["index"], inplace=True)
        output.set_index("index", inplace=True)
        return output

    @staticmethod
    def _cluster_info(
        df, threshold=0.5, user_id_col="user_id", purpose_col="imputed_purpose"
    ) -> pd.DataFrame:
        """
        Computes cluster labels and importance of visited places.

        Args:
            - df: DataFrame with cluster labels. Columns: 'user_id', 'imputed_purpose', 'cluster', 'cluster_size'
            - threshold: Threshold for frequent/occasional visits.
            - user_id_col: Column name for user IDs.
            - purpose_col: Column name for activity purpose.

        Returns:
            - DataFrame with labeled places.
        """

        output = pd.DataFrame()
        df["cluster_info"] = np.nan

        for user_id in df[user_id_col].unique():
            for purpose in df.loc[df[user_id_col] == user_id][purpose_col].unique():
                sub_gdf = df.loc[
                    (df[user_id_col] == user_id) & (df[purpose_col] == purpose)
                ]
                sub_gdf = sub_gdf.copy(deep=True)
                sub_gdf.reset_index(inplace=True)

                max_size = sub_gdf["cluster_size"][sub_gdf.cluster != -1].max()

                sub_gdf.loc[sub_gdf.cluster == -1, "cluster_info"] = "Visited once"
                sub_gdf.loc[
                    (sub_gdf.cluster_size == max_size) & (sub_gdf.cluster != -1),
                    "cluster_info",
                ] = "Most visited"
                sub_gdf.loc[
                    (sub_gdf.cluster_size >= max_size * threshold)
                    & (sub_gdf.cluster_info != "Most visited")
                    & (sub_gdf.cluster != -1),
                    "cluster_info",
                ] = "Frequent visit"
                sub_gdf.loc[
                    (sub_gdf.cluster_size < max_size * threshold)
                    & (sub_gdf.cluster_info != "Most visited")
                    & (sub_gdf.cluster != -1),
                    "cluster_info",
                ] = "Occasional visit"

                output = pd.concat([output, sub_gdf], ignore_index=True)
        output.sort_values(by=["index"], inplace=True)
        output.set_index("index", inplace=True)

        return output

    def spatial_clustering(self, staypoint: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates locations of most visited places.

        Args:
            - staypoint: DataFrame containing activity locations. Columns: 'activity_id', 'started_at', 'finished_at', 'purpose', 'user_id', 'lon', 'lat'

        Returns:
            - DataFrame with clustered locations and labels.
        """

        self.verify_columns(df=staypoint, expected_columns=self.expect_col_spt)

        new_spt = staypoint.copy()

        # Aggregate the locations of most visited places per user_id and per imputed_purpose
        new_spt = self._spatial_clustering(new_spt, purpose_col="purpose")

        # Label the Most Visited Places
        new_spt = self._cluster_info(new_spt, purpose_col="purpose")

        # Aggregate the 100m-neighbooring lon,lat pairs
        db = DBSCAN(
            eps=100 / 3671000, min_samples=2, metric="haversine", algorithm="ball_tree"
        )
        cl = db.fit_predict(np.deg2rad(new_spt[["lon", "lat"]]))
        for cluster in np.unique(cl):
            if cluster != -1:
                new_spt.loc[cl == cluster, "lon"] = new_spt.loc[
                    cl == cluster, "lon"
                ].mean()
                new_spt.loc[cl == cluster, "lat"] = new_spt.loc[
                    cl == cluster, "lat"
                ].mean()
        # Add location id for unique lon,lat pairs
        new_spt["location_id"] = np.nan
        for counter, lon, lat in (
            new_spt.groupby(["lon", "lat"], as_index=False)
            .size()[["lon", "lat"]]
            .itertuples(index=True)
        ):
            new_spt.loc[
                (new_spt.lon == lon) & (new_spt.lat == lat), "location_id"
            ] = counter
        return new_spt

    @staticmethod
    def _get_distance(origin, destination, od_matrix) -> float:
        """
        Calculate distance between two geographical coordinates.

        Args:
            - lon1: Longitude of the first point.
            - lat1: Latitude of the first point.
            - lon2: Longitude of the second point.
            - lat2: Latitude of the second point.

        Returns:
            - Distance between the points (in kilometers).
        """

        if math.isnan(origin) or math.isnan(destination):
            output = np.nan
        else:
            output = od_matrix.loc[origin, destination] * 1000
        return output

    def get_metrics(self, staypoint: pd.DataFrame, leg: pd.DataFrame) -> pd.DataFrame:
        """
        Computes additional variables and metrics.

        Args:
            - staypoint: DataFrame with activity data. Columns: 'activity_id', 'started_at', 'finished_at', 'purpose', 'user_id', 'lon', 'lat'
            - leg: DataFrame with leg data. Columns: 'leg_id', 'started_at', 'finished_at', 'detected_mode', 'mode', 'user_id', 'geometry', 'next_activity_id', 'length', 'duration'

        Returns:
            - DataFrame with computed metrics.
        """

        self.verify_columns(df=staypoint, expected_columns=self.expect_col_spt)
        self.verify_columns(df=leg, expected_columns=self.expect_col_leg)
        # Derive additional variables
        df = staypoint.copy()
        # DAILY USER_ID: Add user_ids per day
        df.insert(
            1,
            "user_id_day",
            df["user_id"]
            + "_"
            + df.started_at.dt.year.astype(str)
            + df.started_at.dt.month.astype(str).str.zfill(2)
            + df.started_at.dt.day.astype(str).str.zfill(2),
        )
        # PEAK HOURS: Add boolean if trip starts (i.e. activity ends) in peak hour
        morning_hours = range(6, 10)  # 6:00 AM to 9:59 AM
        noon_hours = range(12, 15)    # 12:00 PM to 2:59 PM
        evening_hours = range(16, 20) # 4:00 PM to 7:59 PM

        df["peak"] = "off_peak"  # Assuming initially all times are off-peak
        df.loc[
            df['finished_at'].dt.hour.isin(morning_hours),
            "peak"
        ] = "morning_peak"
        df.loc[
            df['finished_at'].dt.hour.isin(noon_hours),
            "peak"
        ] = "noon_peak"
        df.loc[
            df['finished_at'].dt.hour.isin(evening_hours),
            "peak"
        ] = "evening_peak"
        # Get the time of first departure / last arrival
        df["first_dep"] = np.nan
        df["last_arr"] = np.nan
        df["home_loop"] = 0
        df["daily_trip_dist"] = np.nan
        df["num_trip"] = np.nan
        df["max_dist"] = np.nan
        df["min_dist"] = np.nan
        df["max_dist_from_home"] = np.nan
        df["dist_from_home"] = np.nan
        df["home_location_id"] = np.nan

        location = (
            staypoint[["lon", "lat", "location_id"]]
            .copy()
            .sort_values("location_id")
            .reset_index(drop=True)
        )
        location.drop_duplicates(ignore_index=True, inplace=True)
        od_matrix_kms = pd.DataFrame(
            DistanceMetric.get_metric("haversine").pairwise(
                location[["lat", "lon"]].to_numpy()
            )
            * 6373,
            columns=location.location_id.unique(),
            index=location.location_id.unique(),
        )

        # RUN PARRALEL FUNCTIONS FUNC0. FUNC1 & FUNC2
        # BE CAREFUL THIS PART CAN BE LONG
        # MAKE A PROGRESS BAR ?

        df.rename(columns={"purpose": "imputed_purpose"}, inplace=True)

        for func in [self._func1, self._func2, self._func3]:
            cores = mp.cpu_count()
            # split the df in as many array as the machine has cores
            user_ids = np.array_split(df.user_id_day.unique(), cores, axis=0)
            df_split = []
            for u in user_ids:
                df_split.append(df.loc[df.user_id_day.isin(u.tolist())])
            # create the multiprocessing pool
            pool = Pool(cores)
            # process the DataFrame by mapping function to each df across the pool
            if func == self._func2:
                func2_partial = partial(self._func2, leg=leg)
                df_out = np.vstack(pool.map(func2_partial, df_split))
            elif func == self._func3:
                func3_partial = partial(self._func3, od_matrix_kms=od_matrix_kms)
                df_out = np.vstack(pool.map(func3_partial, df_split))
            else:
                df_out = np.vstack(pool.map(func, df_split))

            # return the df
            df = pd.DataFrame(df_out, columns=df.columns)

            # close down the pool and join
            pool.close()
            pool.join()
            pool.clear()

            if func == self._func2:
                # drop the days with only one obesrvation and small connection duration
                df.drop(
                    df[(df.first_dep.isna()) & (df.duration < 43200)].index,
                    inplace=True,
                )  # 43200sec is 12 hours
                # Add weekdays
                df["weekday"] = df.started_at.dt.weekday
                # SORT VALUES
                df.sort_values(
                    by=["user_id_day", "started_at"], inplace=True, ignore_index=True
                )
                df["cluster_size"] = df["cluster_size"].astype(int)

            if func == self._func3:
                df.reset_index(inplace=True, drop=True)
                for index, row in df.iterrows():
                    df.loc[index, "dist_from_home"] = self._get_distance(
                        row["location_id"], row["home_location_id"], od_matrix_kms
                    )
        return df

    @staticmethod
    def _func1(df) -> pd.DataFrame:
        import datetime
        import pandas as pd
        import numpy as np

        ## Clean up the first / last activity of the day
        # CASE 1. Manage cases where the first/last act is at home but started sometime in the morning/afternoon --> set started_at == 00:00:01 / finished_at == 23:59:59
        # Set a df with only the first/last activities of user-days (NB DO NOT RESET OR IGNORE INDEXES):
        first_act = df.loc[
            (df.drop_duplicates(subset=["user_id_day"], keep="first").index)
        ].copy()
        last_act = df.loc[
            (df.drop_duplicates(subset=["user_id_day"], keep="last").index)
        ].copy()

        case1f = first_act[
            (first_act.started_at.dt.time > datetime.time(0, 0, 1))
            & (first_act.started_at.dt.time < datetime.time(12, 0, 0))
            & (first_act.imputed_purpose == "Home")
        ]
        case1l = last_act[
            (last_act.finished_at.dt.time > datetime.time(12, 0, 0))
            & (last_act.finished_at.dt.time < datetime.time(23, 59, 59))
            & (last_act.imputed_purpose == "Home")
        ]
        df.loc[case1f.index, "started_at"] = pd.to_datetime(
            df.loc[case1f.index, "started_at"].dt.date.astype(str) + "T00:00:01Z",
            format="%Y-%m-%dT%H:%M:%SZ",
        )
        df.loc[case1l.index, "finished_at"] = pd.to_datetime(
            df.loc[case1l.index, "finished_at"].dt.date.astype(str) + "T23:59:59Z",
            format="%Y-%m-%dT%H:%M:%SZ",
        )
        # Recalculate the durations
        df.loc[case1l.index.append(case1f.index), "duration"] = (
            df.loc[case1l.index.append(case1f.index), "finished_at"]
            - df.loc[case1l.index.append(case1f.index), "started_at"]
        ) / np.timedelta64(1, "s")

        # CASE 2. Drop all the user_id_day for wich the first / last activity does not starts / ends at 00:00:01 / 23:59:59
        len_before = len(df)
        # Set a df with only the first/last activities of user-days (NB DO NOT RESET OR IGNORE INDEXES):
        first_act = df.loc[
            (df.drop_duplicates(subset=["user_id_day"], keep="first").index)
        ].copy()
        last_act = df.loc[
            (df.drop_duplicates(subset=["user_id_day"], keep="last").index)
        ].copy()

        # Clean the user_id_day with home only
        index_to_drop = []
        for user_id in df.user_id_day.unique():
            try:
                if (
                    len(df.loc[df.user_id_day == user_id, "location_id"].unique()) == 1
                ) & (
                    "Home"
                    in df.loc[df.user_id_day == user_id, "imputed_purpose"].unique()
                ):
                    index_to_drop.extend(
                        df.loc[(df.user_id_day == user_id)].index[1:].tolist()
                    )
                    df.loc[(df.user_id_day == user_id), "duration"] == df.loc[
                        (df.user_id_day == user_id), "duration"
                    ].sum()
                    df.loc[(df.user_id_day == user_id), "started_at"] == df.loc[
                        (df.user_id_day == user_id), "started_at"
                    ].min()
                    df.loc[(df.user_id_day == user_id), "finished_at"] == df.loc[
                        (df.user_id_day == user_id), "finished_at"
                    ].max()
            except ValueError:
                continue
        df = df.loc[~df.index.isin(index_to_drop)].copy()
        df.reset_index(inplace=True, drop=True)

        df = df.loc[
            (
                ~df.user_id_day.isin(
                    first_act.loc[
                        first_act.started_at.dt.time != datetime.time(0, 0, 1),
                        "user_id_day",
                    ].array
                )
            )
        ]
        df = df.loc[
            (
                ~df.user_id_day.isin(
                    last_act.loc[
                        last_act.finished_at.dt.time != datetime.time(23, 59, 59),
                        "user_id_day",
                    ].array
                )
            )
        ]
        # len_after = len(df)
        # print('Warning: clean up operation reduced the df lenght by -' + str("{:.1f}".format((len_before-len_after)*100/len_before)) + ' %')
        ##Most user-days start at home:
        # print('But note that there are still some weird cases like a day starting with Shopping. Those cases are however very few :' + "\n" + df.drop_duplicates(subset=['user_id_day'], keep='first').groupby(by='imputed_purpose').count()['user_id_day'].to_string())

        return df

    @staticmethod
    def _func2(df, leg) -> pd.DataFrame:
        import datetime
        import geopandas

        # Compute miscellaneous additional variables
        for user_id in df.user_id_day.unique():
            try:
                if len(df.loc[(df.user_id_day == user_id)]) > 1:
                    df.loc[(df.user_id_day == user_id), "first_dep"] = df.loc[
                        (df.user_id_day == user_id), "finished_at"
                    ].dt.time.min()
                    df.loc[(df.user_id_day == user_id), "last_arr"] = df.loc[
                        (df.user_id_day == user_id), "started_at"
                    ].dt.time.max()
                if (
                    sum(
                        df.loc[(df.user_id_day == user_id), "imputed_purpose"].isin(
                            ["Home", "home"]
                        )
                    )
                    > 1
                ):
                    df.loc[(df.user_id_day == user_id), "home_loop"] = (
                        sum(
                            df.loc[(df.user_id_day == user_id), "imputed_purpose"].isin(
                                ["Home", "home"]
                            )
                        )
                        - 1
                    )
                # find from legs the actual trip distance
                date = datetime.datetime.strptime(
                    user_id[-8:], "%Y%m%d"
                ).date()  # retrieve the date of the concerned activities from the user_id_day string
                condition1 = leg.next_activity_id.isin(
                    df.loc[(df.user_id_day == user_id), "activity_id"].tolist()
                )  # spot all the legs tracked to reach the activities
                condition2 = leg.started_at.dt.date == date  # match the dates
                df.loc[(df.user_id_day == user_id), "daily_trip_dist"] = leg.loc[
                    (condition1) & (condition2), "length"
                ].sum()
                # return the number of trips between activities
                df.loc[(df.user_id_day == user_id), "num_trip"] = (
                    len(df[df.user_id_day == user_id]) - 1
                )
            except ValueError:
                pass

        return df

    @staticmethod
    def _func3(df, od_matrix_kms) -> pd.DataFrame:
        from sklearn.metrics import DistanceMetric
        import numpy as np
        import math

        # Compute the distances between locations
        for user_id in df.user_id_day.unique():
            try:
                # Compute max/min distance between all activity locations
                od_pairs = df.loc[
                    df.user_id_day == user_id, ["lon", "lat"]
                ].drop_duplicates(ignore_index=True)
                od_dist = (
                    DistanceMetric.get_metric("haversine").pairwise(
                        od_pairs[["lat", "lon"]].to_numpy()
                    )
                    * 6373
                )
                if len(od_dist) > 1:
                    df.loc[df.user_id_day == user_id, "max_dist"] = (
                        od_dist.max().astype(int) * 1000
                    )
                    df.loc[df.user_id_day == user_id, "min_dist"] = (
                        od_dist[np.nonzero(od_dist)].min().astype(int) * 1000
                    )  # get the min among non-null values
                else:
                    df.loc[df.user_id_day == user_id, "max_dist"] = 0
                    df.loc[df.user_id_day == user_id, "min_dist"] = 0
                # Compute max distance from home
                home_locations = df.loc[
                    (df.user_id_day == user_id)
                    & (df.imputed_purpose.str.lower() == "home"),
                    ["location_id", "cluster_size"],
                ]
                if len(home_locations.location_id.unique()) > 1:
                    home_id = home_locations.loc[
                        home_locations.cluster_size.idxmax(), "location_id"
                    ]
                else:
                    home_id = home_locations.location_id.mean()
                if math.isnan(home_id) == False:
                    all_id = df.loc[
                        (df.user_id_day == user_id) & (df.location_id != home_id),
                        "location_id",
                    ]
                    df.loc[df.user_id_day == user_id, "max_dist_from_home"] = (
                        od_matrix_kms.loc[
                            od_matrix_kms.index.isin(all_id), home_id
                        ].max()
                        * 1000
                    )
                    df.loc[df.user_id_day == user_id, "home_location_id"] = home_id
            except ValueError:
                continue

        return df

    def get_daily_metrics(self, ext_staypoint: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a matrix of daily descriptive statistics.

        Args:
            - ext_staypoint: DataFrame with relevant columns. Columns: 'started_at', 'finished_at', 'user_id', 'duration', 'cluster', 'cluster_size', 'cluster_info', 'location_id', 'peak', 'first_dep', 'last_arr', 'home_loop', 'daily_trip_dist', 'num_trip', 'max_dist', 'min_dist', 'max_dist_from_home', 'dist_from_home', 'home_location_id', 'weekday'

        Returns:
            - DataFrame with computed daily profiles.
        """

        self.verify_columns(df=ext_staypoint, expected_columns=self.expect_col_ext_stp)
        # Select relevant columns
        daily_act = ext_staypoint[
            [
                "user_id_day",
                "first_dep",
                "last_arr",
                "home_loop",
                "daily_trip_dist",
                "peak",
                "num_trip",
                "max_dist_from_home",
                "weekday",
            ]
        ].copy()

        # Convert 'first_dep' and 'last_arr' to minutes since midnight
        daily_act.loc[daily_act.first_dep.notnull(), "first_dep"] = (
            (
                pd.to_datetime(
                    daily_act.loc[daily_act.first_dep.notnull(), "first_dep"],
                    format="%H:%M:%S",
                )
                - np.datetime64("1900-01-01")
            )
            .dt.total_seconds()
            .div(60)
            .astype(int)
        )
        daily_act.loc[daily_act.last_arr.notnull(), "last_arr"] = (
            (
                pd.to_datetime(
                    daily_act.loc[daily_act.last_arr.notnull(), "last_arr"],
                    format="%H:%M:%S",
                )
                - np.datetime64("1900-01-01")
            )
            .dt.total_seconds()
            .div(60)
            .astype(int)
        )

        # Remove duplicate rows based on 'user_id_day'
        daily_act.drop_duplicates(
            subset=["user_id_day"], keep="first", ignore_index=True, inplace=True
        )

        # Create new columns 'am_peak', 'pm_peak', and 'noon_peak'
        daily_act["am_peak"] = daily_act["pm_peak"] = daily_act["noon_peak"] = 0
        daily_act.loc[daily_act.peak == "morning_peak", "am_peak"] = 1
        daily_act.loc[daily_act.peak == "evening_peak", "pm_peak"] = 1
        daily_act.loc[daily_act.peak == "noon_peak", "noon_peak"] = 1

        # Drop the 'peak' column
        daily_act.drop("peak", inplace=True, axis=1)

        # Set 'user_id_day' as the index
        daily_act.set_index("user_id_day", inplace=True)

        return daily_act
