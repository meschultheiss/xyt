# Data Manipulation and Analysis
import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj

# Visualization
import plotly.graph_objects as go
import seaborn as sns

import matplotlib.pyplot as plt
import contextily as ctx
from pointpats.centrography import (
    mean_center,
    weighted_mean_center,
    euclidean_median,
    dtot,
)
from shapely.geometry import MultiPoint
from matplotlib.patches import Ellipse
from scipy.spatial import distance
from shapely.geometry import Polygon, Point
import folium
import networkx as nx
import pointpats


class GPStoActionspace:
    """
    A class for analyzing GPS-based user activity and generating action spaces.

    This class provides methods to process GPS data, compute action spaces for users,
    visualize action spaces using Matplotlib and Folium, calculate covariance matrices,
    and compute innovation rates based on user activity motifs.
    """

    def __init__(self) -> None:
        plt.rcParams["figure.figsize"] = [5, 5]
        plt.rcParams["figure.dpi"] = 120  # 200 e.g. is really fine, but slower
        sns.set_theme(style="white")
        sns.set_palette("Pastel2_r")
        cm = sns.light_palette("#fff4ac", as_cmap=True)

    @staticmethod
    def verify_columns(df: pd.DataFrame, expected_columns: list[str]) -> None:
        """
        Verify if expected columns are present in the DataFrame.

        Args:
            - df (pandas.DataFrame): DataFrame to check.
            - expected_columns (list[str]): List of column names expected in the DataFrame.

        Raises:
            - ValueError: If columns are missing in the DataFrame.
        """

        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Columns missing in DataFrame: {', '.join(missing_columns)}"
            )

    @staticmethod
    def modified_std_distance(pp, center) -> float:
        """
        Calculate standard distance of a point array like std_distance() does in PySAL with the mean center, but we can specify here a different center.

        Args:
            - pp: point pattern
            - center: array([ x, y])

        Returns:
            - float: standard distance
        """

        return np.sqrt(
            1
            / len(pp)
            * (
                np.sum((pp.geometry.x - center[0]) ** 2)
                + np.sum((pp.geometry.y - center[1]) ** 2)
            )
        )

    def compute_action_space(
        self, act: pd.DataFrame, aggregation_method: str
    ) -> pd.DataFrame:
        """
        Compute action space Args for each user.

        Args:
            - act (pandas.DataFrame): DataFrame with user activity data.
            - aggregation_method (str): Aggregation method ('user_id' or 'user_id_day').

        Returns:
            - pandas.DataFrame: DataFrame with action space Args for each user.
        """

        action_space = []
        cols = [
            "user_id",
            "lon",
            "lat",
            "cluster_size",
            "cluster_info",
            "location_id",
            "home_location_id",
            "num_trip",
            "main_home_location_id",
            "dist_from_home",
            "imputed_purpose",
        ]

        self.verify_columns(act, cols)

        # check if aggregation method is user_id or user_id_day

        if aggregation_method not in ["user_id", "user_id_day"]:
            raise ValueError(
                f"aggreg_method must be either user_id or user_id_day, not {aggregation_method}"
            )

        for user_id_ in act[aggregation_method].unique():
            act_ = act.loc[act[aggregation_method] == user_id_, cols].copy()
            # get the main home id at user_id scale
            main_home_id = act_.home_location_id.value_counts().idxmax()

            if main_home_id == "not_detected":
                continue
            # elif np.any(act_.dist_from_home > 3000000):
            #    continue
            elif act_.main_home_location_id.unique() != act_.home_location_id.unique():
                continue
            elif np.all(act_.num_trip == 0):
                continue
            else:
                act_.drop(columns=["num_trip", "dist_from_home"], inplace=True)

                # Assuming your DataFrame is called df
                act_ = gpd.GeoDataFrame(
                    act_,
                    geometry=gpd.points_from_xy(act_["lon"], act_["lat"]),
                    crs="EPSG:4326",
                )

                # Change the CRS to EPSG:2056
                act_ = act_.to_crs(
                    "EPSG:2056"
                )  # This should be a parameter to change in the method, but default set to this one

                # Now, you can access the transformed x and y coordinates
                act_["x"], act_["y"] = act_.geometry.x, act_.geometry.y

                # And drop duplicated locations (information of visit count is in cluster_size)
                act_ = act_.drop_duplicates(ignore_index=True)

                # Differentiate most frequent locations from all locations
                act_freq = act_.loc[
                    act_.cluster_info.isin(["Most visited", "Frequent visit"])
                ].reset_index(drop=True)

                if len(act_) == 0:
                    continue
                else:
                    # Find mean center of a point array
                    mc_ = mean_center(act_[["x", "y"]])
                    # Find weighted mean center of a marked point pattern.
                    wmc_ = weighted_mean_center(act_[["x", "y"]], act_.cluster_size)

                    # Calculate Args of standard deviational ellipse for a point pattern
                    if len(act_) > 2:
                        sx, sy, theta = pointpats.centrography.ellipse(act_[["x", "y"]])
                        theta_degree = np.degrees(
                            theta
                        )  # need degree of rotation to plot the ellipse
                        # Calculate the surface of the SD ellipse of the action space, in square meters
                        surface_ellipse = np.pi * sx * sy
                    else:
                        surface_ellipse = 0
                    # sx, sy, theta, theta_degree

                    # Calculate the ellipse motplotlib objet
                    e = Ellipse(
                        xy=wmc_,
                        width=sx * 2,
                        height=sy * 2,
                        angle=-theta_degree,
                        alpha=0.2,
                        facecolor="#c9cfbc",
                    )  # angle is rotation in degrees (anti-clockwise)

                    # Find the main home coordinates
                    find_home_geom = act_.loc[
                        act_.location_id == act_.home_location_id.unique()[0],
                        "geometry",
                    ]
                    if len(find_home_geom) == 1:
                        home_loc = np.array(
                            [find_home_geom.x, find_home_geom.y], dtype=float
                        )
                    elif len(find_home_geom) > 1:
                        home_loc = np.array(
                            [find_home_geom[0].x, find_home_geom[0].y], dtype=float
                        )

                    # Sum of Euclidean distances between event points and a selected point.
                    sum_dist_home = dtot(home_loc, act_[["x", "y"]].to_numpy())

                    # Compute location regularity
                    n_all = len(act_)
                    n_freq = len(act_freq)
                    regularity = n_freq / n_all

                    # Calculate standard distance between all nodes and home
                    std_dist_all = self.modified_std_distance(act_, home_loc)
                    # Calculate standard distance between freq nodes and home
                    if len(act_freq) > 0:
                        std_dist_freq = self.modified_std_distance(act_freq, home_loc)
                    else:
                        std_dist_freq = np.nan

                    # Calculate the proximity
                    proximity = std_dist_freq / std_dist_all

                    # Calculate the Euclidean median for a point pattern.
                    median_ = euclidean_median(act_[["x", "y"]].to_numpy())[0]

                    # Get area of convex hull for frequent (i.e. habitual) locations in square meters
                    # And Coefficient of intensity
                    if len(act_freq.location_id.unique()) > 2:
                        freq_hull_area = MultiPoint(act_freq.geometry).convex_hull.area
                        intensity = (
                            MultiPoint(act_freq.geometry).convex_hull.area
                            / MultiPoint(act_.geometry).convex_hull.area
                        )
                    elif len(act_freq.location_id.unique()) == 2:
                        loc1 = (
                            act_freq[["x", "y"]]
                            .loc[
                                act_freq.location_id == act_freq.location_id.unique()[0]
                            ]
                            .drop_duplicates()
                        )
                        loc2 = (
                            act_freq[["x", "y"]]
                            .loc[
                                act_freq.location_id == act_freq.location_id.unique()[1]
                            ]
                            .drop_duplicates()
                        )

                        coord1 = loc1[["x", "y"]].to_numpy().flatten()
                        coord2 = loc2[["x", "y"]].to_numpy().flatten()

                        freq_hull_area = distance.euclidean(coord1, coord2)
                        intensity = 0
                    else:
                        freq_hull_area = 0
                        intensity = 0

                    # Calculate euclidean distance between weighted mean center and main home location in meters
                    home_shift = distance.euclidean(wmc_, home_loc.flatten())

                    action_space_ = np.array(
                        [
                            user_id_,
                            main_home_id,
                            sum_dist_home,
                            proximity,
                            std_dist_all,
                            std_dist_freq,
                            median_,
                            intensity,
                            regularity,
                            n_all,
                            n_freq,
                            freq_hull_area,
                            home_shift,
                            e,
                            surface_ellipse,
                            wmc_[0],
                            wmc_[1],
                            sx,
                            sy,
                            theta_degree,
                        ]
                    )
                    action_space.append(action_space_)

        columns_ = [
            aggregation_method,
            "main_home_id",
            "sum_dist_home",
            "proximity",
            "std_dist_all",
            "std_dist_freq",
            "median_",
            "intensity",
            "regularity",
            "n_all",
            "n_freq",
            "freq_hull_area",
            "home_shift",
            "ellipse_2056",
            "surface_ellipse",
            "wmc_x",
            "wmc_y",
            "sx",
            "sy",
            "theta_degree",
        ]
        action_space = pd.DataFrame(action_space, columns=columns_)
        action_space.set_index(aggregation_method, inplace=True)

        # Parse to float
        col_2 = [
            "sum_dist_home",
            "proximity",
            "std_dist_all",
            "std_dist_freq",
            "median_",
            "intensity",
            "regularity",
            "n_all",
            "n_freq",
            "freq_hull_area",
            "home_shift",
            "surface_ellipse",
        ]
        action_space[col_2] = action_space[col_2].astype(float)

        return action_space

    def plot_ellipses(
        self, action_space: pd.DataFrame, aggregation_method: str
    ) -> folium.Map:
        """
        Plot ellipses on a Folium map based on action space data.

        Args:
            - action_space (pandas.DataFrame): DataFrame with action space data.
            - aggregation_method (str): Aggregation method ('user_id' or 'user_id_day').

        Returns:
            - folium.Map: Folium map with ellipses.
        """

        # Create a list to store ellipse geometries
        ellipse_geometries = []

        # Define the source and target coordinate reference systems (CRS)
        source_crs = "EPSG:2056"
        target_crs = "EPSG:4326"

        # Iterate through the DataFrame and add ellipses to the list
        for idx, row in action_space.reset_index().iterrows():
            # Extract ellipse information
            user_id = row[aggregation_method]
            ellipse = row["ellipse_2056"]

            # Convert ellipse to Polygon geometry
            ellipse_polygon = Polygon(ellipse.get_verts())
            ellipse_coords = [[x, y] for x, y in ellipse_polygon.exterior.coords]

            ellipse_polygon = Polygon(ellipse.get_verts())
            ellipse_geometries.append(ellipse_polygon)

        # Create a GeoDataFrame from the list of ellipse geometries
        ellipse_gdf = gpd.GeoDataFrame(
            geometry=ellipse_geometries, crs=source_crs
        ).to_crs(target_crs)

        # Calculate the mean centroid of all ellipses
        mean_lat = ellipse_gdf.geometry.apply(lambda geom: geom.centroid.x).mean()
        mean_lon = ellipse_gdf.geometry.apply(lambda geom: geom.centroid.y).mean()

        # Create a Folium map centered around the mean centroid
        mymap = folium.Map(
            location=[mean_lon, mean_lat], zoom_start=12, control_scale=True
        )

        # Iterate through the GeoDataFrame and add ellipses to the map
        for idx, row in ellipse_gdf.iterrows():
            # Extract ellipse information
            user_id = idx  # Assuming user_id is the index in the GeoDataFrame
            ellipse_polygon = row["geometry"]

            # Convert ellipse to coordinates for Folium
            ellipse_coords = list(ellipse_polygon.exterior.coords)
            ellipse_coords_t = [[y, x] for x, y in ellipse_coords]

            # Add the ellipse to the map
            folium.Polygon(
                locations=ellipse_coords_t,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.02,
                popup=f"User ID: {user_id}",
            ).add_to(mymap)

        # Display the map
        return mymap  # .save("ellipse_map.html")

    def covariance_matrix(
        self,
        action_space: pd.DataFrame,
        title: str = "",
        annot: bool = False,
        cmap=sns.diverging_palette(360, 65, l=80, as_cmap=True),
    ):
        """
        Map the correlation, covariance, or p-values of a set of observed variables.

        Args:
            - action_space (pandas.DataFrame): DataFrame with action space data.
            - title (str): Title for the heatmap.
            - annot (bool): If True, return the values in the heatmap cells.
            - cmap: Color map for the heatmap.

        Returns:
        - Heatmap of correlation, covariance, or p-values of observed variables.
        """

        cols = [
            "sum_dist_home",
            "proximity",
            "std_dist_all",
            "std_dist_freq",
            "median_",
            "intensity",
            "regularity",
            "n_all",
            "n_freq",
            "freq_hull_area",
            "home_shift",
            "surface_ellipse",
        ]

        self.verify_columns(action_space, cols)

        X = action_space[cols]

        X_ = X.corr()
        vmin = None
        title2 = "Correlation matrix of X_"

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(X_, dtype=bool), 0)

        # Set the title
        if len(title) < 1:
            title = title2  # pre-set titles else specified titles

        # Set up the matplotlib figure
        f, ax = plt.subplots(ncols=1, figsize=(7, 7))
        plt.title(title)

        # Draw the heatmap with the mask and correct aspect ratio
        fig = sns.heatmap(
            X_,
            mask=mask,
            cmap=cmap,
            vmin=vmin,
            vmax=None,
            center=0,
            square=True,
            linewidths=0.1,
            cbar_kws={"shrink": 0.7},
            annot=annot,
            fmt=".2g",
        )
        return plt.show()

    def plot_action_space(
        self,
        act: pd.DataFrame,
        action_space: pd.DataFrame,
        user: str,
        how: str = "vignette",
        save: bool = False,
    ):
        """
        Plot the action space for a specific user.

        Args:
            - act (pandas.DataFrame): DataFrame with user activity data.
            - action_space (pandas.DataFrame): DataFrame with action space data.
            - user (str): User ID for whom the action space will be plotted.
            - how (str): Plotting method ('vignette' or 'folium').
            - save (bool): If True, save the plot.

        Returns:
            - Visualization of the action space.
        """

        cols = [
            "user_id",
            "lon",
            "lat",
            "cluster_size",
            "cluster_info",
            "location_id",
            "home_location_id",
            "num_trip",
            "main_home_location_id",
            "dist_from_home",
            "imputed_purpose",
        ]

        # Determine aggregation method and check if it exists in the DataFrame
        if "user_id" in action_space.columns or "user_id" == action_space.index.name:
            aggregation_method = "user_id"
        elif (
            "user_id_day" in action_space.columns
            or "user_id_day" == action_space.index.name
        ):
            aggregation_method = "user_id_day"
        else:
            raise ValueError("No user_id_day or user_id column in act DataFrame")

        # check if user exists in the DataFrame
        if user not in act[aggregation_method].unique():
            raise ValueError(f"User {user} not in the DataFrame")

        act_ = act.loc[act[aggregation_method] == user, cols].copy()

        act_.drop(columns=["num_trip", "dist_from_home"], inplace=True)

        # Assuming your DataFrame is called df
        act_ = gpd.GeoDataFrame(
            act_, geometry=gpd.points_from_xy(act_["lon"], act_["lat"]), crs="EPSG:4326"
        )

        # Change the CRS to EPSG:2056
        act_ = act_.to_crs(
            "EPSG:2056"
        )  # This should be a parameter to change in the method, but default set to this one

        # Now, you can access the transformed x and y coordinates
        act_["x"], act_["y"] = act_.geometry.x, act_.geometry.y

        # And drop duplicated locations (information of visit count is in cluster_size)
        act_ = act_.drop_duplicates(ignore_index=True)

        if how == "vignette":
            self._plot_action_space_vignette(act_, action_space, user, save)
        elif how == "folium":
            return self._plot_action_space_folium(act_, action_space, user, save)
        else:
            raise ValueError(f"how must be either vignette or folium, not {how}")

    def _plot_action_space_vignette(
        self,
        act_: pd.DataFrame,
        action_space: pd.DataFrame,
        user: str,
        save: bool = False,
    ):
        """
        Plot the action space for a specific user using a vignette style.

        Args:
            - act_ (pandas.DataFrame): DataFrame with user activity data.
            - action_space (pandas.DataFrame): DataFrame with action space data.
            - user (str): User ID for whom the action space will be plotted.
            - save (bool): If True, save the plot.

        Returns:
            - Visualization of the action space using a vignette style.
        """

        # Create a GeoDataFrame for the points from act_
        geometry = [Point(x, y) for x, y in zip(act_["x"], act_["y"])]
        gdf = gpd.GeoDataFrame(act_, geometry=geometry, crs="EPSG:2056")

        # Create a Matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot Ellipse

        e = action_space.loc[user, "ellipse_2056"]

        ellipse_polygon = Polygon(e.get_verts())
        ellipse_coords = np.array(ellipse_polygon.exterior.coords.xy)

        # ellipse_coords_4326 = np.column_stack(transformer.transform(x, y) for x, y in zip(ellipse_coords[0], ellipse_coords[1]))
        ax.fill(
            ellipse_coords[0],
            ellipse_coords[1],
            color="red",
            alpha=0.2,
            edgecolor="red",
        )

        # Plot points from act_
        ax.scatter(
            gdf["x"],
            gdf["y"],
            color="red",
            marker="o",
            s=50,
            label="Points from act",
        )

        # Plot wmc_ in green
        ax.scatter(
            action_space.loc[user, "wmc_x"],
            action_space.loc[user, "wmc_y"],
            color="green",
            marker="o",
            s=50,
            label="Weighted mean center",
        )
        find_home_geom = act_.loc[
            act_.location_id == act_.home_location_id.unique()[0], "geometry"
        ]
        if len(find_home_geom) == 1:
            home_loc = np.array([find_home_geom.x, find_home_geom.y], dtype=float)
        elif len(find_home_geom) > 1:
            home_loc = np.array([find_home_geom[0].x, find_home_geom[0].y], dtype=float)
        # Plot home_loc from act_ with a customized icon
        ax.scatter(
            home_loc[0],
            home_loc[1],
            color="black",
            marker="^",
            s=100,
            label="Home Location",
        )

        # Add legend
        ax.legend()

        # Set the extent of the plot based on the bounding box of the GeoDataFrame
        minx, miny, maxx, maxy = gdf.total_bounds
        ax.set_xlim(minx * 0.998, maxx * 1.002)
        ax.set_ylim(miny * 0.998, maxy * 1.002)

        # Plot basemap using OpenStreetMap
        ctx.add_basemap(ax, crs="EPSG:2056", source=ctx.providers.OpenStreetMap.Mapnik)

        # Save and show the figure
        if save:
            plt.savefig("action_space_vignette_" + user + ".png", dpi=300)

        plt.show()

    def _plot_action_space_folium(
        self,
        act_: pd.DataFrame,
        action_space: pd.DataFrame,
        user: str,
        save: bool = False,
    ):
        """
        Plot the action space for a specific user using Folium for interactive maps.

        Args:
            - act_ (pandas.DataFrame): DataFrame with user activity data.
            - action_space (pandas.DataFrame): DataFrame with action space data.
            - user (str): User ID for whom the action space will be plotted.
            - save (bool): If True, save the interactive map.

        Returns:
            - Folium interactive map displaying the action space for the given user.
        """

        # Find weighted mean center of a marked point pattern.
        wmc_ = weighted_mean_center(act_[["x", "y"]], act_.cluster_size)

        # Convert wmc_ to 'EPSG:4326'
        wmc_4326 = (
            gpd.GeoSeries(Point(wmc_[0], wmc_[1]), crs="EPSG:2056")
            .to_crs("EPSG:4326")
            .iloc[0]
        )
        wmc_4326 = [wmc_4326.y, wmc_4326.x]  # Flip lat and lon to match Folium format

        # Create a GeoDataFrame for the points from act_
        geometry = [Point(lon, lat) for lon, lat in zip(act_["lon"], act_["lat"])]
        gdf = gpd.GeoDataFrame(act_, geometry=geometry, crs="EPSG:4326")

        # Find the main home coordinates
        find_home_geom = act_.loc[
            act_.location_id == act_.home_location_id.unique()[0], "geometry"
        ]
        if len(find_home_geom) == 1:
            home_loc = np.array([find_home_geom.x, find_home_geom.y], dtype=float)
        elif len(find_home_geom) > 1:
            home_loc = np.array([find_home_geom[0].x, find_home_geom[0].y], dtype=float)

        # Create a GeoDataFrame for the home_loc
        home_loc_geometry = [Point(home_loc[0], home_loc[1])]
        home_loc_gdf = gpd.GeoDataFrame(
            geometry=home_loc_geometry, crs="EPSG:2056"
        ).to_crs("EPSG:4326")

        # Create a GeoDataFrame for the ellipse
        # Define the source and target coordinate reference systems (CRS)
        source_crs = "EPSG:2056"
        target_crs = "EPSG:4326"
        # Create a PyProj transformer for the conversion
        transformer = pyproj.Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )

        # Create Ellipse
        e = action_space.loc[user, "ellipse_2056"]
        ellipse_polygon = Polygon(e.get_verts())
        ellipse_coords = [[x, y] for x, y in ellipse_polygon.exterior.coords]
        # Convert coordinates to EPSG:4326
        ellipse_coords_4326 = [transformer.transform(x, y) for x, y in ellipse_coords]
        ellipse_coords_4326_t = [[y, x] for x, y in ellipse_coords_4326]
        ellipse_gdf = gpd.GeoDataFrame(
            geometry=[Polygon(ellipse_coords_4326)], crs="EPSG:4326"
        )

        # Create a Folium map centered around the mean coordinates
        mean_lat, mean_lon = gdf["lat"].mean(), gdf["lon"].mean()
        map_center = [mean_lat, mean_lon]
        mymap = folium.Map(location=map_center, zoom_start=12, control_scale=True)

        # Add basemap using OpenStreetMap
        folium.TileLayer("openstreetmap").add_to(mymap)

        # Add Ellipse
        # folium.Polygon(locations=ellipse.exterior.coords, color='#c9cfbc', fill=True, fill_color='#c9cfbc', fill_opacity=0.2).add_to(mymap)
        folium.GeoJson(
            ellipse_gdf,
            name="polygon",
            zoom_on_click=True,
            style_function=lambda x: {"color": "red"},
        ).add_to(mymap)

        # Add points from act_ with CircleMarkers
        for idx, row in gdf.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.7,
                popup=f"{row['user_id']}",
            ).add_to(mymap)

        # Add wmc_ in green
        folium.CircleMarker(
            location=wmc_4326,
            radius=5,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=1,
            popup="Weighted mean ceter",
        ).add_to(mymap)

        # Add home_loc from act_ with a customized icon
        folium.Marker(
            location=[home_loc_gdf.geometry.y.iloc[0], home_loc_gdf.geometry.x.iloc[0]],
            icon=folium.Icon(color="red", icon="home"),
        ).add_to(mymap)

        if save:
            mymap.save("action_space_folium_" + user + ".html")

        # Display the map
        return mymap

    # def get_inno_rate_per_phase(
    #     action_space,
    #     multiday_graph,
    #     user_id_,
    #     col_digraph_motif="DiGraph_motif",
    #     phase=None,
    #     treatment=None,
    # ):
    #     """
    #     Compute the innovation rate from motif
    #     user_id: list of users to compute the innovation rate
    #     mtf_useridday: df of DiGraph_motifs with user_id_days in index
    #     """

    #     mtf_useridday = pd.merge(
    #         multiday_graph[["DiGraph_motif", "motif_flat"]],
    #         action_space,
    #         left_index=True,
    #         right_index=True,
    #         how="left",
    #     )

    #     # df_innov_rate = pd.DataFrame(index=range(100),columns=user_id)

    #     # isolate rows for phase X only
    #     if phase != None:
    #         mtf_useridday = mtf_useridday[mtf_useridday.phase == phase]
    #     # isolate rows for treatment X only
    #     if treatment != None:
    #         mtf_useridday = mtf_useridday[mtf_useridday.treatment == treatment]
    #     # resample for continuous days
    #     graphs = (
    #         mtf_useridday[mtf_useridday.user_id == user_id_]
    #         .set_index("date")
    #         .resample("D")
    #         .ffill()
    #     )
    #     # read the first graph G0 for user_id
    #     G0 = graphs[col_digraph_motif][0]
    #     # init inno rate list
    #     innovation_rate = []
    #     # then compute the next graph and compare with G0
    #     for G in graphs[col_digraph_motif][1:]:
    #         innovation_rate.append(nx.compose(G0, G).number_of_nodes())
    #         G0 = nx.compose(G0, G)
    #     # x = np.arange(0, len(innovation_rate), 1).tolist()
    #     # y = innovation_rate

    #     return innovation_rate
