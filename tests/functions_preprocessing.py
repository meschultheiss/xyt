from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
import math
from scipy.cluster.hierarchy import dendrogram


def find_next_activity_id(
    leg_df,
    staypoint_df,
    user_id_col="user_id",
    started_at_col="started_at",
    finished_at_col="finished_at",
    next_activity_col="next_activity_id",
):
    """
    Find the next activity for each leg based on the finished_at time.

    Args:
    - leg_df (DataFrame): DataFrame containing leg data.
    - staypoint_df (DataFrame): DataFrame containing staypoint data.
    - user_id_col (str, optional): Name of the column containing user IDs. Defaults to 'user_id'.
    - started_at_col (str, optional): Name of the column containing activity start times. Defaults to 'started_at'.
    - finished_at_col (str, optional): Name of the column containing activity finish times. Defaults to 'finished_at'.
    - next_activity_col (str, optional): Name of the column to store the next activity IDs. Defaults to 'next_activity_id'.

    Returns:
    - DataFrame: Original leg DataFrame with an additional column 'next_activity_id' indicating the next activity for each leg.

    Note:
    - The function performs a nearest-merge of leg and staypoint DataFrames based on finished_at and started_at times.
    - Forward fills NaN values in the 'next_activity_id' column to propagate the last known activity to subsequent rows.
    """
    result_df = pd.DataFrame()  # Initialize an empty DataFrame to store the results

    # Sort staypoint DataFrame once outside the loop
    staypoint_df_sorted = staypoint_df.sort_values(by=[user_id_col, started_at_col])

    for user_id in leg_df[user_id_col].unique():
        # Subset leg DataFrame for the current user
        leg_subset = leg_df[leg_df[user_id_col] == user_id].sort_values(finished_at_col)

        # Subset staypoint DataFrame for the current user
        staypoint_subset = staypoint_df_sorted[
            staypoint_df_sorted[user_id_col] == user_id
        ]

        # Merge the leg and staypoint DataFrames based on finished_at and started_at
        merged_df = pd.merge_asof(
            leg_subset,
            staypoint_subset[["activity_id", started_at_col]],
            left_on=finished_at_col,
            right_on=started_at_col,
            direction="nearest",
        )

        # Rename the columns
        merged_df = merged_df.rename(columns={"activity_id": next_activity_col})

        # Check if merged_df is empty before attempting to merge
        if not merged_df.empty:
            # Concatenate the result DataFrame with merged_df
            result_df = pd.concat(
                [result_df, merged_df[[next_activity_col, finished_at_col]]]
            )

    # Forward fill NaN values in the next_activity_id column
    # result_df[next_activity_col] = result_df.groupby(user_id_col)[next_activity_col].ffill()

    # Merge the original leg DataFrame with the result DataFrame
    result_df = pd.merge(
        leg_df, result_df, how="left", left_on=finished_at_col, right_on=finished_at_col
    )

    return result_df


def split_overnight(df, time_columns=["started_at", "finished_at"]):
    """
    Description:
        1. Split activities that go over midnight into two activities
        2. Allocate the same geolocation and activity purpose to the splitted activity
        3. Compute the duration of the splitted activities

    Args:
        df: DataFrame containing the activities to be splitted
        time_columns: specify here the name of the columns of activity start and end times
    Returns:
        DataFrame with sequence of activities all finishing at 23:59:59
        and all starting at 00:00:01, and a new 'duration' column
    """

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
        df.apply(split_activity, axis=1).tolist(), ignore_index=True
    )

    # Compute the duration in seconds
    split_activities["duration"] = (
        split_activities[time_columns[1]] - split_activities[time_columns[0]]
    ).dt.total_seconds() / 60

    return split_activities


def parse_time_geo_data(
    df,
    datetime_format="%Y-%m-%dT%H:%M:%SZ",
    time_columns=["started_at", "finished_at"],
    geo_columns=["geo_x", "geo_y"],
    CRS1="EPSG:2056",
    CRS2="EPSG:4326",
):
    """
    Descritpion:
        1. Parse to datetime
        2. Parse (x,y) tuples in CRS1 to geopandas lon/lat geometries in CRS2

    Args:
        df: DataFrame containing time and geolocation columns to be parsed. Please specify the column names if necessary.
        datetime_format: all the columns must have the same format (see format code https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
        time_columns: specify here the name of the columns to be parsed in datetime
        geo_columns: specify here the name of the (x,y) columns to be parsed in lon/lat geometries
        CRS1: Original projection of (x,y)
        CRS2: New projection for lon/lat

    Returns:
        GeoDataFrame with datetimes
    """
    # Manage dates
    for column in time_columns:
        df[column] = pd.to_datetime(df[column], format=datetime_format)
    # Manage locations
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[geo_columns[0]], df[geo_columns[1]]),
        crs=CRS1,
    )
    gdf.to_crs(CRS2, inplace=True)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    # gdf.drop(columns=[geo_columns[0],geo_columns[1]], inplace=True, axis=1)
    # gdf.rename(columns={'geometry':'original_geometry'}, inplace=True)
    return gdf


def spatial_clustering(
    gdf,
    eps=300,
    minpts=2,
    lon_lat_columns=["lon", "lat"],
    user_id_col="user_id",
    purpose_col="imputed_purpose",
):
    """
    Desription:
        1. Use the Density-based spatial clustering of applications with noise (DBSCAN) method to aggregate neighboring nodes
        2. Label the clusters with -1 being noise, and 0, 1, ..., n the number of the cluster. NB. 0 is not necessarily the denser cluster
        3. Aggregate the lon/lat to the mean of all nodes in a same cluster
    NB. clustering done for a given user_id and a given imputed_purpose

    Args:
        gdf: GeoDataFrame containing Lon and Lat information for a series of nodes. Please specify the column names if necessary.
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        minpts: The number of nodes in a neighborhood for a point to be considered as a core point. This includes the point itself.
        lon_lat_columns: refer to the longitude and latitute columns of nodes
        user_id_col: str of the column-name containing the user_ids
        purpose_cal: str of the column-name containing the purpose of the acitvity

    Returns:
        GeoDataFrame with clustered lon/lat and cluster labels
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
                    ] = sub_gdf[lon_lat_columns[0]][sub_gdf.cluster == cluster].mean()
                    sub_gdf.loc[
                        sub_gdf.cluster == cluster, lon_lat_columns[1]
                    ] = sub_gdf[lon_lat_columns[1]][sub_gdf.cluster == cluster].mean()

            output = pd.concat([output, sub_gdf], ignore_index=True)

    output.sort_values(by=["index"], inplace=True)
    output.set_index("index", inplace=True)
    return output


def cluster_info(
    df, threshold=0.5, user_id_col="user_id", purpose_col="imputed_purpose"
):
    """
    Description:
        1. Count the cluster labels for a given user_id and a given activity purpose
        2. Define the frequence of visits to the node

    Args:
        df: DataFrame containing some cluster labels
        threshold: threshold between Frequent / Occasinal visit i.e. a node is set as "frequently visited" if the number of visits to this node is >= to threshold * the number of visits to the most visited node
        user_id_col: str of the column-name containing the user_ids
        purpose_cal: str of the column-name containing the purpose of the acitvity

    Returns:
        Same DataFrame as input but with an extra column labelling the "importance of visited places"

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


# Calculate network-based distance between each node and return the OD matrix (in meter)
def network_distance_matrix(u, G, vs):
    dists = [
        nx.dijkstra_path_length(G, source=u, target=v, weight="length") for v in vs
    ]
    return pd.Series(dists, index=vs)


def OD_distance(gdf, G):
    # attach nearest network node to each POI
    gdf["nearest_node"] = ox.get_nearest_nodes(
        G, X=gdf["lon"], Y=gdf["lat"], method="balltree"
    )
    # Get distances for each pair of nodes
    nodes_unique = pd.Series(gdf["nearest_node"].unique())
    nodes_unique.index = nodes_unique.values
    # convert MultiDiGraph to DiGraph for simpler faster distance matrix computation
    G_dm = nx.DiGraph(G)
    # create node-based distance matrix
    output = nodes_unique.apply(network_distance_matrix, G=G_dm, vs=nodes_unique)
    output = output.astype(int)
    return output


def get_distance(origin, destination, od_matrix):
    if math.isnan(origin) or math.isnan(destination):
        output = np.nan
    else:
        output = od_matrix.loc[origin, destination] * 1000
    return output


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
