import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import Point, LineString
import random


def plot_gps_on_map(
    df, trace_type=None, home_col=None, work_col=None, geo_columns=None
):
    """
    Plot location-based data on a Folium map with different colors for each group.

    Args:
        - df (pd.DataFrame): A Pandas DataFrame containing the location-based data to be plotted.
        - trace_type (str, optional): The type of data to be plotted ('Stay', 'Track', or 'Waypoint'). If provided, only data of the specified trace type will be plotted.
        - home_col (str, optional): Name of the column containing the home coordinates. Default is None.
        - work_col (str, optional): Name of the column containing the work coordinates. Default is None.
        - geo_columns (str or list, optional): Name of the column(s) containing the latitude and longitude coordinates. Can be a string (e.g., 'geometry') or a list (e.g., ['latitude', 'longitude']). Default is None.

    Returns:
        - Interactive Folium map.
    """
    if geo_columns:
        if isinstance(geo_columns, str):
            geom_column = geo_columns
        elif isinstance(geo_columns, list) and len(geo_columns) == 2:
            df["geometry"] = df.apply(
                lambda row: Point(row[geo_columns[1]], row[geo_columns[0]]), axis=1
            )
            geom_column = "geometry"
        else:
            raise ValueError(
                "Invalid geo_columns parameter. It must be a string or a list of two strings."
            )
    elif "geometry" in df:
        geom_column = "geometry"
    elif "longitude" in df and "latitude" in df:
        df["geometry"] = df.apply(
            lambda row: Point(row["latitude"], row["longitude"]), axis=1
        )
        geom_column = "geometry"
    else:
        raise ValueError(
            "DataFrame must have 'geometry' column or 'longitude' and 'latitude' columns."
        )

    if df.empty:
        return None

    if isinstance(df[geom_column].iloc[0], Point):
        m = folium.Map(
            location=[df[geom_column].iloc[0].y, df[geom_column].iloc[0].x],
            zoom_start=10,
        )
    elif isinstance(df[geom_column].iloc[0], LineString):
        m = folium.Map(
            location=[
                df[geom_column].iloc[0].coords[0][1],
                df[geom_column].iloc[0].coords[0][0],
            ],
            zoom_start=10,
        )

    user_colors = {}  # To store colors for each user ID
    plotted_home = set()  # To track which users' home locations have been plotted
    plotted_work = set()  # To track which users' work locations have been plotted

    for index, row in df.iterrows():
        if trace_type and row["type"] != trace_type:
            continue  # Skip rows that don't match the specified trace type

        user_id = row["user_id"]
        if user_id not in user_colors:
            user_colors[user_id] = "#{:02x}{:02x}{:02x}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )

        color = user_colors[user_id]
        popup_text = f"Type: {row['type']}<br>ID: {row['user_id']}"

        if row["type"] == "Staypoint":
            lat, lon = (
                row[geom_column].y,
                row[geom_column].x,
            )  # Swap latitude and longitude
            folium.CircleMarker(
                location=[lat, lon],
                color="red",
                radius=4,
                fill=True,
                fill_color="red",
                popup=popup_text,
            ).add_to(m)
        elif row["type"] == "Waypoint" or row["type"] == "Leg":
            if isinstance(row[geom_column], Point):
                lat, lon = (
                    row[geom_column].y,
                    row[geom_column].x,
                )  # Swap latitude and longitude
                folium.CircleMarker(
                    location=[lat, lon],
                    color=color,
                    radius=2,
                    fill=True,
                    fill_color=color,
                    popup=popup_text,
                ).add_to(m)
            elif isinstance(row[geom_column], LineString):
                linestring = [
                    (point[1], point[0]) for point in row[geom_column].coords
                ]  # Swap latitude and longitude
                folium.PolyLine(
                    locations=linestring, color=color, popup=popup_text
                ).add_to(m)

        if home_col and user_id not in plotted_home and home_col in df:
            home_location = df[df["user_id"] == user_id][home_col].iloc[0]
            folium.Marker(
                location=[home_location.y, home_location.x],
                icon=folium.Icon(color="red", icon="home", prefix="fa"),
            ).add_to(m)
            plotted_home.add(user_id)

        if work_col and user_id not in plotted_work and work_col in df:
            work_location = df[df["user_id"] == user_id][work_col].iloc[0]
            folium.Marker(
                location=[work_location.y, work_location.x],
                icon=folium.Icon(color="red", icon="briefcase", prefix="fa"),
            ).add_to(m)
            plotted_work.add(user_id)

    folium.plugins.Fullscreen().add_to(m)
    return m
