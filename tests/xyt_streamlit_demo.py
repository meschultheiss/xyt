import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium, folium_static
from xyt import FakeDataGenerator, GPSDataProcessor
from xyt import plot_gps_on_map


st.title("Welcome to XYT")
st.subheader("Generate fake GPS data")
st.write(
    "We first propose to populate a dataframe with fake GPS data in order to be able to play and getting familiar with the app without infringing on anybody's privacy"
)
# Example usage


st.code(
    """
fakegps = FakeDataGenerator(location_name="Suisse", num_users=15)
"""
)
st.markdown(
    """From this class, we can generate different types of data :
    - wayopints
    - get_legs
    - staypoints"""
)
st.code(
    """
    waypoints = fakegps.generate_waypoints(num_rows=12, num_extra_od_points=9, max_displacement_meters = 5)
    """
)


@st.cache_data
def get_waypoints(loc="Suisse"):
    fakegps = FakeDataGenerator(location_name=loc, num_users=5)
    waypoints = fakegps.generate_waypoints(
        num_rows=12, num_extra_od_points=12, max_displacement_meters=12
    )
    return waypoints


# legs = fakegps.generate_legs(num_rows=12)
# stays = fakegps.generate_staypoints(num_rows=12)

# st.dataframe(waypoints.head(5))
# st_data = folium_static(plot_gps_on_map(waypoints))
waypoints = get_waypoints()

st.dataframe(waypoints.head(3))
map = folium_static(plot_gps_on_map(waypoints))

st.subheader("Transform your raw GPS {x,y,t} into proper mobility data")
st.markdown("""Create an instance of GPSDataProcessor""")

st.code("""data_processor = GPSDataProcessor(radius=0.03)""")

data_processor = GPSDataProcessor(radius=0.03, min_samples=3, time_gap=100)

st.markdown(
    """Apply Gaussian smoothing, staypoint / leg segmentation, and some basic mode inference to your waypoints DataFrame"""
)

st.code(
    """
smoothed_df = data_processor.smooth(waypoints)
segmented_df = data_processor.segment(smoothed_df)
mode_df = data_processor.mode_detection(segmented_df)
"""
)

st.markdown("""You can eventually get a proper leg and staypoint data structure""")

st.code(
    """
legs = data_processor.get_legs(df = mode_df)
"""
)


@st.cache_data
def get_processed_waypoints():
    data_processor = GPSDataProcessor(radius=0.025)
    poi_waypoints = data_processor.guess_home_work(waypoints, cell_size=0.3)
    smoothed_df = data_processor.smooth(poi_waypoints, sigma=100)
    segmented_df = data_processor.segment(smoothed_df)
    mode_df = data_processor.mode_detection(segmented_df)
    return data_processor.get_legs(df=mode_df)  #


legs = get_processed_waypoints()
st.dataframe(legs.head(3))
# st_data = folium_static(plot_gps_on_map(legs,home_col='home_loc',work_col='work_loc'))
