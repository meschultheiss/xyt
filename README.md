# xyt-demo

## Usage

Install the xyt library

```bash
cd xyt-demo
pip install -e .
```

This repository uses black for code formatting. To format your code, run

```bash
black .
```

## Input gps format

Here are some specific things to know regarding geodata processing

- CRS is important – this is the geographic projection system which is set to `EPSG:4327` (internatioinal standard, aka WGS 84) or `EPSG:2056`  (Swiss standard, aka CH1903+ / LV95) – please stick to WGS 84
- WGS 84 is in degrees, CH1903+ in in meter – this matters when computing distances, dbscna, etc.
- `datetimes`  are set to a specific time zone (UTC+1 for Switzerland)
- Typically there are three types of geometries:
    - `type==waypoints` is a shapely point() unlabeled i.e., raw gps data
    - `type==staypoint` is a shapely point() detected as an activity
    - `type==leg` is a shapely linestring() detected as a trip
- Waypoints typically have the following columns. Note that some other columns such as `<'detected_mode'>` or `<'detected_purpose'>` may be inferred from the gps data provider

```python
['user_id', 'type', 'tracked_at', 'latitude', 'longitude', 'accuracy']
```

- Leg and / or Staypoint df typically have the following columns :

```python
['user_id', 'type', 'started_at', 'finished_at', 'timezone',
		    'length_meters', 'detected_mode', 'purpose', 'geometry',
				'home_location', 'work_location']
```

## Keep in mind

- You work in degrees
- Write docstrings for documentation generation later on
- Document the structure of the input df so we know what column and column name we should find in the input df for each method
- Test the instances – without spending too much time on corner cases, we aim at a Minimum Viable Product here which will be further enhanced and maintained later on

## List of instances and ‘public’ methods

List of instances in the python library xyt:

```python
from xyt import FakeDataGenerator, GPSDataProcessor, GPSDataPrivacy, GPSAnalytics, GPStoGraph, GPStoActionspace, plot_gps_on_map
```

- `FakeDataGenerator()`

A fake gps data generator to play with the library without infringing on users’ privacy

```python
fakegps = FakeDataGenerator(location_name="Suisse", num_users=5, home_radius_km = 20)
waypoints = fakegps.generate_waypoints(num_rows=12, num_extra_od_points=10, max_displacement_meters = 10)
legs = fakegps.generate_legs(num_rows=12)
stays = fakegps.generate_staypoints(num_rows=12)
```

- `GPSDataProcessor()`

A geotagging instance to transform raw gps data into mobility data

```python
data_processor = GPSDataProcessor(radius=0.03)

poi_waypoints = data_processor.guess_home_work(waypoints_df, cell_size=0.3)
smoothed_df = data_processor.smooth(poi_waypoints, sigma=10)
segmented_df = data_processor.segment(smoothed_df)
mode_df = data_processor.mode_detection(segmented_df)
legs_ = data_processor.get_legs(df = mode_df)
```

- `plot_gps_on_map()`

A function to plot easily xyt’s instances outputs

```python
plot_gps_on_map(poi_waypoints, home_col='home_loc', work_col='work_loc')
```

- `GPSDataPrivacy()`

An instance to artificially degrade the data for privacy purposes

```python
data_privacy = GPSDataPrivacy()

df_obfuscated = data_privacy.obfuscate()
utility = data_privacy.get_obfuscation_utility()
df_aggergated = data_privacy.aggregate()
```

- `GPSAnalytics()`

An instance to perform space-based and time-based analytics on the mobility data

```python
metrics = GPSAnalytics()

metrics.check_inputs()
staypoint1 = metrics.split_overnight(staypoint)
staypoint2 = metrics.spatial_clustering(staypoint1)
extended_staypoint = metrics.get_metrics(staypoint2)
day_staypoint = metrics.get_daily_metrics(extended_staypoint)
```

- `GPStoGraph()`

An instance to abstract mobility diaries as a graph

```python
graphs = GPStoGraph()

multiday_graph = graphs.get_graphs(extended_staypoint)
graphs.plot_motif(multiday_graph)
graphs.plot_graph(multiday_graph)
motif_seq = graphs.motif_sequence(multiday_graph)
```

- `GPStoActionspace()`

An instance to generate key metrics characterizing the activity space for a more in-depth exploration of spatial familiarity.

```python
action_space = GPStoActionspace()

aggregation_method = 'user_id'  # Change this to 'user_id' or 'user_id_day'
act_spc, act_modified = action_space.compute_action_space(act, aggregation_method=aggregation_method)
mymap = action_space.plot_ellipses(act_spc, aggregation_method=aggregation_method)
action_space.covariance_matrix(action_space=act_spc)
action_space.plot_action_space(act, act_spc, user="CH16871", how="vignette", save=False)
action_space.plot_action_space(act, act_spc, user="CH16871", how="folium", save=False)
```
