import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiLineString
import glob
import json
import os
from datetime import datetime
from shapely.ops import linemerge
from geopy.distance import geodesic

# Define constants
BUFFER_SIZE = 100  # Buffer size in meters for route adherence

# Define station lists for Mumbai routes
ROUTES = {
    'mumbai_thane': [
        'CST', 'Masjid', 'Sandhurst Road', 'Byculla', 'Chinchpokli', 'Currey Road',
        'Parel', 'Dadar', 'Matunga', 'Sion', 'Kurla', 'Vidyavihar', 'Ghatkopar',
        'Vikhroli', 'Kanjur Marg', 'Bhandup', 'Nahur', 'Mulund', 'Thane'
    ],
    'mumbai_bhayandar': [
        'Churchgate', 'Marine Lines', 'Charni Road', 'Grant Road', 'Mumbai Central',
        'Lower Parel', 'Elphinstone Road', 'Dadar', 'Matunga Road', 'Mahim', 'Bandra',
        'Khar Road', 'Santacruz', 'Veli Parle', 'Andheri', 'Jogeshwari', 'Ram Mandir', 'Goregaon',
        'Malad', 'Kandivali', 'Borivali', 'Dahisar', 'Mira Road', 'Bhayandhar'
    ],
    'mumbai_panvel': [
        'CST', 'Masjid', 'Sandhurst Road', 'Dockyard Road', 'Reay Road', 'Cotton Green',
        'Sewri', 'Vadala Nagar', 'GTB Nagar', 'Chunabhatti', 'Kurla', 'Tilak Nagar',
        'Chembur', 'Govandi', 'Mankhurd', 'Vashi'
    ]
}

def are_stations_in_same_route(station1, station2):
    """Check if two stations are on the same route.

    Args:
        station1, station2 (str): Names of the stations.

    Returns:
        bool: True if both stations are on the same route, False otherwise.
    """
    for route in ROUTES.values():
        if station1 in route and station2 in route:
            return True
    return False

def get_route_for_station_pair(station1, station2):
    """Get the route name for a pair of stations.

    Args:
        station1, station2 (str): Names of the stations.

    Returns:
        str or None: Route name if stations are on the same route, None otherwise.
    """
    for route_name, stations in ROUTES.items():
        if station1 in stations and station2 in stations:
            return route_name
    return None

def load_railway_routes():
    """Load railway routes from GeoJSON files.

    Returns:
        dict: Mapping of route names to LineString or MultiLineString geometries.
    """
    railway_routes = {}
    route_files = {
        'mumbai_thane': 'train_routes/MumbaiThane2.geojson',
        'mumbai_bhayandar': 'train_routes/MumbaiBhayandar.geojson',
        'mumbai_panvel': 'train_routes/MumbaiPanvel2.geojson'
    }

    for route_name, file_path in route_files.items():
        try:
            with open(file_path, 'r') as f:
                geojson_data = json.load(f)
            for feature in geojson_data['features']:
                if feature['geometry']['type'] == 'MultiLineString':
                    lines = [LineString(segment) for segment in feature['geometry']['coordinates']]
                    railway_routes[route_name] = linemerge(lines)
                    break
                elif feature['geometry']['type'] == 'LineString':
                    railway_routes[route_name] = LineString(feature['geometry']['coordinates'])
                    break
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return railway_routes

def calculate_distance_to_railway(point, railway_line):
    """Calculate signed distance from a point to a railway line in meters.

    Args:
        point (tuple): (latitude, longitude) of the point.
        railway_line (LineString or MultiLineString): Railway geometry.

    Returns:
        float or None: Signed distance in meters, or None if invalid.
    """
    if railway_line is None:
        return None

    point_geom = Point(point[1], point[0])  # Convert to (lon, lat) for Shapely
    if isinstance(railway_line, MultiLineString):
        min_distance = float('inf')
        nearest_line = None
        for line in railway_line.geoms:
            dist = point_geom.distance(line)
            if dist < min_distance:
                min_distance = dist
                nearest_line = line
        railway_line = nearest_line

    nearest_point = railway_line.interpolate(railway_line.project(point_geom))
    distance_meters = geodesic(
        (point[0], point[1]),
        (nearest_point.y, nearest_point.x)
    ).meters

    coords = list(railway_line.coords)
    nearest_segment_idx = 0
    min_distance = float('inf')

    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        dist = point_geom.distance(segment)
        if dist < min_distance:
            min_distance = dist
            nearest_segment_idx = i

    p1 = coords[nearest_segment_idx]
    p2 = coords[nearest_segment_idx + 1]
    segment_vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    point_vector = np.array([point_geom.x - nearest_point.x, point_geom.y - nearest_point.y])
    cross_product = np.cross(segment_vector, point_vector)
    sign = np.sign(cross_product)

    return sign * distance_meters

def is_point_in_station(lat, lon, station_polygons):
    """Check if a point is within a station's polygon.

    Args:
        lat, lon (float): Latitude and longitude of the point.
        station_polygons (dict): Mapping of station names to Polygon geometries.

    Returns:
        str or None: Station name if point is within a station polygon, None otherwise.
    """
    point = Point(lon, lat)
    for station, polygon in station_polygons.items():
        if polygon.contains(point):
            return station
    return None

def process_trajectory_file(file_path, station_polygons, railway_routes):
    """Process a trajectory file to check route adherence within 100m buffer.

    Args:
        file_path (str): Path to the trajectory CSV file.
        station_polygons (dict): Mapping of station names to Polygon geometries.
        railway_routes (dict): Mapping of route names to railway geometries.

    Returns:
        dict: Statistics for the processed file.
    """
    # Extract date from filename
    file_basename = os.path.basename(file_path)
    file_date = file_basename.split('_')[-1].split('.')[0]

    # Load trajectory data
    trajectories = pd.read_csv(file_path)
    print(f"\nProcessing {file_path}: {len(trajectories)} trajectory points")

    # Standardize timestamp format
    if 'timestamp' in trajectories.columns:
        if trajectories['timestamp'].iloc[0].find('T') >= 0:
            trajectories['timestamp'] = trajectories['timestamp'].apply(
                lambda x: datetime.fromisoformat(x).strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, str) else x
            )

    transitions = []
    journey_validity = {}
    valid_device_ids = set()
    count = 0
    total_processed = 0

    # Process trajectories
    for (device_id, date), device_data in trajectories.groupby(['device_aid', 'date']):
        device_data = device_data.sort_values('timestamp')
        device_data['station'] = None
        count += 1
        total_processed += len(device_data)

        if count % 100 == 0:
            print(f"Processed {count} device-date pairs, {total_processed} total points")

        # Label points with their station
        for i, row in device_data.iterrows():
            station = is_point_in_station(row['latitude'], row['longitude'], station_polygons)
            device_data.at[i, 'station'] = station

        # Identify station sequences
        station_sequences = []
        current_station = None
        station_points = []

        for i, row in device_data.iterrows():
            if row['station'] != current_station:
                if current_station is not None and station_points:
                    station_sequences.append({
                        'station': current_station,
                        'points': station_points.copy()
                    })
                current_station = row['station']
                station_points = []
            if row['station'] is not None:
                station_points.append({
                    'lat': row['latitude'],
                    'lng': row['longitude'],
                    'timestamp': row['timestamp']
                })

        if current_station is not None and station_points:
            station_sequences.append({
                'station': current_station,
                'points': station_points.copy()
            })

        # Process transitions
        has_valid_journey = False
        for i in range(len(station_sequences) - 1):
            from_seq = station_sequences[i]
            to_seq = station_sequences[i + 1]

            if from_seq['station'] is None or to_seq['station'] is None:
                continue
            if from_seq['station'] == to_seq['station']:
                continue
            if not are_stations_in_same_route(from_seq['station'], to_seq['station']):
                continue

            from_timestamps = [datetime.fromisoformat(p['timestamp'].replace(' ', 'T')) if 'T' not in p['timestamp'] else datetime.fromisoformat(p['timestamp']) for p in from_seq['points']]
            to_timestamps = [datetime.fromisoformat(p['timestamp'].replace(' ', 'T')) if 'T' not in p['timestamp'] else datetime.fromisoformat(p['timestamp']) for p in to_seq['points']]

            if not from_timestamps or not to_timestamps:
                continue

            mean_departure_time = from_timestamps[0] + (from_timestamps[-1] - from_timestamps[0]) / 2
            mean_arrival_time = to_timestamps[0] + (to_timestamps[-1] - to_timestamps[0]) / 2

            from_lats = [p['lat'] for p in from_seq['points']]
            from_lngs = [p['lng'] for p in from_seq['points']]
            to_lats = [p['lat'] for p in to_seq['points']]
            to_lngs = [p['lng'] for p in to_seq['points']]

            mean_from_lat = sum(from_lats) / len(from_lats)
            mean_from_lng = sum(from_lngs) / len(from_lngs)
            mean_to_lat = sum(to_lats) / len(to_lats)
            mean_to_lng = sum(to_lngs) / len(to_lngs)

            journey_key = (device_id, date, from_seq['station'], to_seq['station'])
            journey_validity[journey_key] = True

            route_name = get_route_for_station_pair(from_seq['station'], to_seq['station'])
            railway_line = railway_routes.get(route_name)

            last_from_point = from_seq['points'][-1]
            first_to_point = to_seq['points'][0]
            last_from_time = datetime.fromisoformat(last_from_point['timestamp'].replace(' ', 'T')) if 'T' not in last_from_point['timestamp'] else datetime.fromisoformat(last_from_point['timestamp'])
            first_to_time = datetime.fromisoformat(first_to_point['timestamp'].replace(' ', 'T')) if 'T' not in first_to_point['timestamp'] else datetime.fromisoformat(first_to_point['timestamp'])

            points_between = []
            for j, row in device_data.iterrows():
                row_time_str = row['timestamp']
                row_time = datetime.fromisoformat(row_time_str.replace(' ', 'T')) if 'T' not in row_time_str else datetime.fromisoformat(row_time_str)
                if row_time > last_from_time and row_time < first_to_time and row['station'] is None:
                    points_between.append((row['latitude'], row['longitude']))

            transition_info = {
                'device_id': device_id,
                'date': date,
                'from_station': from_seq['station'],
                'to_station': to_seq['station'],
                'exit_time': mean_departure_time.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_lat': mean_from_lat,
                'exit_lng': mean_from_lng,
                'entry_time': mean_arrival_time.strftime('%Y-%m-%d %H:%M:%S'),
                'entry_lat': mean_to_lat,
                'entry_lng': mean_to_lng,
                'is_valid': True,
                'file_date': file_date,
                'journey_time_seconds': (mean_arrival_time - mean_departure_time).total_seconds()
            }

            if not points_between:
                transition_info['is_valid'] = True
            else:
                all_points_within_buffer = True
                for point in points_between:
                    signed_distance = calculate_distance_to_railway(point, railway_line)
                    abs_distance = abs(signed_distance) if signed_distance is not None else float('inf')
                    if abs_distance > BUFFER_SIZE:
                        all_points_within_buffer = False
                        break
                transition_info['is_valid'] = all_points_within_buffer

            journey_validity[journey_key] = transition_info['is_valid']
            if transition_info['is_valid']:
                has_valid_journey = True
            transitions.append(transition_info)

        if has_valid_journey:
            valid_device_ids.add(device_id)

    # Save valid transitions
    output_dir = f"final_4months/output_{file_date}"
    os.makedirs(output_dir, exist_ok=True)
    transitions_df = pd.DataFrame(transitions)
    if not transitions_df.empty:
        valid_transitions_df = transitions_df[transitions_df['is_valid'] == True]
        valid_transitions_df.to_csv(f'{output_dir}/transitions_valid_{file_date}.csv', index=False)
        print(f"Saved {len(valid_transitions_df)} valid transitions to {output_dir}/transitions_valid_{file_date}.csv")

    # Save filtered trajectories
    filtered_trajectories = trajectories[trajectories['device_aid'].isin(valid_device_ids)]
    filtered_trajectories.to_csv(f'{output_dir}/filtered_railway_trajectories_{file_date}.csv', index=False)
    print(f"Saved {len(filtered_trajectories)} trajectory points from {len(valid_device_ids)} valid devices to {output_dir}/filtered_railway_trajectories_{file_date}.csv")

    return {
        'file_date': file_date,
        'total_device_date_pairs': count,
        'total_valid_devices': len(valid_device_ids),
        'total_valid_journeys': sum(journey_validity.values()),
        'valid_journey_percentage': (sum(journey_validity.values()) / len(journey_validity) * 100) if journey_validity else 0
    }

def main():
    """Orchestrate route adherence processing for trajectory files."""
    # Load railway routes and station polygons
    railway_routes = load_railway_routes()
    print("Loaded railway routes:")
    for route_name, route_geom in railway_routes.items():
        print(f"  {route_name}: {type(route_geom)}")

    buffer_files = glob.glob('station_buffers/*.geojson')
    station_polygons = {}
    for buffer_file in buffer_files:
        with open(buffer_file) as f:
            geojson_data = json.load(f)
            for feature in geojson_data['features']:
                station_name = feature['properties']['station']
                coordinates = feature['geometry']['coordinates'][0]
                polygon = Polygon(coordinates)
                station_polygons[station_name] = polygon

    # Process trajectory files
    trajectory_files = glob.glob('filtered_data_4months_nonoise/filtered_data_*.csv')
    if not trajectory_files:
        print("No trajectory files found. Please check the path.")
        return

    print(f"Found {len(trajectory_files)} trajectory files to process")
    all_stats = []
    for file_path in trajectory_files:
        file_stats = process_trajectory_file(file_path, station_polygons, railway_routes)
        all_stats.append(file_stats)

    # Save processing statistics
    output_dir = "final_4months/combined_output"
    os.makedirs(output_dir, exist_ok=True)
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(f'{output_dir}/route_adherence_stats.csv', index=False)
    print(f"Saved route adherence statistics to {output_dir}/route_adherence_stats.csv")

    print("\nRoute Adherence Summary:")
    print(f"Total files processed: {len(trajectory_files)}")
    print(f"Average valid journey percentage: {stats_df['valid_journey_percentage'].mean():.2f}%")
    print(f"Total valid journeys: {stats_df['total_valid_journeys'].sum()}")

if __name__ == "__main__":
    main()