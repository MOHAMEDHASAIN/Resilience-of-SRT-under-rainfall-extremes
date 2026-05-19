import os
import glob
import pandas as pd
import logging
import numpy as np
import geopandas as gpd
from datetime import datetime
from shapely.geometry import Point

# Configure logging for detailed monitoring and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_daily_data(day_folder, output_base_dir):
    """Process raw location data from CSV files into a formatted CSV with essential columns.

    Args:
        day_folder (str): Path to the folder containing daily location data.
        output_base_dir (str): Base directory for saving processed output.

    Returns:
        str or False: Path to the processed CSV file or False if no data is processed.
    """
    # Extract day and month names from folder paths
    day_name = os.path.basename(day_folder)
    month_name = os.path.basename(os.path.dirname(day_folder))
    logging.info(f"Processing data for month: {month_name}, day: {day_name}")

    # Create output directory for the day
    day_output_dir = os.path.join(output_base_dir, month_name, day_name)
    os.makedirs(day_output_dir, exist_ok=True)

    # Find all location CSV part files
    part_files = glob.glob(os.path.join(day_folder, "*", "locations-*-part*.csv"))
    if not part_files:
        logging.warning(f"No location part files found in {day_folder}")
        return False
    logging.info(f"Found {len(part_files)} location part files")

    processed_file = os.path.join(day_output_dir, f"{day_name}_processed.csv")
    all_data = []

    # Process each CSV file
    for file in part_files:
        try:
            # Read CSV and split raw data into columns
            df = pd.read_csv(file, header=None, names=["raw_data"])
            split_columns = df['raw_data'].str.split(r'\s+', expand=True)

            # Limit to maximum expected columns
            max_columns = 16
            if split_columns.shape[1] > max_columns:
                split_columns = split_columns.iloc[:, :max_columns]

            # Define expected column names
            column_names = [
                "timestamp", "device_aid", "device_aid_type", "latitude", "longitude",
                "horizontal_accuracy", "altitude", "altitude_accuracy", "location_method",
                "ip", "user_agent", "OS", "OS_version", "manufacturer", "model", "carrier"
            ]
            split_columns.columns = column_names[:split_columns.shape[1]]
            all_data.append(split_columns)
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")

    if not all_data:
        logging.warning(f"No data could be processed for {day_folder}")
        return False

    # Combine and clean data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df["timestamp"] = pd.to_numeric(combined_df["timestamp"], errors="coerce")
    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], unit='s')
    combined_df["latitude"] = pd.to_numeric(combined_df["latitude"], errors="coerce")
    combined_df["longitude"] = pd.to_numeric(combined_df["longitude"], errors="coerce")
    combined_df = combined_df.dropna(subset=["timestamp", "device_aid", "latitude", "longitude"])
    combined_df = combined_df[["timestamp", "device_aid", "latitude", "longitude"]]

    # Save processed data
    combined_df.to_csv(processed_file, index=False)
    logging.info(f"Saved processed data to {processed_file}")
    return processed_file

def run_trajectory_analysis(processed_file, output_dir, route_geojson_mapping):
    """Analyze processed data to identify trajectories intersecting at least 2 stations.

    Args:
        processed_file (str): Path to the processed CSV file.
        output_dir (str): Directory to save trajectory output.
        route_geojson_mapping (dict): Mapping of route names to GeoJSON buffer files.

    Returns:
        str or None: Path to trajectory CSV or None if no valid trajectories.
    """
    # Load route buffers
    routes_gdf = load_all_route_buffers(route_geojson_mapping)

    # Load processed data
    device_df = pd.read_csv(processed_file)
    device_df['timestamp'] = pd.to_datetime(device_df['timestamp'])

    # Find trajectories with at least 2 stations and 4 points
    trajectories_df = find_device_trajectories_optimized(device_df, routes_gdf)

    if not trajectories_df.empty:
        trajectory_output = os.path.join(output_dir, "trajectories.csv")
        trajectories_df.to_csv(trajectory_output, index=False)
        logging.info(f"Saved {len(trajectories_df)} trajectory records to {trajectory_output}")
        return trajectory_output
    else:
        logging.info(f"No valid trajectories found for {processed_file}")
        return None

def load_all_route_buffers(route_geojson_mapping):
    """Load route buffer GeoJSON files into a single GeoDataFrame.

    Args:
        route_geojson_mapping (dict): Mapping of route names to GeoJSON file paths.

    Returns:
        GeoDataFrame: Combined GeoDataFrame of route buffers.
    """
    logging.info("Loading all route buffers into a single GeoDataFrame")
    route_dfs = []
    for route_name, path in route_geojson_mapping.items():
        gdf = gpd.read_file(path)
        gdf['route'] = route_name
        route_dfs.append(gdf[['route', 'station', 'geometry']])
    combined_gdf = gpd.GeoDataFrame(pd.concat(route_dfs, ignore_index=True), crs='EPSG:4326')
    combined_gdf['route'] = combined_gdf['route'].astype('category')
    combined_gdf['station'] = combined_gdf['station'].astype('category')
    logging.info(f"Loaded {len(combined_gdf)} buffers from {len(route_geojson_mapping)} routes")
    combined_gdf.to_file('combined_buffers.geojson', driver='GeoJSON')
    return combined_gdf

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between coordinate sets in kilometers.

    Args:
        lat1, lon1, lat2, lon2 (array-like): Latitude and longitude coordinates.

    Returns:
        array: Distances in kilometers.
    """
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def calculate_trajectory_metrics(df):
    """Calculate distance, time difference, and speed for trajectories.

    Args:
        df (DataFrame): DataFrame with timestamp, latitude, and longitude.

    Returns:
        DataFrame: DataFrame with added distance, time_diff, and speed columns.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    prev = df.shift(1)
    df['distance_km'] = haversine_vectorized(prev['latitude'], prev['longitude'], df['latitude'], df['longitude'])
    df['time_diff_sec'] = (df['timestamp'] - prev['timestamp']).dt.total_seconds()
    df['speed_kmh'] = (df['distance_km'] / df['time_diff_sec'].replace(0, np.nan)) * 3600
    df[['distance_km', 'time_diff_sec', 'speed_kmh']] = df[['distance_km', 'time_diff_sec', 'speed_kmh']].fillna(0)
    return df

def find_device_trajectories_optimized(device_df, routes_gdf):
    """Identify trajectories with at least 2 stations and 4 points.

    Args:
        device_df (DataFrame): Processed device location data.
        routes_gdf (GeoDataFrame): Route buffers GeoDataFrame.

    Returns:
        DataFrame: Combined trajectory data for qualifying devices.
    """
    logging.info("Creating device points GeoDataFrame")
    device_gdf = gpd.GeoDataFrame(
        device_df,
        geometry=gpd.points_from_xy(device_df.longitude, device_df.latitude),
        crs=routes_gdf.crs
    )

    logging.info("Performing spatial join with route buffers")
    joined = gpd.sjoin(device_gdf, routes_gdf, how='inner', predicate='within')

    logging.info("Identifying qualifying device-route pairs")
    qualifiers = (
        joined.groupby(['device_aid', 'route'])
        .agg(stations=('station', 'nunique'))
        .query('stations >= 2')
        .reset_index()
    )

    if qualifiers.empty or qualifiers['device_aid'].nunique() < 4:
        logging.info("Less than 4 unique device_aid found or empty qualifiers")
        return pd.DataFrame()

    logging.info("Processing trajectories for qualifying devices")
    device_trajs = []
    grouped = device_df.groupby('device_aid')

    for (dev_id, route), _ in qualifiers.groupby(['device_aid', 'route']):
        try:
            traj_df = grouped.get_group(dev_id)
            traj_df = calculate_trajectory_metrics(traj_df)
            if len(traj_df) < 4:
                continue
            stations = joined[(joined.device_aid == dev_id) & (joined.route == route)]['station'].unique().tolist()
            traj_df['route'] = route
            traj_df['stations'] = [stations] * len(traj_df)
            device_trajs.append(traj_df)
        except KeyError:
            continue

    logging.info(f"Found {len(device_trajs)} valid trajectories")
    if device_trajs:
        return pd.concat(device_trajs, ignore_index=True)
    else:
        return pd.DataFrame()

def generate_summary_report(all_trajectory_files, output_dir):
    """Generate a summary report for trajectories across multiple days.

    Args:
        all_trajectory_files (list): List of trajectory CSV file paths.
        output_dir (str): Directory to save the summary report.
    """
    if not all_trajectory_files:
        logging.warning("No trajectory files to generate report")
        return

    logging.info("Generating summary report")
    all_dfs = []

    for traj_file in all_trajectory_files:
        day_name = os.path.basename(os.path.dirname(traj_file))
        month_name = os.path.basename(os.path.dirname(os.path.dirname(traj_file)))
        df = pd.read_csv(traj_file)
        df['day'] = f"{month_name}_{day_name}"
        all_dfs.append(df)

    if not all_dfs:
        logging.warning("No valid trajectory data to analyze")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    route_summary = combined_df.groupby(['day', 'route']).agg(
        device_count=('device_aid', 'nunique'),
        avg_speed=('speed_kmh', 'mean'),
        total_distance=('distance_km', 'sum')
    ).reset_index()

    summary_file = os.path.join(output_dir, "multi_day_summary.csv")
    route_summary.to_csv(summary_file, index=False)
    logging.info(f"Summary report saved to {summary_file}")

def main():
    """Orchestrate raw data processing, trajectory analysis, and summary generation."""
    # Define base directories
    base_data_dir = "C:/Users/vivek/Desktop/Traffic_Flow_Data"
    output_base_dir = "initial_filtered_data_4months"
    os.makedirs(output_base_dir, exist_ok=True)

    # Define route GeoJSON mappings
    route_geojson_mapping = {
        "Mumbai Panvel": "station_buffers/MumbaiPanvel_buffers.geojson",
        "Mumbai Thane": "station_buffers/MumbaiThane_buffers.geojson",
        "Mumbai Bhayander": "station_buffers/MumbaiBhayander_buffers.geojson"
    }

    # Find month and day folders
    month_folders = [f.path for f in os.scandir(base_data_dir) if f.is_dir() and os.path.basename(f.path) in ["06", "07", "08"]]
    day_folders = []
    for month_folder in month_folders:
        day_folders.extend([f.path for f in os.scandir(month_folder) if f.is_dir()])

    logging.info(f"Found {len(day_folders)} day folders across {len(month_folders)} months")
    successful_trajectory_files = []

    # Process each day folder
    for day_folder in day_folders:
        try:
            processed_file = process_daily_data(day_folder, output_base_dir)
            if processed_file:
                day_output_dir = os.path.dirname(processed_file)
                trajectory_file = run_trajectory_analysis(processed_file, day_output_dir, route_geojson_mapping)
                if trajectory_file:
                    successful_trajectory_files.append(trajectory_file)
        except Exception as e:
            logging.error(f"Error processing day folder {day_folder}: {e}")

    # Generate summary report
    generate_summary_report(successful_trajectory_files, output_base_dir)
    logging.info("All processing complete")

if __name__ == "__main__":
    main()