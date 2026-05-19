import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

# Define constants
RAINFALL_THRESHOLD = 10  # Rainfall threshold in mm for high rainfall events
MAX_JOURNEY_TIME = 100  # Maximum journey time in minutes between the station as the upper limit
WINDOW_MINUTES = 30  # Time window for trajectory extraction and transition filtering
TIME_ZONE_OFFSET = timedelta(hours=5, minutes=30)  # IST offset for AWS data

def load_nearest_aws():
    """Load AWS station data with associated railway stations.

    Returns:
        DataFrame: AWS stations with railway station mappings.
    """
    try:
        aws_df = pd.read_csv("formatted_stations.csv")
        print("Loaded nearest AWS stations data")
        return aws_df
    except Exception as e:
        print(f"Error loading formatted_stations.csv: {e}")
        return pd.DataFrame()

def load_all_aws_data(data_dir="merged_rainfall_data"):
    """Load rainfall data from AWS stations.

    Args:
        data_dir (str): Directory containing AWS rainfall CSV files.

    Returns:
        dict: Mapping of station names to rainfall DataFrames.
    """
    aws_data = {}
    for file in glob.glob(os.path.join(data_dir, "*.csv")):
        station_name = os.path.basename(file).replace('.csv', '')
        try:
            df = pd.read_csv(file)
            try:
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="mixed")
            except:
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="%m/%d/%Y %I:%M %p")
            df = df[['Date', 'Time', 'datetime', 'Rain']]
            df['datetime'] = df['datetime'] - TIME_ZONE_OFFSET
            df.set_index('datetime', inplace=True)
            aws_data[station_name] = df
            print(f"Loaded AWS data for {station_name}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return aws_data

def load_stations():
    """Load railway station data with expected travel times.

    Returns:
        DataFrame: Railway stations with line, distance, and time information.
    """
    try:
        stations_df = pd.read_csv("stations.csv", header=None)
        stations_df.columns = ['station_name', 'code', 'line', 'distance', 'time']
        print("Loaded stations data")
        return stations_df
    except Exception as e:
        print(f"Error loading stations.csv: {e}")
        return pd.DataFrame()

def load_transitions(date_str):
    """Load valid transition data for a specific date.

    Args:
        date_str (str): Date string in YYYY-MM-DD format.

    Returns:
        DataFrame: Valid transitions with journey times.
    """
    transition_file = f"final_4months/output_{date_str}/transitions_valid_{date_str}.csv"
    try:
        transitions_df = pd.read_csv(transition_file)

        # STANDARDIZE DEVICE COLUMN
        if 'device_aid' in transitions_df.columns:
            transitions_df.rename(columns={'device_aid': 'device_id'}, inplace=True)
        transitions_df['exit_time'] = pd.to_datetime(transitions_df['exit_time'])
        transitions_df['entry_time'] = pd.to_datetime(transitions_df['entry_time'])
        transitions_df = transitions_df[transitions_df['is_valid'] == True]
        transitions_df['journey_time_minutes'] = (transitions_df['entry_time'] - transitions_df['exit_time']).dt.total_seconds() / 60
        transitions_df = transitions_df[transitions_df['journey_time_minutes'] <= MAX_JOURNEY_TIME]
        print(f"Loaded {len(transitions_df)} valid transitions for {date_str}")
        return transitions_df
    except Exception as e:
        print(f"Error loading transitions for {date_str}: {e}")
        return pd.DataFrame()

def load_trajectories(date_str):
    """Load filtered trajectory data for a specific date.

    Args:
        date_str (str): Date string in YYYY-MM-DD format.

    Returns:
        DataFrame: Trajectory data with timestamps.
    """
    trajectory_file = f"final_4months/output_{date_str}/filtered_railway_trajectories_{date_str}.csv"
    try:
        trajectories_df = pd.read_csv(trajectory_file)

        # STANDARDIZE DEVICE COLUMN
        if 'device_aid' in trajectories_df.columns:
            trajectories_df.rename(columns={'device_aid': 'device_id'}, inplace=True)

        trajectories_df['timestamp'] = pd.to_datetime(
            trajectories_df['timestamp']
        )
        print(f"Loaded {len(trajectories_df)} trajectory points for {date_str}")
        return trajectories_df
    except Exception as e:
        print(f"Error loading trajectories for {date_str}: {e}")
        return pd.DataFrame()

def identify_high_rainfall_events(aws_data, threshold=RAINFALL_THRESHOLD):
    """Identify high rainfall events exceeding the threshold.

    Args:
        aws_data (dict): Mapping of station names to rainfall DataFrames.
        threshold (float): Rainfall threshold in mm.

    Returns:
        DataFrame: High rainfall events with station and time information.
    """
    high_rainfall_events = []
    for station, data in aws_data.items():
        high_rain = data[data['Rain'] >= threshold].copy()
        if not high_rain.empty:
            high_rain['station'] = station
            high_rainfall_events.append(high_rain)
    return pd.concat(high_rainfall_events) if high_rainfall_events else pd.DataFrame()

def find_nearest_railway_stations(aws_df, high_rainfall_events):
    """Map AWS stations to nearby railway stations.

    Args:
        aws_df (DataFrame): AWS station data with railway station mappings.
        high_rainfall_events (DataFrame): High rainfall events.

    Returns:
        DataFrame: Events with associated railway stations.
    """
    events_with_stations = []
    for _, event in high_rainfall_events.iterrows():
        station = event['station']
        aws_station = aws_df[aws_df['Place'].str.lower().str.strip() == station.lower().strip()]
        if not aws_station.empty:
            railway_stations_str = aws_station.iloc[0]['Railway_Station']
            railway_stations = railway_stations_str.split(',') if ',' in railway_stations_str else railway_stations_str.split(';') if ';' in railway_stations_str else [railway_stations_str]
            for railway in railway_stations:
                event_copy = event.copy()
                event_copy['railway_station'] = railway.strip()
                events_with_stations.append(event_copy)
    return pd.DataFrame(events_with_stations) if events_with_stations else pd.DataFrame()

def find_transitions_during_rainfall(events_with_stations, transitions_df):
    """Find transitions occurring during high rainfall periods.

    Args:
        events_with_stations (DataFrame): High rainfall events with railway stations.
        transitions_df (DataFrame): Valid transitions.

    Returns:
        DataFrame: Transitions during rainfall events.
    """
    rainfall_transitions = []
    for _, event in events_with_stations.iterrows():
        event_time = event.name
        event_start = event_time - timedelta(minutes=WINDOW_MINUTES)
        event_end = event_time + timedelta(minutes=WINDOW_MINUTES)
        relevant_transitions = transitions_df[
            ((transitions_df['from_station'] == event['railway_station']) &
             (transitions_df['exit_time'] >= event_start) & (transitions_df['exit_time'] <= event_end)) |
            ((transitions_df['to_station'] == event['railway_station']) &
             (transitions_df['entry_time'] >= event_start) & (transitions_df['entry_time'] <= event_end))
        ].copy()
        if not relevant_transitions.empty:
            relevant_transitions['rainfall_station'] = event['station']
            relevant_transitions['rainfall_time'] = event_time
            relevant_transitions['rainfall_amount'] = event['Rain']
            rainfall_transitions.append(relevant_transitions)
    return pd.concat(rainfall_transitions) if rainfall_transitions else pd.DataFrame()

def has_intermediate_points(device_id, from_station, to_station, trajectories_df, exit_time, entry_time, device_column):
    """Check if there are intermediate points between stations.

    Args:
        device_id (str): Device identifier.
        from_station, to_station (str): Station names.
        trajectories_df (DataFrame): Trajectory data.
        exit_time, entry_time (datetime): Transition times.
        device_column (str): Column name for device ID.

    Returns:
        bool: True if intermediate points exist, False otherwise.
    """
    trajectory_between = trajectories_df[
        (trajectories_df['device_id'] == device_id) &
        (trajectories_df['timestamp'] > exit_time) &
        (trajectories_df['timestamp'] < entry_time)
    ]
    return not trajectory_between.empty

def calculate_expected_time(from_station, to_station, stations_df):
    """Calculate expected journey time between stations.

    Args:
        from_station, to_station (str): Station names.
        stations_df (DataFrame): Railway station data.

    Returns:
        float or None: Expected journey time in minutes, or None if unavailable.
    """
    try:
        from_indices = stations_df[(stations_df['station_name'] == from_station) | 
                                  (stations_df['code'] == from_station)].index
        to_indices = stations_df[(stations_df['station_name'] == to_station) | 
                                (stations_df['code'] == to_station)].index
        expected_time = None
        for from_index in from_indices:
            for to_index in to_indices:
                from_line = stations_df.iloc[from_index]['line']
                to_line = stations_df.iloc[to_index]['line']
                if from_line == to_line:
                    time_series = []
                    station_range = stations_df.iloc[from_index + 1: to_index + 1] if from_index < to_index else stations_df.iloc[to_index + 1: from_index + 1]
                    station_range = station_range[station_range['line'] == from_line]
                    for time_str in station_range['time']:
                        if isinstance(time_str, str) and 'min' in time_str:
                            try:
                                minutes = int(time_str.split()[0])
                                time_series.append(minutes)
                            except (ValueError, IndexError):
                                pass
                    current_expected_time = sum(time_series) if time_series else None
                    if current_expected_time is not None and (expected_time is None or current_expected_time < expected_time):
                        expected_time = current_expected_time
        return expected_time
    except (IndexError, KeyError):
        return None

def extract_device_trajectories(rainfall_transitions, trajectories_df, stations_df, window_minutes=WINDOW_MINUTES):
    """Extract trajectories within a time window around valid transitions.

    Args:
        rainfall_transitions (DataFrame): Transitions during rainfall.
        trajectories_df (DataFrame): Trajectory data.
        stations_df (DataFrame): Railway station data.
        window_minutes (int): Time window size in minutes.

    Returns:
        tuple: (trajectories DataFrame, valid transitions DataFrame).
    """
    fast_stations = [
        "Bhayandhar", "Borivali", "Andheri", "Bandra", "Dadar", "Mumbai Central",
        "Churchgate", "Vashi", "Kurla", "Vadala", "Nagar", "CST", "Thane",
        "Ghatkopar", "Kurla", "Dadar", "Byculla", "CST"
    ]
    device_first_transitions = {}
    valid_transitions = []
    all_device_trajectories = []

    for _, transition in rainfall_transitions.iterrows():
        device_id = transition['device_id']
        from_station = transition['from_station']
        to_station = transition['to_station']
        exit_time = transition['exit_time']
        entry_time = transition['entry_time']
        journey_time = (entry_time - exit_time).total_seconds() / 60
        expected_time = calculate_expected_time(from_station, to_station, stations_df)

        if expected_time is None:
            continue

        if journey_time > (4 * expected_time) and not has_intermediate_points(
            device_id, from_station, to_station, trajectories_df, exit_time, entry_time, 'device_id'
        ):
            continue

        valid_transitions.append(transition)
        rainfall_time = transition['rainfall_time']
        reference_time = exit_time if abs((exit_time - rainfall_time).total_seconds()) < abs((entry_time - rainfall_time).total_seconds()) else entry_time
        start_time = reference_time - timedelta(minutes=window_minutes) - timedelta(minutes=15)
        end_time = reference_time + timedelta(minutes=window_minutes)

        device_trajectory = trajectories_df[
            (trajectories_df['device_id'] == device_id) &
            (trajectories_df['timestamp'] >= start_time) &
            (trajectories_df['timestamp'] <= end_time)
        ].copy()

        if not device_trajectory.empty:
            device_trajectory['transition_id'] = transition.name
            device_trajectory['from_station'] = transition['from_station']
            device_trajectory['to_station'] = transition['to_station']
            device_trajectory['exit_time'] = transition['exit_time']
            device_trajectory['entry_time'] = transition['entry_time']
            device_trajectory['rainfall_station'] = transition['rainfall_station']
            device_trajectory['rainfall_time'] = transition['rainfall_time']
            device_trajectory['rainfall_amount'] = transition['rainfall_amount']
            all_device_trajectories.append(device_trajectory)

    valid_transitions_df = pd.DataFrame(valid_transitions) if valid_transitions else pd.DataFrame()
    return pd.concat(all_device_trajectories) if all_device_trajectories else pd.DataFrame(), valid_transitions_df

def calculate_journey_metrics(valid_rainfall_transitions, device_trajectories, stations_df):
    """Calculate metrics for transitions during rainfall.

    Args:
        valid_rainfall_transitions (DataFrame): Valid transitions during rainfall.
        device_trajectories (DataFrame): Trajectory data within time windows.
        stations_df (DataFrame): Railway station data.

    Returns:
        DataFrame: Journey metrics for each transition.
    """
    metrics = []

    for _, transition in valid_rainfall_transitions.iterrows():
        device_id = transition['device_id']
        transition_id = transition.name
        from_station = transition['from_station']
        to_station = transition['to_station']
        exit_time = transition['exit_time']
        entry_time = transition['entry_time']
        journey_time = (entry_time - exit_time).total_seconds() / 60
        expected_time = calculate_expected_time(from_station, to_station, stations_df)
        delay = journey_time - expected_time if expected_time is not None else None

        window_start = exit_time - timedelta(minutes=WINDOW_MINUTES)
        window_end = exit_time + timedelta(minutes=WINDOW_MINUTES)
        trajectory_data = device_trajectories[
            (device_trajectories['transition_id'] == transition_id)
        ].sort_values('timestamp')

        avg_speed = max_speed = total_distance = transition_distance = None
        data_points = len(trajectory_data)

        if not trajectory_data.empty:
            if 'speed_kmh' in trajectory_data.columns:
                avg_speed = trajectory_data['speed_kmh'].mean()
                max_speed = trajectory_data['speed_kmh'].max()
            if 'distance_km' in trajectory_data.columns:
                total_distance = trajectory_data['distance_km'].sum()
                transition_trajectory = trajectory_data[
                    (trajectory_data['timestamp'] >= exit_time) &
                    (trajectory_data['timestamp'] <= entry_time)
                ]
                transition_distance = transition_trajectory['distance_km'].sum() if not transition_trajectory.empty else 0

        metrics.append({
            'device_id': device_id,
            'transition_id': transition_id,
            'from_station': from_station,
            'to_station': to_station,
            'exit_time': exit_time,
            'entry_time': entry_time,
            'journey_time_minutes': journey_time,
            'expected_time_minutes': expected_time,
            'delay_minutes': delay,
            'window_start': window_start,
            'window_end': window_end,
            'window_duration_minutes': 2 * WINDOW_MINUTES,
            'avg_speed_kmh': avg_speed,
            'max_speed_kmh': max_speed,
            'total_distance_km': total_distance,
            'transition_distance_km': transition_distance,
            'data_points_in_window': data_points
        })

    return pd.DataFrame(metrics)

def main():
    """Analyze the impact of high rainfall events on railway transitions."""
    output_dir = "rainfall_analysis_events_4months"
    os.makedirs(output_dir, exist_ok=True)

    aws_gdf = load_nearest_aws()
    if aws_gdf.empty:
        return
    aws_data = load_all_aws_data()
    if not aws_data:
        return
    stations_df = load_stations()
    if stations_df.empty:
        return

    all_journey_metrics = []
    all_window_trajectories = []
    start_date = datetime(2023, 6, 1)
    end_date = datetime(2023, 9, 30)
    specific_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days + 1)]

    for date_str in specific_dates:
        print(f"Processing {date_str}...")
        transitions_df = load_transitions(date_str)
        if transitions_df.empty:
            continue
        trajectories_df = load_trajectories(date_str)
        if trajectories_df.empty:
            continue

        high_rainfall_events = identify_high_rainfall_events(aws_data)
        if high_rainfall_events.empty:
            print(f"No high rainfall events on {date_str}")
            continue

        events_with_stations = find_nearest_railway_stations(aws_gdf, high_rainfall_events)
        if events_with_stations.empty:
            print(f"No railway stations found near high rainfall events on {date_str}")
            continue

        rainfall_transitions = find_transitions_during_rainfall(events_with_stations, transitions_df)
        if rainfall_transitions.empty:
            print(f"No transitions during high rainfall events on {date_str}")
            continue

        window_trajectories, valid_rainfall_transitions = extract_device_trajectories(
            rainfall_transitions, trajectories_df, stations_df
        )
        if window_trajectories.empty:
            print(f"No valid device trajectories found for {date_str}")
            continue

        journey_metrics = calculate_journey_metrics(valid_rainfall_transitions, window_trajectories, stations_df)
        journey_metrics.to_csv(os.path.join(output_dir, f"journey_metrics_{date_str}.csv"), index=False)
        window_trajectories.to_csv(os.path.join(output_dir, f"window_trajectories_{date_str}.csv"), index=False)
        valid_rainfall_transitions.to_csv(os.path.join(output_dir, f"valid_rainfall_transitions_{date_str}.csv"), index=False)

        all_journey_metrics.append(journey_metrics)
        all_window_trajectories.append(window_trajectories)
        print(f"Completed processing for {date_str}. Found {len(journey_metrics)} transition windows.")

    if all_journey_metrics:
        combined_journey_metrics = pd.concat(all_journey_metrics)
        combined_journey_metrics.to_csv(os.path.join(output_dir, "all_journey_metrics.csv"), index=False)
        print(f"Saved combined journey metrics for {len(combined_journey_metrics)} transition windows.")

    if all_window_trajectories:
        combined_window_trajectories = pd.concat(all_window_trajectories)
        combined_window_trajectories.to_csv(os.path.join(output_dir, "all_window_trajectories.csv"), index=False)
        print(f"Saved combined window trajectories with {len(combined_window_trajectories)} data points.")

    print("Rainfall impact analysis complete!")

if __name__ == "__main__":
    main()