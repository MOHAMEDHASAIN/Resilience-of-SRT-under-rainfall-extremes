import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from geopy.distance import geodesic

# Define constants
DATA_DIR = ""


def load_journey_data():
    """Load and preprocess rainfall and non-rainfall journey data.

    Returns:
        tuple: (rainfall journeys DataFrame, non-rainfall journeys DataFrame)
    """
    rainfall_file = os.path.join(DATA_DIR, "rainfall_analysis_events_4months/all_journey_metrics.csv")
    non_rainfall_file = os.path.join(DATA_DIR, "non_rainfall_analysis_4months_10_updated/all_journey_metrics.csv")

    try:
        transition_journeys_rainfall = pd.read_csv(rainfall_file)
        transition_journeys_non_rainfall = pd.read_csv(non_rainfall_file)
        print(f"Loaded {len(transition_journeys_rainfall)} rainfall journeys and {len(transition_journeys_non_rainfall)} non-rainfall journeys")
    except Exception as e:
        print(f"Error loading journey data: {e}")
        return pd.DataFrame(), pd.DataFrame()

    for df in [transition_journeys_rainfall, transition_journeys_non_rainfall]:
        df.drop_duplicates(subset=["transition_id", "device_id", "exit_time", "entry_time"], keep="first", inplace=True)
        df["from_station"] = df["from_station"].str.strip().str.title()
        df["to_station"] = df["to_station"].str.strip().str.title()
        try:
            df["exit_time"] = pd.to_datetime(df["exit_time"], format="%Y-%m-%d %H:%M:%S")
            df["entry_time"] = pd.to_datetime(df["entry_time"], format="%Y-%m-%d %H:%M:%S")
            df["window_start"] = pd.to_datetime(df["window_start"])
            df["window_end"] = pd.to_datetime(df["window_end"])
        except Exception as e:
            print(f"Error parsing datetime columns: {e}")
            df["exit_time"] = pd.to_datetime(df["exit_time"], format="mixed")
            df["entry_time"] = pd.to_datetime(df["entry_time"], format="mixed")
            df["window_start"] = pd.to_datetime(df["window_start"], format="mixed")
            df["window_end"] = pd.to_datetime(df["window_end"], format="mixed")

    return transition_journeys_rainfall, transition_journeys_non_rainfall

def load_station_data():
    """Load and preprocess station data.

    Returns:
        tuple: (station times DataFrame, station distances DataFrame)
    """
    station_times_file = os.path.join(DATA_DIR, "Stations.csv")
    try:
        station_times = pd.read_csv(station_times_file)
        station_times["Station"] = station_times["Station"].str.strip().str.title()
        print(f"Loaded station times for {len(station_times)} stations")
    except Exception as e:
        print(f"Error loading Stations.csv: {e}")
        return pd.DataFrame(), pd.DataFrame()

    distance_files = [
        os.path.join(DATA_DIR, "station_distances_MumbaiBhayandar.csv"),
        os.path.join(DATA_DIR, "station_distances_MumbaiPanvel.csv"),
        os.path.join(DATA_DIR, "station_distances_MumbaiThane.csv")
    ]
    station_distances_list = []
    for file in distance_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                df["station1"] = df["station1"].str.strip().str.title()
                df["station2"] = df["station2"].str.strip().str.title()
                station_distances_list.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    station_distances = pd.concat(station_distances_list, ignore_index=True) if station_distances_list else pd.DataFrame()
    print(f"Loaded distances for {len(station_distances)} station pairs")
    return station_times, station_distances

def load_trajectories(date_str):
    """Load trajectory data for a specific date.

    Args:
        date_str (str): Date string in YYYY-MM-DD format.

    Returns:
        DataFrame: Trajectory data with timestamps and coordinates.
    """
    trajectory_file = os.path.join(DATA_DIR, f"final_4months/output_{date_str}/filtered_railway_trajectories_{date_str}.csv")
    try:
        trajectories_df = pd.read_csv(trajectory_file)
        trajectories_df['timestamp'] = pd.to_datetime(trajectories_df['timestamp'])
        print(f"Loaded {len(trajectories_df)} trajectory points for {date_str}")
        return trajectories_df
    except Exception as e:
        print(f"Error loading trajectories for {date_str}: {e}")
        return pd.DataFrame()

def print_debug_info(station_times, transition_journeys_rainfall, transition_journeys_non_rainfall):
    """Print debugging information about stations and lines."""
    print("\nUnique stations in Stations.csv:")
    print(sorted(station_times["Station"].unique()))
    print("\nUnique from_station in rainfall journeys:")
    print(sorted(transition_journeys_rainfall["from_station"].unique()))
    print("\nUnique to_station in rainfall journeys:")
    print(sorted(transition_journeys_rainfall["to_station"].unique()))
    print("\nUnique from_station in non-rainfall journeys:")
    print(sorted(transition_journeys_non_rainfall["from_station"].unique()))
    print("\nUnique to_station in non-rainfall journeys:")
    print(sorted(transition_journeys_non_rainfall["to_station"].unique()))
    print("\nLines in Stations.csv:")
    print(station_times[["Station", "line"]].drop_duplicates().to_string(index=False))

def get_station_pairs(from_station, to_station, station_times):
    """Get ordered station pairs for a journey, respecting travel direction.

    Args:
        from_station, to_station (str): Start and end stations.
        station_times (DataFrame): Station data with line information.

    Returns:
        list: Ordered list of station pairs.
    """
    try:
        from_lines = station_times[station_times["Station"] == from_station]["line"].unique()
        to_lines = station_times[station_times["Station"] == to_station]["line"].unique()
        common_lines = set(from_lines).intersection(to_lines)
        
        if not common_lines:
            raise ValueError(f"Stations {from_station} and {to_station} not on same line or not found.")
        
        for line in common_lines:
            line_stations = station_times[station_times["line"] == line][["Station"]].reset_index(drop=True)
            if from_station not in line_stations["Station"].values or to_station not in line_stations["Station"].values:
                continue
            from_idx = line_stations[line_stations["Station"] == from_station].index[0]
            to_idx = line_stations[line_stations["Station"] == to_station].index[0]
            
            if from_idx < to_idx:
                stations = line_stations.iloc[from_idx:to_idx + 1]["Station"].tolist()
            else:
                stations = line_stations.iloc[to_idx:from_idx + 1]["Station"].tolist()
                stations.reverse()
            
            pairs = [(stations[i], stations[i + 1]) for i in range(len(stations) - 1)]
            
            # if from_station == "Elphinstone Road" and to_station == "Mumbai Central":
            #     print(f"\nDEBUG - Journey: {from_station} -> {to_station}")
            #     print(f"Line: {line}")
            #     print(f"From index: {from_idx}, To index: {to_idx}")
            #     print(f"Stations sequence: {stations}")
            #     print(f"Station pairs: {pairs}")
            
            return pairs
        
        raise ValueError(f"Stations {from_station} and {to_station} not found in any common line.")
    except (IndexError, KeyError, ValueError) as e:
        with open(os.path.join(DATA_DIR, "skipped_journeys.log"), "a") as f:
            f.write(f"Skipped journey {from_station} to {to_station}: {str(e)}\n")
        return []

def get_expected_time(station1, station2, station_times):
    """Calculate expected time between two consecutive stations.

    Args:
        station1, station2 (str): Station names.
        station_times (DataFrame): Station data with time information.

    Returns:
        float: Expected time in minutes, or 0 if unavailable.
    """
    try:
        station1_lines = station_times[station_times["Station"] == station1]["line"].unique()
        station2_lines = station_times[station_times["Station"] == station2]["line"].unique()
        common_lines = set(station1_lines).intersection(station2_lines)
        
        if not common_lines:
            return 0
        
        for line in common_lines:
            line_stations = station_times[station_times["line"] == line].reset_index(drop=True)
            try:
                idx1 = line_stations[line_stations["Station"] == station1].index[0]
                idx2 = line_stations[line_stations["Station"] == station2].index[0]
                if idx2 == idx1 + 1:
                    time_str = line_stations.iloc[idx2]["Time taken From Previous of the Line"]
                    return float(str(time_str).replace("min", "").strip())
                elif idx1 == idx2 + 1:
                    time_str = line_stations.iloc[idx1]["Time taken From Previous of the Line"]
                    return float(str(time_str).replace("min", "").strip())
            except (IndexError, KeyError, ValueError):
                continue
        return 0
    except (IndexError, KeyError, ValueError):
        return 0

def interpolate_station_timings(device_id, from_station, to_station, exit_time, entry_time, station_times, trajectories_df):
    """Interpolate timestamps for station pairs using trajectory data.

    Args:
        device_id (str): Device identifier.
        from_station, to_station (str): Start and end stations.
        exit_time, entry_time (datetime): Journey start and end times.
        station_times (DataFrame): Station data with coordinates.
        trajectories_df (DataFrame): Trajectory data with coordinates and timestamps.

    Returns:
        list: List of dictionaries with station pair timings.
    """
    device_column = 'device_id' if 'device_id' in trajectories_df.columns else 'device_aid'
    station_pairs = get_station_pairs(from_station, to_station, station_times)
    if not station_pairs:
        return []

    trajectory = trajectories_df[
        (trajectories_df[device_column] == device_id) &
        (trajectories_df['timestamp'] >= exit_time) &
        (trajectories_df['timestamp'] <= entry_time)
    ].sort_values('timestamp')

    if trajectory.empty:
        return []

    pair_timings = []
    for station1, station2 in station_pairs:
        station1_data = station_times[station_times["Station"] == station1]
        station2_data = station_times[station_times["Station"] == station2]
        if station1_data.empty or station2_data.empty:
            continue

        station1_coords = (station1_data.iloc[0]["Latitude"], station1_data.iloc[0]["Longitude"])
        station2_coords = (station2_data.iloc[0]["Latitude"], station2_data.iloc[0]["Longitude"])

        trajectory['dist_to_station1'] = trajectory.apply(
            lambda row: geodesic((row['latitude'], row['longitude']), station1_coords).km, axis=1)
        trajectory['dist_to_station2'] = trajectory.apply(
            lambda row: geodesic((row['latitude'], row['longitude']), station2_coords).km, axis=1)

        closest_to_station1 = trajectory[trajectory['dist_to_station1'] <= 0.5]
        closest_to_station2 = trajectory[trajectory['dist_to_station2'] <= 0.5]

        pair_start_time = closest_to_station1['timestamp'].min() if not closest_to_station1.empty else exit_time
        pair_end_time = closest_to_station2['timestamp'].min() if not closest_to_station2.empty else entry_time

        if pair_start_time is pd.NaT or pair_end_time is pd.NaT or pair_start_time >= pair_end_time:
            continue

        expected_time = get_expected_time(station1, station2, station_times)
        if expected_time == 0:
            continue

        pair_duration = (pair_end_time - pair_start_time).total_seconds() / 60
        pair_delay = pair_duration - expected_time if pair_duration is not None else None

        pair_timings.append({
            "from_station": station1,
            "to_station": station2,
            "pair_start_time": pair_start_time,
            "pair_end_time": pair_end_time,
            "pair_expected_time_minutes": expected_time,
            "pair_delay_minutes": pair_delay
        })

    return pair_timings

def main():
    """Process journeys to calculate station pair delays using trajectory interpolation."""
    transition_journeys_rainfall, transition_journeys_non_rainfall = load_journey_data()
    if transition_journeys_rainfall.empty and transition_journeys_non_rainfall.empty:
        return
    station_times, station_distances = load_station_data()
    if station_times.empty:
        return

    print_debug_info(station_times, transition_journeys_rainfall, transition_journeys_non_rainfall)
    
    station_pair_delays = []
    original_journeys = []
    broken_down_journeys = []
    start_date = datetime(2023, 6, 1)
    end_date = datetime(2023, 9, 30)
    specific_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days + 1)]

    for journey_type, journeys in [
        ("rainfall", transition_journeys_rainfall),
        ("non_rainfall", transition_journeys_non_rainfall)
    ]:
        for _, journey in journeys.iterrows():
            from_station = journey["from_station"]
            to_station = journey["to_station"]
            actual_delay = journey["delay_minutes"]
            journey_time = journey["journey_time_minutes"]
            exit_time = journey["exit_time"]
            entry_time = journey["entry_time"]
            transition_id = journey["transition_id"]
            device_id = journey["device_id"]
            date_str = exit_time.strftime("%Y-%m-%d")

            total_expected_time = journey.get("expected_time_minutes", None)
            if total_expected_time is None or total_expected_time == 0:
                pairs = get_station_pairs(from_station, to_station, station_times)
                total_expected_time = sum(get_expected_time(p[0], p[1], station_times) for p in pairs)
                if total_expected_time == 0:
                    with open(os.path.join(DATA_DIR, "skipped_journeys.log"), "a") as f:
                        f.write(f"Skipped journey {from_station} to {to_station} on {date_str}: No expected time available.\n")
                    continue

            if journey_time > (4 * total_expected_time):
                trajectories_df = load_trajectories(date_str)
                if trajectories_df.empty:
                    continue
                device_column = 'device_id' if 'device_id' in trajectories_df.columns else 'device_aid'
                if not has_intermediate_points(device_id, from_station, to_station, trajectories_df, exit_time, entry_time, device_column):
                    with open(os.path.join(DATA_DIR, "skipped_journeys.log"), "a") as f:
                        f.write(f"Skipped journey {from_station} to {to_station} on {date_str}: Journey time > 4x expected and no intermediate points.\n")
                    continue

            trajectories_df = load_trajectories(date_str)
            if trajectories_df.empty:
                continue

            pairs = get_station_pairs(from_station, to_station, station_times)
            if not pairs:
                with open(os.path.join(DATA_DIR, "skipped_journeys.log"), "a") as f:
                    f.write(f"Skipped journey {from_station} to {to_station} on {date_str}: No valid station pairs.\n")
                continue

            pair_timings = interpolate_station_timings(device_id, from_station, to_station, exit_time, entry_time, station_times, trajectories_df)
            if not pair_timings:
                with open(os.path.join(DATA_DIR, "skipped_journeys.log"), "a") as f:
                    f.write(f"Skipped journey {from_station} to {to_station} on {date_str}: No valid interpolated timings.\n")
                continue

            original_journey_data = {
                "journey_type": journey_type,
                "transition_id": transition_id,
                "device_id": device_id,
                "from_station": from_station,
                "to_station": to_station,
                "pair_start_time": exit_time.strftime("%d-%m-%Y %H:%M:%S"),
                "pair_end_time": entry_time.strftime("%d-%m-%Y %H:%M:%S"),
                "pair_expected_time_minutes": total_expected_time,
                "pair_delay_minutes": actual_delay,
                "original_from_station": from_station,
                "original_to_station": to_station,
                "original_journey_start": exit_time.strftime("%d-%m-%Y %H:%M:%S"),
                "original_journey_end": entry_time.strftime("%d-%m-%Y %H:%M:%S"),
                "is_original_journey": True
            }
            station_pair_delays.append(original_journey_data)
            if len(pairs) > 1:
                original_journeys.append(original_journey_data)

            for i, timing in enumerate(pair_timings):
                broken_down_pair_data = {
                    "journey_type": journey_type,
                    "transition_id": transition_id,
                    "device_id": device_id,
                    "from_station": timing["from_station"],
                    "to_station": timing["to_station"],
                    "pair_start_time": timing["pair_start_time"].strftime("%d-%m-%Y %H:%M:%S"),
                    "pair_end_time": timing["pair_end_time"].strftime("%d-%m-%Y %H:%M:%S"),
                    "pair_expected_time_minutes": timing["pair_expected_time_minutes"],
                    "pair_delay_minutes": timing["pair_delay_minutes"],
                    "original_from_station": from_station,
                    "original_to_station": to_station,
                    "original_journey_start": exit_time.strftime("%d-%m-%Y %H:%M:%S"),
                    "original_journey_end": entry_time.strftime("%d-%m-%Y %H:%M:%S"),
                    "is_original_journey": False,
                    "pair_sequence": i + 1
                }
                station_pair_delays.append(broken_down_pair_data)
                broken_down_journeys.append(broken_down_pair_data)

    station_pair_delays_df = pd.DataFrame(station_pair_delays)
    broken_down_journeys_df = pd.DataFrame(broken_down_journeys)

    output_file = os.path.join(DATA_DIR, "station_pair_delays_4months_10_interpolated.csv")
    station_pair_delays_df.to_csv(output_file, index=False)
    print(f"\nSaved all station pair delays to {output_file}")

    broken_down_output_file = os.path.join(DATA_DIR, "broken_down_station_pairs_4months_10_interpolated.csv")
    broken_down_journeys_df.to_csv(broken_down_output_file, index=False)
    print(f"Saved broken-down station pairs only to {broken_down_output_file}")

    print("\nSummary:")
    if not station_pair_delays_df.empty:
        print(f"Total station pair delays calculated: {len(station_pair_delays_df)}")
        print(f"Original journeys: {len(station_pair_delays_df[station_pair_delays_df['is_original_journey'] == True])}")
        print(f"Broken-down station pairs: {len(broken_down_journeys_df)}")
        print(f"Rainfall journey pairs: {len(station_pair_delays_df[station_pair_delays_df['journey_type'] == 'rainfall'])}")
        print(f"Non-rainfall journey pairs: {len(station_pair_delays_df[station_pair_delays_df['journey_type'] == 'non_rainfall'])}")
        print("\nSample from original file (all data):")
        print(station_pair_delays_df.head(10))
        print("\nSample from broken-down journeys only:")
        print(broken_down_journeys_df.head(10))
        sample_transition = station_pair_delays_df[station_pair_delays_df['transition_id'] == 576]
        if not sample_transition.empty:
            print(f"\nFixed example for transition_id 576:")
            print(sample_transition[['from_station', 'to_station', 'is_original_journey', 'pair_sequence']].to_string(index=False))
    else:
        print("No station pair delays calculated. Check skipped_journeys.log for details.")

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
        (trajectories_df[device_column] == device_id) &
        (trajectories_df['timestamp'] > exit_time) &
        (trajectories_df['timestamp'] < entry_time)
    ]
    return not trajectory_between.empty

if __name__ == "__main__":
    main()