import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import interp1d

# Define constants
DATA_DIR = "..."
OUTPUT_DIR = os.path.join(DATA_DIR, "...")
SKIPPED_LOG = os.path.join(OUTPUT_DIR, "...")
MIN_JOURNEYS = 5
THRESHOLD_PCT = 10
EPSILON = 1e-6

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_delay(delay):
    """Normalize delay using 1/(1+Delay) formula, handling negative values."""
    if pd.isna(delay):
        return delay
    adjusted_delay = max(-0.99, float(delay))
    normalized = 1 / (1 + adjusted_delay + 1e-10)
    return max(0.0, min(1.0, normalized))

def load_aws_rainfall_data(data_dir):
    """Load AWS rainfall data with IST timestamps."""
    aws_data = {}
    for file in glob.glob(os.path.join(data_dir, "merged_rainfall_data", "*.csv")):
        station_name = os.path.basename(file).replace('.csv', '')
        try:
            df = pd.read_csv(file)
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="mixed", errors="coerce")
            df = df[['datetime', 'Rain']].set_index('datetime')
            aws_data[station_name] = df
            print(f"Loaded rainfall data for {station_name}")
        except Exception as e:
            print(f"Error loading {station_name}: {e}")
    print(f"Loaded AWS rainfall data for {len(aws_data)} stations in IST")
    return aws_data

def load_station_mapping(data_dir):
    """Load station to AWS mapping."""
    mapping_file = os.path.join(data_dir, "formatted_stations.csv")
    if not os.path.exists(mapping_file):
        print(f"Warning: Station mapping file {mapping_file} not found.")
        return {}
    try:
        mapping_df = pd.read_csv(mapping_file)
        station_map = {}
        for _, row in mapping_df.iterrows():
            place = row['Place']
            railway_stations = str(row['Railway_Station']).split(",")
            for station in railway_stations:
                station = station.strip().title()
                if station:
                    station_map[station] = place
        print(f"Loaded {len(station_map)} railway stations mapped to AWS stations")
        return station_map
    except Exception as e:
        print(f"Error loading station mapping: {e}")
        return {}

def get_rainfall_at_time(station_name, timestamp, aws_data):
    """Get rainfall value at a specific time for a station."""
    if station_name not in aws_data or aws_data[station_name].empty:
        return 0
    try:
        if isinstance(timestamp, pd.Series):
            timestamp = timestamp.iloc[0]
        closest_idx = aws_data[station_name].index.get_indexer([timestamp], method='nearest')[0]
        if 0 <= closest_idx < len(aws_data[station_name]):
            time_diff = abs((aws_data[station_name].index[closest_idx] - timestamp).total_seconds()) / 60
            if time_diff <= 30:
                return aws_data[station_name].iloc[closest_idx]['Rain']
            print(f"Warning: Closest rainfall data point for {station_name} is {time_diff:.1f} minutes away")
    except Exception as e:
        print(f"Error getting rainfall data for {station_name}: {e}")
    return 0

def time_to_seconds(time_obj):
    """Convert time to seconds since midnight."""
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def seconds_to_datetime(seconds, date):
    """Convert seconds since midnight to datetime."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = int(seconds % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return datetime.combine(date, datetime.strptime(time_str, "%H:%M:%S").time())

def get_delay_at_time(timestamp, journey_times, journey_delays):
    """Interpolate delay at a specific time."""
    timestamp_seconds = time_to_seconds(timestamp.time())
    if len(journey_times) >= 2:
        try:
            journey_seconds = [time_to_seconds(t.time()) for t in journey_times]
            sorted_indices = np.argsort(journey_seconds)
            sorted_seconds = np.array(journey_seconds)[sorted_indices]
            sorted_delays = np.array(journey_delays)[sorted_indices]
            if len(np.unique(sorted_seconds)) >= 2:
                interp_func = interp1d(sorted_seconds, sorted_delays, kind='linear', bounds_error=False, fill_value="extrapolate")
                return float(interp_func(timestamp_seconds))
        except Exception as e:
            print(f"Error in delay interpolation: {e}")
    closest_journey = min(journey_times, key=lambda x: abs((x - timestamp).total_seconds()))
    closest_idx = journey_times.index(closest_journey)
    return journey_delays[closest_idx]

def time_diff_in_seconds(t2, t1, time_format="%d-%m-%Y %H:%M"):
    """Calculate time difference in seconds."""
    t2_dt = datetime.strptime(t2, time_format)
    t1_dt = datetime.strptime(t1, time_format)
    return (t2_dt - t1_dt).total_seconds()

def add_minutes_to_timestamp(timestamp, minutes, time_format="%d-%m-%Y %H:%M"):
    """Add minutes to a timestamp and return formatted string."""
    dt = datetime.strptime(timestamp, time_format)
    dt += timedelta(minutes=minutes)
    return dt.strftime(time_format)

def compute_metrics(row):
    """Compute new performance metrics using normalized delays."""
    t0_delay = row['t0_normalized']
    td_delay = row['td_normalized']
    tr_delay = row['tr_normalized']
    tns_delay = row['tns_normalized']

    tr_minus_td = time_diff_in_seconds(row['tr'], row['td'])
    tns_minus_tr = time_diff_in_seconds(row['tns'], row['tr'])
    tns_minus_td = time_diff_in_seconds(row['tns'], row['td'])
    t0_minus_tr = time_diff_in_seconds(row['t0'], row['tr'])

    rapl_dp = abs((td_delay - tr_delay) / (tr_minus_td if tr_minus_td != 0 else EPSILON))
    rapg_rp = abs((tns_delay - tr_delay) / (tns_minus_tr if tns_minus_tr != 0 else EPSILON))
    denominator_ra = t0_delay - tr_delay
    ra = abs((tns_delay - tr_delay) / (denominator_ra if denominator_ra != 0 else EPSILON))
    tapl = abs((t0_delay - td_delay) + (t0_delay - tns_delay)) / 2

    return pd.Series({
        'RAPL_DP': rapl_dp,
        'RAPG_RP': rapg_rp,
        'RA': ra,
        'TAPL': tapl
    })


def recompute_gr(row):
    """Compute GR metric."""
    R = row['tr_normalized']
    rapl_dp = row['RAPL_DP'] if row['RAPL_DP'] != 0 else EPSILON
    tapl = row['TAPL'] if row['TAPL'] != 0 else EPSILON
    return R * (row['RAPG_RP'] / rapl_dp) * (1 / tapl) * row['RA']

def identify_key_points(rainfall_journeys, event_start, event_end):
    """Identify key points (t0, td, tr, tns) without adjustments."""
    if len(rainfall_journeys) < MIN_JOURNEYS:
        return None, None, None, None, None, None, None, None, None, None, None, None

    rainfall_journeys = rainfall_journeys.sort_values("pair_start_time")
    baseline_delays = rainfall_journeys.iloc[0:min(2, len(rainfall_journeys))]["pair_delay_minutes"].mean()
    threshold = baseline_delays * (1 + THRESHOLD_PCT/100)

    t0 = rainfall_journeys["pair_start_time"].iloc[0]
    t0_delay = rainfall_journeys["pair_delay_minutes"].iloc[0]
    t0_normalized = rainfall_journeys["normalized_delay"].iloc[0]

    tr_idx = rainfall_journeys["pair_delay_minutes"].idxmax()
    tr = rainfall_journeys.loc[tr_idx, "pair_start_time"]
    tr_delay = rainfall_journeys.loc[tr_idx, "pair_delay_minutes"]
    tr_normalized = rainfall_journeys.loc[tr_idx, "normalized_delay"]

    td = None
    td_delay = None
    td_normalized = None
    for _, row in rainfall_journeys.iterrows():
        if row["pair_delay_minutes"] > threshold and row["pair_start_time"] > t0:
            td = row["pair_start_time"]
            td_delay = row["pair_delay_minutes"]
            td_normalized = row["normalized_delay"]
            break

    tns = None
    tns_delay = None
    tns_normalized = None
    post_peak = rainfall_journeys[rainfall_journeys["pair_start_time"] >= tr]
    for _, row in post_peak[::-1].iterrows():
        if row["pair_delay_minutes"] <= threshold and row["pair_start_time"] > tr:
            tns = row["pair_start_time"]
            tns_delay = row["pair_delay_minutes"]
            tns_normalized = row["normalized_delay"]
            break

    if any(x is None for x in [t0, td, tr, tns]):
        return None, None, None, None, None, None, None, None, None, None, None, None

    return t0, t0_delay, t0_normalized, td, td_delay, td_normalized, tr, tr_delay, tr_normalized, tns, tns_delay, tns_normalized

def calculate_area(times, delays):
    """Calculate area under delay curve."""
    if len(times) < 2:
        return 0
    times_seconds = [time_to_seconds(t.time()) for t in times]
    area = 0
    for i in range(len(times) - 1):
        width = times_seconds[i+1] - times_seconds[i]
        height_avg = (delays[i] + delays[i+1]) / 2
        area += width * height_avg
    return area

def main():
    """Process rainfall events to analyze delays and calculate enhanced metrics."""
    aws_rainfall_data = load_aws_rainfall_data(DATA_DIR)
    station_mapping = load_station_mapping(DATA_DIR)

    events_file = os.path.join(DATA_DIR, "heavy_rain_periods_10.csv")
    events_df = pd.read_csv(events_file)
    events_df["start_time"] = pd.to_datetime(events_df["start_time"])
    events_df["end_time"] = pd.to_datetime(events_df["end_time"])

    station_pair_delays_file = os.path.join(DATA_DIR, "broken_down_station_pairs_4months_10_interpolated.csv")
    station_pair_delays = pd.read_csv(station_pair_delays_file)
    station_pair_delays["pair_start_time"] = pd.to_datetime(station_pair_delays["pair_start_time"], format="%d-%m-%Y %H:%M:%S")
    station_pair_delays["pair_end_time"] = pd.to_datetime(station_pair_delays["pair_end_time"], format="%d-%m-%Y %H:%M:%S")
    station_pair_delays["normalized_delay"] = station_pair_delays["pair_delay_minutes"].apply(normalize_delay)

    dataset = []
    skipped_no_pairs = 0
    skipped_missing_points = 0

    for event_index, event in events_df.iterrows():
        railway_station = event["railway_station"].strip().title()
        event_start = event["start_time"]
        event_end = event["end_time"]
        event_date = event_start.date()
        aws_station = station_mapping.get(railway_station)
        rainfall_at_start = get_rainfall_at_time(aws_station, event_start, aws_rainfall_data) if aws_station else 0
        extended_start = event_start - timedelta(minutes=30)
        extended_end = event_end + timedelta(minutes=30)

        print(f"\nProcessing rainfall event {event_index} for {railway_station} on {event_date} from {event_start.strftime('%H:%M')} to {event_end.strftime('%H:%M')} IST")

        station_journeys = station_pair_delays[
            (station_pair_delays["from_station"].str.strip().str.title() == railway_station) |
            (station_pair_delays["to_station"].str.strip().str.title() == railway_station)
        ]

        station_pairs = station_journeys[
            (station_journeys["journey_type"] == "rainfall") &
            (station_journeys["pair_start_time"] >= extended_start) &
            (station_journeys["pair_start_time"] <= extended_end) &
            (station_journeys["pair_start_time"].dt.date == event_date)
        ][["from_station", "to_station"]].drop_duplicates().values.tolist()

        station_pairs = [(pair[0].strip().title(), pair[1].strip().title()) for pair in station_pairs]

        if not station_pairs:
            print(f"Skipped: No station pairs with rainfall data found for {railway_station}.")
            with open(SKIPPED_LOG, "a") as f:
                f.write(f"Event {event_index} ({railway_station}, {event_date}): No station pairs found.\n")
            skipped_no_pairs += 1
            continue

        print(f"Station pairs with rainfall data for {railway_station}: {station_pairs}")

        for from_station, to_station in station_pairs:
            station_pair = f"{from_station}–{to_station}"
            print(f"Processing pair {station_pair}")

            rainfall_journeys = station_journeys[
                (station_journeys["journey_type"] == "rainfall") &
                (station_journeys["pair_start_time"] >= extended_start) &
                (station_journeys["pair_start_time"] <= extended_end) &
                (station_journeys["pair_start_time"].dt.date == event_date) &
                (station_journeys["from_station"].str.strip().str.title() == from_station) &
                (station_journeys["to_station"].str.strip().str.title() == to_station)
            ].copy()

            if len(rainfall_journeys) < MIN_JOURNEYS:
                print(f"  Skipping {station_pair}: Only {len(rainfall_journeys)} rainfall journeys (minimum {MIN_JOURNEYS} required)")
                with open(SKIPPED_LOG, "a") as f:
                    f.write(f"Event {event_index} ({station_pair}, {event_date}): Only {len(rainfall_journeys)} rainfall journeys.\n")
                skipped_missing_points += 1
                continue

            t0, t0_delay, t0_normalized, td, td_delay, td_normalized, tr, tr_delay, tr_normalized, tns, tns_delay, tns_normalized = identify_key_points(rainfall_journeys, event_start, event_end)

            if any(x is None for x in [t0, td, tr, tns]):
                print(f"  Skipping {station_pair}: Missing key points (t0={t0 is not None}, td={td is not None}, tr={tr is not None}, tns={tns is not None})")
                with open(SKIPPED_LOG, "a") as f:
                    f.write(f"Event {event_index} ({station_pair}, {event_date}): Missing key points (t0={t0 is not None}, td={td is not None}, tr={tr is not None}, tns={tns is not None}).\n")
                skipped_missing_points += 1
                continue

            # Adjust timestamps and delays for strict ordering
            t0_str = t0.strftime("%d-%m-%Y %H:%M")
            td_str = td.strftime("%d-%m-%Y %H:%M")
            tr_str = tr.strftime("%d-%m-%Y %H:%M")
            tns_str = tns.strftime("%d-%m-%Y %H:%M")

            if t0_str == td_str:
                td_str = add_minutes_to_timestamp(td_str, 2)
                tr_str = add_minutes_to_timestamp(tr_str, 1)
                tns_str = add_minutes_to_timestamp(tns_str, 1)
                if t0_delay == td_delay:
                    td_delay += 1
                    td_normalized = normalize_delay(td_delay)

            if td_str == tr_str:
                tr_str = add_minutes_to_timestamp(tr_str, 2)
                tns_str = add_minutes_to_timestamp(tns_str, 1)
                if td_delay == tr_delay:
                    tns_delay -= 1
                    tns_normalized = normalize_delay(tns_delay)

            if tr_str == tns_str:
                tns_str = add_minutes_to_timestamp(tns_str, 2)
                if tr_delay == tns_delay:
                    tns_delay -= 1
                    tns_normalized = normalize_delay(tns_delay)

            # Convert back to datetime for calculations
            t0 = datetime.strptime(t0_str, "%d-%m-%Y %H:%M")
            td = datetime.strptime(td_str, "%d-%m-%Y %H:%M")
            tr = datetime.strptime(tr_str, "%d-%m-%Y %H:%M")
            tns = datetime.strptime(tns_str, "%d-%m-%Y %H:%M")

            rainfall_at_t0 = get_rainfall_at_time(aws_station, t0, aws_rainfall_data) if aws_station else 0
            rainfall_at_td = get_rainfall_at_time(aws_station, td, aws_rainfall_data) if aws_station else 0
            rainfall_at_tr = get_rainfall_at_time(aws_station, tr, aws_rainfall_data) if aws_station else 0
            rainfall_at_tns = get_rainfall_at_time(aws_station, tns, aws_rainfall_data) if aws_station else 0

            rainfall_values = []
            current_time = event_start
            while current_time <= event_end:
                rainfall_values.append(get_rainfall_at_time(aws_station, current_time, aws_rainfall_data) if aws_station else 0)
                current_time += timedelta(minutes=15)

            # Original metrics
            RAPDL = (tr_delay - td_delay) / ((tr - td).total_seconds() / 60) if (tr - td).total_seconds() > 0 else 0
            RAPDL_normalized = (td_normalized - tr_normalized) / ((tr - td).total_seconds() / 60) if (tr - td).total_seconds() > 0 else 0
            RAPG = (tns_delay - tr_delay) / ((tns - tr).total_seconds() / 60) if (tns - tr).total_seconds() > 0 else 0
            RAPG_normalized = (tns_normalized - tr_normalized) / ((tns - tr).total_seconds() / 60) if (tns - tr).total_seconds() > 0 else 0
            RA_old = (tns_delay - tr_delay) / (tr_delay - t0_delay) if (tr_delay - t0_delay) != 0 else 0
            RA_normalized_old = (tns_normalized - tr_normalized) / (t0_normalized - tr_normalized) if (t0_normalized - tr_normalized) != 0 else 0

            times_between = []
            delays_between = []
            current_time = td
            while current_time <= tns:
                times_between.append(current_time)
                delays_between.append(get_delay_at_time(current_time, rainfall_journeys["pair_start_time"].tolist(), rainfall_journeys["pair_delay_minutes"].values))
                current_time += timedelta(minutes=1)
            net_performance_loss = calculate_area(times_between, delays_between)

            dataset_row = {
                "Station_Pair": station_pair,
                "AWS_Station": aws_station,
                "Rainfall_Date": event_date.strftime("%Y-%m-%d"),
                "Timezone": "IST",
                "t0": t0_str,
                "t0_delay": t0_delay,
                "t0_normalized": t0_normalized,
                "td": td_str,
                "td_delay": td_delay,
                "td_normalized": td_normalized,
                "tr": tr_str,
                "tr_delay": tr_delay,
                "tr_normalized": tr_normalized,
                "tns": tns_str,
                "tns_delay": tns_delay,
                "tns_normalized": tns_normalized,
                "Rainfall_at_t0": rainfall_at_t0,
                "Rainfall_at_td": rainfall_at_td,
                "Rainfall_at_tr": rainfall_at_tr,
                "Rainfall_at_tns": rainfall_at_tns,
                "Rainfall_15min_intervals": ','.join(map(str, rainfall_values)),
                "RAPDL": RAPDL,
                "RAPDL_normalized": RAPDL_normalized,
                "RAPG": RAPG,
                "RAPG_normalized": RAPG_normalized,
                "RA_old": RA_old,
                "RA_normalized_old": RA_normalized_old,
                "Net_Performance_Loss": net_performance_loss,
                "Rainfall_Journeys": len(rainfall_journeys)
            }

            # Compute new metrics
            metrics = compute_metrics(pd.Series(dataset_row))
            dataset_row.update(metrics)

            dataset.append(dataset_row)

    dataset_df = pd.DataFrame(dataset)


    output_file = os.path.join(OUTPUT_DIR, "rainfall_delay_dataset_enhanced_patterns_4months_ist_normalized.csv")
    dataset_df.to_csv(output_file, index=False)
    print(f"\nDataset saved to {output_file}")

    print("\nSummary:")
    print(f"Processed {len(events_df)} rainfall events in IST")
    print(f"Events skipped (no station pairs): {skipped_no_pairs}")
    print(f"Events skipped (missing key points or insufficient journeys): {skipped_missing_points}")
    print(f"Dataset rows generated: {len(dataset_df)}")
    for metric in ["RAPDL", "RAPG", "RA_old", "Net_Performance_Loss", "RAPDL_normalized", "RAPG_normalized", "RA_normalized_old", "RAPL_DP", "RAPG_RP", "RA", "TAPL", "GR"]:
        if metric in dataset_df.columns:
            valid_values = dataset_df[metric].dropna()
            if len(valid_values) > 0:
                print(f"{metric} - Mean: {valid_values.mean():.2f}, Min: {valid_values.min():.2f}, Max: {valid_values.max():.2f}")
            else:
                print(f"{metric} - No valid values")

if __name__ == "__main__":
    main()