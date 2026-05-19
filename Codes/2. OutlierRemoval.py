import os
import pandas as pd
import numpy as np

# Define constants
SPEED_THRESHOLD = 80  # Speed threshold in km/h
DISTANCE_THRESHOLD = 9  # Distance threshold in km

def load_and_merge_data(folder_path):
    """Load and merge CSV files from a directory.

    Args:
        folder_path (str): Path to the directory containing CSV files.

    Returns:
        DataFrame: Merged DataFrame of all CSV files.
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not all_files:
        print(f"No CSV files found in {folder_path}")
        return pd.DataFrame()
    df_list = [pd.read_csv(file) for file in all_files]
    return pd.concat(df_list, ignore_index=True)

def detect_outliers_fixed_threshold(df, column, threshold):
    """Detect outliers based on a fixed threshold.

    Args:
        df (DataFrame): Input DataFrame.
        column (str): Column to check for outliers.
        threshold (float): Threshold value for outliers.

    Returns:
        DataFrame: Subset of rows exceeding the threshold.
    """
    print(f"Threshold for {column}: {threshold}")
    return df[df[column] > threshold]

def calculate_noise_statistics(df, speed_outliers, distance_outliers):
    """Calculate noise statistics based on speed and distance outliers.

    Args:
        df (DataFrame): Input DataFrame.
        speed_outliers (DataFrame): Rows with speed outliers.
        distance_outliers (DataFrame): Rows with distance outliers.

    Returns:
        DataFrame: Noise statistics per device and date.
    """
    noise_indices = speed_outliers.index.union(distance_outliers.index)
    noise_df = df.loc[noise_indices]
    noise_df['date'] = pd.to_datetime(noise_df['timestamp']).dt.date
    noise_stats = noise_df.groupby(['device_aid', 'date']).size().reset_index(name='noise_count')
    print("Noise statistics per device ID per day:")
    print(noise_stats)

    noise_stats.to_csv('noise_statistics.csv', index=False)
    noise_summary = noise_stats.groupby('noise_count').size().reset_index(name='num_devices')
    noise_summary = noise_summary.sort_values(by='noise_count')

    all_device_ids = df[['device_aid', 'timestamp']].copy()
    all_device_ids['date'] = pd.to_datetime(all_device_ids['timestamp']).dt.date
    all_device_ids = all_device_ids.groupby(['device_aid', 'date']).size().reset_index(name='count')
    total_device_ids = all_device_ids.shape[0]
    print("Total unique device IDs:", total_device_ids)

    zero_noise_count = total_device_ids - noise_stats['device_aid'].nunique()
    one_noise_count = noise_stats[noise_stats['noise_count'] == 1]['device_aid'].nunique()
    two_noise_count = noise_stats[noise_stats['noise_count'] == 2]['device_aid'].nunique()
    three_or_more_noise_count = noise_stats[noise_stats['noise_count'] >= 3]['device_aid'].nunique()

    final_summary = pd.DataFrame({
        'noise_count': ['0', '1', '2', '3+'],
        'num_devices': [zero_noise_count, one_noise_count, two_noise_count, three_or_more_noise_count]
    })

    final_summary.to_csv('final_noise_summary.csv', index=False)
    print("Final noise summary:")
    print(final_summary)

    return noise_stats

def filter_and_save_data(df, noise_stats):
    """Filter out noisy data and save by date.

    Args:
        df (DataFrame): Input DataFrame.
        noise_stats (DataFrame): Noise statistics per device and date.
    """
    print("Starting filter_and_save_data function")
    noise_stats_filtered = noise_stats[noise_stats['noise_count'] >= 1]
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    noise_tuples = set(noise_stats_filtered[['device_aid', 'date']].itertuples(index=False, name=None))
    mask = df.apply(lambda row: (row['device_aid'], row['date']) not in noise_tuples, axis=1)
    filtered_df = df[mask]
    print("Filtered the dataframe")

    grouped = filtered_df.groupby('date')
    output_dir = 'filtered_data_4months_nonoise'
    os.makedirs(output_dir, exist_ok=True)

    for date, group in grouped:
        group.to_csv(f'{output_dir}/filtered_data_{date}.csv', index=False)
        print(f"Filtered data for {date} saved to {output_dir}/filtered_data_{date}.csv")
    print("Completed filter_and_save_data function")

def main():
    """Orchestrate outlier removal and data filtering."""
    folder_path = "../3_months_data"
    df = load_and_merge_data(folder_path)
    if df.empty:
        print("No data to process. Exiting.")
        return

    non_zero_df = df[(df['speed_kmh'] >= 0) & (df['distance_km'] >= 0)]
    speed_outliers = detect_outliers_fixed_threshold(non_zero_df, 'speed_kmh', SPEED_THRESHOLD)
    distance_outliers = detect_outliers_fixed_threshold(non_zero_df, 'distance_km', DISTANCE_THRESHOLD)
    noise_stats = calculate_noise_statistics(df, speed_outliers, distance_outliers)
    filter_and_save_data(df, noise_stats)

if __name__ == "__main__":
    main()