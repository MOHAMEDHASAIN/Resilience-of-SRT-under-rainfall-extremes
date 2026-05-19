import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import glob
import os

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

def generate_od_matrices(file_path, combined_od_matrices):
    """Generate OD matrices for a single trajectory file.

    Args:
        file_path (str): Path to the valid transitions CSV file.
        combined_od_matrices (dict): Dictionary of combined OD matrices per route.

    Returns:
        dict: File-specific OD matrices.
    """
    file_basename = os.path.basename(file_path)
    file_date = file_basename.split('_')[-1].split('.')[0]
    transitions_df = pd.read_csv(file_path)

    file_od_matrices = {}
    for route_name, stations in ROUTES.items():
        file_od_matrices[route_name] = pd.DataFrame(0, index=stations, columns=stations)

    for _, row in transitions_df.iterrows():
        from_station = row['from_station']
        to_station = row['to_station']
        for route_name, stations in ROUTES.items():
            if from_station in stations and to_station in stations:
                file_od_matrices[route_name].at[from_station, to_station] += 1
                combined_od_matrices[route_name].at[from_station, to_station] += 1
                break

    output_dir = f"final_4months/output_{file_date}"
    os.makedirs(output_dir, exist_ok=True)
    for route_name, od_matrix in file_od_matrices.items():
        od_matrix.to_csv(f'{output_dir}/od_matrix_{route_name}_{file_date}.csv')
        plt.figure(figsize=(12, 10))
        plt.imshow(od_matrix, cmap='Blues')
        plt.colorbar(label='Number of trips')
        plt.title(f'Origin-Destination Matrix: {route_name} ({file_date})')
        plt.xlabel('Destination')
        plt.ylabel('Origin')
        stations = od_matrix.index
        plt.xticks(range(len(stations)), stations, rotation=90)
        plt.yticks(range(len(stations)), stations)
        for i in range(len(stations)):
            for j in range(len(stations)):
                value = od_matrix.iloc[i, j]
                if value > 0:
                    plt.text(j, i, str(int(value)), ha='center', va='center',
                             color='white' if value > od_matrix.values.max()/2 else 'black')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/od_matrix_{route_name}_{file_date}.png')
        plt.close()

    journey_stats = transitions_df.groupby(['from_station', 'to_station'])['journey_time_seconds'].agg(['mean', 'median', 'std', 'count']).reset_index()
    journey_stats.to_csv(f'{output_dir}/journey_time_stats_{file_date}.csv', index=False)
    print(f"Saved OD matrices and journey stats for {file_date} to {output_dir}")

    return file_od_matrices

def main():
    """Orchestrate OD matrix generation and journey time statistics."""
    output_dir = "final_4months/combined_output"
    os.makedirs(output_dir, exist_ok=True)

    combined_od_matrices = {}
    for route_name, stations in ROUTES.items():
        combined_od_matrices[route_name] = pd.DataFrame(0, index=stations, columns=stations)

    transition_files = glob.glob('final_4months/output_*/transitions_valid_*.csv')
    if not transition_files:
        print("No valid transition files found. Please check the path.")
        return

    print(f"Found {len(transition_files)} transition files to process")
    for file_path in transition_files:
        generate_od_matrices(file_path, combined_od_matrices)

    for route_name, od_matrix in combined_od_matrices.items():
        od_matrix.to_csv(f'{output_dir}/combined_od_matrix_{route_name}.csv')
        plt.figure(figsize=(12, 10))
        plt.imshow(od_matrix, cmap='Blues')
        plt.colorbar(label='Number of trips')
        plt.title(f'Combined Origin-Destination Matrix: {route_name}')
        plt.xlabel('Destination')
        plt.ylabel('Origin')
        stations = od_matrix.index
        plt.xticks(range(len(stations)), stations, rotation=90)
        plt.yticks(range(len(stations)), stations)
        for i in range(len(stations)):
            for j in range(len(stations)):
                value = od_matrix.iloc[i, j]
                if value > 0:
                    plt.text(j, i, str(int(value)), ha='center', va='center',
                             color='white' if value > od_matrix.values.max()/2 else 'black')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/combined_od_matrix_{route_name}.png')
        plt.close()

    all_transitions = []
    for transition_file in transition_files:
        try:
            df = pd.read_csv(transition_file)
            all_transitions.append(df)
        except Exception as e:
            print(f"Error reading {transition_file}: {e}")

    if all_transitions:
        combined_transitions = pd.concat(all_transitions, ignore_index=True)
        combined_journey_stats = combined_transitions.groupby(['from_station', 'to_station'])['journey_time_seconds'].agg(['mean', 'median', 'std', 'count']).reset_index()
        combined_journey_stats.to_csv(f'{output_dir}/combined_journey_time_stats.csv', index=False)
        print(f"Saved combined journey time statistics to {output_dir}/combined_journey_time_stats.csv")

if __name__ == "__main__":
    main()