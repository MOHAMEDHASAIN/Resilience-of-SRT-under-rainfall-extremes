# Resilience of Megacity Rail Transit to Sub-Hourly Rainfall Extremes in Emerging Economies

## 0. Contact

### Main Contact Information

**Name:** Prof. Subimal Ghosh  
**Institution:** Indian Institute of Technology Bombay  
**Email:** subimal@iitb.ac.in  

### Alternate Contact Information

**Name:** Dr. Mohamed Hasain N  
**Institution:** Indian Institute of Technology Bombay  
**Email:** mohamedhasaincivil@gmail.com

**Name:** Dr. Aniket Navalkar  
**Institution:** Lead Researcher (Climate Risk, Adaptation & Resilience), Avlokan Analytics  
**Email:** aunavalkar@gmail.com

---

## 1. Overview

This repository contains the official implementation and analysis framework for the paper:

**“Resilience of Megacity Rail Transit to Sub-Hourly Rainfall Extremes in Emerging Economies”**

The framework investigates the resilience of Mumbai’s Suburban Rail Transit (SRT) system under high-intensity, short-duration rainfall extremes using GPS trajectory data, Automatic Weather Station (AWS) rainfall observations, trajectory interpolation, and resilience-based performance metrics.

The repository provides a complete workflow for:

- GPS trajectory preprocessing
- Outlier and noise removal
- Route adherence filtering
- Origin–Destination (OD) matrix generation
- Rainfall-event and non-rainfall journey extraction
- Station-pair delay interpolation
- Resilience metric estimation

---

## 2. Abstract

Suburban Rail Transit (SRT) systems are increasingly vulnerable to high-intensity, short-duration rainfall, a signature of climate change that triggers rapid urban waterlogging. In megacities like Mumbai, where 7.6 million daily commuters face SRT disruptions due to recurrent monsoon extremes, the resilience of SRT networks to sub-hourly precipitation remains poorly understood.

Using GPS-based commuter trajectories, we quantified segment-level delays and resilience metrics across varying rainfall intensities and antecedent conditions. Using the CATBOOST machine-learning algorithm, we demonstrate that a 10 mm/15-minute extreme induces an excess delay of 21–50 seconds/km.

We find that delay, recovery time, and resilience are significantly governed by segment location and rainfall characteristics, rather than event timing. Specifically, central Mumbai junctions emerge as critical delay hotspots with the lowest resilience.

Our framework provides a scalable diagnostic for adaptive scheduling and infrastructure reinforcement, offering a roadmap to weather-ready transit systems against intensifying urban precipitation extremes in emerging economies.

---

## 3. Authors

- N Mohamed Hasain
- Aniket Navalkar
- Vivek Gangwani
- Avijit Maji
- Raghu Murtugudde
- Subimal Ghosh

---

# 4. Step-by-Step Analysis Workflow

---

## Step 1 - GPS Data Processing and Railway Trajectory Filtering

Run:

```bash
python "1. DataProcessingAndMultiStationFilter.py"
```

This script processes raw mobile trajectory data and identifies railway-related trajectories across Mumbai suburban rail routes.

### Key Functionalities

- Processes raw GPS location CSV files
- Extracts:
  - Timestamp
  - Latitude
  - Longitude
  - Device identifiers
- Uses GeoJSON railway buffers for spatial filtering
- Identifies trajectories intersecting multiple railway stations
- Computes:
  - Distance
  - Speed
  - Trajectory metrics
- Filters trajectories containing:
  - At least 2 stations
  - At least 4 valid GPS points

### Routes Processed

- Mumbai–Thane
- Mumbai–Panvel
- Mumbai–Bhayandar

### Output

- Processed trajectory CSV files
- Railway-filtered trajectories
- Spatially validated device movements

---

## Step 2 - Outlier and Noise Removal

Run:

```bash
python "2. OutlierRemoval.py"
```

This script removes noisy and unrealistic trajectory points using speed and distance thresholds.

### Key Functionalities

- Loads and merges trajectory CSV files
- Detects:
  - Speed outliers above 80 km/h
  - Distance outliers above 9 km
- Computes noise statistics per:
  - Device
  - Date
- Summarizes:
  - Devices with 0 noisy points
  - Devices with 1 noisy point
  - Devices with 2 noisy points
  - Devices with 3+ noisy points
- Removes device-date pairs containing noisy observations

### Output

- Noise statistics CSV files
- Filtered trajectory datasets
- Cleaned GPS trajectories stored in:

```text
filtered_data_4months_nonoise
```

---

## Step 3 - Route Adherence and Valid Transition Detection

Run:

```bash
python "3. RouteAdherence.py"
```

This script validates whether trajectories adhere to Mumbai suburban railway routes within a 100 m spatial buffer.

### Key Functionalities

- Loads railway routes from GeoJSON files:
  - MumbaiThane2.geojson
  - MumbaiBhayandar.geojson
  - MumbaiPanvel2.geojson
- Loads station polygons from:
  - `station_buffers`
- Identifies station entry and exit points
- Detects valid transitions between stations on the same railway corridor
- Uses Haversine distance calculations to ensure intermediate trajectory points remain within 100 m of railway alignments
- Filters invalid off-route movements

### Route Coverage

- Mumbai–Thane
- Mumbai–Bhayandar
- Mumbai–Panvel

### Output

- Valid station transitions
- Route-adherent trajectories
- Transition statistics
- Filtered railway movement datasets stored in:

```text
route_adherence_4months/output_{date}
```

---

## Step 4 - Origin–Destination (OD) Matrix Generation

Run:

```bash
python "4. OD_Matrix_Generation.py"
```

This script generates OD matrices and journey-time statistics for Mumbai suburban railway routes.

### Key Functionalities

- Generates station-wise OD matrices
- Produces:
  - Route-specific matrices
  - Combined matrices
- Computes:
  - Mean journey time
  - Median journey time
  - Standard deviation
  - Journey counts
- Produces OD heatmaps and travel-pattern visualizations

### Output

- Daily OD matrices
- Combined OD matrices
- Journey-time statistics
- OD heatmap visualizations

---

## Step 5A - Heavy Rainfall Journey Extraction

Run:

```bash
python "5a. HeavyRainJourneys.py"
```

This module extracts and analyzes journeys affected by heavy rainfall events.

### Key Functionalities

- Loads AWS rainfall observations
- Detects high rainfall events:
  - Threshold: >10 mm in 15 minutes
- Maps AWS stations to nearby railway stations
- Identifies rainfall-affected transitions within ±30-minute windows
- Extracts valid railway trajectories
- Computes:
  - Journey delay
  - Travel speed
  - Distance metrics
  - Transition statistics

### Study Period

- June 2023
- July 2023
- August 2023
- September 2023

### Output

- Rainfall journey metrics
- Transition datasets
- Trajectory datasets
- Combined rainfall-event statistics

---

## Step 5B - Non-Rainfall Journey Extraction

Run:

```bash
python "5b. NonRainJourneys.py"
```

This script extracts journeys occurring under non-rainfall conditions for baseline comparison.

### Key Functionalities

- Identifies rainfall-affected transitions
- Removes journeys affected by rainfall events
- Filters:
  - Direct rainfall impacts
  - Intermediate station rainfall impacts
- Extracts non-rainfall railway trajectories
- Computes journey metrics for baseline conditions

### Filtering Criteria

- ±30-minute rainfall-event windows
- UTC and IST-based rainfall matching
- Intermediate station rainfall screening

### Output

- Non-rainfall journey metrics
- Baseline transition datasets
- Exclusion logs
- Cleaned non-event trajectories

---

## Step 6 - Station Pair Delay Interpolation

Run:

```bash
python "6. StationPairBreakdown.py"
```

This module decomposes journeys into consecutive station pairs and interpolates delays using trajectory data.

### Key Functionalities

- Splits journeys into consecutive station pairs
- Preserves travel direction
- Uses trajectory interpolation for timestamp estimation
- Matches GPS trajectories to station coordinates
- Computes:
  - Pair-wise delays
  - Expected travel times
  - Pair start and end times

### Output

- Station-pair delay datasets
- Interpolated timings
- Journey decomposition files

---

## Step 7 - Dataset Preparation and Resilience Metrics

Run:

```bash
python "7. DatasetPreparation.py"
```

This module computes resilience and performance metrics under rainfall-induced disruptions.

### Key Functionalities

- Identifies critical resilience points:
  - t0 (initial state)
  - td (degradation point)
  - tr (maximum disruption)
  - tns (recovered state)

- Computes normalized delays

- Calculates resilience metrics:
  - RAPL_DP
  - RAPG_RP
  - RA
  - TAPL
  - GR

- Integrates rainfall observations at 15-minute intervals
- Generates final resilience datasets

### Analysis Features

- IST timestamps
- ±30-minute event windows
- Segment-wise resilience analysis

### Output

- Final resilience datasets
- Recovery profiles
- Delay-recovery metrics
- Segment-wise resilience indicators

---

## 5. Data Sources

The analysis integrates multiple data sources:

- GPS trajectory data
- AWS rainfall observations
- Railway station metadata
- GeoJSON route buffers
- Station distance datasets
- Journey transition datasets

---

## 6. Study Area

The study focuses on the Mumbai Suburban Rail Network, covering major corridors:

- Mumbai–Thane
- Mumbai–Panvel
- Mumbai–Bhayandar

---

## 7. Requirements

Recommended Python Version:

```bash
Python >= 3.10
```

### Main Python Libraries

```bash
pandas
numpy
geopandas
matplotlib
scipy
scikit-learn
catboost
shapely
geopy
networkx
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 8. Citation

If you find this repository useful, please cite our paper:

```text
Hasain, N. M., Navalkar, A., Gangwani, V., Maji, A.,
Murtugudde, R., & Ghosh, S.

"Resilience of Megacity Rail Transit to Sub-Hourly Rainfall Extremes in Emerging Economies".

Nature Cities (under review).
```

---

## 9. License

This repository is intended for academic and research purposes only. Please cite the associated paper when using the code, methodology, or derived datasets.