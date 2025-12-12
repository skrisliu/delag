# DELAG
Daily Land Surface Temperature Reconstruction in Landsat Cross-Track Areas Using Deep Ensemble Learning With Uncertainty Quantification

## Code and data coming soon

This is a GitHub repo at [github.com/skrisliu/delag](https://github.com/skrisliu/delag)


## Preprocessing

Step 1: Unzip all Landsat data in a year

Step 1b optional: Reproject data to the same UTM zone (for London only)

Step 2: Clip data based on UTM xy

Step 3: Reorder data as YYYYMMDD, with LC08 or LC09 as sensor indicator

Step 4: Make datacube: lsts, clearmasks, meanbands, era5lst

## Tasks

Task #1: Clear-Sky Situations With Real-World Cloud Patterns

Task #2: Under Heavily-Cloudy Situations

Task #3: Indirect Validation via Estimating Near-Surface Air Temperature


# Project File Structure

This repository contains city-specific geospatial and meteorological analysis scripts:

- `nyc` → New York City
- `ldn` → London 
- `hkg` → Hong Kong 


All scripts follow a consistent numeric workflow (01 → 22).

## Project Structure

<details>
<summary>View directory contents</summary>
<br/>

```text
.
├── nyc_01_ReadData.py
├── nyc_01b_Reproject.py
├── nyc_02_Clip.py
├── nyc_03_Order.py
├── nyc_04_DataCube.py
├── nyc_05_Train1.py
├── nyc_07_GetMean95.py
├── nyc_10_CombineSmall.py
├── nyc_12_GP.py
├── nyc_19_AirTempTest.py
├── nyc_21_90cloud_gp.py
└── nyc_22_clear_gp.py