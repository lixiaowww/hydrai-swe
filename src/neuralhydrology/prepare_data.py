import xarray as xr
import pandas as pd
import numpy as np
import os
import rioxarray
import yaml

def prepare_data_for_neuralhydrology(processed_dir, output_dir):
    """
    Prepares the data for NeuralHydrology using ECC Manitoba snow data.

    This function loads ECC snow data and HYDAT streamflow data, merges them,
    and saves it as a CSV file in the format required by NeuralHydrology.
    
    Available data sources:
    - Historical ECCC snow data: 1979-1998 (20 years of rich historical patterns)
    - Recent ECCC data: 2020-2024 (modern observations)
    - HYDAT streamflow: 2020-2024 (synthetic database for testing)

    Args:
        processed_dir (str): The directory containing the processed files.
        output_dir (str): The directory to save the prepared data to.
    """
    print("Preparing data for NeuralHydrology using ECC Manitoba snow data...")
    print("📊 Available data: Historical (1979-1998) + Recent (2020-2024) for comprehensive analysis")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data ---
    eccc_snow_data_path = os.path.join(processed_dir, "eccc_manitoba_snow_processed.csv")
    hydat_data_path = os.path.join(processed_dir, "hydat_streamflow_processed.csv")

    try:
        # Load ECC snow data
        eccc_snow_data = pd.read_csv(eccc_snow_data_path)
        print(f"✅ Loaded ECC snow data: {len(eccc_snow_data)} records")
        
        # Load HYDAT streamflow data
        hydat_data = pd.read_csv(hydat_data_path, index_col='date', parse_dates=True)
        print(f"✅ Loaded HYDAT streamflow data: {len(hydat_data)} records")
        
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        print("Please run the main ETL script first to generate the data.")
        return

    # --- Process ECC Snow Data ---
    print("Processing ECC snow data...")
    
    # Convert date column
    eccc_snow_data['date'] = pd.to_datetime(eccc_snow_data['date'])
    
    # Group by date and calculate daily averages for snow data
    daily_snow_data = eccc_snow_data.groupby('date').agg({
        'Total Snow (cm)': 'mean',
        'Snow on Grnd (cm)': 'mean'
    }).reset_index()
    
    # Convert snow depth from cm to mm for consistency
    daily_snow_data['Total Snow (mm)'] = daily_snow_data['Total Snow (cm)'] * 10
    daily_snow_data['Snow on Grnd (mm)'] = daily_snow_data['Snow on Grnd (cm)'] * 10
    
    # Set date as index
    daily_snow_data.set_index('date', inplace=True)
    
    print(f"Daily snow data: {len(daily_snow_data)} records")
    print(f"Snow depth range: {daily_snow_data['Snow on Grnd (mm)'].min():.1f} - {daily_snow_data['Snow on Grnd (mm)'].max():.1f} mm")

    # --- Merge Snow and Streamflow Data ---
    print("Merging snow and streamflow data...")
    
    # Merge on date
    merged_data = daily_snow_data.join(hydat_data, how='inner')
    
    if merged_data.empty:
        print("⚠️ No overlapping dates between snow and streamflow data")
        print("Creating complete dataset using ECC snow data...")
        
        # 系统禁止使用合成数据
        print("❌ 系统禁止使用合成数据")
        print("💡 请提供真实的径流观测数据")
        raise ValueError("Missing streamflow data. Synthetic data generation is prohibited.")
    else:
        print(f"Merged data: {len(merged_data)} records")
    
    # --- Optional: merge ECCC GRIB2 weather (temperature, precipitation) ---
    print("Merging ECCC weather (if available)...")

    weather_file = os.path.join("data", "processed", "eccc_weather_processed.nc")
    weather_daily_df = None
    if os.path.exists(weather_file):
        try:
            wds = xr.open_dataset(weather_file)
            # Expect variables: temperature, precipitation; reduce over spatial dims
            def reduce_to_daily_mean(da):
                reduce_dims = [d for d in da.dims if d != 'time']
                if reduce_dims:
                    da = da.mean(dim=reduce_dims)
                # Ensure daily frequency
                df = da.to_series().reset_index()
                df['time'] = pd.to_datetime(df['time'])
                return df.set_index('time').resample('D').mean()

            def reduce_to_daily_sum(da):
                reduce_dims = [d for d in da.dims if d != 'time']
                if reduce_dims:
                    da = da.mean(dim=reduce_dims)
                df = da.to_series().reset_index()
                df['time'] = pd.to_datetime(df['time'])
                return df.set_index('time').resample('D').sum()

            weather_parts = []
            if 'temperature' in wds.variables:
                t_daily = reduce_to_daily_mean(wds['temperature'])
                # Convert K->C if values appear in Kelvin
                temp_vals = t_daily.values
                # heuristic: Kelvin if > 150
                if np.nanmean(temp_vals) > 150:
                    t_daily = t_daily - 273.15
                t_daily = t_daily.rename('temperature_c')
                weather_parts.append(t_daily)
            if 'precipitation' in wds.variables:
                p_daily = reduce_to_daily_sum(wds['precipitation'])
                # Convert m->mm if small values
                if np.nanmean(p_daily.values) < 10:
                    p_daily = p_daily * 1000.0
                p_daily = p_daily.rename('precipitation_mm')
                weather_parts.append(p_daily)

            if weather_parts:
                weather_daily_df = pd.concat(weather_parts, axis=1)
                weather_daily_df.index.name = 'date'
        except Exception as e:
            print(f"Warning: could not merge weather data: {e}")

    # Fallback: use recent combined ECCC daily CSV if GRIB2 not available
    if (weather_daily_df is None or weather_daily_df.empty):
        try:
            recent_path = os.path.join("data", "raw", "eccc_recent", "eccc_recent_combined.csv")
            if os.path.exists(recent_path):
                rdf = pd.read_csv(recent_path)
                date_col = 'date' if 'date' in rdf.columns else ('Date/Time' if 'Date/Time' in rdf.columns else None)
                if date_col is not None:
                    rdf[date_col] = pd.to_datetime(rdf[date_col], errors='coerce')
                    # Optional spatial filter
                    if {'Latitude (y)', 'Longitude (x)'}.issubset(rdf.columns):
                        lon_min, lat_min, lon_max, lat_max = (-97.5, 49.0, -96.5, 50.5)
                        rdf = rdf[(rdf['Longitude (x)'] >= lon_min) & (rdf['Longitude (x)'] <= lon_max) & (rdf['Latitude (y)'] >= lat_min) & (rdf['Latitude (y)'] <= lat_max)]
                    parts = []
                    if 'Mean Temp (°C)' in rdf.columns:
                        t = rdf[[date_col, 'Mean Temp (°C)']].groupby(date_col)['Mean Temp (°C)'].mean().rename('temperature_c')
                        parts.append(t)
                    if 'Total Precip (mm)' in rdf.columns:
                        p = rdf[[date_col, 'Total Precip (mm)']].groupby(date_col)['Total Precip (mm)'].sum().rename('precipitation_mm')
                        parts.append(p)
                    if parts:
                        weather_daily_df = pd.concat(parts, axis=1)
                        weather_daily_df.index.name = 'date'
        except Exception as e:
            print(f"Warning: could not use recent ECCC CSV for weather: {e}")

    # --- Format data for NeuralHydrology ---
    print("Formatting data for NeuralHydrology...")
    
    # Ensure we have a single date index and proper data structure
    if isinstance(merged_data.index, pd.MultiIndex):
        # If we have a MultiIndex, take the first level (date)
        merged_data = merged_data.reset_index(level=1, drop=True)
        print("Fixed MultiIndex - using date as single index")
    
    # Ensure the index is a proper datetime
    if not isinstance(merged_data.index, pd.DatetimeIndex):
        merged_data.index = pd.to_datetime(merged_data.index)
    
    # Ensure we have the required columns for NeuralHydrology
    required_columns = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 'day_of_year', 'month', 'year']
    
    # Create missing columns if they don't exist
    if 'snow_depth_mm' not in merged_data.columns:
        if 'Snow on Grnd (mm)' in merged_data.columns:
            merged_data['snow_depth_mm'] = merged_data['Snow on Grnd (mm)']
        else:
            merged_data['snow_depth_mm'] = 0.0
            print("Warning: snow_depth_mm column not found, using default value")
    
    if 'snow_fall_mm' not in merged_data.columns:
        if 'Total Snow (mm)' in merged_data.columns:
            merged_data['snow_fall_mm'] = merged_data['Total Snow (mm)']
    else:
            merged_data['snow_fall_mm'] = 0.0
            print("Warning: snow_fall_mm column not found, using default value")
    
    if 'snow_water_equivalent_mm' not in merged_data.columns:
        # Estimate SWE from snow depth (typical ratio: 0.3)
        merged_data['snow_water_equivalent_mm'] = merged_data['snow_depth_mm'] * 0.3
        print("Created snow_water_equivalent_mm from snow depth estimation")
    
    # Add time-based features
    merged_data['day_of_year'] = merged_data.index.dayofyear
    merged_data['month'] = merged_data.index.month
    merged_data['year'] = merged_data.index.year
    
    # Select the streamflow column (use the first available station)
    streamflow_col = None
    for col in merged_data.columns:
        if col != 'date' and 'streamflow' in col.lower():
            streamflow_col = col
            break
    
    if streamflow_col:
        print(f"Using streamflow data from station: {streamflow_col}")
        # Rename to standard name
        merged_data = merged_data.rename(columns={streamflow_col: 'streamflow_m3s'})
    else:
        print("No streamflow column found")
        # 系统禁止使用合成数据
        raise ValueError("Missing streamflow data. Synthetic data generation is prohibited.")
    
    # Clean the data
    print(f"Before cleaning: {len(merged_data)} records")
    
    # Check for missing values
    missing_counts = merged_data.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"Missing {col}: {count}")
    
    # Fill missing values
    merged_data = merged_data.fillna(0)
    
    print(f"After filling missing values: {len(merged_data)} records")
    
    # Ensure all numeric columns are float
    for col in merged_data.columns:
        if col != 'date' and merged_data[col].dtype in ['object', 'int64']:
            merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
    
    # Final data validation
    print("\n🔍 Data Validation:")
    print(f"  Index type: {type(merged_data.index)}")
    print(f"  Index length: {len(merged_data.index)}")
    print(f"  Columns: {list(merged_data.columns)}")
    print(f"  Data types: {merged_data.dtypes.to_dict()}")
    
    # Verify required columns exist
    missing_required = [col for col in required_columns if col not in merged_data.columns]
    if missing_required:
        print(f"❌ Missing required columns: {missing_required}")
        return None
    else:
        print("✅ All required columns present")
    
    # Final data summary
    print(f"\n📊 Data Summary:")
    print(f"  Total records: {len(merged_data)}")
    print(f"  Date range: {merged_data.index.min()} to {merged_data.index.max()}")
    
    if 'snow_depth_mm' in merged_data.columns:
        print(f"  Snow depth range: {merged_data['snow_depth_mm'].min():.1f} - {merged_data['snow_depth_mm'].max():.1f} mm")
    
    if 'streamflow_m3s' in merged_data.columns:
        print(f"  Streamflow range: {merged_data['streamflow_m3s'].min():.1f} - {merged_data['streamflow_m3s'].max():.1f} m³/s")
    
    # Save the prepared data
    output_file = os.path.join(output_dir, "red_river_basin", "timeseries.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Reset index to make date a column and ensure proper format
    merged_data_reset = merged_data.reset_index()
    merged_data_reset['date'] = merged_data_reset['date'].dt.strftime('%Y-%m-%d')
    
    # Save with proper encoding
    merged_data_reset.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Saving prepared data to {output_file}...")
    
    # Create basins file
    basins_file = os.path.join(output_dir, "basins.txt")
    with open(basins_file, 'w') as f:
        f.write("red_river_basin\n")
    
    print(f"Created basins file: {basins_file}")
    
    print("✅ Data preparation for NeuralHydrology finished.")
    return output_file

if __name__ == "__main__":
    processed_data_dir = "data/processed"
    neuralhydrology_data_dir = "src/neuralhydrology/data"
    prepare_data_for_neuralhydrology(processed_data_dir, neuralhydrology_data_dir)


def prepare_eval_dataset(processed_dir, output_dir):
    """
    Prepares evaluation dataset using overlapping dates between ECC snow and HYDAT streamflow.
    Prioritizes historical data (1979-1998) for scenario analysis when available.
    
    Saves a CSV at {output_dir}/red_river_basin/timeseries_eval.csv
    Returns the output file path.
    """
    print("Preparing evaluation dataset (overlapping dates only, no synthetic flow)...")

    os.makedirs(output_dir, exist_ok=True)

    # Priority 1: Use historical ECCC data (1979-1998) for rich scenario analysis
    historical_path = os.path.join(processed_dir, "eccc_manitoba_snow_processed.csv")
    # Priority 2: Use recent combined ECCC (2020-2024) if historical not available
    recent_combined_path = os.path.join("data", "raw", "eccc_recent", "eccc_recent_combined.csv")
    
    # Choose data source based on availability
    if os.path.exists(historical_path):
        eccc_snow_data_path = historical_path
        print("✅ Using historical ECCC data (1979-1998) for rich scenario analysis")
    elif os.path.exists(recent_combined_path):
        eccc_snow_data_path = recent_combined_path
        print("✅ Using recent ECCC data (2020-2024) as fallback")
    else:
        raise FileNotFoundError("No ECCC snow data available")
    
    hydat_data_path = os.path.join(processed_dir, "hydat_streamflow_processed.csv")

    # Load data
    snow = pd.read_csv(eccc_snow_data_path)
    if 'date' in snow.columns:
        snow['date'] = pd.to_datetime(snow['date'])
    elif 'Date/Time' in snow.columns:
        snow['date'] = pd.to_datetime(snow['Date/Time'])
    else:
        raise ValueError("ECCC snow data missing date column")

    # Optional: filter ECCC snow to Red River Basin region to reduce spatial mismatch
    try:
        with open("config/geographic_regions.yml", 'r') as f:
            cfg = yaml.safe_load(f)
        rr = cfg.get('red_river_basin', {})
        bbox = rr.get('bounding_box', None)
        if bbox and {'Latitude (y)', 'Longitude (x)'}.issubset(set(snow.columns)):
            lon_min, lat_min, lon_max, lat_max = bbox
            snow = snow[(snow['Longitude (x)'] >= lon_min) & (snow['Longitude (x)'] <= lon_max) & (snow['Latitude (y)'] >= lat_min) & (snow['Latitude (y)'] <= lat_max)]
    except Exception:
        pass

    # Aggregate to daily mean snow metrics
    snow_daily = snow.groupby('date').agg({
        'Total Snow (cm)': 'mean' if 'Total Snow (cm)' in snow.columns else 'first',
        'Snow on Grnd (cm)': 'mean' if 'Snow on Grnd (cm)' in snow.columns else 'first'
    }).reset_index()

    # Convert to mm (handle missing columns gracefully)
    if 'Total Snow (cm)' in snow_daily.columns:
        snow_daily['Total Snow (mm)'] = snow_daily['Total Snow (cm)'] * 10.0
    else:
        snow_daily['Total Snow (mm)'] = 0.0
    if 'Snow on Grnd (cm)' in snow_daily.columns:
        snow_daily['Snow on Grnd (mm)'] = snow_daily['Snow on Grnd (cm)'] * 10.0
    else:
        snow_daily['Snow on Grnd (mm)'] = 0.0

    # Load HYDAT (multiple station columns possible)
    hydat = pd.read_csv(hydat_data_path, index_col='date', parse_dates=True)
    # Pick first numeric column as streamflow
    numeric_cols = [c for c in hydat.columns if pd.api.types.is_numeric_dtype(hydat[c])]
    if not numeric_cols:
        # Fallback: if there is a 'streamflow_m3s' column
        target_col = 'streamflow_m3s' if 'streamflow_m3s' in hydat.columns else None
    else:
        target_col = numeric_cols[0]
    if target_col is None:
        raise ValueError("HYDAT data missing numeric streamflow column")
    hydat_df = hydat[[target_col]].rename(columns={target_col: 'streamflow_m3s'}).copy()
    hydat_df = hydat_df.reset_index()

    # Merge by inner join on date
    merged = pd.merge(
        snow_daily[['date', 'Total Snow (mm)', 'Snow on Grnd (mm)']],
        hydat_df[['date', 'streamflow_m3s']],
        on='date',
        how='inner'
    )

    if merged.empty:
        raise ValueError("No overlapping dates between ECCC snow and HYDAT streamflow for evaluation dataset")

    # Build NH-style frame
    df = pd.DataFrame()
    df['date'] = merged['date']
    df['snow_depth_mm'] = merged['Snow on Grnd (mm)'].fillna(0)
    df['snow_fall_mm'] = merged['Total Snow (mm)'].fillna(0)
    df['snow_water_equivalent_mm'] = df['snow_depth_mm'] * 0.3
    df['streamflow_m3s'] = merged['streamflow_m3s']
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    df = df.sort_values('date').dropna(subset=['streamflow_m3s'])

    basin_dir = os.path.join(output_dir, "red_river_basin")
    os.makedirs(basin_dir, exist_ok=True)
    out_file = os.path.join(basin_dir, "timeseries_eval.csv")
    print(f"Saving evaluation data to {out_file}...")
    df.to_csv(out_file, index=False)
    print("✅ Evaluation dataset prepared.")
    return out_file


def use_eval_as_training_data(eval_csv_path: str, nh_data_dir: str = "src/neuralhydrology/data") -> str:
    """
    Replace the NeuralHydrology timeseries.csv with the evaluation dataset for CV training.
    Returns the path to the written timeseries.csv
    """
    basin_dir = os.path.join(nh_data_dir, "red_river_basin")
    os.makedirs(basin_dir, exist_ok=True)
    out_file = os.path.join(basin_dir, "timeseries.csv")
    df = pd.read_csv(eval_csv_path)
    # Ensure expected columns exist
    required = [
        'date', 'snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm',
        'day_of_year', 'month', 'year', 'streamflow_m3s'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Evaluation CSV missing required columns: {missing}")
    df = df.sort_values('date').drop_duplicates(subset=['date'])
    df.to_csv(out_file, index=False)
    return out_file
