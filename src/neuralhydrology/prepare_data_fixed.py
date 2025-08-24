import xarray as xr
import pandas as pd
import numpy as np
import os
import rioxarray
import yaml

def prepare_data_for_neuralhydrology(processed_dir, output_dir):
    """
    Prepares the data for NeuralHydrology using ECC Manitoba snow data.
    Fixed version to ensure proper data format.
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
    
    print(f"Daily snow data: {len(daily_snow_data)} records")
    
    # Convert cm to mm
    daily_snow_data['Total Snow (mm)'] = daily_snow_data['Total Snow (cm)'] * 10.0
    daily_snow_data['Snow on Grnd (mm)'] = daily_snow_data['Snow on Grnd (cm)'] * 10.0
    
    # Set index to date for merging
    daily_snow_data = daily_snow_data.set_index('date')
    
    # Check snow depth range
    if 'Snow on Grnd (mm)' in daily_snow_data.columns:
        snow_range = daily_snow_data['Snow on Grnd (mm)']
        print(f"Snow depth range: {snow_range.min():.1f} - {snow_range.max():.1f} mm")

    # --- Merge Snow and Streamflow Data ---
    print("Merging snow and streamflow data...")
    
    # Check for overlapping dates
    snow_dates = set(daily_snow_data.index)
    hydat_dates = set(hydat_data.index)
    overlapping_dates = snow_dates.intersection(hydat_dates)
    
    if overlapping_dates:
        print(f"Found {len(overlapping_dates)} overlapping dates")
        # Use overlapping dates only
        merged_data = daily_snow_data.loc[list(overlapping_dates)].copy()
        merged_data = merged_data.join(hydat_data.loc[list(overlapping_dates)], how='inner')
    else:
        print("⚠️ No overlapping dates between snow and streamflow data")
        print("Creating complete dataset using ECC snow data...")
        
        # 系统禁止使用合成数据
        print("❌ 系统禁止使用合成数据")
        print("💡 请提供真实的径流观测数据")
        raise ValueError("Missing streamflow data. Synthetic data generation is prohibited.")
    
    # --- Format Data for NeuralHydrology ---
    print("Formatting data for NeuralHydrology...")
    
    # Ensure we have the required columns
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
    
    # Ensure proper date format and save
    merged_data_reset = merged_data.reset_index()
    merged_data_reset['date'] = pd.to_datetime(merged_data_reset['date']).dt.strftime('%Y-%m-%d')
    
    # Select only the required columns for NeuralHydrology
    final_columns = ['date'] + required_columns + ['streamflow_m3s']
    final_data = merged_data_reset[final_columns].copy()
    
    # Save with proper encoding
    final_data.to_csv(output_file, index=False, encoding='utf-8')
    
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
