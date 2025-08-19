#!/usr/bin/env python3
"""
Complete training pipeline for HydrAI-SWE project.
This script runs the full pipeline: ETL -> Data Preparation -> Model Training
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from src.data.etl import run_etl
from src.neuralhydrology.prepare_data import prepare_data_for_neuralhydrology
from src.models.train import train_model_with_neuralhydrology
from src.neuralhydrology.convert_to_netcdf import (
    convert_csv_to_netcdf,
    create_basin_metadata,
)
from src.models.predict import predict_with_neuralhydrology

def run_full_training_pipeline(region_name=None):
    """
    Run the complete training pipeline:
    1. ETL: Fetch and process real data
    2. Data Preparation: Prepare data for NeuralHydrology
    3. Model Training: Train the LSTM model
    
    Args:
        region_name (str): Geographic region name for data processing
    """
    print("🚀 Starting HydrAI-SWE Full Training Pipeline")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv('config/credentials.env')
    
    nasa_username = os.getenv('NASA_EARTHDATA_USERNAME')
    nasa_password = os.getenv('NASA_EARTHDATA_PASSWORD')
    
    if not nasa_username or not nasa_password:
        print("❌ Error: NASA Earthdata credentials not found")
        print("Please check config/credentials.env file")
        sys.exit(1)
    
    print(f"✅ NASA Earthdata credentials loaded for user: {nasa_username}")
    
    # Geographic region selection
    if region_name is None:
        print("\n🌍 Geographic Region Selection:")
        print("Available regions:")
        print("1. manitoba_province - Manitoba Province (~650,000 km²)")
        print("2. red_river_basin - Red River Basin (~116,000 km²) [RECOMMENDED]")
        print("3. winnipeg_metro - Winnipeg Metropolitan Area (~5,300 km²)")
        print("4. winnipeg_city - Winnipeg City Core (~465 km²)")
        
        # Default to red river basin (recommended for SWE modeling)
        region_name = "red_river_basin"
        print(f"\nUsing recommended region: {region_name}")
    
    try:
        # Step 1: ETL Process
        print("\n📊 Step 1: Running ETL Process")
        print("-" * 30)
        run_etl(nasa_username=nasa_username, nasa_password=nasa_password, region_name=region_name)
        print("✅ ETL process completed")
        
        # Step 2: Data Preparation
        print("\n🔧 Step 2: Preparing Data for NeuralHydrology")
        print("-" * 40)
        processed_data_dir = "data/processed"
        neuralhydrology_data_dir = "src/neuralhydrology/data"
        
        # Check if processed data exists
        if not os.path.exists(processed_data_dir):
            print(f"❌ Error: Processed data directory not found: {processed_data_dir}")
            print("Please run the ETL process first")
            sys.exit(1)
        
        prepare_data_for_neuralhydrology(processed_data_dir, neuralhydrology_data_dir)
        print("✅ Data preparation completed")

        # Step 2.5: Convert prepared CSV to NetCDF and create basin metadata
        print("\n🗂️  Step 2.5: Converting prepared data to NetCDF and creating basin metadata")
        print("-" * 55)
        netcdf_path = convert_csv_to_netcdf()
        create_basin_metadata()
        print(f"✅ NetCDF generated at: {netcdf_path}")
        
        # Step 3: Model Training
        print("\n🤖 Step 3: Training NeuralHydrology Model")
        print("-" * 35)
        train_model_with_neuralhydrology()
        print("✅ Model training completed")
        
        # Step 4: Evaluate model and print regression metrics if available
        print("\n📈 Step 4: Evaluating Model (metrics)")
        print("-" * 34)
        # Try to locate the latest run dir and evaluate
        from src.models.predict_service import RunDirectoryLocator
        locator = RunDirectoryLocator(runs_root="src/neuralhydrology/runs")
        latest_run = locator.find_latest_run_dir("hydrai_swe_experiment")
        if latest_run is not None:
            predict_with_neuralhydrology(str(latest_run))
        else:
            print("No run directory found for evaluation.")

        print("\n🎉 Full training pipeline completed successfully!")
        print("Check the 'runs' directory for training results and trained models.")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("Please check the error details and fix any issues before retrying.")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HydrAI-SWE Full Training Pipeline')
    parser.add_argument('--region', '-r', 
                       choices=['manitoba_province', 'red_river_basin', 'winnipeg_metro', 'winnipeg_city'],
                       default=None,
                       help='Geographic region for data processing (default: red_river_basin)')
    
    args = parser.parse_args()
    
    run_full_training_pipeline(region_name=args.region)
