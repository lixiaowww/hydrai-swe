#!/usr/bin/env python3
"""
Test script to verify NASA Earthdata authentication
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from src.data.etl import fetch_nasa_data

def test_nasa_authentication(region_name="red_river_basin"):
    """Test NASA Earthdata authentication"""
    print("Testing NASA Earthdata authentication...")
    
    # Load credentials
    load_dotenv('config/credentials.env')
    
    username = os.getenv('NASA_EARTHDATA_USERNAME')
    password = os.getenv('NASA_EARTHDATA_PASSWORD')
    
    if not username or not password:
        print("Error: NASA Earthdata credentials not found in environment variables")
        print("Please check config/credentials.env file")
        return False
    
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password)} (hidden)")
    
    # Load geographic configuration
    try:
        with open('config/geographic_regions.yml', 'r') as file:
            import yaml
            config = yaml.safe_load(file)
            region_config = config.get(region_name, config['red_river_basin'])
            bounding_box = region_config['bounding_box']
            print(f"Testing with region: {region_config['name']}")
            print(f"Bounding box: {bounding_box}")
    except Exception as e:
        print(f"Warning: Could not load geographic config, using default: {e}")
        bounding_box = (-97.5, 49.0, -96.5, 50.5)  # Red River Basin default
    
    # Test with a small data request
    try:
        print("\nAttempting to authenticate and search for data...")
        
        # This will test authentication and data search
        fetch_nasa_data(
            short_name="MOD10A1",
            version="061",
            bounding_box=bounding_box,
            start_date="2024-03-01",
            end_date="2024-03-03",  # Short period for testing
            output_dir="test_output",
            username=username,
            password=password
        )
        
        print("Authentication test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Authentication test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NASA Earthdata Authentication')
    parser.add_argument('--region', '-r', 
                       choices=['manitoba_province', 'red_river_basin', 'winnipeg_metro', 'winnipeg_city'],
                       default='red_river_basin',
                       help='Geographic region for testing (default: red_river_basin)')
    
    args = parser.parse_args()
    
    success = test_nasa_authentication(region_name=args.region)
    if success:
        print("\n✅ NASA Earthdata authentication is working!")
    else:
        print("\n❌ NASA Earthdata authentication failed!")
        sys.exit(1)
