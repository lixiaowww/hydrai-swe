#!/usr/bin/env python3
"""
Debug script to test various data sources and identify issues
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

def test_nasa_data_availability():
    """Test NASA data availability with different parameters"""
    print("🔍 Testing NASA Data Availability")
    print("=" * 40)
    
    try:
        import earthaccess
        
        # Load credentials
        load_dotenv('config/credentials.env')
        username = os.getenv('NASA_EARTHDATA_USERNAME')
        password = os.getenv('NASA_EARTHDATA_PASSWORD')
        
        if username and password:
            os.environ['EARTHDATA_USERNAME'] = username
            os.environ['EARTHDATA_PASSWORD'] = password
            
            # Test authentication
            auth = earthaccess.login(strategy="environment")
            if auth.authenticated:
                print("✅ NASA authentication successful")
                
                # Test different data products
                test_products = [
                    ("MOD10A1", "061", "MODIS Snow Cover"),
                    ("MOD10A2", "061", "MODIS Snow Cover 8-day"),
                    ("MYD10A1", "061", "MODIS Aqua Snow Cover"),
                    ("VNP10A1", "001", "VIIRS Snow Cover"),
                ]
                
                for short_name, version, description in test_products:
                    print(f"\nTesting {description} ({short_name} v{version})...")
                    
                    try:
                        # Test with Red River Basin coordinates
                        results = earthaccess.search_data(
                            short_name=short_name,
                            version=version,
                            bounding_box=(-97.5, 49.0, -96.5, 50.5),
                            temporal=("2024-01-01", "2024-12-31"),
                        )
                        
                        if results:
                            print(f"  ✅ Found {len(results)} granules")
                            # Show first few results
                            for i, result in enumerate(results[:3]):
                                print(f"    {i+1}. {result.metadata.get('title', 'No title')}")
                        else:
                            print(f"  ❌ No data found")
                            
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
                        
            else:
                print("❌ NASA authentication failed")
        else:
            print("❌ NASA credentials not found")
            
    except ImportError:
        print("❌ earthaccess library not available")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_geographic_config():
    """Test geographic configuration loading"""
    print("\n🌍 Testing Geographic Configuration")
    print("=" * 40)
    
    try:
        import yaml
        
        config_path = "config/geographic_regions.yml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            print("✅ Geographic config loaded successfully")
            print(f"Default region: {config.get('default_region', 'Not set')}")
            
            for region_name, region_data in config.items():
                if isinstance(region_data, dict) and 'name' in region_data:
                    print(f"  - {region_name}: {region_data['name']}")
                    if 'bounding_box' in region_data:
                        print(f"    Bounding box: {region_data['bounding_box']}")
        else:
            print("❌ Geographic config file not found")
            
    except Exception as e:
        print(f"❌ Error loading geographic config: {e}")

def test_data_directories():
    """Test data directory structure"""
    print("\n📁 Testing Data Directory Structure")
    print("=" * 40)
    
    required_dirs = [
        "data/raw/nasa_modis_snow",
        "data/raw/eccc_grib", 
        "data/raw",
        "data/processed",
        "config"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - Creating...")
            os.makedirs(dir_path, exist_ok=True)
            print(f"  ✅ Created {dir_path}")

def test_hydat_config():
    """Test HYDAT database configuration"""
    print("\n💧 Testing HYDAT Configuration")
    print("=" * 40)
    
    # Align with ETL's HYDAT path
    hydat_path = "data/raw/Hydat_with_snow.sqlite3"
    if os.path.exists(hydat_path):
        print(f"✅ HYDAT database found: {hydat_path}")
        file_size = os.path.getsize(hydat_path)
        print(f"  File size: {file_size / (1024*1024):.1f} MB")
    else:
        print(f"❌ HYDAT database not found: {hydat_path}")
        print("  Please download from: https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/")

def test_requirements():
    """Test if all required packages are available"""
    print("\n📦 Testing Required Packages")
    print("=" * 40)
    
    required_packages = [
        'earthaccess', 'xarray', 'rasterio', 'cfgrib', 
        'neuralhydrology', 'fastapi', 'pandas', 'numpy'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not available")

def main():
    """Run all tests"""
    print("🚀 HydrAI-SWE Data Source Debug Tool")
    print("=" * 50)
    
    test_requirements()
    test_data_directories()
    test_geographic_config()
    test_hydat_config()
    test_nasa_data_availability()
    
    print("\n" + "=" * 50)
    print("🎯 Debug Summary:")
    print("1. Check package availability above")
    print("2. Ensure data directories exist")
    print("3. Verify geographic configuration")
    print("4. Download HYDAT database if missing")
    print("5. Test NASA data access with different products")

if __name__ == "__main__":
    main()
