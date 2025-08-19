#!/bin/bash

# HydrAI-SWE Quick Start Script
# This script helps you get started with the project

echo "🚀 HydrAI-SWE Quick Start Script"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/raw/nasa_modis_snow data/raw/eccc_grib data/processed

# Geographic region selection
echo ""
echo "🌍 Geographic Region Selection:"
echo "Available regions:"
echo "1. manitoba_province - Manitoba Province (~650,000 km²)"
echo "2. red_river_basin - Red River Basin (~116,000 km²) [RECOMMENDED]"
echo "3. winnipeg_metro - Winnipeg Metropolitan Area (~5,300 km²)"
echo "4. winnipeg_city - Winnipeg City Core (~465 km²)"
echo ""
echo "Using recommended region: red_river_basin"

# Test NASA authentication with selected region
echo "🔐 Testing NASA Earthdata authentication..."
python3 test_nasa_auth.py --region red_river_basin

echo ""
echo "🎉 Setup completed!"
echo ""
echo "Next steps:"
echo "1. Download ECCC GRIB2 data to data/raw/eccc_grib/"
echo "2. Download HYDAT database to data/raw/Hydat.sqlite3"
echo "3. Run the full training pipeline:"
echo "   - Default (Red River Basin): python3 run_full_training.py"
echo "   - Specific region: python3 run_full_training.py --region winnipeg_metro"
echo ""
echo "For detailed instructions, see DATA_ACQUISITION.md"
