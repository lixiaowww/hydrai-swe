# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

HydrAI-SWE is a hydrology AI system focused on Snow Water Equivalent (SWE) and runoff prediction for Manitoba/Canadian basins. It integrates satellite, weather, and hydrometric data with NeuralHydrology models and provides a production-ready FastAPI + Web UI.

## Architecture

### Core Components

**API Layer** (`src/api/`)
- FastAPI application with multiple UI versions (/ui, /ui/vnext, /ui/legacy)
- Three main router modules: SWE forecasting, flood warning, and cross-validation
- Gzip middleware enabled for performance

**Data Processing** (`src/data/`, `src/neuralhydrology/`)
- Multi-source data integration: NASA MODIS, ECCC weather, HYDAT streamflow
- High-resolution data pipeline supporting Sentinel-2 (10m) and DEM (30m)
- NeuralHydrology integration with NetCDF conversion and basin metadata

**ML Models** (`src/models/`)
- LSTM-based prediction models using NeuralHydrology framework
- Cross-validation system with time-forward validation to prevent data leakage
- Deterministic fallback when ML models unavailable
- Multiple prediction modes: nowcast, scenario, and forecast

**Geographic Regions**
- Four precision levels: Red River Basin (recommended, 116k km²), Winnipeg Metro (5.3k km²), Winnipeg City (465 km²), Manitoba Province (650k km²)
- Configurable resolution from 100m to 1000m based on region

## Common Development Tasks

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

### Running the Application
```bash
# Start API server with UI
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Access points:
# http://localhost:8000/ui (default UI)
# http://localhost:8000/ui/vnext (next-gen UI)
# http://localhost:8000/docs (API documentation)
```

### Training Pipeline

**Full training with high-resolution data (recommended):**
```bash
python run_full_training_with_high_resolution.py
```

**Classic training pipeline:**
```bash
python run_full_training.py --region red_river_basin
```

**Regional options:** `red_river_basin`, `winnipeg_metro`, `winnipeg_city`, `manitoba_province`

### Testing
```bash
# Test API endpoints
python -m pytest tests/api/

# Test specific components
python test_cross_validation_system.py
python test_flood_warning_system.py
python test_frontend_performance.py

# Data format testing
python test_data_format.py
```

### Single Test Execution
```bash
# Run specific test file
python -m pytest tests/api/test_swe.py -v

# Run specific test method
python -m pytest tests/api/test_swe.py::test_runoff_forecast -v
```

## Development Workflow

### Data Credentials Setup
- Copy `config/credentials.env.template` to `config/credentials.env`
- Add NASA_EARTHDATA_USERNAME and NASA_EARTHDATA_PASSWORD
- Never commit credentials file to version control

### High-Resolution Data Integration
The system includes advanced data integration supporting:
- **Sentinel-2**: NDSI processing for snow cover (10m resolution)
- **DEM**: Terrain features from SRTM/ASTER (30m resolution)
- **Automatic fallback**: Simulators when network/credentials unavailable

### Model Training Architecture
- **NeuralHydrology Integration**: Uses standardized LSTM/TCN/GRU models
- **Time-Forward Cross-Validation**: Prevents data leakage in temporal data
- **Multi-Resolution Training**: Adapts to different geographic precision levels
- **Automatic Model Versioning**: Results stored in `src/neuralhydrology/runs/`

### API Design Patterns
- **Three Prediction Modes**:
  - `nowcast`: ML models with deterministic fallback
  - `scenario`: Historical snow depth with melt-based flow estimates  
  - `forecast`: Future predictions using trained models
- **Smart Date Selection**: Intelligent range suggestions and climatology fallbacks
- **Regional Filtering**: Automatic coordinate-based data filtering

## Key Configuration Files

- `src/neuralhydrology/config.yml`: LSTM training parameters and data paths
- `src/core/config.py`: Application settings and database configuration
- `requirements.txt`: Python dependencies including TensorFlow and NeuralHydrology

## Data Pipeline Flow

1. **ETL Process**: Multi-source data ingestion (NASA, ECCC, HYDAT)
2. **High-Resolution Integration**: Optional Sentinel-2 and DEM enhancement
3. **NeuralHydrology Preparation**: Convert to NetCDF format with basin metadata
4. **Model Training**: LSTM training with cross-validation
5. **Prediction Service**: Real-time forecasting with multiple fallback strategies

## Geographic Context

This system is specifically designed for Canadian hydrology, particularly Manitoba basins. All coordinate systems use UTM Zone 14N, and the default region (Red River Basin) provides optimal balance between coverage and resolution for SWE modeling.

## Important Notes

- **Bilingual Codebase**: Contains both English and Chinese documentation and comments
- **Production-Ready**: Includes comprehensive error handling, logging, and performance monitoring
- **Multi-UI Strategy**: Three UI versions for different use cases and progressive rollout
- **Research Integration**: Includes planning for agriculture module development based on Kaggle/GitHub research

## Performance Considerations

- **Processing Times**: 2-4 hours for daily data ingestion, 6-12 hours for model training
- **Resource Requirements**: Minimum 8GB RAM, recommended 16GB+ with GPU support
- **API Response**: <200ms for standard queries, <1 minute for 7-day forecasts
