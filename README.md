# HydrAI-SWE Project

This project implements the HydrAI-SWE system as described in the project documentation.

## Overview

The HydrAI-SWE project aims to develop an advanced model for predicting hydrological responses in Canadian watersheds to optimize water resource management, hydropower operations, and flood warnings.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- NASA Earthdata account (configured in `config/credentials.env`)

### Quick Setup
```bash
# Run the quick start script
./quick_start.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Requirements
Before running the training pipeline, you need to download:
1. **ECCC GRIB2 weather data** → `data/raw/eccc_grib/`
2. **HYDAT hydrological database** → `data/raw/Hydat.sqlite3`

See `DATA_ACQUISITION.md` for detailed instructions.

### Run Training
```bash
# Full pipeline (ETL + Data Prep + Training)
# Default: Red River Basin (recommended)
python3 run_full_training.py

# Specify geographic region:
python3 run_full_training.py --region red_river_basin      # Red River Basin
python3 run_full_training.py --region winnipeg_metro      # Winnipeg Metro
python3 run_full_training.py --region winnipeg_city       # Winnipeg City
python3 run_full_training.py --region manitoba_province   # Manitoba Province

# Or step by step:
python3 src/data/etl.py                    # Data ETL
python3 src/neuralhydrology/prepare_data.py # Data preparation
python3 src/models/train.py                # Model training
```

## 📊 Project Status

**Current Progress: ~75% Complete**

- ✅ **Project Design**: 100% (Complete documentation)
- ✅ **Core Architecture**: 100% (FastAPI + NeuralHydrology)
- ✅ **Data Pipeline**: 90% (ETL + Processing)
- ✅ **Model Framework**: 80% (LSTM + Configuration)
- 🔄 **Data Acquisition**: 60% (NASA auth working, need manual downloads)
- 🔄 **Training Pipeline**: 70% (Ready, needs real data)
- ⏳ **Deployment**: 30% (Basic Terraform config)

## 📁 Project Structure

- `data/`: Contains raw and processed data
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `src/`: Main source code for the project
  - `api/`: FastAPI application for serving the model's predictions
  - `core/`: Core components like configuration
  - `data/`: Data processing and ETL pipelines
  - `models/`: Machine learning model for SWE prediction
  - `neuralhydrology/`: NeuralHydrology framework integration
  - `utils/`: Utility functions
- `tests/`: Unit and integration tests
- `infrastructure/`: Infrastructure as Code (Terraform) for deploying the project
- `config/`: Configuration files including credentials
- `Dockerfile`: For containerizing the application
- `requirements.txt`: Python dependencies

## 🔧 Key Features

- **Real-time Data Processing**: NASA MODIS + ECCC weather + HYDAT hydrological data
- **Advanced ML Model**: LSTM-based neural network for SWE prediction
- **Production API**: FastAPI-based REST API for model serving
- **Cloud Ready**: Terraform configuration for cloud deployment
- **Data Versioning**: DVC integration for data and model versioning

## 📚 Documentation

- `GEOGRAPHIC_REGIONS.md`: Geographic region selection guide
- `TRAINING_README.md`: Detailed training guide
- `DATA_ACQUISITION.md`: Data source and download instructions
- `docs/`: Project requirements and specifications

## 🎯 Next Steps

1. **Download required data** (ECCC + HYDAT)
2. **Run full training pipeline** with real data
3. **Evaluate model performance**
4. **Deploy to cloud infrastructure**
5. **Integrate with production systems**

## 🤝 Contributing

This project follows strict quality standards. See user rules for development guidelines.
