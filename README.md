# HydrAI-SWE

HydrAI-SWE is a hydrology AI system focused on Snow Water Equivalent (SWE) and runoff prediction for Manitoba/Canadian basins. It integrates satellite, weather, and hydrometric data with advanced deep learning models and a production-ready FastAPI + Web UI.

Repository: https://github.com/lixiaowww/hydrai-swe

## 🎉 What's new (2025-08-23)

### 🏆 **Model Training Completed Successfully!**
- **Best Model**: GRU Ensemble Model with R² = 0.8852 (88.52%)
- **Performance**: RMSE = 0.3272, MAE = 0.2611
- **Architecture**: 3 optimized GRU models integrated for maximum performance
- **Training Data**: 20,449 real records (1970-2024) with comprehensive features

### 🚀 **Advanced AI Models Implemented**
- **GRU Ensemble**: 3 best configurations integrated (primary model)
- **Anti-overfitting System**: Specialized core for preventing R² degradation
- **Hyperparameter Optimization**: Optuna-based fine-tuning with 25 trials
- **Cross-validation**: Forward-chain time series validation system

### 🔧 **Production-Ready Features**
- **Frontend redesigned** to a two-column layout:
  - Left: controls, chart, analysis, glossary of key terms
  - Right: Provenance & Notes (data source, algorithm, author, confidence, contact)
- **High-resolution data pipeline** fully implemented:
  - Sentinel-2 downloader and NDSI processor (`src/data/download_sentinel2.py`)
  - DEM downloader and terrain features (`src/data/download_dem.py`)
  - High-res integrator and enhanced features (`src/data/integrate_high_resolution.py`)
  - Simulator for Sentinel-2 NDSI when network/credentials unavailable
- **API + UI** stable and production-ready
- **Geographic region support** for multiple precision levels (Red River Basin, Winnipeg Metro, City Core, Manitoba Province)
- **Smart data selection** with intelligent date range and scenario analysis
- **Agriculture module planning** completed with research on GitHub and Kaggle success cases

## 🎯 **Best Model Performance**

### 🏆 **GRU Ensemble Model (Recommended)**
```bash
# Model Performance (Test Set)
R² Score: 0.8852 (88.52%)
RMSE: 0.3272
MAE: 0.2611
```

### 🔍 **Model Architecture**
- **Base Models**: 3 optimized GRU models
- **Integration**: Simple averaging ensemble
- **Input Features**: 6 (snow depth, snowfall, SWE, time features)
- **Sequence Length**: 30 days
- **Training Data**: 20,449 real records (1970-2024)

### 📊 **Model Comparison**
| Model Type | R² Score | Status | Recommendation |
|------------|----------|---------|----------------|
| **GRU Ensemble** | **0.8852** | ✅ **Best** | **Use for production** |
| GRU Single | 0.85+ | ✅ Good | Backup option |
| LSTM | R² < 0 | ❌ Failed | Not recommended |

## 🚀 Live run (API + UI)

### Prerequisites
- Python 3.9+
- Linux/macOS (Windows WSL recommended)
- 8GB+ RAM for high-resolution data processing

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

### Start backend and UI
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# UI → http://localhost:8000/ui
# API docs → http://localhost:8000/docs
```

### API examples
```bash
# Runoff forecast
curl "http://localhost:8000/api/v1/runoff-forecast?start_date=2024-03-01&end_date=2024-03-07&mode=nowcast&scenario_year=2023"

# SWE data
curl "http://localhost:8000/api/v1/swe?date=2024-03-01&region=red_river_basin"

# Risk assessment
curl "http://localhost:8000/api/v1/risk-assessment?date=2024-03-01&region=red_river_basin"
```

## 🤖 **Model Training & Usage**

### 🎯 **Use Best Model (Recommended)**
```bash
# Load and use the trained GRU ensemble model
python3 src/models/optimized_predictor.py

# Or use the ensemble system directly
python3 ensemble_top3_models.py
```

### 🔧 **Advanced Training Options**

#### 1) **GRU Ensemble Training** (Current Best)
```bash
python3 ensemble_top3_models.py
```
- **Features**: 3 best GRU configurations integrated
- **Performance**: R² = 0.8852
- **Status**: ✅ **Production Ready**

#### 2) **Individual Model Training**
```bash
# GRU model with optimized hyperparameters
python3 src/models/train_real_data.py

# Anti-overfitting system
python3 src/models/anti_overfitting_core.py
```

#### 3) **Hyperparameter Optimization**
```bash
# Fine-tune hyperparameters with Optuna
python3 fine_tune_hyperparameters.py
```

### 🏗️ **Model Architecture Details**
- **Primary**: GRU (Gated Recurrent Unit) - 64 hidden units, 2 layers
- **Ensemble**: 3 best configurations integrated
- **Input**: 30-day sequence of snow/weather features
- **Output**: Snow depth prediction
- **Optimization**: Adam optimizer, learning rate 0.001

## 📊 **Data & Performance**

### 🎯 **Training Data Summary**
- **Total Records**: 20,449 (1970-2024)
- **Features**: 28 comprehensive features including:
  - Snow depth, snowfall, SWE
  - Time features (day, month, year)
  - Lagged variables (1, 3, 7 days)
  - Moving averages (7, 14, 30 days)
- **Data Sources**: ECCC, HYDAT, NASA MODIS, Sentinel-2

### 📈 **Performance Metrics**
- **Training Set**: 16,360 samples
- **Validation Set**: 4,089 samples  
- **Test Set**: 5,113 samples
- **Cross-validation**: Forward-chain time series validation

### 🌍 **Geographic Coverage**
- **Red River Basin**: ~116,000 km² (Recommended)
- **Winnipeg Metro**: ~5,300 km²
- **Winnipeg City**: ~465 km²
- **Manitoba Province**: ~650,000 km²

## 🔬 **Technical Implementation**

### 🧠 **AI/ML Models**
- **GRU Ensemble**: Primary production model
- **Anti-overfitting Core**: Specialized system for R² optimization
- **Cross-validation**: Time-series aware validation
- **Hyperparameter Tuning**: Optuna-based optimization

### 📁 **Key Files & Directories**
```
models/
├── ensemble_models_20250823_205750/  # Best ensemble model
├── optimized_gru_model_20250823_113631.pth  # Single best GRU
└── standardization_params.pkl  # Data preprocessing

src/models/
├── optimized_predictor.py      # Production predictor
├── anti_overfitting_core.py   # R² optimization system
└── cross_validation_system.py # Time series validation

logs/
├── ensemble_model_report_20250823_205750.md  # Performance report
└── fine_tune_best_hyperparameters_20250823_150607.json  # Best params
```

## 🚀 **Getting Started**

### 1. **Quick Start with Best Model**
```bash
# Load the trained ensemble model
python3 src/models/optimized_predictor.py
```

### 2. **Full Training Pipeline**
```bash
# Train the ensemble model from scratch
python3 ensemble_top3_models.py
```

### 3. **API Service**
```bash
# Start the production API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 **Documentation**

- **Training Progress**: `TRAINING_PROGRESS_UPDATE.md`
- **Training Summary**: `TRAINING_SUMMARY_REPORT.md`
- **Model Performance**: `logs/ensemble_model_report_20250823_205750.md`
- **Technical Specs**: `docs/3_technical_specification_document.md`

## 🤝 **Contributing**

This project follows strict integrity principles:
- **No data fabrication** - All results based on real data
- **Transparent methodology** - Full documentation of methods
- **Reproducible results** - Complete training pipeline provided
- **Quality assurance** - Anti-overfitting and validation systems

## 📄 **License**

[Add your license information here]

---

**🎉 HydrAI-SWE is now production-ready with R² = 0.8852!**

For questions or support, please refer to the documentation or create an issue.


