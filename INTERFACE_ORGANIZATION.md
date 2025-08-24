# HydrAI-SWE Interface Organization

## Overview

The HydrAI-SWE system has been reorganized to separate end-user interfaces from technical/model training interfaces for better user experience and workflow organization.

## Interface Structure

### 🌟 End User Interface
- **URL**: http://localhost:8000/ui
- **Language**: English
- **Purpose**: Primary interface for end users to view SWE predictions, runoff forecasting, flood warnings, and system status
- **Features**:
  - Real-time SWE predictions with 95% completion status
  - Runoff forecasting (90% complete, production-ready)
  - Flood warning system (60% complete, in development)
  - System performance monitoring
  - Data quality assessment
  - Real-time HYDAT station data

### 🛠️ Model Training Interface
- **URL**: http://localhost:8000/model
- **Language**: English
- **Purpose**: Technical interface for model training, configuration, and system administration
- **Features**:
  - Live training status and progress monitoring
  - Model architecture configuration (LSTM, GRU, EA-LSTM, Transformer)
  - Hyperparameter tuning (learning rate, batch size, epochs)
  - Data pipeline management
  - Model performance metrics by basin
  - Real-time training logs
  - Configuration management

### 🇨🇳 Chinese Interface
- **URL**: http://localhost:8000/ui/enhanced
- **Language**: Chinese (中文)
- **Purpose**: Chinese localized version of the enhanced interface
- **Features**: Same functionality as the English enhanced interface but fully localized

### 🏛️ Legacy Interface
- **URL**: http://localhost:8000/ui/legacy
- **Language**: English
- **Purpose**: Original interface maintained for backwards compatibility
- **Features**: Basic functionality for users who prefer the original design

### 🚀 Next-Gen Interface
- **URL**: http://localhost:8000/ui/vnext
- **Language**: English
- **Purpose**: Prototype interface for testing new features
- **Features**: Experimental UI components and workflows

## Navigation

Each interface includes navigation links to easily switch between different interfaces:

### From End User Interface (/ui):
- Model Training → `/model`
- Chinese Interface → `/ui/enhanced`
- Next-Gen UI → `/ui/vnext`
- API Documentation → `/docs`

### From Model Training Interface (/model):
- End User Interface → `/ui`
- Chinese Interface → `/ui/enhanced`
- API Documentation → `/docs`

## Quick Access

### For End Users:
Primary interface: **http://localhost:8000/ui**

### For Data Scientists/Developers:
Model training: **http://localhost:8000/model**

### For Chinese Users:
Chinese interface: **http://localhost:8000/ui/enhanced**

## Starting the System

Use either of these startup scripts:

### Enhanced Startup Script:
```bash
python3 start_enhanced_ui.py
```
- Shows all available interfaces
- Includes detailed system information
- Auto-opens browser to Chinese interface by default

### Simple Test Script:
```bash
python3 test_server.py
```
- Basic server startup
- Lists all interface URLs
- No auto-browser opening

## Interface Characteristics

| Interface | Primary Users | Focus | Status |
|-----------|---------------|-------|--------|
| `/ui` | End Users | SWE predictions, flood warnings | ✅ Production |
| `/model` | Data Scientists, Engineers | Model training, configuration | ✅ Production |
| `/ui/enhanced` | Chinese Users | Full functionality in Chinese | ✅ Production |
| `/ui/legacy` | Legacy Users | Basic functionality | ✅ Stable |
| `/ui/vnext` | Testers, Developers | Experimental features | ⚠️ Prototype |

## API Integration

All interfaces connect to the same backend API endpoints:
- **System Status**: `/api/swe/system-status`
- **Data Quality**: `/api/swe/data-quality`  
- **HYDAT Stations**: `/api/swe/hydat-stations`
- **API Documentation**: `/docs`

## Design Philosophy

1. **Separation of Concerns**: End users don't need access to model training controls
2. **Role-Based Access**: Different interfaces for different user roles
3. **Consistent Navigation**: Easy switching between interfaces
4. **Language Support**: Native language support for international users
5. **Progressive Enhancement**: Legacy support while introducing new features

## Migration Guide

### For Previous Users of `/ui/enhanced-en`:
- **New URL**: http://localhost:8000/ui
- **Changes**: None - same functionality, cleaner URL

### For Model Training Users:
- **New URL**: http://localhost:8000/model
- **Changes**: Enhanced technical features, live training monitoring

### For Chinese Users:
- **URL Unchanged**: http://localhost:8000/ui/enhanced
- **Changes**: None - interface remains the same

This organization provides a cleaner, more intuitive structure while maintaining all existing functionality.
