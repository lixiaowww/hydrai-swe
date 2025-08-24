#!/usr/bin/env python3
"""
HydrAI-SWE 农业模块API路由
集成土壤水分预测、作物推荐、产量预测等功能
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# 添加农业模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'agriculture'))

from soil_moisture_predictor import SoilMoisturePredictor, AgricultureDataProcessor

# 创建路由器
router = APIRouter(tags=["agriculture"])

# 数据模型
class SoilMoistureRequest(BaseModel):
    """土壤水分预测请求"""
    location: str
    start_date: str
    end_date: str
    features: Optional[List[str]] = None

class CropRecommendationRequest(BaseModel):
    """作物推荐请求"""
    location: str
    temperature: float
    precipitation: float
    soil_moisture: float
    soil_type: Optional[str] = "loam"

class YieldPredictionRequest(BaseModel):
    """产量预测请求"""
    crop_type: str
    location: str
    planting_date: str
    weather_conditions: Dict[str, float]

# 全局农业模块实例
soil_moisture_predictor = None
agriculture_data_processor = None

def get_agriculture_modules():
    """获取农业模块实例"""
    global soil_moisture_predictor, agriculture_data_processor
    
    if soil_moisture_predictor is None:
        soil_moisture_predictor = SoilMoisturePredictor()
        
    if agriculture_data_processor is None:
        agriculture_data_processor = AgricultureDataProcessor()
    
    return soil_moisture_predictor, agriculture_data_processor

@router.get("/health")
async def agriculture_health_check():
    """农业模块健康检查"""
    return {
        "status": "healthy",
        "module": "agriculture",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "soil_moisture_prediction",
            "crop_recommendation", 
            "yield_prediction",
            "irrigation_optimization"
        ]
    }

@router.get("/model/status")
async def get_model_status():
    """检查模型训练状态"""
    try:
        predictor, data_processor = get_agriculture_modules()
        
        is_trained = predictor.model is not None
        model_config = {
            "input_size": predictor.config.get('input_size', 0),
            "hidden_size": predictor.config.get('hidden_size', 64),
            "num_layers": predictor.config.get('num_layers', 2)
        }
        
        return {
            "status": "success",
            "model_trained": is_trained,
            "model_config": model_config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型状态失败: {str(e)}")

@router.post("/model/train")
async def train_soil_moisture_model():
    """预训练土壤水分预测模型"""
    try:
        predictor, data_processor = get_agriculture_modules()
        
        # 加载数据
        data_path = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
        
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="数据文件不存在")
        
        # 准备数据
        X_train, y_train, X_val, y_val, X_test, y_test, scalers = \
            data_processor.prepare_soil_moisture_data(data_path)
        
        # 从data_processor获取input_size并设置到predictor
        predictor.config['input_size'] = data_processor.config['input_size']
        print(f"🔧 设置predictor input_size: {predictor.config['input_size']}")
        
        # 训练模型
        training_history = predictor.train_model(X_train, y_train, X_val, y_val)
        
        return {
            "status": "success",
            "message": "模型训练完成",
            "model_info": {
                "type": "LSTM",
                "input_features": predictor.config['input_size'],
                "hidden_size": predictor.config['hidden_size'],
                "layers": predictor.config['num_layers'],
                "training_data_shape": [len(X_train), X_train.shape[1] if len(X_train) > 0 else 0]
            },
            "training_stats": {
                "epochs_completed": len(training_history) if training_history else 0,
                "final_loss": training_history[-1] if training_history else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")

@router.post("/soil-moisture/predict")
async def predict_soil_moisture(request: SoilMoistureRequest):
    """
    预测土壤水分
    
    Args:
        request: 预测请求参数
        
    Returns:
        dict: 预测结果
    """
    try:
        predictor, data_processor = get_agriculture_modules()
        
        # 加载数据
        data_path = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
        
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="数据文件不存在")
        
        # 准备数据
        X_train, y_train, X_val, y_val, X_test, y_test, scalers = \
            data_processor.prepare_soil_moisture_data(data_path)
        
        # 从data_processor获取input_size并设置到predictor
        predictor.config['input_size'] = data_processor.config['input_size']
        print(f"🔧 设置predictor input_size: {predictor.config['input_size']}")
        
        # 训练模型（如果还没有训练）
        if predictor.model is None:
            training_history = predictor.train_model(X_train, y_train, X_val, y_val)
        
        # 进行预测
        predictions = predictor.predict(X_test, scalers[1])
        
        # 计算预测统计
        prediction_stats = {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "predictions_count": len(predictions)
        }
        
        return {
            "status": "success",
            "location": request.location,
            "prediction_date": datetime.now().isoformat(),
            "prediction_stats": prediction_stats,
            "model_info": {
                "type": "LSTM",
                "input_features": predictor.config['input_size'],
                "hidden_size": predictor.config['hidden_size'],
                "layers": predictor.config['num_layers']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"土壤水分预测失败: {str(e)}")

@router.post("/crop/recommend")
async def recommend_crops(request: CropRecommendationRequest):
    """
    推荐适合的作物
    
    Args:
        request: 作物推荐请求参数
        
    Returns:
        dict: 推荐结果
    """
    try:
        # 作物数据库（基于SoilWeatherPredictor项目）
        crop_database = {
            'Corn': {
                'temp_range': (8, 40),
                'precip_range': (100, 700),
                'moisture_range': (15, 60),
                'description': '玉米，适合温暖气候，需要充足水分'
            },
            'Wheat': {
                'temp_range': (0, 30),
                'precip_range': (100, 500),
                'moisture_range': (10, 50),
                'description': '小麦，耐寒作物，适合温带气候'
            },
            'Rice': {
                'temp_range': (10, 35),
                'precip_range': (300, 1000),
                'moisture_range': (30, 80),
                'description': '水稻，需要大量水分，适合湿润气候'
            },
            'Soybeans': {
                'temp_range': (8, 40),
                'precip_range': (150, 600),
                'moisture_range': (20, 60),
                'description': '大豆，适应性强的豆类作物'
            },
            'Barley': {
                'temp_range': (0, 25),
                'precip_range': (80, 400),
                'moisture_range': (10, 40),
                'description': '大麦，耐寒耐旱，适合北方种植'
            },
            'Sorghum': {
                'temp_range': (10, 45),
                'precip_range': (50, 400),
                'moisture_range': (5, 45),
                'description': '高粱，耐旱作物，适合干旱地区'
            }
        }
        
        # 评估作物适宜性
        suitable_crops = []
        crop_scores = {}
        
        for crop_name, crop_data in crop_database.items():
            score = 0
            reasons = []
            
            # 温度适宜性评分
            temp_min, temp_max = crop_data['temp_range']
            if temp_min <= request.temperature <= temp_max:
                score += 30
                reasons.append("温度适宜")
            elif abs(request.temperature - (temp_min + temp_max) / 2) <= 5:
                score += 20
                reasons.append("温度接近适宜范围")
            else:
                reasons.append("温度不适宜")
            
            # 降水适宜性评分
            precip_min, precip_max = crop_data['precip_range']
            if precip_min <= request.precipitation <= precip_max:
                score += 30
                reasons.append("降水适宜")
            elif abs(request.precipitation - (precip_min + precip_max) / 2) <= 50:
                score += 20
                reasons.append("降水接近适宜范围")
            else:
                reasons.append("降水不适宜")
            
            # 土壤水分适宜性评分
            moisture_min, moisture_max = crop_data['moisture_range']
            if moisture_min <= request.soil_moisture <= moisture_max:
                score += 40
                reasons.append("土壤水分适宜")
            elif abs(request.soil_moisture - (moisture_min + moisture_max) / 2) <= 5:
                score += 25
                reasons.append("土壤水分接近适宜范围")
            else:
                reasons.append("土壤水分不适宜")
            
            crop_scores[crop_name] = {
                'score': score,
                'reasons': reasons,
                'description': crop_data['description']
            }
            
            if score >= 60:  # 适宜性阈值
                suitable_crops.append(crop_name)
        
        # 按评分排序
        sorted_crops = sorted(crop_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        return {
            "status": "success",
            "location": request.location,
            "environmental_conditions": {
                "temperature": request.temperature,
                "precipitation": request.precipitation,
                "soil_moisture": request.soil_moisture,
                "soil_type": request.soil_type
            },
            "recommendations": {
                "highly_suitable": [crop for crop, data in sorted_crops if data['score'] >= 80],
                "suitable": [crop for crop, data in sorted_crops if 60 <= data['score'] < 80],
                "moderately_suitable": [crop for crop, data in sorted_crops if 40 <= data['score'] < 60]
            },
            "crop_details": crop_scores,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"作物推荐失败: {str(e)}")

@router.post("/yield/predict")
async def predict_crop_yield(request: YieldPredictionRequest):
    """
    预测作物产量
    
    Args:
        request: 产量预测请求参数
        
    Returns:
        dict: 预测结果
    """
    try:
        # 基于crop_yield_prediction项目的简化实现
        # 这里使用简化的产量预测模型
        
        # 基础产量（吨/公顷）
        base_yields = {
            'corn': 8.5,
            'wheat': 3.2,
            'rice': 4.8,
            'soybeans': 2.8,
            'barley': 4.1,
            'sorghum': 3.9
        }
        
        crop_type = request.crop_type.lower()
        if crop_type not in base_yields:
            raise HTTPException(status_code=400, detail=f"不支持的作物类型: {request.crop_type}")
        
        base_yield = base_yields[crop_type]
        
        # 环境因子调整
        weather = request.weather_conditions
        
        # 温度影响因子
        temp_factor = 1.0
        if 'temperature' in weather:
            temp = weather['temperature']
            if 15 <= temp <= 25:
                temp_factor = 1.2  # 最适温度
            elif 10 <= temp <= 30:
                temp_factor = 1.0  # 适宜温度
            else:
                temp_factor = 0.7  # 不适宜温度
        
        # 降水影响因子
        precip_factor = 1.0
        if 'precipitation' in weather:
            precip = weather['precipitation']
            if 200 <= precip <= 600:
                precip_factor = 1.1  # 适宜降水
            elif 100 <= precip <= 800:
                precip_factor = 1.0  # 可接受降水
            else:
                precip_factor = 0.8  # 不适宜降水
        
        # 土壤水分影响因子
        moisture_factor = 1.0
        if 'soil_moisture' in weather:
            moisture = weather['soil_moisture']
            if 20 <= moisture <= 50:
                moisture_factor = 1.1  # 适宜土壤水分
            elif 15 <= moisture <= 60:
                moisture_factor = 1.0  # 可接受土壤水分
            else:
                moisture_factor = 0.8  # 不适宜土壤水分
        
        # 计算预测产量
        predicted_yield = base_yield * temp_factor * precip_factor * moisture_factor
        
        # 不确定性估计
        uncertainty = predicted_yield * 0.15  # 15%的不确定性
        
        return {
            "status": "success",
            "crop_type": request.crop_type,
            "location": request.location,
            "planting_date": request.planting_date,
            "predicted_yield": {
                "value": round(predicted_yield, 2),
                "unit": "tonnes/hectare",
                "uncertainty": round(uncertainty, 2),
                "confidence_interval": [
                    round(predicted_yield - uncertainty, 2),
                    round(predicted_yield + uncertainty, 2)
                ]
            },
            "environmental_factors": {
                "temperature_factor": round(temp_factor, 2),
                "precipitation_factor": round(precip_factor, 2),
                "soil_moisture_factor": round(moisture_factor, 2)
            },
            "model_info": {
                "type": "Environmental Factor Model",
                "base_yield": base_yield,
                "uncertainty_level": "15%"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"产量预测失败: {str(e)}")

@router.get("/data/available-features")
async def get_available_features():
    """获取可用的农业数据特征"""
    try:
        # 尝试多个可能的数据路径
        data_paths = [
            "src/neuralhydrology/data/red_river_basin/timeseries.csv",
            "../../neuralhydrology/data/red_river_basin/timeseries.csv",
            "neuralhydrology/data/red_river_basin/timeseries.csv"
        ]
        
        df = None
        for path in data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"✅ 找到数据文件: {path}")
                break
        
        if df is None:
            # 如果没有找到数据文件，返回默认特征
            print("⚠️ 未找到数据文件，返回默认特征")
            return {
                "status": "success",
                "total_features": 5,
                "feature_categories": {
                    "weather": ["temperature", "precipitation", "wind_speed"],
                    "snow": ["snow_depth_mm", "snow_water_equivalent_mm"],
                    "temporal": ["date", "year", "month", "day"],
                    "other": ["soil_moisture"]
                },
                "all_features": ["temperature", "precipitation", "wind_speed", "snow_depth_mm", "snow_water_equivalent_mm", "date", "year", "month", "day", "soil_moisture"],
                "data_shape": [0, 10],
                "timestamp": datetime.now().isoformat()
            }
        
        features = df.columns.tolist()
        
        # 分类特征
        feature_categories = {
            "weather": [col for col in features if any(x in col.lower() for x in ['temp', 'precip', 'wind'])],
            "snow": [col for col in features if 'snow' in col.lower()],
            "temporal": [col for col in features if any(x in col.lower() for x in ['date', 'year', 'month', 'day'])],
            "other": [col for col in features if col not in [col for cat in ['weather', 'snow', 'temporal'] for col in feature_categories.get(cat, [])]]
        }
        
        return {
            "status": "success",
            "total_features": len(features),
            "feature_categories": feature_categories,
            "all_features": features,
            "data_shape": df.shape,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征失败: {str(e)}")

@router.get("/models/status")
async def get_models_status():
    """获取农业模型状态"""
    try:
        predictor, _ = get_agriculture_modules()
        
        model_status = {
            "soil_moisture_predictor": {
                "status": "ready" if predictor.model is not None else "not_trained",
                "config": predictor.config if hasattr(predictor, 'config') else None,
                "training_history": bool(predictor.training_history)
            }
        }
        
        return {
            "status": "success",
            "models": model_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型状态失败: {str(e)}")

@router.post("/models/train")
async def train_agriculture_models():
    """训练农业模型"""
    try:
        predictor, data_processor = get_agriculture_modules()
        
        # 加载数据
        data_path = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
        
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="数据文件不存在")
        
        # 准备数据
        X_train, y_train, X_val, y_val, X_test, y_test, scalers = \
            data_processor.prepare_soil_moisture_data(data_path)
        
        # 从data_processor获取input_size并设置到predictor
        predictor.config['input_size'] = data_processor.config['input_size']
        print(f"🔧 设置predictor input_size: {predictor.config['input_size']}")
        
        # 训练模型
        training_history = predictor.train_model(X_train, y_train, X_val, y_val)
        
        # 评估模型
        predictions, actual, metrics = predictor.evaluate_model(X_test, y_test, scalers[1])
        
        return {
            "status": "success",
            "message": "农业模型训练完成",
            "training_results": {
                "final_train_loss": training_history['train_losses'][-1],
                "final_val_loss": training_history['val_losses'][-1],
                "test_metrics": metrics
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")
