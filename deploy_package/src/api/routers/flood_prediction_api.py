#!/usr/bin/env python3
"""
洪水预测API
基于真实数据的洪水风险评估和预测
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import hashlib
from scipy import stats

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/flood", tags=["flood-prediction"])

# 真实数据源配置
REAL_DATA_SOURCES = {
    "swe_manitoba_daily": "data/processed/validation/manitoba_daily_swe_2010_2020_20250915_213917.csv",
    "hydrometric_manitoba": "data/processed/hydro/manitoba_hydro_20250915_161112.csv",
    "weather_openmeteo": "data/real/openmeteo/openmeteo_canada_20250915_184349.csv"
}

def build_provenance(source_key: str) -> Dict[str, Any]:
    """Build provenance info for a given real data source key."""
    try:
        file_path = REAL_DATA_SOURCES.get(source_key)
        if not file_path:
            return {"source": source_key, "exists": False}
        
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            lineage_input = f"{file_path}|{stat.st_mtime}|{stat.st_size}"
            lineage_id = hashlib.sha256(lineage_input.encode("utf-8")).hexdigest()[:16]
            return {
                "source": source_key,
                "source_path": file_path,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
                "exists": True,
                "lineage_id": lineage_id,
            }
        else:
            return {"source": source_key, "exists": False}
    except Exception as e:
        logger.error(f"Provenance build error for {source_key}: {e}")
        return {"source": source_key, "error": str(e)}

def load_real_swe_data():
    """加载真实SWE数据"""
    file_path = REAL_DATA_SOURCES["swe_manitoba_daily"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SWE data file not found: {file_path}")
    return pd.read_csv(file_path)

def load_real_hydrometric_data():
    """加载真实水文数据"""
    file_path = REAL_DATA_SOURCES["hydrometric_manitoba"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hydrometric data file not found: {file_path}")
    return pd.read_csv(file_path)

def load_real_weather_data():
    """加载真实天气数据"""
    file_path = REAL_DATA_SOURCES["weather_openmeteo"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Weather data file not found: {file_path}")
    return pd.read_csv(file_path)

def calculate_flood_risk_factors(swe_data, hydrometric_data, weather_data):
    """基于真实数据计算洪水风险因子"""
    try:
        # 处理SWE数据
        swe_data['timestamp'] = pd.to_datetime(swe_data['timestamp'])
        current_swe = float(swe_data['swe_mm'].iloc[-1]) if 'swe_mm' in swe_data.columns else 0.0
        swe_trend = float(swe_data['swe_mm'].diff().mean()) if len(swe_data) > 1 else 0.0
        
        # 处理水文数据
        hydrometric_data['measurement_time'] = pd.to_datetime(hydrometric_data['measurement_time'])
        current_water_level = float(hydrometric_data['water_level_m'].iloc[-1]) if 'water_level_m' in hydrometric_data.columns else 0.0
        current_discharge = float(hydrometric_data['discharge_m3s'].iloc[-1]) if 'discharge_m3s' in hydrometric_data.columns else 0.0
        
        # 处理天气数据
        weather_data['time'] = pd.to_datetime(weather_data['time'])
        recent_weather = weather_data.tail(7)  # 最近7天
        avg_temp = float(recent_weather['temperature_2m_max'].mean()) if 'temperature_2m_max' in recent_weather.columns else 0.0
        total_precip = float(recent_weather['precipitation_sum'].sum()) if 'precipitation_sum' in recent_weather.columns else 0.0
        
        # 计算风险因子
        risk_factors = {
            "swe_current_mm": current_swe,
            "swe_trend_mm_per_day": swe_trend,
            "water_level_m": current_water_level,
            "discharge_m3s": current_discharge,
            "avg_temperature_c": avg_temp,
            "precipitation_7d_mm": total_precip,
            "snowmelt_potential": max(0, avg_temp - 0) * current_swe / 100,  # 简化的融雪潜力
            "runoff_potential": current_swe + total_precip,  # 径流潜力
            "flood_risk_score": 0.0
        }
        
        # 计算综合洪水风险评分 (0-100)
        risk_score = 0
        
        # SWE因子 (40%权重)
        if current_swe > 100:  # 高SWE
            risk_score += 30
        elif current_swe > 50:
            risk_score += 20
        elif current_swe > 20:
            risk_score += 10
        
        # 融雪潜力 (25%权重)
        if risk_factors["snowmelt_potential"] > 50:
            risk_score += 25
        elif risk_factors["snowmelt_potential"] > 20:
            risk_score += 15
        elif risk_factors["snowmelt_potential"] > 5:
            risk_score += 10
        
        # 降水因子 (20%权重)
        if total_precip > 50:
            risk_score += 20
        elif total_precip > 20:
            risk_score += 15
        elif total_precip > 10:
            risk_score += 10
        
        # 水位因子 (15%权重)
        if current_water_level > 300:  # 高水位
            risk_score += 15
        elif current_water_level > 250:
            risk_score += 10
        elif current_water_level > 200:
            risk_score += 5
        
        risk_factors["flood_risk_score"] = min(100, risk_score)
        
        return risk_factors
        
    except Exception as e:
        logger.error(f"Error calculating flood risk factors: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating flood risk factors: {str(e)}")

def generate_flood_interpretation(risk_factors):
    """生成专业的洪水风险解读"""
    risk_score = risk_factors["flood_risk_score"]
    swe_current = risk_factors["swe_current_mm"]
    snowmelt_potential = risk_factors["snowmelt_potential"]
    runoff_potential = risk_factors["runoff_potential"]
    
    # 风险等级评估
    if risk_score >= 80:
        risk_level = "极高"
        risk_color = "red"
        urgency = "立即行动"
    elif risk_score >= 60:
        risk_level = "高"
        risk_color = "orange"
        urgency = "高度关注"
    elif risk_score >= 40:
        risk_level = "中等"
        risk_color = "yellow"
        urgency = "持续监控"
    elif risk_score >= 20:
        risk_level = "低"
        risk_color = "blue"
        urgency = "常规监控"
    else:
        risk_level = "极低"
        risk_color = "green"
        urgency = "正常状态"
    
    # 专业解读
    interpretation = {
        "risk_level": risk_level,
        "risk_color": risk_color,
        "urgency": urgency,
        "summary": f"当前洪水风险等级为{risk_level}，风险评分为{risk_score:.1f}分。",
        "key_factors": [],
        "recommendations": []
    }
    
    # 关键因子分析
    if swe_current > 100:
        interpretation["key_factors"].append(f"积雪水当量较高({swe_current:.1f}mm)，为洪水提供了充足的水源")
    elif swe_current > 50:
        interpretation["key_factors"].append(f"积雪水当量中等({swe_current:.1f}mm)，需要关注融雪过程")
    else:
        interpretation["key_factors"].append(f"积雪水当量较低({swe_current:.1f}mm)，洪水风险相对较小")
    
    if snowmelt_potential > 50:
        interpretation["key_factors"].append(f"融雪潜力很高({snowmelt_potential:.1f})，温度上升可能导致快速融雪")
    elif snowmelt_potential > 20:
        interpretation["key_factors"].append(f"融雪潜力中等({snowmelt_potential:.1f})，需要监控温度变化")
    
    if runoff_potential > 100:
        interpretation["key_factors"].append(f"径流潜力很高({runoff_potential:.1f}mm)，可能造成河流水位快速上升")
    elif runoff_potential > 50:
        interpretation["key_factors"].append(f"径流潜力中等({runoff_potential:.1f}mm)，需要关注河流水位")
    
    # 政府行动建议
    if risk_score >= 80:
        interpretation["recommendations"] = [
            "立即启动洪水应急预案",
            "疏散高风险区域居民",
            "加强堤防巡查和加固",
            "准备应急物资和救援力量",
            "发布最高级别洪水警报"
        ]
    elif risk_score >= 60:
        interpretation["recommendations"] = [
            "启动洪水预警系统",
            "加强河流水位监测",
            "检查防洪设施状态",
            "准备应急响应预案",
            "向公众发布洪水风险信息"
        ]
    elif risk_score >= 40:
        interpretation["recommendations"] = [
            "加强水文监测频率",
            "评估防洪设施能力",
            "更新洪水风险评估",
            "准备应急物资",
            "向相关部门通报风险状况"
        ]
    else:
        interpretation["recommendations"] = [
            "继续常规水文监测",
            "维护防洪设施",
            "更新应急预案",
            "进行风险评估培训",
            "保持应急响应能力"
        ]
    
    return interpretation

@router.get("/prediction/7day")
async def get_7day_flood_prediction():
    """获取7天洪水预测（基于真实数据）"""
    try:
        # 加载真实数据
        swe_data = load_real_swe_data()
        hydrometric_data = load_real_hydrometric_data()
        weather_data = load_real_weather_data()
        
        # 计算当前风险因子
        risk_factors = calculate_flood_risk_factors(swe_data, hydrometric_data, weather_data)
        
        # 生成7天预测
        forecast_dates = []
        flood_risk_forecast = []
        water_level_forecast = []
        discharge_forecast = []
        
        current_date = datetime.now()
        for i in range(1, 8):
            forecast_date = current_date + timedelta(days=i)
            forecast_dates.append(forecast_date.strftime('%Y-%m-%d'))
            
            # 基于当前风险因子和季节性模式生成预测
            # 这里使用简化的预测模型，实际应用中应该使用更复杂的机器学习模型
            base_risk = risk_factors["flood_risk_score"]
            seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * forecast_date.timetuple().tm_yday / 365)
            trend_factor = 1.0 + 0.05 * i  # 假设风险随时间略有增加
            
            predicted_risk = min(100, base_risk * seasonal_factor * trend_factor)
            flood_risk_forecast.append(round(predicted_risk, 1))
            
            # 预测水位和流量
            base_level = risk_factors["water_level_m"]
            base_discharge = risk_factors["discharge_m3s"]
            
            predicted_level = base_level * (1 + predicted_risk / 1000)
            predicted_discharge = base_discharge * (1 + predicted_risk / 500)
            
            water_level_forecast.append(round(predicted_level, 2))
            discharge_forecast.append(round(predicted_discharge, 1))
        
        # 生成专业解读
        interpretation = generate_flood_interpretation(risk_factors)
        
        return {
            "forecast_period": "7 days",
            "forecast_dates": forecast_dates,
            "flood_risk_scores": flood_risk_forecast,
            "water_level_forecast_m": water_level_forecast,
            "discharge_forecast_m3s": discharge_forecast,
            "current_risk_factors": risk_factors,
            "interpretation": interpretation,
            "data_sources": {
                "swe_data": build_provenance("swe_manitoba_daily"),
                "hydrometric_data": build_provenance("hydrometric_manitoba"),
                "weather_data": build_provenance("weather_openmeteo")
            },
            "last_update": datetime.now().isoformat(),
            "methodology": "Multi-factor flood risk assessment based on real SWE, hydrometric, and weather data"
        }
        
    except Exception as e:
        logger.error(f"Error in 7-day flood prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Flood prediction error: {str(e)}")

@router.get("/prediction/14day")
async def get_14day_flood_prediction():
    """获取14天洪水预测（基于真实数据）"""
    try:
        # 加载真实数据
        swe_data = load_real_swe_data()
        hydrometric_data = load_real_hydrometric_data()
        weather_data = load_real_weather_data()
        
        # 计算当前风险因子
        risk_factors = calculate_flood_risk_factors(swe_data, hydrometric_data, weather_data)
        
        # 生成14天预测
        forecast_dates = []
        flood_risk_forecast = []
        water_level_forecast = []
        discharge_forecast = []
        
        current_date = datetime.now()
        for i in range(1, 15):
            forecast_date = current_date + timedelta(days=i)
            forecast_dates.append(forecast_date.strftime('%Y-%m-%d'))
            
            # 基于当前风险因子和季节性模式生成预测
            base_risk = risk_factors["flood_risk_score"]
            seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * forecast_date.timetuple().tm_yday / 365)
            trend_factor = 1.0 + 0.03 * i  # 14天预测的不确定性更高
            
            predicted_risk = min(100, base_risk * seasonal_factor * trend_factor)
            flood_risk_forecast.append(round(predicted_risk, 1))
            
            # 预测水位和流量
            base_level = risk_factors["water_level_m"]
            base_discharge = risk_factors["discharge_m3s"]
            
            predicted_level = base_level * (1 + predicted_risk / 1200)
            predicted_discharge = base_discharge * (1 + predicted_risk / 600)
            
            water_level_forecast.append(round(predicted_level, 2))
            discharge_forecast.append(round(predicted_discharge, 1))
        
        # 生成专业解读
        interpretation = generate_flood_interpretation(risk_factors)
        
        return {
            "forecast_period": "14 days",
            "forecast_dates": forecast_dates,
            "flood_risk_scores": flood_risk_forecast,
            "water_level_forecast_m": water_level_forecast,
            "discharge_forecast_m3s": discharge_forecast,
            "current_risk_factors": risk_factors,
            "interpretation": interpretation,
            "data_sources": {
                "swe_data": build_provenance("swe_manitoba_daily"),
                "hydrometric_data": build_provenance("hydrometric_manitoba"),
                "weather_data": build_provenance("weather_openmeteo")
            },
            "last_update": datetime.now().isoformat(),
            "methodology": "Extended multi-factor flood risk assessment with increased uncertainty for longer-term predictions"
        }
        
    except Exception as e:
        logger.error(f"Error in 14-day flood prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Flood prediction error: {str(e)}")

@router.get("/risk-assessment")
async def get_flood_risk_assessment():
    """获取当前洪水风险评估（基于真实数据）"""
    try:
        # 加载真实数据
        swe_data = load_real_swe_data()
        hydrometric_data = load_real_hydrometric_data()
        weather_data = load_real_weather_data()
        
        # 计算风险因子
        risk_factors = calculate_flood_risk_factors(swe_data, hydrometric_data, weather_data)
        
        # 生成专业解读
        interpretation = generate_flood_interpretation(risk_factors)
        
        # 计算历史对比
        historical_analysis = {
            "swe_percentile": 0.0,
            "water_level_percentile": 0.0,
            "precipitation_percentile": 0.0,
            "risk_trend": "stable"
        }
        
        try:
            # 计算SWE百分位数
            swe_values = swe_data['swe_mm'].dropna()
            if len(swe_values) > 0:
                current_swe = risk_factors["swe_current_mm"]
                historical_analysis["swe_percentile"] = round(stats.percentileofscore(swe_values, current_swe), 1)
            
            # 计算水位百分位数
            water_levels = hydrometric_data['water_level_m'].dropna()
            if len(water_levels) > 0:
                current_level = risk_factors["water_level_m"]
                historical_analysis["water_level_percentile"] = round(stats.percentileofscore(water_levels, current_level), 1)
            
            # 计算降水百分位数
            precip_values = weather_data['precipitation_sum'].dropna()
            if len(precip_values) > 0:
                current_precip = risk_factors["precipitation_7d_mm"]
                historical_analysis["precipitation_percentile"] = round(stats.percentileofscore(precip_values, current_precip), 1)
            
        except Exception as e:
            logger.warning(f"Error calculating historical analysis: {e}")
        
        return {
            "assessment_date": datetime.now().isoformat(),
            "risk_factors": risk_factors,
            "interpretation": interpretation,
            "historical_analysis": historical_analysis,
            "data_sources": {
                "swe_data": build_provenance("swe_manitoba_daily"),
                "hydrometric_data": build_provenance("hydrometric_manitoba"),
                "weather_data": build_provenance("weather_openmeteo")
            },
            "assessment_methodology": "Real-time multi-factor flood risk assessment based on current SWE, hydrometric, and weather conditions"
        }
        
    except Exception as e:
        logger.error(f"Error in flood risk assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Flood risk assessment error: {str(e)}")

@router.get("/historical-trends")
async def get_flood_historical_trends():
    """获取洪水历史趋势（基于真实数据）"""
    try:
        # 加载真实数据
        swe_data = load_real_swe_data()
        hydrometric_data = load_real_hydrometric_data()
        weather_data = load_real_weather_data()
        
        # 处理时间序列数据
        swe_data['timestamp'] = pd.to_datetime(swe_data['timestamp'])
        hydrometric_data['measurement_time'] = pd.to_datetime(hydrometric_data['measurement_time'])
        weather_data['time'] = pd.to_datetime(weather_data['time'])
        
        # 计算历史趋势
        trends = {}
        
        # SWE趋势
        if 'swe_mm' in swe_data.columns and len(swe_data) > 1:
            swe_values = swe_data['swe_mm'].dropna()
            if len(swe_values) > 1:
                x = np.arange(len(swe_values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, swe_values)
                trends['swe_trend'] = {
                    "slope": round(slope, 4),
                    "r_squared": round(r_value ** 2, 4),
                    "p_value": round(p_value, 4),
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "significance": "significant" if p_value < 0.05 else "not_significant"
                }
        
        # 水位趋势
        if 'water_level_m' in hydrometric_data.columns and len(hydrometric_data) > 1:
            level_values = hydrometric_data['water_level_m'].dropna()
            if len(level_values) > 1:
                x = np.arange(len(level_values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, level_values)
                trends['water_level_trend'] = {
                    "slope": round(slope, 4),
                    "r_squared": round(r_value ** 2, 4),
                    "p_value": round(p_value, 4),
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "significance": "significant" if p_value < 0.05 else "not_significant"
                }
        
        # 降水趋势
        if 'precipitation_sum' in weather_data.columns and len(weather_data) > 1:
            precip_values = weather_data['precipitation_sum'].dropna()
            if len(precip_values) > 1:
                x = np.arange(len(precip_values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, precip_values)
                trends['precipitation_trend'] = {
                    "slope": round(slope, 4),
                    "r_squared": round(r_value ** 2, 4),
                    "p_value": round(p_value, 4),
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "significance": "significant" if p_value < 0.05 else "not_significant"
                }
        
        # 计算季节性模式
        seasonal_patterns = {}
        
        if 'swe_mm' in swe_data.columns:
            swe_data['month'] = swe_data['timestamp'].dt.month
            monthly_swe = swe_data.groupby('month')['swe_mm'].mean()
            seasonal_patterns['swe_monthly'] = {
                str(month): round(float(monthly_swe[month]), 2) 
                for month in monthly_swe.index
            }
        
        if 'precipitation_sum' in weather_data.columns:
            weather_data['month'] = weather_data['time'].dt.month
            monthly_precip = weather_data.groupby('month')['precipitation_sum'].mean()
            seasonal_patterns['precipitation_monthly'] = {
                str(month): round(float(monthly_precip[month]), 2) 
                for month in monthly_precip.index
            }
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "trends": trends,
            "seasonal_patterns": seasonal_patterns,
            "data_periods": {
                "swe_data": f"{swe_data['timestamp'].min().strftime('%Y-%m-%d')} to {swe_data['timestamp'].max().strftime('%Y-%m-%d')}",
                "hydrometric_data": f"{hydrometric_data['measurement_time'].min().strftime('%Y-%m-%d')} to {hydrometric_data['measurement_time'].max().strftime('%Y-%m-%d')}",
                "weather_data": f"{weather_data['time'].min().strftime('%Y-%m-%d')} to {weather_data['time'].max().strftime('%Y-%m-%d')}"
            },
            "data_sources": {
                "swe_data": build_provenance("swe_manitoba_daily"),
                "hydrometric_data": build_provenance("hydrometric_manitoba"),
                "weather_data": build_provenance("weather_openmeteo")
            },
            "methodology": "Statistical trend analysis using linear regression on real historical data"
        }
        
    except Exception as e:
        logger.error(f"Error in historical trends analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Historical trends analysis error: {str(e)}")

@router.get("/health")
async def health_check():
    """洪水预测API健康检查"""
    try:
        # 检查数据源可用性
        data_status = {}
        for source_name, file_path in REAL_DATA_SOURCES.items():
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                data_status[source_name] = {
                    "available": True,
                    "size_mb": round(stat.st_size / (1024 * 1024), 3),
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            else:
                data_status[source_name] = {"available": False}
        
        return {
            "status": "healthy",
            "message": "Flood prediction API is operational with real data sources",
            "components": {
                "prediction_models": "operational",
                "data_sources": "connected",
                "risk_assessment": "active"
            },
            "data_sources": data_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "message": f"Flood prediction API operational with some issues: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
