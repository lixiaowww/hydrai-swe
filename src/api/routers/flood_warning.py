#!/usr/bin/env python3
"""
洪水预警API端点
提供洪水风险评估和预警服务
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class FloodWarningService:
    """洪水预警服务类"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_path = "models/flood_warning_model.pkl"
        self.scaler_path = "models/flood_warning_scaler.pkl"
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("✅ 洪水预警模型加载成功")
                
                # 设置特征名称
                self.feature_names = [
                    'Snow on Grnd (cm)', 'snow_change', 'Max Temp (°C)', 'Min Temp (°C)', 
                    'Mean Temp (°C)', 'temp_anomaly', 'Total Rain (mm)', 'rain_cumulative',
                    '05OC001', 'flow_change', 'flow_anomaly', 'season_fall', 'season_winter',
                    'day_of_year_sin', 'day_of_year_cos'
                ]
                return True
            else:
                logger.warning("⚠️ 洪水预警模型文件不存在")
                return False
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def prepare_features_from_merged(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """从已合并的数据准备特征"""
        try:
            # 确保Date/Time列是datetime类型
            if 'Date/Time' in merged_data.columns:
                merged_data['Date/Time'] = pd.to_datetime(merged_data['Date/Time'])
            
            # 直接处理合并后的数据
            features = []
            
            # 积雪相关特征
            if 'Snow on Grnd (cm)' in merged_data.columns:
                features.append('Snow on Grnd (cm)')
                merged_data['snow_change'] = merged_data['Snow on Grnd (cm)'].diff()
                features.append('snow_change')
            
            # 温度相关特征
            if 'Max Temp (°C)' in merged_data.columns:
                features.append('Max Temp (°C)')
                features.append('Min Temp (°C)')
                features.append('Mean Temp (°C)')
                merged_data['temp_anomaly'] = merged_data['Mean Temp (°C)'] - merged_data['Mean Temp (°C)'].rolling(30).mean()
                features.append('temp_anomaly')
            
            # 降水相关特征
            if 'Total Rain (mm)' in merged_data.columns:
                features.append('Total Rain (mm)')
                merged_data['rain_cumulative'] = merged_data['Total Rain (mm)'].rolling(7).sum()
                features.append('rain_cumulative')
            
            # 径流相关特征
            flow_columns = [col for col in merged_data.columns if col.startswith('05OC')]
            if flow_columns:
                main_flow_column = flow_columns[0]
                features.append(main_flow_column)
                merged_data['flow_change'] = merged_data[main_flow_column].pct_change()
                features.append('flow_change')
                merged_data['flow_anomaly'] = merged_data[main_flow_column] / merged_data[main_flow_column].rolling(30).mean()
                features.append('flow_anomaly')
            
            # 季节性特征
            if 'Month' not in merged_data.columns:
                merged_data['Date/Time'] = pd.to_datetime(merged_data['Date/Time'])
                merged_data['Month'] = merged_data['Date/Time'].dt.month
            
            merged_data['season'] = merged_data['Month'].map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            })
            
            season_encoding = pd.get_dummies(merged_data['season'], prefix='season')
            merged_data = pd.concat([merged_data, season_encoding], axis=1)
            
            # 确保所有需要的季节列都存在
            for season in ['season_fall', 'season_winter']:
                if season not in merged_data.columns:
                    merged_data[season] = 0
            
            features.extend(['season_fall', 'season_winter'])
            
            # 时间特征
            if 'DayOfYear' not in merged_data.columns:
                merged_data['DayOfYear'] = merged_data['Date/Time'].dt.dayofyear
            
            merged_data['day_of_year_sin'] = np.sin(2 * np.pi * merged_data['DayOfYear'] / 365)
            merged_data['day_of_year_cos'] = np.cos(2 * np.pi * merged_data['DayOfYear'] / 365)
            features.extend(['day_of_year_sin', 'day_of_year_cos'])
            
            # 选择特征
            feature_data = merged_data[features].fillna(0)
            
            # 移除无穷值
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            feature_data = feature_data.fillna(0)
            
            return feature_data
            
        except Exception as e:
            logger.error(f"特征准备失败: {e}")
            raise

    def prepare_features(self, weather_data: pd.DataFrame, flow_data: pd.DataFrame) -> pd.DataFrame:
        """准备预测特征"""
        try:
            # 数据预处理
            weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])
            weather_data['Year'] = weather_data['Date/Time'].dt.year
            weather_data['Month'] = weather_data['Date/Time'].dt.month
            weather_data['Day'] = weather_data['Date/Time'].dt.day
            weather_data['DayOfYear'] = weather_data['Date/Time'].dt.dayofyear
            
            flow_data['date'] = pd.to_datetime(flow_data['date'])
            
            # 合并数据
            merged_data = pd.merge(
                weather_data, 
                flow_data, 
                left_on='Date/Time', 
                right_on='date', 
                how='inner'
            )
            
            # 特征工程
            features = []
            
            # 积雪相关特征
            if 'Snow on Grnd (cm)' in merged_data.columns:
                features.append('Snow on Grnd (cm)')
                merged_data['snow_change'] = merged_data['Snow on Grnd (cm)'].diff()
                features.append('snow_change')
            
            # 温度相关特征
            if 'Max Temp (°C)' in merged_data.columns:
                features.append('Max Temp (°C)')
                features.append('Min Temp (°C)')
                features.append('Mean Temp (°C)')
                merged_data['temp_anomaly'] = merged_data['Mean Temp (°C)'] - merged_data['Mean Temp (°C)'].rolling(30).mean()
                features.append('temp_anomaly')
            
            # 降水相关特征
            if 'Total Rain (mm)' in merged_data.columns:
                features.append('Total Rain (mm)')
                merged_data['rain_cumulative'] = merged_data['Total Rain (mm)'].rolling(7).sum()
                features.append('rain_cumulative')
            
            # 径流相关特征
            flow_columns = [col for col in merged_data.columns if col.startswith('05OC')]
            if flow_columns:
                main_flow_column = flow_columns[0]
                features.append(main_flow_column)
                merged_data['flow_change'] = merged_data[main_flow_column].pct_change()
                features.append('flow_change')
                merged_data['flow_anomaly'] = merged_data[main_flow_column] / merged_data[main_flow_column].rolling(30).mean()
                features.append('flow_anomaly')
            
            # 季节性特征
            merged_data['season'] = merged_data['Month'].map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            })
            
            season_encoding = pd.get_dummies(merged_data['season'], prefix='season')
            merged_data = pd.concat([merged_data, season_encoding], axis=1)
            
            # 确保所有需要的季节列都存在
            for season in ['season_fall', 'season_winter']:
                if season not in merged_data.columns:
                    merged_data[season] = 0
            
            features.extend(['season_fall', 'season_winter'])
            
            # 时间特征
            merged_data['day_of_year_sin'] = np.sin(2 * np.pi * merged_data['DayOfYear'] / 365)
            merged_data['day_of_year_cos'] = np.cos(2 * np.pi * merged_data['DayOfYear'] / 365)
            features.extend(['day_of_year_sin', 'day_of_year_cos'])
            
            # 选择特征
            feature_data = merged_data[features].fillna(0)
            
            # 移除无穷值
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            feature_data = feature_data.fillna(0)
            
            return feature_data
            
        except Exception as e:
            logger.error(f"特征准备失败: {e}")
            raise
    
    def predict_flood_risk(self, features_data: pd.DataFrame) -> Dict:
        """预测洪水风险"""
        try:
            if self.model is None or self.scaler is None:
                raise ValueError("模型未加载")
            
            # 数据标准化
            features_scaled = self.scaler.transform(features_data)
            
            # 预测
            risk_prediction = self.model.predict(features_scaled)
            risk_probability = self.model.predict_proba(features_scaled)[:, 1]
            
            return {
                'risk_level': risk_prediction.tolist(),
                'risk_probability': risk_probability.tolist()
            }
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise

# 创建服务实例
flood_service = FloodWarningService()

@router.get("/health")
async def health_check():
    """健康检查"""
    model_loaded = flood_service.model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/risk-assessment")
async def assess_flood_risk(
    days: int = Query(7, description="预测天数", ge=1, le=30),
    region: str = Query("red-river-basin", description="评估区域")
):
    """洪水风险评估"""
    try:
        if flood_service.model is None:
            raise HTTPException(status_code=503, detail="洪水预警模型未加载")
        
        # 加载优化后的数据
        optimized_path = "data/processed/flood_warning/flood_warning_optimized.csv"
        
        if os.path.exists(optimized_path):
            # 使用优化后的数据
            merged_data = pd.read_csv(optimized_path)
            # 直接使用合并后的数据，不需要再次合并
            features_data = flood_service.prepare_features_from_merged(merged_data)
        else:
            # 回退到原始数据
            weather_path = "data/raw/eccc_recent/eccc_recent_combined.csv"
            flow_path = "data/processed/hydat_streamflow_processed.csv"
            
            if not os.path.exists(weather_path) or not os.path.exists(flow_path):
                raise HTTPException(status_code=404, detail="数据文件不存在")
            
            weather_data = pd.read_csv(weather_path)
            flow_data = pd.read_csv(flow_path)
            
            # 准备特征
            features_data = flood_service.prepare_features(weather_data, flow_data)
        
        # 预测风险
        prediction_result = flood_service.predict_flood_risk(features_data)
        
        # 计算风险统计
        risk_levels = prediction_result['risk_level']
        risk_probs = prediction_result['risk_probability']
        
        high_risk_count = sum(1 for x in risk_levels if x == 1)
        total_count = len(risk_levels)
        avg_risk_prob = np.mean(risk_probs)
        
        # 确定整体风险等级 - 修复逻辑
        high_risk_ratio = high_risk_count / total_count if total_count > 0 else 0
        
        if high_risk_ratio > 0.2:  # 降低阈值到20%
            overall_risk = "HIGH"
        elif high_risk_ratio > 0.05:  # 降低阈值到5%
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"
        
        # 添加风险一致性检查
        if high_risk_count > 0 and overall_risk == "LOW":
            # 如果检测到高风险区域但整体风险为低，调整为中等风险
            overall_risk = "MEDIUM"
        
        return {
            "status": "success",
            "region": region,
            "assessment_date": datetime.now().isoformat(),
            "prediction_days": days,
            "overall_risk": overall_risk,
            "risk_statistics": {
                "total_samples": total_count,
                "high_risk_count": high_risk_count,
                "high_risk_percentage": round(high_risk_count / total_count * 100, 2),
                "average_risk_probability": round(avg_risk_prob * 100, 2)
            },
            "model_info": {
                "model_type": "Random Forest",
                "features_count": len(flood_service.feature_names),
                "training_samples": len(features_data) if features_data is not None else 0,
                "accuracy": "Model loaded successfully - accuracy based on training data"
            }
        }
        
    except Exception as e:
        logger.error(f"风险评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"风险评估失败: {str(e)}")

@router.get("/real-time-risk")
async def get_real_time_risk():
    """获取实时洪水风险"""
    try:
        if flood_service.model is None:
            raise HTTPException(status_code=503, detail="洪水预警模型未加载")
        
        # 加载优化后的数据
        optimized_path = "data/processed/flood_warning/flood_warning_optimized.csv"
        
        if os.path.exists(optimized_path):
            # 使用优化后的数据
            merged_data = pd.read_csv(optimized_path)
            # 获取最新一天的数据
            merged_data['Date/Time'] = pd.to_datetime(merged_data['Date/Time'])
            latest_date = merged_data['Date/Time'].max()
            latest_data = merged_data[merged_data['Date/Time'] == latest_date]
            
            if latest_data.empty:
                raise HTTPException(status_code=404, detail="最新数据不可用")
            
            # 准备特征
            features_data = flood_service.prepare_features_from_merged(latest_data)
        else:
            # 回退到原始数据
            weather_path = "data/raw/eccc_recent/eccc_recent_combined.csv"
            flow_path = "data/processed/hydat_streamflow_processed.csv"
            
            weather_data = pd.read_csv(weather_path)
            flow_data = pd.read_csv(flow_path)
            
            # 获取最新一天的数据
            weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])
            latest_date = weather_data['Date/Time'].max()
            latest_weather = weather_data[weather_data['Date/Time'] == latest_date]
            
            flow_data['date'] = pd.to_datetime(flow_data['date'])
            latest_flow = flow_data[flow_data['date'] == latest_date]
            
            if latest_weather.empty or latest_flow.empty:
                raise HTTPException(status_code=404, detail="最新数据不可用")
            
            # 准备特征
            features_data = flood_service.prepare_features(latest_weather, latest_flow)
        
        # 预测风险
        prediction_result = flood_service.predict_flood_risk(features_data)
        
        risk_level = prediction_result['risk_level'][0]
        risk_probability = prediction_result['risk_probability'][0]
        
        risk_status = "HIGH" if risk_level == 1 else "LOW"
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data_date": latest_date.isoformat(),
            "current_risk": {
                "level": risk_status,
                "probability": round(risk_probability * 100, 2),
                "description": {
                    "zh": "洪水风险高，需要关注" if risk_level == 1 else "洪水风险低，正常状态",
                    "en": "High flood risk, attention required" if risk_level == 1 else "Low flood risk, normal status",
                    "fr": "Risque d'inondation élevé, attention requise" if risk_level == 1 else "Risque d'inondation faible, état normal"
                }
            },
            "recommendation": {
                "action": {
                    "zh": "立即关注水位变化，准备应急措施" if risk_level == 1 else "继续正常监测",
                    "en": "Monitor water levels immediately and prepare emergency measures" if risk_level == 1 else "Continue normal monitoring",
                    "fr": "Surveiller immédiatement les niveaux d'eau et préparer les mesures d'urgence" if risk_level == 1 else "Continuer la surveillance normale"
                },
                "monitoring": {
                    "zh": "建议增加监测频率" if risk_level == 1 else "保持常规监测频率",
                    "en": "Recommend increasing monitoring frequency" if risk_level == 1 else "Maintain regular monitoring frequency",
                    "fr": "Recommandé d'augmenter la fréquence de surveillance" if risk_level == 1 else "Maintenir la fréquence de surveillance régulière"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"实时风险评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"实时风险评估失败: {str(e)}")

@router.get("/flood-timeline")
async def get_flood_timeline(
    days: int = Query(30, description="预测天数", ge=7, le=90),
    region: str = Query("red-river-basin", description="评估区域")
):
    """获取洪水预警时间线"""
    try:
        if flood_service.model is None:
            raise HTTPException(status_code=503, detail="洪水预警模型未加载")
        
        # 加载优化后的数据
        optimized_path = "data/processed/flood_warning/flood_warning_optimized.csv"
        
        if os.path.exists(optimized_path):
            # 使用优化后的数据
            merged_data = pd.read_csv(optimized_path)
            # 直接使用合并后的数据，不需要再次合并
            features_data = flood_service.prepare_features_from_merged(merged_data)
        else:
            # 回退到原始数据
            weather_path = "data/raw/eccc_recent/eccc_recent_combined.csv"
            flow_path = "data/processed/hydat_streamflow_processed.csv"
            
            if not os.path.exists(weather_path) or not os.path.exists(flow_path):
                raise HTTPException(status_code=404, detail="数据文件不存在")
            
            weather_data = pd.read_csv(weather_path)
            flow_data = pd.read_csv(flow_path)
            
            # 准备特征
            features_data = flood_service.prepare_features(weather_data, flow_data)
        
        # 预测风险
        prediction_result = flood_service.predict_flood_risk(features_data)
        
        # 生成时间线数据
        timeline_data = []
        for i, (risk_level, risk_prob) in enumerate(zip(
            prediction_result['risk_level'], 
            prediction_result['risk_probability']
        )):
            # 计算日期（假设数据是按时间顺序排列的）
            if os.path.exists(optimized_path):
                if i < len(merged_data):
                    date = merged_data.iloc[i]['Date/Time']
                else:
                    date = datetime.now() + timedelta(days=i)
            else:
                if i < len(weather_data):
                    date = weather_data.iloc[i]['Date/Time']
                else:
                    date = datetime.now() + timedelta(days=i)
            
            timeline_data.append({
                "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                "risk_level": "HIGH" if risk_level == 1 else "LOW",
                "risk_probability": round(risk_prob * 100, 2),
                "alert_level": "WARNING" if risk_prob > 0.7 else "INFO" if risk_prob > 0.3 else "NORMAL"
            })
        
        return {
            "status": "success",
            "region": region,
            "timeline": timeline_data[:days],  # 只返回请求的天数
            "summary": {
                "total_days": len(timeline_data[:days]),
                "high_risk_days": sum(1 for x in timeline_data[:days] if x['risk_level'] == 'HIGH'),
                "average_risk": round(np.mean([x['risk_probability'] for x in timeline_data[:days]]), 2)
            }
        }
        
    except Exception as e:
        logger.error(f"洪水预警时间线生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"洪水预警时间线生成失败: {str(e)}")

@router.get("/flood-history")
async def get_flood_history(
    start_date: str = Query("2020-01-01", description="开始日期"),
    end_date: str = Query(None, description="结束日期"),
    region: str = Query("red-river-basin", description="评估区域")
):
    """获取历史洪水事件分析"""
    try:
        if flood_service.model is None:
            raise HTTPException(status_code=503, detail="洪水预警模型未加载")
        
        # 设置默认结束日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 加载优化后的数据
        optimized_path = "data/processed/flood_warning/flood_warning_optimized.csv"
        
        if os.path.exists(optimized_path):
            # 使用优化后的数据
            merged_data = pd.read_csv(optimized_path)
            # 过滤日期范围
            merged_data['Date/Time'] = pd.to_datetime(merged_data['Date/Time'])
            
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            filtered_data = merged_data[
                (merged_data['Date/Time'] >= start_dt) & 
                (merged_data['Date/Time'] <= end_dt)
            ]
            
            # 准备特征
            features_data = flood_service.prepare_features_from_merged(filtered_data)
        else:
            # 回退到原始数据
            weather_path = "data/raw/eccc_recent/eccc_recent_combined.csv"
            flow_path = "data/processed/hydat_streamflow_processed.csv"
            
            if not os.path.exists(weather_path) or not os.path.exists(flow_path):
                raise HTTPException(status_code=404, detail="数据文件不存在")
            
            weather_data = pd.read_csv(weather_path)
            flow_data = pd.read_csv(flow_path)
            
            # 过滤日期范围
            weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])
            flow_data['date'] = pd.to_datetime(flow_data['date'])
            
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            weather_filtered = weather_data[
                (weather_data['Date/Time'] >= start_dt) & 
                (weather_data['Date/Time'] <= end_dt)
            ]
            flow_filtered = flow_data[
                (flow_data['date'] >= start_dt) & 
                (flow_data['date'] <= end_dt)
            ]
            
            # 准备特征
            features_data = flood_service.prepare_features(weather_filtered, flow_filtered)
        
        # 预测风险
        prediction_result = flood_service.predict_flood_risk(features_data)
        
        # 分析历史模式
        high_risk_periods = []
        for i, (risk_level, risk_prob) in enumerate(zip(
            prediction_result['risk_level'], 
            prediction_result['risk_probability']
        )):
            if risk_level == 1 and risk_prob > 0.8:  # 高风险且概率>80%
                if os.path.exists(optimized_path):
                    if i < len(filtered_data):
                        date = filtered_data.iloc[i]['Date/Time']
                        high_risk_periods.append({
                            "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                            "risk_probability": round(risk_prob * 100, 2),
                            "severity": "CRITICAL" if risk_prob > 0.9 else "HIGH"
                        })
                else:
                    if i < len(weather_filtered):
                        date = weather_filtered.iloc[i]['Date/Time']
                        high_risk_periods.append({
                            "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                            "risk_probability": round(risk_prob * 100, 2),
                            "severity": "CRITICAL" if risk_prob > 0.9 else "HIGH"
                        })
        
        return {
            "status": "success",
            "region": region,
            "period": {"start": start_date, "end": end_date},
            "high_risk_events": high_risk_periods,
            "statistics": {
                "total_days": len(prediction_result['risk_level']),
                "high_risk_days": sum(1 for x in prediction_result['risk_level'] if x == 1),
                "critical_events": len([x for x in high_risk_periods if x['severity'] == 'CRITICAL']),
                "average_risk": round(np.mean(prediction_result['risk_probability']) * 100, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"历史洪水事件分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"历史洪水事件分析失败: {str(e)}")

@router.get("/model-status")
async def get_model_status():
    """获取模型状态"""
    try:
        model_loaded = flood_service.model is not None
        scaler_loaded = flood_service.scaler is not None
        
        return {
            "status": "success",
            "model": {
                "loaded": model_loaded,
                "type": "Random Forest Classifier" if model_loaded else "None",
                "features": len(flood_service.feature_names) if model_loaded else 0,
                "training_samples": 650 if model_loaded else 0
            },
            "scaler": {
                "loaded": scaler_loaded,
                "type": "StandardScaler" if scaler_loaded else "None"
            },
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取模型状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型状态失败: {str(e)}")
