#!/usr/bin/env python3
"""
高级洪水预警API端点
集成RNN神经网络、聚类分析和高级预警功能
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
import logging

# 导入高级洪水预警系统
from models.advanced_flood_warning import advanced_flood_system

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        model_loaded = advanced_flood_system.model is not None
        cluster_loaded = advanced_flood_system.cluster_model is not None
        
        return {
            "status": "healthy" if model_loaded and cluster_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "cluster_model_loaded": cluster_loaded,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/advanced-risk-assessment")
async def advanced_flood_risk_assessment(
    days: int = Query(7, description="预测天数", ge=1, le=30),
    region: str = Query("red-river-basin", description="评估区域")
):
    """高级洪水风险评估"""
    try:
        if advanced_flood_system.model is None:
            raise HTTPException(status_code=503, detail="高级洪水预警模型未加载")
        
        # 优先加载测试数据，然后是优化数据
        test_data_path = "data/processed/flood_warning/flood_warning_test_data.csv"
        optimized_path = "data/processed/flood_warning/flood_warning_optimized.csv"
        
        if os.path.exists(test_data_path):
            logger.info("使用测试数据")
            merged_data = pd.read_csv(test_data_path)
        elif os.path.exists(optimized_path):
            logger.info("使用优化数据")
            merged_data = pd.read_csv(optimized_path)
        else:
            raise HTTPException(status_code=404, detail="数据文件不存在")
        
        # 准备高级特征
        features_data = advanced_flood_system.prepare_advanced_features(merged_data)
        
        # 预测高级风险
        prediction_result = advanced_flood_system.predict_advanced_flood_risk(features_data)
        
        # 生成综合报告
        comprehensive_report = advanced_flood_system.generate_comprehensive_report(
            features_data, prediction_result
        )
        
        return {
            "status": "success",
            "region": region,
            "assessment_date": datetime.now().isoformat(),
            "prediction_days": days,
            "comprehensive_report": comprehensive_report,
            "advanced_metrics": prediction_result['advanced_metrics'],
            "alert_levels": prediction_result['alert_levels'][:days],  # 只返回请求的天数
            "cluster_analysis": prediction_result['cluster_analysis'],
            "trend_analysis": prediction_result['trend_analysis']
        }
        
    except Exception as e:
        logger.error(f"高级风险评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"高级风险评估失败: {str(e)}")

@router.get("/real-time-advanced-risk")
async def get_real_time_advanced_risk():
    """获取实时高级洪水风险"""
    try:
        if advanced_flood_system.model is None:
            raise HTTPException(status_code=503, detail="高级洪水预警模型未加载")
        
        # 优先加载修复后的测试数据，然后是优化数据
        fixed_test_path = "data/processed/flood_warning/flood_warning_fixed_features.csv"
        test_data_path = "data/processed/flood_warning/flood_warning_test_data.csv"
        optimized_path = "data/processed/flood_warning/flood_warning_optimized.csv"
        
        if os.path.exists(fixed_test_path):
            logger.info("使用修复后的测试数据")
            merged_data = pd.read_csv(fixed_test_path)
        elif os.path.exists(test_data_path):
            logger.info("使用测试数据")
            merged_data = pd.read_csv(test_data_path)
        elif os.path.exists(optimized_path):
            logger.info("使用优化数据")
            merged_data = pd.read_csv(optimized_path)
        else:
            raise HTTPException(status_code=404, detail="数据文件不存在")
        
        # 获取最新一天的数据
        merged_data['Date/Time'] = pd.to_datetime(merged_data['Date/Time'])
        latest_date = merged_data['Date/Time'].max()
        latest_data = merged_data[merged_data['Date/Time'] == latest_date]
        
        if latest_data.empty:
            raise HTTPException(status_code=404, detail="最新数据不可用")
        
        # 准备高级特征
        features_data = advanced_flood_system.prepare_advanced_features(latest_data)
        
        # 预测高级风险
        prediction_result = advanced_flood_system.predict_advanced_flood_risk(features_data)
        
        # 获取最新风险信息
        latest_risk = prediction_result['risk_level'][0]
        latest_probability = prediction_result['risk_probability'][0]
        latest_alert = prediction_result['alert_levels'][0]
        
        # 基于概率值判断风险等级，而不是分类结果
        risk_status = "HIGH" if latest_probability >= 0.5 else "LOW"
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data_date": latest_date.isoformat(),
            "current_risk": {
                "level": risk_status,
                "probability": round(latest_probability * 100, 2),
                "alert_level": latest_alert['alert_level'],
                "alert_color": latest_alert['color'],
                "action": latest_alert['action'],
                "intensity": latest_alert['intensity']
            },
            "advanced_metrics": {
                "risk_stability": prediction_result['advanced_metrics'].get('risk_stability', {}).get('stability_level', 'UNKNOWN'),
                "cluster_count": len(prediction_result['cluster_analysis']),
                "trend_direction": prediction_result['trend_analysis'].get('risk_trend', {}).get('direction', 'UNKNOWN')
            },
            "recommendations": advanced_flood_system._generate_recommendations(prediction_result)
        }
        
    except Exception as e:
        logger.error(f"实时高级风险评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"实时高级风险评估失败: {str(e)}")

@router.get("/advanced-timeline")
async def get_advanced_flood_timeline(
    days: int = Query(30, description="预测天数", ge=7, le=90),
    region: str = Query("red-river-basin", description="评估区域")
):
    """获取高级洪水预警时间线"""
    try:
        if advanced_flood_system.model is None:
            raise HTTPException(status_code=503, detail="高级洪水预警模型未加载")
        
        # 优先加载修复后的测试数据，然后是优化数据
        fixed_test_path = "data/processed/flood_warning/flood_warning_fixed_features.csv"
        test_data_path = "data/processed/flood_warning/flood_warning_test_data.csv"
        optimized_path = "data/processed/flood_warning/flood_warning_optimized.csv"
        
        if os.path.exists(fixed_test_path):
            logger.info("使用修复后的测试数据")
            merged_data = pd.read_csv(fixed_test_path)
        elif os.path.exists(test_data_path):
            logger.info("使用测试数据")
            merged_data = pd.read_csv(test_data_path)
        elif os.path.exists(optimized_path):
            logger.info("使用优化数据")
            merged_data = pd.read_csv(optimized_path)
        else:
            raise HTTPException(status_code=404, detail="数据文件不存在")
        
        # 准备高级特征
        features_data = advanced_flood_system.prepare_advanced_features(merged_data)
        
        # 预测高级风险
        prediction_result = advanced_flood_system.predict_advanced_flood_risk(features_data)
        
        # 生成高级时间线数据
        timeline_data = []
        for i, (risk_level, risk_prob, alert) in enumerate(zip(
            prediction_result['risk_level'], 
            prediction_result['risk_probability'],
            prediction_result['alert_levels']
        )):
            # 计算日期
            if i < len(merged_data):
                date = merged_data.iloc[i]['Date/Time']
            else:
                date = datetime.now() + timedelta(days=i)
            
            # 获取聚类信息
            cluster_info = None
            if 'cluster_label' in features_data.columns:
                cluster_id = features_data.iloc[i]['cluster_label'] if i < len(features_data) else 0
                cluster_info = prediction_result['cluster_analysis'].get(f'cluster_{int(cluster_id)}', {})
            
            timeline_data.append({
                "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                "risk_level": "HIGH" if risk_prob >= 0.5 else "LOW",  # 基于概率值判断
                "risk_probability": round(risk_prob * 100, 2),
                "alert_level": alert['alert_level'],
                "alert_color": alert['color"],
                "action": alert['action'],
                "intensity": alert['intensity'],
                "cluster_info": cluster_info
            })
        
        return {
            "status": "success",
            "region": region,
            "timeline": timeline_data[:days],
            "summary": {
                "total_days": len(timeline_data[:days]),
                "high_risk_days": sum(1 for x in timeline_data[:days] if x['risk_level'] == 'HIGH'),
                "average_risk": round(np.mean([x['risk_probability'] for x in timeline_data[:days]]), 2),
                "alert_distribution": {
                    'critical': sum(1 for x in timeline_data[:days] if x['alert_level'] == 'CRITICAL'),
                    'high': sum(1 for x in timeline_data[:days] if x['alert_level'] == 'HIGH'),
                    'medium': sum(1 for x in timeline_data[:days] if x['alert_level'] == 'MEDIUM'),
                    'low': sum(1 for x in timeline_data[:days] if x['alert_level'] == 'LOW'),
                    'normal': sum(1 for x in timeline_data[:days] if x['alert_level'] == 'NORMAL')
                }
            }
        }
        
    except Exception as e:
        logger.error(f"高级洪水预警时间线生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"高级洪水预警时间线生成失败: {str(e)}")

@router.get("/cluster-analysis")
async def get_cluster_analysis(
    region: str = Query("red-river-basin", description="评估区域")
):
    """获取聚类分析结果"""
    try:
        if advanced_flood_system.model is None:
            raise HTTPException(status_code=503, detail="高级洪水预警模型未加载")
        
        # 优先加载修复后的测试数据，然后是优化数据
        fixed_test_path = "data/processed/flood_warning/flood_warning_fixed_features.csv"
        test_data_path = "data/processed/flood_warning/flood_warning_test_data.csv"
        optimized_path = "data/processed/flood_warning/flood_warning_optimized.csv"
        
        if os.path.exists(fixed_test_path):
            logger.info("使用修复后的测试数据")
            merged_data = pd.read_csv(fixed_test_path)
        elif os.path.exists(test_data_path):
            logger.info("使用测试数据")
            merged_data = pd.read_csv(test_data_path)
        elif os.path.exists(optimized_path):
            logger.info("使用优化数据")
            merged_data = pd.read_csv(optimized_path)
        else:
            raise HTTPException(status_code=404, detail="数据文件不存在")
        
        # 准备高级特征
        features_data = advanced_flood_system.prepare_advanced_features(merged_data)
        
        # 预测高级风险
        prediction_result = advanced_flood_system.predict_advanced_flood_risk(features_data)
        
        return {
            "status": "success",
            "region": region,
            "timestamp": datetime.now().isoformat(),
            "cluster_analysis": prediction_result['cluster_analysis'],
            "cluster_summary": {
                "total_clusters": len(prediction_result['cluster_analysis']),
                "cluster_sizes": [c['size'] for c in prediction_result['cluster_analysis'].values()],
                "cluster_percentages": [c['percentage'] for c in prediction_result['cluster_analysis'].values()]
            }
        }
        
    except Exception as e:
        logger.error(f"聚类分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"聚类分析失败: {str(e)}")

@router.get("/trend-analysis")
async def get_trend_analysis(
    region: str = Query("red-river-basin", description="评估区域")
):
    """获取趋势分析结果"""
    try:
        if advanced_flood_system.model is None:
            raise HTTPException(status_code=503, detail="高级洪水预警模型未加载")
        
        # 加载优化后的数据
        optimized_path = "data/processed/flood_warning/flood_warning_optimized.csv"
        
        if not os.path.exists(optimized_path):
            raise HTTPException(status_code=404, detail="优化数据文件不存在")
        
        # 读取数据
        merged_data = pd.read_csv(optimized_path)
        
        # 准备高级特征
        features_data = advanced_flood_system.prepare_advanced_features(merged_data)
        
        # 预测高级风险
        prediction_result = advanced_flood_system.predict_advanced_flood_risk(features_data)
        
        return {
            "status": "success",
            "region": region,
            "timestamp": datetime.now().isoformat(),
            "trend_analysis": prediction_result['trend_analysis'],
            "time_series_analysis": prediction_result['advanced_metrics'].get('time_series', {})
        }
        
    except Exception as e:
        logger.error(f"趋势分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"趋势分析失败: {str(e)}")

@router.get("/comprehensive-report")
async def get_comprehensive_report(
    region: str = Query("red-river-basin", description="评估区域")
):
    """获取综合报告"""
    try:
        if advanced_flood_system.model is None:
            raise HTTPException(status_code=503, detail="高级洪水预警模型未加载")
        
        # 优先加载修复后的测试数据，然后是优化数据
        fixed_test_path = "data/processed/flood_warning/flood_warning_fixed_features.csv"
        test_data_path = "data/processed/flood_warning/flood_warning_test_data.csv"
        optimized_path = "data/processed/flood_warning/flood_warning_optimized.csv"
        
        if os.path.exists(fixed_test_path):
            logger.info("使用修复后的测试数据")
            merged_data = pd.read_csv(fixed_test_path)
        elif os.path.exists(test_data_path):
            logger.info("使用测试数据")
            merged_data = pd.read_csv(test_data_path)
        elif os.path.exists(optimized_path):
            logger.info("使用优化数据")
            merged_data = pd.read_csv(optimized_path)
        else:
            raise HTTPException(status_code=404, detail="数据文件不存在")
        
        # 准备高级特征
        features_data = advanced_flood_system.prepare_advanced_features(merged_data)
        
        # 预测高级风险
        prediction_result = advanced_flood_system.predict_advanced_flood_risk(features_data)
        
        # 生成综合报告
        comprehensive_report = advanced_flood_system.generate_comprehensive_report(
            features_data, prediction_result
        )
        
        return {
            "status": "success",
            "region": region,
            "timestamp": datetime.now().isoformat(),
            "comprehensive_report": comprehensive_report
        }
        
    except Exception as e:
        logger.error(f"综合报告生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"综合报告生成失败: {str(e)}")

@router.get("/model-status")
async def get_advanced_model_status():
    """获取高级模型状态"""
    try:
        model_loaded = advanced_flood_system.model is not None
        cluster_loaded = advanced_flood_system.cluster_model is not None
        scaler_loaded = advanced_flood_system.scaler is not None
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "models": {
                "main_model": {
                    "loaded": model_loaded,
                    "type": "Advanced Random Forest Classifier" if model_loaded else "None",
                    "features": len(advanced_flood_system.feature_names) if hasattr(advanced_flood_system, 'feature_names') else 0
                },
                "cluster_model": {
                    "loaded": cluster_loaded,
                    "type": "KMeans Clustering" if cluster_loaded else "None",
                    "n_clusters": advanced_flood_system.cluster_model.n_clusters if cluster_loaded else 0
                },
                "scaler": {
                    "loaded": scaler_loaded,
                    "type": "StandardScaler" if scaler_loaded else "None"
                }
            },
            "alert_levels": list(advanced_flood_system.alert_levels.keys()),
            "capabilities": [
                "Advanced feature engineering",
                "Clustering analysis",
                "Trend analysis",
                "Multi-level alerts",
                "Comprehensive reporting",
                "Real-time monitoring"
            ]
        }
        
    except Exception as e:
        logger.error(f"获取高级模型状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取高级模型状态失败: {str(e)}")
