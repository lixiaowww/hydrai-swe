#!/usr/bin/env python3
"""
高级洪水预警API路由 - 已修复数据同步问题
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
import numpy as np

# 添加项目根目录到Python路径
sys.path.append('/home/sean/hydrai_swe')

from models.advanced_flood_warning_system import AdvancedFloodWarningSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/advanced-flood", tags=["Advanced Flood Warning"])

# 初始化高级洪水预警系统
advanced_flood_system = AdvancedFloodWarningSystem()

def load_flood_data_with_pipeline_sync():
    """使用数据管道同步加载洪水数据"""
    merged_data = None
    
    # 1. 尝试从数据管道获取最新数据
    try:
        from models.real_data_loader import RealDataLoader
        data_loader = RealDataLoader()
        
        # 尝试获取实时洪水数据
        pipeline_data = data_loader._try_pipeline_data_sync("flood_warning", 30)
        if pipeline_data and 'data' in pipeline_data:
            merged_data = pipeline_data['data']
            logger.info(f"使用数据管道同步数据: {pipeline_data.get('source', 'unknown')}")
    except Exception as e:
        logger.warning(f"数据管道同步失败: {e}")
    
    # 2. 如果管道数据不可用，使用静态文件（按优先级）
    if merged_data is None:
        static_paths = [
            "data/processed/flood_warning/flood_warning_fixed_features.csv",
            "data/processed/flood_warning/flood_warning_test_data.csv", 
            "data/processed/flood_warning/flood_warning_optimized.csv"
        ]
        
        for path in static_paths:
            if os.path.exists(path):
                merged_data = pd.read_csv(path)
                logger.warning(f"使用静态数据文件: {path} (数据可能过时)")
                break
        
        if merged_data is None:
            raise HTTPException(status_code=404, detail="无可用数据文件")
    
    return merged_data

@router.get("/advanced-risk-assessment")
async def advanced_flood_risk_assessment(
    days: int = Query(7, description="预测天数", ge=1, le=30),
    region: str = Query("red-river-basin", description="评估区域")
):
    """高级洪水风险评估"""
    try:
        if advanced_flood_system.model is None:
            raise HTTPException(status_code=503, detail="高级洪水预警模型未加载")
        
        # 使用数据管道同步加载数据
        merged_data = load_flood_data_with_pipeline_sync()
        
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
                "probability": round(latest_probability, 3),
                "alert_level": latest_alert
            },
            "region": region,
            "prediction_days": days,
            "data_source": "pipeline_sync" if merged_data is not None else "static_file"
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
        
        # 使用数据管道同步加载数据
        merged_data = load_flood_data_with_pipeline_sync()
        
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
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data_date": latest_date.isoformat(),
            "current_risk": {
                "level": latest_risk,
                "probability": round(latest_probability, 3),
                "alert_level": latest_alert
            },
            "data_source": "pipeline_sync" if merged_data is not None else "static_file"
        }
        
    except Exception as e:
        logger.error(f"实时高级风险评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"实时高级风险评估失败: {str(e)}")

@router.get("/cluster-analysis")
async def get_cluster_analysis(
    region: str = Query("red-river-basin", description="评估区域")
):
    """获取聚类分析结果"""
    try:
        if advanced_flood_system.model is None:
            raise HTTPException(status_code=503, detail="高级洪水预警模型未加载")
        
        # 使用数据管道同步加载数据
        merged_data = load_flood_data_with_pipeline_sync()
        
        # 准备高级特征
        features_data = advanced_flood_system.prepare_advanced_features(merged_data)
        
        # 预测高级风险
        prediction_result = advanced_flood_system.predict_advanced_flood_risk(features_data)
        
        return {
            "status": "success",
            "region": region,
            "timestamp": datetime.now().isoformat(),
            "cluster_analysis": {
                "risk_clusters": prediction_result.get('risk_clusters', []),
                "cluster_centers": prediction_result.get('cluster_centers', []),
                "cluster_labels": prediction_result.get('cluster_labels', [])
            },
            "data_source": "pipeline_sync" if merged_data is not None else "static_file"
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
        
        # 使用数据管道同步加载数据
        merged_data = load_flood_data_with_pipeline_sync()
        
        # 准备高级特征
        features_data = advanced_flood_system.prepare_advanced_features(merged_data)
        
        # 预测高级风险
        prediction_result = advanced_flood_system.predict_advanced_flood_risk(features_data)
        
        return {
            "status": "success",
            "region": region,
            "timestamp": datetime.now().isoformat(),
            "trend_analysis": {
                "risk_trend": prediction_result.get('risk_trend', []),
                "trend_direction": prediction_result.get('trend_direction', 'stable'),
                "trend_strength": prediction_result.get('trend_strength', 0.0)
            },
            "data_source": "pipeline_sync" if merged_data is not None else "static_file"
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
        
        # 使用数据管道同步加载数据
        merged_data = load_flood_data_with_pipeline_sync()
        
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
            "comprehensive_report": comprehensive_report,
            "data_source": "pipeline_sync" if merged_data is not None else "static_file"
        }
        
    except Exception as e:
        logger.error(f"综合报告生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"综合报告生成失败: {str(e)}")

@router.get("/model-status")
async def get_advanced_model_status():
    """获取高级模型状态"""
    try:
        return {
            "status": "success",
            "model_loaded": advanced_flood_system.model is not None,
            "model_type": "Advanced Flood Warning System",
            "timestamp": datetime.now().isoformat(),
            "data_sync_enabled": True
        }
        
    except Exception as e:
        logger.error(f"模型状态获取失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型状态获取失败: {str(e)}")
