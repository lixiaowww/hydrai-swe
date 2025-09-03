#!/usr/bin/env python3
"""
水文分析API路由 - 已修复数据同步问题
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hydrology", tags=["Hydrological Analysis"])

# 数据路径配置
DATA_PATHS = {
    'hydrometric': '/home/sean/hydrai_swe/data/processed/hydat_streamflow_processed.csv',
    'weather': '/home/sean/hydrai_swe/data/processed/eccc_weather_processed.csv',
    'swe': '/home/sean/hydrai_swe/data/processed/swe_processed.csv'
}

def load_hydrological_data_with_pipeline_sync(data_path: str):
    """使用数据管道同步加载水文数据"""
    df = None
    
    # 1. 尝试从数据管道获取最新数据
    try:
        import sys
        sys.path.append('/home/sean/hydrai_swe/src')
        from models.real_data_loader import RealDataLoader
        data_loader = RealDataLoader()
        
        # 尝试获取实时水文数据
        pipeline_data = data_loader._try_pipeline_data_sync("hydrological_analysis", 30)
        if pipeline_data and 'data' in pipeline_data:
            df = pipeline_data['data']
            logger.info(f"使用数据管道同步数据: {pipeline_data.get('source', 'unknown')}")
    except Exception as e:
        logger.warning(f"数据管道同步失败: {e}")
    
    # 2. 如果管道数据不可用，使用静态文件
    if df is None:
        df = pd.read_csv(data_path)
        logger.warning(f"使用静态数据文件: {data_path} (数据可能过时)")
    
    return df

@router.get("/available-data-sources")
async def get_available_data_sources():
    """获取可用的数据源"""
    try:
        available_sources = {}
        
        for source, path in DATA_PATHS.items():
            if os.path.exists(path):
                # 检查文件大小和修改时间
                stat = os.stat(path)
                file_size = stat.st_size
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                age_days = (datetime.now() - mod_time).days
                
                available_sources[source] = {
                    'path': path,
                    'file_size_mb': round(file_size / (1024 * 1024), 2),
                    'last_modified': mod_time.isoformat(),
                    'age_days': age_days,
                    'status': 'recent' if age_days <= 7 else 'stale'
                }
        
        return {
            "status": "success",
            "available_sources": available_sources,
            "total_sources": len(available_sources),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取数据源失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据源失败: {str(e)}")

@router.get("/comprehensive-analysis")
async def get_comprehensive_analysis(
    data_source: str = Query("hydrometric", description="数据源类型"),
    days: int = Query(30, description="分析天数", ge=7, le=365)
):
    """获取综合水文分析"""
    try:
        # 获取数据路径
        data_path = DATA_PATHS.get(data_source)
        if not data_path:
            raise HTTPException(status_code=404, detail=f"不支持的数据源: {data_source}")
        
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"数据文件不存在: {data_path}")
        
        # 使用数据管道同步加载数据
        df = load_hydrological_data_with_pipeline_sync(data_path)
        
        # 初始化水量平衡分析
        from models.hydrological_analysis_system import WatershedWaterBalance
        water_balance = WatershedWaterBalance(df)
        
        # 运行分析
        balance_results = water_balance.calculate_water_balance()
        runoff_analysis = water_balance.analyze_precipitation_runoff_relationship()
        
        return {
            "status": "success",
            "water_balance": balance_results,
            "precipitation_runoff_relationship": runoff_analysis,
            "data_source": os.path.basename(data_path),
            "data_sync_status": "pipeline" if df is not None else "static",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"综合水文分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"综合水文分析失败: {str(e)}")

@router.get("/water-balance")
async def get_water_balance_analysis(
    data_source: str = Query("hydrometric", description="数据源类型"),
    days: int = Query(30, description="分析天数", ge=7, le=365)
):
    """获取水量平衡分析"""
    try:
        # 获取数据路径
        data_path = DATA_PATHS.get(data_source)
        if not data_path:
            raise HTTPException(status_code=404, detail=f"不支持的数据源: {data_source}")
        
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"数据文件不存在: {data_path}")
        
        # 使用数据管道同步加载数据
        df = load_hydrological_data_with_pipeline_sync(data_path)
        
        # 初始化水量平衡分析
        from models.hydrological_analysis_system import WatershedWaterBalance
        water_balance = WatershedWaterBalance(df)
        
        # 运行水量平衡分析
        balance_results = water_balance.calculate_water_balance()
        
        return {
            "status": "success",
            "water_balance": balance_results,
            "data_source": os.path.basename(data_path),
            "data_sync_status": "pipeline" if df is not None else "static",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"水量平衡分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"水量平衡分析失败: {str(e)}")

@router.get("/flood-frequency")
async def get_flood_frequency_analysis(
    data_source: str = Query("hydrometric", description="数据源类型"),
    return_periods: str = Query("2,5,10,25,50,100", description="重现期（年）")
):
    """获取洪水频率分析"""
    try:
        # 获取数据路径
        data_path = DATA_PATHS.get(data_source)
        if not data_path:
            raise HTTPException(status_code=404, detail=f"不支持的数据源: {data_source}")
        
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"数据文件不存在: {data_path}")
        
        # 使用数据管道同步加载数据
        df = load_hydrological_data_with_pipeline_sync(data_path)
        
        # 初始化洪水频率分析
        from models.hydrological_analysis_system import FloodFrequencyAnalysis
        flood_analysis = FloodFrequencyAnalysis(df)
        
        # 解析重现期
        periods = [int(p.strip()) for p in return_periods.split(',')]
        
        # 运行洪水频率分析
        frequency_results = flood_analysis.analyze_flood_frequency(periods)
        
        return {
            "status": "success",
            "flood_frequency": frequency_results,
            "data_source": os.path.basename(data_path),
            "data_sync_status": "pipeline" if df is not None else "static",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"洪水频率分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"洪水频率分析失败: {str(e)}")

@router.get("/precipitation-runoff")
async def get_precipitation_runoff_analysis(
    data_source: str = Query("hydrometric", description="数据源类型"),
    days: int = Query(30, description="分析天数", ge=7, le=365)
):
    """获取降水径流关系分析"""
    try:
        # 获取数据路径
        data_path = DATA_PATHS.get(data_source)
        if not data_path:
            raise HTTPException(status_code=404, detail=f"不支持的数据源: {data_source}")
        
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"数据文件不存在: {data_path}")
        
        # 使用数据管道同步加载数据
        df = load_hydrological_data_with_pipeline_sync(data_path)
        
        # 初始化降水径流分析
        from models.hydrological_analysis_system import PrecipitationRunoffAnalysis
        runoff_analysis = PrecipitationRunoffAnalysis(df)
        
        # 运行降水径流关系分析
        relationship_results = runoff_analysis.analyze_precipitation_runoff_relationship()
        
        return {
            "status": "success",
            "precipitation_runoff_relationship": relationship_results,
            "data_source": os.path.basename(data_path),
            "data_sync_status": "pipeline" if df is not None else "static",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"降水径流关系分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"降水径流关系分析失败: {str(e)}")
