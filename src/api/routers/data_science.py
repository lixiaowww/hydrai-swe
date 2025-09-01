#!/usr/bin/env python3
"""
数据科学分析API端点
提供高级数据分析功能，包括无监督学习、异常检测、聚类分析等
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import json
import os
from datetime import datetime

# 导入数据科学分析器
import sys
sys.path.append('/home/sean/hydrai_swe/src')
from models.data_science_analyzer import DataScienceAnalyzer
from models.exploration.insight_discovery import InsightDiscoveryModule

router = APIRouter(prefix="/data-science", tags=["Data Science Analysis"])

# 全局分析器实例
analyzer_instance = None

class AnalysisRequest(BaseModel):
    """分析请求模型"""
    data_path: Optional[str] = None
    column: str = "snow_water_equivalent_mm"
    analysis_types: List[str] = ["decomposition", "anomaly", "clustering", "dimensionality", "statistical"]
    save_results: bool = True

class AnalysisResponse(BaseModel):
    """分析响应模型"""
    success: bool
    message: str
    analysis_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    timestamp: str

# 存储分析结果的字典
analysis_results_storage = {}


def _json_safe(value):
    """Convert pandas/numpy objects to JSON-serializable primitives."""
    import numpy as _np
    import pandas as _pd

    if isinstance(value, _pd.Series):
        idx = value.index
        try:
            idx = [i.isoformat() if hasattr(i, "isoformat") else str(i) for i in idx]
        except Exception:
            idx = [str(i) for i in idx]
        arr = value.astype(float).to_numpy()
        # Replace non-finite entries (NaN/Inf) with None for JSON compliance
        arr = [_np.nan if _np.isfinite(v) else None for v in arr]
        # Replace numpy.nan with None explicitly
        arr = [None if (isinstance(v, float) and (v != v)) else v for v in arr]
        return {"index": idx, "values": arr}
    if isinstance(value, _pd.DataFrame):
        data = {}
        for col in value.columns:
            data[col] = _json_safe(value[col])
        return {"columns": list(value.columns), "data": data}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (dict,)):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (_np.integer,)):
        return int(value)
    if isinstance(value, (_np.floating,)):
        return float(value)
    if isinstance(value, (_np.ndarray,)):
        arr = value
        arr = _np.where(_np.isfinite(arr), arr, _np.nan).tolist()
        # json cannot carry NaN; map to None
        arr = [None if (isinstance(v, float) and (v != v)) else v for v in arr]
        return arr
    return value

@router.post("/analyze", response_model=AnalysisResponse)
async def run_comprehensive_analysis(request: AnalysisRequest):
    """
    运行数据科学综合分析
    
    Args:
        request: 分析请求参数
        
    Returns:
        AnalysisResponse: 分析结果
    """
    try:
        global analyzer_instance
        
        # 创建分析器实例
        analyzer_instance = DataScienceAnalyzer()
        
        # 确定数据路径
        data_path = request.data_path
        if not data_path:
            # 尝试默认数据路径
            default_paths = [
                "src/neuralhydrology/data/red_river_basin/timeseries.csv",
                "data/processed/eccc_manitoba_snow_processed.csv",
                "data/raw/eccc_recent/eccc_recent_combined.csv"
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    data_path = path
                    break
            
            if not data_path:
                raise HTTPException(status_code=404, detail="未找到数据文件")
        
        # 加载数据
        analyzer_instance.load_data(data_path)
        
        if analyzer_instance.data is None:
            raise HTTPException(status_code=400, detail="数据加载失败")
        
        # 生成分析ID
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 运行分析
        results = {}
        
        if "decomposition" in request.analysis_types:
            results["decomposition"] = _json_safe(analyzer_instance.advanced_time_series_decomposition(request.column))
        
        if "anomaly" in request.analysis_types:
            results["anomaly_detection"] = _json_safe(analyzer_instance.advanced_anomaly_detection(request.column))
        
        if "clustering" in request.analysis_types:
            results["clustering"] = _json_safe(analyzer_instance.clustering_analysis())
        
        if "dimensionality" in request.analysis_types:
            results["dimensionality_reduction"] = _json_safe(analyzer_instance.dimensionality_reduction_analysis())
        
        if "statistical" in request.analysis_types:
            results["statistical_tests"] = _json_safe(analyzer_instance.statistical_hypothesis_testing(request.column))
        
        # 存储结果
        analysis_results_storage[analysis_id] = {
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "request": request.dict()
        }
        
        # 保存结果到文件
        if request.save_results:
            save_path = f"analysis_results/{analysis_id}"
            os.makedirs(save_path, exist_ok=True)
            
            with open(f"{save_path}/results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            # 创建可视化
            visualizations = analyzer_instance.create_interactive_visualizations(save_path)
            
            # 保存可视化信息
            with open(f"{save_path}/visualizations.json", "w", encoding="utf-8") as f:
                json.dump(list(visualizations.keys()), f, ensure_ascii=False, indent=2)
        
        return AnalysisResponse(
            success=True,
            message="数据科学分析完成",
            analysis_id=analysis_id,
            results=results,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_results(analysis_id: str):
    """
    获取分析结果
    
    Args:
        analysis_id: 分析ID
        
    Returns:
        AnalysisResponse: 分析结果
    """
    if analysis_id not in analysis_results_storage:
        raise HTTPException(status_code=404, detail="分析结果不存在")
    
    stored_data = analysis_results_storage[analysis_id]
    
    return AnalysisResponse(
        success=True,
        message="分析结果获取成功",
        analysis_id=analysis_id,
        results=stored_data["results"],
        timestamp=stored_data["timestamp"]
    )

@router.get("/analysis/{analysis_id}/visualizations")
async def get_analysis_visualizations(analysis_id: str):
    """
    获取分析可视化文件列表
    
    Args:
        analysis_id: 分析ID
        
    Returns:
        dict: 可视化文件列表
    """
    save_path = f"analysis_results/{analysis_id}"
    
    if not os.path.exists(save_path):
        raise HTTPException(status_code=404, detail="分析结果不存在")
    
    visualizations = []
    for file in os.listdir(save_path):
        if file.endswith('.html'):
            visualizations.append({
                "name": file.replace('.html', ''),
                "file": file,
                "url": f"/analysis_results/{analysis_id}/{file}"
            })
    
    return {
        "analysis_id": analysis_id,
        "visualizations": visualizations
    }

@router.get("/decomposition")
async def get_time_series_decomposition(
    column: str = Query("snow_water_equivalent_mm", description="要分析的列名"),
    data_path: Optional[str] = Query(None, description="数据文件路径")
):
    """
    获取时间序列分解结果
    
    Args:
        column: 要分析的列名
        data_path: 数据文件路径
        
    Returns:
        dict: 分解结果
    """
    try:
        global analyzer_instance
        
        if analyzer_instance is None or analyzer_instance.data is None:
            analyzer_instance = DataScienceAnalyzer()
            if data_path:
                analyzer_instance.load_data(data_path)
            else:
                # 使用默认数据路径
                default_paths = [
                    "src/neuralhydrology/data/red_river_basin/timeseries.csv",
                    "data/processed/eccc_manitoba_snow_processed.csv"
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        analyzer_instance.load_data(path)
                        break
        
        if analyzer_instance.data is None:
            raise HTTPException(status_code=400, detail="数据未加载")
        
        raw = analyzer_instance.advanced_time_series_decomposition(column)
        results = _json_safe(raw)
        
        return {
            "success": True,
            "column": column,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分解分析失败: {str(e)}")

@router.get("/anomaly-detection")
async def get_anomaly_detection(
    column: str = Query("snow_water_equivalent_mm", description="要分析的列名"),
    data_path: Optional[str] = Query(None, description="数据文件路径")
):
    """
    获取异常检测结果
    
    Args:
        column: 要分析的列名
        data_path: 数据文件路径
        
    Returns:
        dict: 异常检测结果
    """
    try:
        global analyzer_instance
        
        if analyzer_instance is None or analyzer_instance.data is None:
            analyzer_instance = DataScienceAnalyzer()
            if data_path:
                analyzer_instance.load_data(data_path)
            else:
                # 使用默认数据路径
                default_paths = [
                    "src/neuralhydrology/data/red_river_basin/timeseries.csv",
                    "data/processed/eccc_manitoba_snow_processed.csv"
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        analyzer_instance.load_data(path)
                        break
        
        if analyzer_instance.data is None:
            raise HTTPException(status_code=400, detail="数据未加载")
        
        results = _json_safe(analyzer_instance.advanced_anomaly_detection(column))
        
        return {
            "success": True,
            "column": column,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"异常检测失败: {str(e)}")

@router.get("/clustering")
async def get_clustering_analysis(
    columns: Optional[str] = Query(None, description="要分析的列名，用逗号分隔"),
    data_path: Optional[str] = Query(None, description="数据文件路径")
):
    """
    获取聚类分析结果
    
    Args:
        columns: 要分析的列名，用逗号分隔
        data_path: 数据文件路径
        
    Returns:
        dict: 聚类分析结果
    """
    try:
        global analyzer_instance
        
        if analyzer_instance is None or analyzer_instance.data is None:
            analyzer_instance = DataScienceAnalyzer()
            if data_path:
                analyzer_instance.load_data(data_path)
            else:
                # 使用默认数据路径
                default_paths = [
                    "src/neuralhydrology/data/red_river_basin/timeseries.csv",
                    "data/processed/eccc_manitoba_snow_processed.csv"
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        analyzer_instance.load_data(path)
                        break
        
        if analyzer_instance.data is None:
            raise HTTPException(status_code=400, detail="数据未加载")
        
        # 处理列名参数
        column_list = None
        if columns:
            column_list = [col.strip() for col in columns.split(",")]
        
        raw = analyzer_instance.clustering_analysis(column_list)

        def to_list(v):
            try:
                out = list(v)
            except Exception:
                out = []
            # ensure python primitives
            py = []
            for x in out:
                try:
                    if x is None:
                        py.append(None)
                    elif isinstance(x, (int, float, str, bool)):
                        py.append(x)
                    else:
                        py.append(int(x))
                except Exception:
                    try:
                        py.append(float(x))
                    except Exception:
                        py.append(str(x))
            return py

        compact = {}
        if isinstance(raw, dict):
            km = raw.get('kmeans', {})
            db = raw.get('dbscan', {})
            hi = raw.get('hierarchical', {})
            compact['kmeans'] = {
                'labels': to_list(km.get('labels', [])),
                'n_clusters': int(km.get('n_clusters', 0) or 0),
                'silhouette_score': float(km.get('silhouette_score', 0.0) or 0.0)
            }
            compact['dbscan'] = {
                'labels': to_list(db.get('labels', [])),
                'n_clusters': int(db.get('n_clusters', 0) or 0),
                'n_noise': int(db.get('n_noise', 0) or 0)
            }
            compact['hierarchical'] = {
                'labels': to_list(hi.get('labels', [])),
                'n_clusters': int(hi.get('n_clusters', 0) or 0),
                'silhouette_score': float(hi.get('silhouette_score', 0.0) or 0.0)
            }
            compact['features_used'] = [str(x) for x in to_list(raw.get('features_used', []))]
            compact['interpretation'] = raw.get('interpretation', {})
        else:
            compact = {'kmeans': {'labels': [], 'n_clusters': 0, 'silhouette_score': 0.0},
                       'dbscan': {'labels': [], 'n_clusters': 0, 'n_noise': 0},
                       'hierarchical': {'labels': [], 'n_clusters': 0, 'silhouette_score': 0.0},
                       'features_used': []}

        return {
            'success': True,
            'columns': column_list,
            'results': compact,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚类分析失败: {str(e)}")

@router.get("/dimensionality-reduction")
async def get_dimensionality_reduction(
    columns: Optional[str] = Query(None, description="要分析的列名，用逗号分隔"),
    data_path: Optional[str] = Query(None, description="数据文件路径")
):
    """
    获取降维分析结果
    
    Args:
        columns: 要分析的列名，用逗号分隔
        data_path: 数据文件路径
        
    Returns:
        dict: 降维分析结果
    """
    try:
        global analyzer_instance
        
        if analyzer_instance is None or analyzer_instance.data is None:
            analyzer_instance = DataScienceAnalyzer()
            if data_path:
                analyzer_instance.load_data(data_path)
            else:
                # 使用默认数据路径
                default_paths = [
                    "src/neuralhydrology/data/red_river_basin/timeseries.csv",
                    "data/processed/eccc_manitoba_snow_processed.csv"
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        analyzer_instance.load_data(path)
                        break
        
        if analyzer_instance.data is None:
            raise HTTPException(status_code=400, detail="数据未加载")
        
        # 处理列名参数
        column_list = None
        if columns:
            column_list = [col.strip() for col in columns.split(",")]
        
        results = _json_safe(analyzer_instance.dimensionality_reduction_analysis(column_list))
        
        return {
            "success": True,
            "columns": column_list,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"降维分析失败: {str(e)}")

@router.get("/statistical-tests")
async def get_statistical_tests(
    column: str = Query("snow_water_equivalent_mm", description="要分析的列名"),
    data_path: Optional[str] = Query(None, description="数据文件路径")
):
    """
    获取统计假设检验结果
    
    Args:
        column: 要分析的列名
        data_path: 数据文件路径
        
    Returns:
        dict: 统计检验结果
    """
    try:
        global analyzer_instance
        
        if analyzer_instance is None or analyzer_instance.data is None:
            analyzer_instance = DataScienceAnalyzer()
            if data_path:
                analyzer_instance.load_data(data_path)
            else:
                # 使用默认数据路径
                default_paths = [
                    "src/neuralhydrology/data/red_river_basin/timeseries.csv",
                    "data/processed/eccc_manitoba_snow_processed.csv"
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        analyzer_instance.load_data(path)
                        break
        
        if analyzer_instance.data is None:
            raise HTTPException(status_code=400, detail="数据未加载")
        
        raw = analyzer_instance.statistical_hypothesis_testing(column)
        print(f"🔍 统计检验API调用结果: {raw}")
        
        # 直接使用返回的数据结构
        compact = raw if isinstance(raw, dict) else {}
        return {
            'success': True,
            'column': column,
            'results': compact,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"统计检验失败: {str(e)}")

@router.get("/visualizations")
async def create_visualizations(
    analysis_id: Optional[str] = Query(None, description="分析ID"),
    save_path: Optional[str] = Query(None, description="保存路径")
):
    """
    创建交互式可视化
    
    Args:
        analysis_id: 分析ID
        save_path: 保存路径
        
    Returns:
        dict: 可视化结果
    """
    try:
        global analyzer_instance
        
        if analyzer_instance is None or analyzer_instance.data is None:
            raise HTTPException(status_code=400, detail="分析器未初始化")
        
        # 确定保存路径
        if not save_path:
            if analysis_id:
                save_path = f"analysis_results/{analysis_id}"
            else:
                save_path = f"visualizations/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建可视化
        visualizations = analyzer_instance.create_interactive_visualizations(save_path)
        
        return {
            "success": True,
            "save_path": save_path,
            "visualizations": list(visualizations.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"可视化创建失败: {str(e)}")

@router.get("/data-info")
async def get_data_info(data_path: Optional[str] = Query(None, description="数据文件路径")):
    """
    获取数据信息
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        dict: 数据信息
    """
    try:
        global analyzer_instance
        
        if analyzer_instance is None or analyzer_instance.data is None:
            analyzer_instance = DataScienceAnalyzer()
            if data_path:
                analyzer_instance.load_data(data_path)
            else:
                # 使用默认数据路径
                default_paths = [
                    "src/neuralhydrology/data/red_river_basin/timeseries.csv",
                    "data/processed/eccc_manitoba_snow_processed.csv"
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        analyzer_instance.load_data(path)
                        break
        
        if analyzer_instance.data is None:
            raise HTTPException(status_code=400, detail="数据未加载")
        
        data = analyzer_instance.data
        
        return {
            "success": True,
            "data_info": {
                "shape": list(data.shape),
                "columns": data.columns.tolist(),
                "dtypes": {c: str(dtype) for c, dtype in data.dtypes.items()},
                "time_range": {
                    "start": data.index.min().isoformat() if hasattr(data.index.min(), 'isoformat') else str(data.index.min()),
                    "end": data.index.max().isoformat() if hasattr(data.index.max(), 'isoformat') else str(data.index.max())
                },
                "missing_values": {k: int(v) for k, v in data.isnull().sum().to_dict().items()},
                "numeric_columns": data.select_dtypes(include=['number']).columns.tolist()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据信息失败: {str(e)}")

@router.get("/analysis-list")
async def get_analysis_list():
    """
    获取所有分析结果列表
    
    Returns:
        dict: 分析结果列表
    """
    try:
        analysis_list = []
        
        for analysis_id, data in analysis_results_storage.items():
            analysis_list.append({
                "analysis_id": analysis_id,
                "timestamp": data["timestamp"],
                "analysis_types": list(data["results"].keys()),
                "request": data["request"]
            })
        
        # 按时间戳排序
        analysis_list.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "success": True,
            "analysis_list": analysis_list,
            "total_count": len(analysis_list),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取分析列表失败: {str(e)}")

@router.get("/factor-discovery")
async def factor_discovery(
    target: str = Query("snow_water_equivalent_mm", description="目标列名"),
    top_k: int = Query(10, ge=1, le=50),
    data_path: Optional[str] = Query(None, description="数据文件路径")
):
    """无监督冷门影响因素发现接口。"""
    try:
        global analyzer_instance
        if analyzer_instance is None or analyzer_instance.data is None:
            analyzer_instance = DataScienceAnalyzer()
            if data_path:
                analyzer_instance.load_data(data_path)
            else:
                for path in [
                    "src/neuralhydrology/data/red_river_basin/timeseries.csv",
                    "data/processed/eccc_manitoba_snow_processed.csv"
                ]:
                    if os.path.exists(path):
                        analyzer_instance.load_data(path)
                        break
        if analyzer_instance.data is None:
            raise HTTPException(status_code=400, detail="数据未加载")

        raw = analyzer_instance.discover_cold_factors(target, top_k)
        return {"success": True, "results": _json_safe(raw), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"因素发现失败: {str(e)}")

@router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    删除分析结果
    
    Args:
        analysis_id: 分析ID
        
    Returns:
        dict: 删除结果
    """
    try:
        if analysis_id not in analysis_results_storage:
            raise HTTPException(status_code=404, detail="分析结果不存在")
        
        # 从内存中删除
        del analysis_results_storage[analysis_id]
        
        # 删除文件
        save_path = f"analysis_results/{analysis_id}"
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path)
        
        return {
            "success": True,
            "message": f"分析结果 {analysis_id} 已删除",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除分析结果失败: {str(e)}")

@router.get("/unsupervised-insights")
async def get_unsupervised_insights(
    target_column: str = Query("estimated_soil_moisture", description="目标列名"),
    data_path: Optional[str] = Query(None, description="数据文件路径")
):
    """
    获取无监督学习洞察 - 调用真实的InsightDiscoveryModule
    
    Args:
        target_column: 目标分析列名
        data_path: 数据文件路径
        
    Returns:
        dict: 无监督学习洞察结果
    """
    try:
        # 创建无监督探索模块实例
        insight_module = InsightDiscoveryModule()
        
        # 确定数据路径并加载数据
        if not data_path:
            # 尝试默认数据路径
            default_paths = [
                "src/neuralhydrology/data/red_river_basin/timeseries.csv",
                "data/processed/eccc_manitoba_snow_processed.csv",
                "data/raw/eccc_recent/eccc_recent_combined.csv"
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    data_path = path
                    break
        
        if not data_path or not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="未找到数据文件")
        
        # 加载数据
        import pandas as pd
        data = pd.read_csv(data_path)
        
        # 处理日期索引
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        elif 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        
        print(f"✅ 数据加载成功: {len(data)} 条记录")
        print(f"📊 数据列: {list(data.columns)}")
        
        # 运行无监督模式发现
        insights = insight_module.discover_patterns(data, target_column)
        
        # 运行解读洞察结果
        interpretation = insight_module.interpret_insights(insights)
        
        # 组合完整结果
        complete_results = {
            "insights": _json_safe(insights),
            "interpretation": _json_safe(interpretation)
        }
        
        return {
            "success": True,
            "target_column": target_column,
            "data_path": data_path,
            "data_shape": list(data.shape),
            "results": complete_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_detail = f"无监督学习洞察失败: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/health")
async def health_check():
    """
    健康检查端点
    
    Returns:
        dict: 健康状态
    """
    return {
        "status": "healthy",
        "service": "Data Science Analysis API",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
