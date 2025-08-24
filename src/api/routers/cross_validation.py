#!/usr/bin/env python3
"""
交叉验证API端点
提供历史数据交叉验证服务
"""

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path

# 导入交叉验证器
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.flood_risk_cross_validation import FloodRiskCrossValidator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# 全局验证器实例
validator = FloodRiskCrossValidator()

# 存储验证任务状态
validation_tasks: Dict[str, Dict] = {}

@router.post("/start")
async def start_cross_validation(
    background_tasks: BackgroundTasks,
    # 真实模式：通过Query传入
    data_path: Optional[str] = Query(None, description="历史数据文件路径"),
    stations: Optional[str] = Query(None, description="要验证的站点ID列表，逗号分隔"),
    validation_windows: int = Query(5, description="验证窗口数量"),
    forecast_horizon: int = Query(7, description="预测时间范围（天）"),
    # 兼容前端：JSON载荷触发模拟任务
    payload: Optional[Dict] = Body(None)
):
    """
    启动交叉验证任务
    - 若提供 data_path（Query），则运行真实交叉验证（后台）
    - 若提供 JSON 载荷（start_date/end_date/station_id/window_size/step_size），则运行模拟任务（后台）
    """
    task_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 解析前端JSON（用于模拟模式）
    start_date = end_date = station_id = None
    window_size = validation_windows
    step_size = 1
    if payload:
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        station_id = payload.get("station_id")
        window_size = int(payload.get("window_size", window_size))
        step_size = int(payload.get("step_size", step_size))

    # 初始化任务状态
    validation_tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "start_time": datetime.now().isoformat(),
        "parameters": {
            "data_path": data_path,
            "stations": stations,
            "validation_windows": validation_windows,
            "forecast_horizon": forecast_horizon,
            "start_date": start_date,
            "end_date": end_date,
            "station_id": station_id,
            "window_size": window_size,
            "step_size": step_size,
        },
        "result": None,
        "error": None,
    }

    # 分支：真实模式
    if data_path:
        if not Path(data_path).exists():
            raise HTTPException(status_code=400, detail=f"数据文件不存在: {data_path}")
        station_list = [s.strip() for s in stations.split(',')] if stations else None

        async def run_real():
            try:
                validation_tasks[task_id]["status"] = "running"
                validation_tasks[task_id]["progress"] = 10
                report = validator.run_cross_validation(
                    data_path=data_path,
                    stations=station_list,
                    validation_windows=validation_windows,
                    forecast_horizon=forecast_horizon,
                )
                validation_tasks[task_id]["progress"] = 90
                validator.generate_validation_report(f"data/processed/validation_report_{task_id}.json")
                validator.plot_validation_results(f"data/processed/validation_plots_{task_id}")
                validation_tasks[task_id]["status"] = "completed"
                validation_tasks[task_id]["progress"] = 100
                validation_tasks[task_id]["result"] = report
                validation_tasks[task_id]["completion_time"] = datetime.now().isoformat()
            except Exception as e:
                logger.exception("真实交叉验证失败")
                validation_tasks[task_id]["status"] = "failed"
                validation_tasks[task_id]["error"] = str(e)
                validation_tasks[task_id]["completion_time"] = datetime.now().isoformat()

        background_tasks.add_task(run_real)
        return {"task_id": task_id, "status": "queued", "mode": "real"}

    # 系统禁止使用模拟模式
    raise HTTPException(status_code=400, detail="系统禁止使用模拟模式。请提供真实数据路径进行交叉验证。")

@router.post("/quick")
async def quick_cross_validation(
    # 真实模式（可选）
    data_path: Optional[str] = Query(None, description="历史数据文件路径"),
    stations: Optional[str] = Query(None, description="要验证的站点ID列表，逗号分隔"),
    validation_windows: int = Query(3, description="验证窗口数量（快速模式）"),
    forecast_horizon: int = Query(7, description="预测时间范围（天）"),
    # 前端JSON（默认）
    payload: Optional[Dict] = Body(None)
):
    """
    快速交叉验证：
    - 若提供 data_path（Query），同步运行真实小样本验证
    - 否则返回模拟结果（兼容前端）
    """
    # 真实快速模式
    if data_path:
        if not Path(data_path).exists():
            raise HTTPException(status_code=400, detail=f"数据文件不存在: {data_path}")
        station_list = [s.strip() for s in stations.split(',')] if stations else None
        report = validator.run_cross_validation(
            data_path=data_path,
            stations=station_list,
            validation_windows=validation_windows,
            forecast_horizon=forecast_horizon,
        )
        return {"status": "completed", "mode": "real", "result": report}

    # 系统禁止使用模拟模式
    raise HTTPException(status_code=400, detail="系统禁止使用模拟模式。请提供真实数据路径进行交叉验证。")

@router.get("/status/{task_id}")
async def get_validation_status(task_id: str):
    if task_id not in validation_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = validation_tasks[task_id]
    resp = {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "progress": task.get("progress", 0),
        "start_time": task.get("start_time"),
        "parameters": task.get("parameters", {}),
    }
    if task.get("status") == "completed":
        resp["completion_time"] = task.get("completion_time")
        if task.get("result"):
            resp["result_summary"] = {
                "keys": list(task["result"].keys())
            }
        resp["download_urls"] = {
            "report": f"/api/v1/cross-validation/download/{task_id}/report",
            "plots": f"/api/v1/cross-validation/download/{task_id}/plots",
        }
    if task.get("status") == "failed":
        resp["error"] = task.get("error")
        resp["completion_time"] = task.get("completion_time")
    return resp

@router.get("/tasks")
async def list_validation_tasks():
    return {"total_tasks": len(validation_tasks), "tasks": list(validation_tasks.values())}

@router.get("/download/{task_id}/report")
async def download_validation_report(task_id: str):
    if task_id not in validation_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = validation_tasks[task_id]
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    report_path = Path(f"data/processed/validation_report_{task_id}.json")
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))
    # 若无真实报告，则返回内存结果
    if task.get("result"):
        return JSONResponse(content=task["result"])
    raise HTTPException(status_code=404, detail="报告不存在")

@router.get("/download/{task_id}/plots")
async def download_validation_plots(task_id: str):
    if task_id not in validation_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    plots_dir = Path(f"data/processed/validation_plots_{task_id}")
    if not plots_dir.exists():
        return {"plots": []}
    plots = [
        {"filename": p.name, "size": p.stat().st_size}
        for p in plots_dir.glob("*.png")
    ]
    return {"plots": plots}

@router.delete("/task/{task_id}")
async def delete_validation_task(task_id: str):
    if task_id in validation_tasks:
        del validation_tasks[task_id]
        return {"status": "deleted", "task_id": task_id}
    raise HTTPException(status_code=404, detail="任务不存在")

@router.get("/available-data")
async def get_available_data():
    data_sources = []
    for base in ["data/raw", "data/processed", "data"]:
        p = Path(base)
        if not p.exists():
            continue
        for fp in list(p.rglob("*.csv")) + list(p.rglob("*.json")):
            try:
                data_sources.append({
                    "path": str(fp),
                    "size_mb": round(fp.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(fp.stat().st_mtime).isoformat(),
                })
            except Exception:
                continue
    return {"total_sources": len(data_sources), "available_data_sources": data_sources}

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cross_validation_api"}

@router.get("/")
async def root():
    return {
        "service": "Cross Validation API",
        "version": "1.0.0",
    }
