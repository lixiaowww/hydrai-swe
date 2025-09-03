"""
水资源管理决策支持API路由
提供水库调度优化、水资源配置优化等功能
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import os
from datetime import datetime

import sys
sys.path.append('/home/sean/hydrai_swe/src')
from models.water_resources_management_system import WaterResourcesManagementSystem

router = APIRouter()

# 全局水资源管理系统实例
water_management_system = WaterResourcesManagementSystem()

class ReservoirDataRequest(BaseModel):
    inflow: List[float]
    demand: List[float]
    capacity: float = 1000.0
    min_level: float = 0.1
    max_level: float = 0.9

class AllocationDataRequest(BaseModel):
    users: List[str]
    priorities: Dict[str, float] = {}
    efficiency_factors: Dict[str, float] = {}
    total_water: float = 1000.0

class OptimizationRequest(BaseModel):
    objectives: List[str] = ["reliability", "efficiency", "safety"]
    constraints: Optional[Dict[str, Any]] = None

class OptimizationResponse(BaseModel):
    status: str
    optimization_type: str
    results: Dict[str, Any]
    recommendations: List[str]
    timestamp: str

@router.post("/api/v1/water-resources/load-reservoir-data")
async def load_reservoir_data(request: Optional[ReservoirDataRequest] = None, use_real_data: bool = True, reservoir_name: str = "red_river", days: int = 30):
    """加载水库数据"""
    try:
        if use_real_data and request is None:
            # 使用真实数据
            result = water_management_system.load_reservoir_data(
                use_real_data=True, 
                reservoir_name=reservoir_name, 
                days=days
            )
            
            return {
                "status": "success",
                "message": result['message'],
                "data_source": result.get('data_source', 'unknown'),
                "data_points": result.get('data_points', 0),
                "date_range": result.get('date_range', 'unknown'),
                "reservoir_name": reservoir_name,
                "days": days,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # 使用提供的数据
            data = {
                'inflow': request.inflow,
                'demand': request.demand,
                'capacity': request.capacity,
                'min_level': request.min_level,
                'max_level': request.max_level
            }
            
            result = water_management_system.load_reservoir_data(data, use_real_data=False)
            
            return {
                "status": "success",
                "message": "水库数据加载成功",
                "data_summary": {
                    "inflow_periods": len(request.inflow),
                    "demand_periods": len(request.demand),
                    "capacity": request.capacity,
                    "level_range": f"{request.min_level*100:.1f}% - {request.max_level*100:.1f}%"
                },
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"水库数据加载失败: {str(e)}")

@router.post("/api/v1/water-resources/load-allocation-data")
async def load_allocation_data(request: Optional[AllocationDataRequest] = None, use_real_data: bool = True, region: str = "manitoba"):
    """加载水资源配置数据"""
    try:
        if use_real_data and request is None:
            # 使用真实数据
            result = water_management_system.load_allocation_data(
                use_real_data=True, 
                region=region
            )
            
            return {
                "status": "success",
                "message": result['message'],
                "data_source": result.get('data_source', 'unknown'),
                "region": result.get('region', 'unknown'),
                "data_quality": result.get('data_quality', 'unknown'),
                "timestamp": datetime.now().isoformat()
            }
        else:
            # 使用提供的数据
            data = {
                'users': request.users,
                'priorities': request.priorities,
                'efficiency_factors': request.efficiency_factors,
                'total_water': request.total_water
            }
            
            result = water_management_system.load_allocation_data(data, use_real_data=False)
            
            return {
                "status": "success",
                "message": "配置数据加载成功",
                "data_summary": {
                    "users": len(request.users),
                    "total_water": request.total_water,
                    "priorities_defined": len(request.priorities),
                    "efficiency_factors_defined": len(request.efficiency_factors)
                },
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"配置数据加载失败: {str(e)}")

@router.post("/api/v1/water-resources/reservoir-optimization", response_model=OptimizationResponse)
async def run_reservoir_optimization(request: OptimizationRequest):
    """运行水库调度优化"""
    try:
        result = water_management_system.run_reservoir_optimization(request.objectives)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return OptimizationResponse(
            status="success",
            optimization_type="reservoir_optimization",
            results=result,
            recommendations=[
                "优化完成，建议定期更新水文数据",
                "考虑气候变化对水库调度的影响",
                "建立多情景分析以应对不确定性"
            ],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"水库调度优化失败: {str(e)}")

@router.post("/api/v1/water-resources/allocation-optimization", response_model=OptimizationResponse)
async def run_allocation_optimization():
    """运行水资源配置优化"""
    try:
        result = water_management_system.run_allocation_optimization()
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return OptimizationResponse(
            status="success",
            optimization_type="allocation_optimization",
            results=result,
            recommendations=[
                "配置优化完成，建议监控实际用水情况",
                "定期评估用户优先级和效率系数",
                "建立动态调整机制以适应需求变化"
            ],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"水资源配置优化失败: {str(e)}")

@router.get("/management-report")
async def get_management_report():
    """获取水资源管理报告"""
    try:
        report = water_management_system.generate_management_report()
        
        return {
            "status": "success",
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"管理报告生成失败: {str(e)}")

@router.get("/optimization-methods")
async def get_optimization_methods():
    """获取可用的优化方法"""
    return {
        "status": "success",
        "optimization_methods": {
            "reservoir_optimization": {
                "name": "水库调度优化",
                "description": "基于多目标优化的水库放水策略优化",
                "objectives": {
                    "reliability": "供水可靠性 - 最大化满足用水需求",
                    "efficiency": "用水效率 - 最小化水资源浪费",
                    "safety": "防洪安全 - 控制水库水位在安全范围内"
                },
                "constraints": [
                    "蓄水量约束",
                    "放水量约束",
                    "水位安全约束"
                ],
                "algorithms": ["多目标优化", "差分进化算法", "SLSQP优化器"]
            },
            "allocation_optimization": {
                "name": "水资源配置优化",
                "description": "多用户水资源公平高效配置",
                "objectives": {
                    "efficiency": "配置效率 - 最大化整体用水效率",
                    "equity": "公平性 - 最小化用户间配置差异",
                    "priority": "优先级 - 满足用户优先级要求"
                },
                "constraints": [
                    "总水量约束",
                    "最小配置约束",
                    "用户需求约束"
                ],
                "algorithms": ["多目标优化", "约束优化", "权重平衡"]
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/real-data")
async def get_real_data(reservoir_name: str = "red_river", region: str = "manitoba", days: int = 30):
    """获取真实数据"""
    try:
        # 获取真实水库数据
        reservoir_data = water_management_system.real_data_loader.load_reservoir_data(reservoir_name, days)
        
        # 获取真实配置数据
        allocation_data = water_management_system.real_data_loader.load_water_allocation_data(region)
        
        # 获取数据源摘要
        data_summary = water_management_system.real_data_loader.get_data_summary()
        
        return {
            "status": "success",
            "real_data": {
                "reservoir_data": {
                    "inflow": reservoir_data['inflow'][:10],  # 只返回前10个数据点
                    "demand": reservoir_data['demand'][:10],
                    "capacity": reservoir_data['capacity'],
                    "min_level": reservoir_data['min_level'],
                    "max_level": reservoir_data['max_level'],
                    "source": reservoir_data['source'],
                    "data_points": reservoir_data['data_points'],
                    "date_range": reservoir_data['date_range']
                },
                "allocation_data": {
                    "users": allocation_data['users'],
                    "priorities": allocation_data['priorities'],
                    "efficiency_factors": allocation_data['efficiency_factors'],
                    "total_water": allocation_data['total_water'],
                    "source": allocation_data['source'],
                    "region": allocation_data['region'],
                    "data_quality": allocation_data['data_quality']
                }
            },
            "data_summary": data_summary,
            "usage_examples": {
                "reservoir_optimization": {
                    "step1": "POST /api/v1/water-resources/load-reservoir-data?use_real_data=true",
                    "step2": "POST /api/v1/water-resources/reservoir-optimization",
                    "description": "加载真实水库数据后运行调度优化"
                },
                "allocation_optimization": {
                    "step1": "POST /api/v1/water-resources/load-allocation-data?use_real_data=true", 
                    "step2": "POST /api/v1/water-resources/allocation-optimization",
                    "description": "加载真实配置数据后运行配置优化"
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取真实数据失败: {str(e)}")

@router.get("/sample-data")
async def get_sample_data():
    """获取示例数据（兼容性保留）"""
    return {
        "status": "success",
        "sample_data": {
            "reservoir_data": {
                "inflow": [50, 60, 45, 70, 55, 40, 65, 50, 75, 60],
                "demand": [30, 35, 25, 40, 30, 20, 35, 30, 45, 35],
                "capacity": 1000.0,
                "min_level": 0.1,
                "max_level": 0.9
            },
            "allocation_data": {
                "users": ["农业用水", "工业用水", "生活用水", "生态用水"],
                "priorities": {
                    "生活用水": 1.0,
                    "农业用水": 0.8,
                    "工业用水": 0.7,
                    "生态用水": 0.6
                },
                "efficiency_factors": {
                    "生活用水": 0.9,
                    "农业用水": 0.7,
                    "工业用水": 0.8,
                    "生态用水": 0.6
                },
                "total_water": 1000.0
            }
        },
        "usage_examples": {
            "reservoir_optimization": {
                "step1": "POST /api/v1/water-resources/load-reservoir-data",
                "step2": "POST /api/v1/water-resources/reservoir-optimization",
                "description": "加载水库数据后运行调度优化"
            },
            "allocation_optimization": {
                "step1": "POST /api/v1/water-resources/load-allocation-data", 
                "step2": "POST /api/v1/water-resources/allocation-optimization",
                "description": "加载配置数据后运行配置优化"
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/system-status")
async def get_system_status():
    """获取系统状态"""
    try:
        report = water_management_system.generate_management_report()
        
        return {
            "status": "success",
            "system_status": report['system_status'],
            "optimization_history_count": len(report['optimization_history']),
            "last_optimization": report['optimization_history'][-1] if report['optimization_history'] else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"系统状态获取失败: {str(e)}")
