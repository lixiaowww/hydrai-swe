#!/usr/bin/env python3
"""
HydrAI-SWE 预测验证API接口
提供预测结果验证的REST API服务
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
import asyncio

# 导入验证器
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.validation.prediction_validator import PredictionQualityValidator, ValidationResult
from models.validation.real_time_validator import RealTimeValidator, RealTimeValidationResult

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/prediction-validation", tags=["prediction-validation"])

# 全局验证器实例
prediction_validator = None
real_time_validator = None

# 数据模型
class ValidationRequest(BaseModel):
    """验证请求数据模型"""
    predictions: List[Dict[str, Any]] = Field(..., description="预测结果数据")
    variable_type: str = Field(..., description="变量类型")
    source_name: str = Field(..., description="数据源名称")
    prediction_id: Optional[str] = Field(None, description="预测ID")
    include_historical_validation: bool = Field(True, description="是否包含历史数据验证")

class MultiSourceValidationRequest(BaseModel):
    """多源验证请求数据模型"""
    predictions: Dict[str, List[Dict[str, Any]]] = Field(..., description="多数据源预测结果")
    variable_type: str = Field(..., description="变量类型")

class RealTimeValidationRequest(BaseModel):
    """实时验证请求数据模型"""
    predictions: List[Dict[str, Any]] = Field(..., description="预测结果数据")
    variable_type: str = Field(..., description="变量类型")
    source_name: str = Field(..., description="数据源名称")
    prediction_id: Optional[str] = Field(None, description="预测ID")

class ValidationResponse(BaseModel):
    """验证响应数据模型"""
    success: bool
    message: str
    validation_result: Optional[Dict[str, Any]] = None
    timestamp: datetime

class RealTimeValidationResponse(BaseModel):
    """实时验证响应数据模型"""
    success: bool
    message: str
    validation_result: Optional[Dict[str, Any]] = None
    timestamp: datetime

class ValidationStatusResponse(BaseModel):
    """验证状态响应数据模型"""
    success: bool
    status: Dict[str, Any]
    timestamp: datetime

# 初始化函数
def initialize_validators():
    """初始化验证器"""
    global prediction_validator, real_time_validator
    
    try:
        # 初始化预测质量验证器
        prediction_validator = PredictionQualityValidator()
        logger.info("✅ 预测质量验证器初始化完成")
        
        # 初始化实时验证器
        real_time_validator = RealTimeValidator()
        logger.info("✅ 实时验证器初始化完成")
        
    except Exception as e:
        logger.error(f"❌ 验证器初始化失败: {e}")
        raise

# 启动时初始化
@router.on_event("startup")
async def startup_event():
    """启动事件"""
    initialize_validators()

# 工具函数
def convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """将字典列表转换为DataFrame"""
    try:
        df = pd.DataFrame(data)
        
        # 尝试解析时间列
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"❌ 数据转换失败: {e}")
        raise HTTPException(status_code=400, detail=f"数据格式错误: {e}")

def get_historical_data(variable_type: str) -> Optional[pd.DataFrame]:
    """获取历史数据用于验证"""
    try:
        # 这里应该根据实际数据源获取历史数据
        # 目前使用示例数据
        if variable_type == 'soil_moisture':
            # 从已训练的数据中获取历史数据
            historical_file = "data/processed/manitoba_agriculture_fixed.csv"
            if os.path.exists(historical_file):
                df = pd.read_csv(historical_file)
                if 'estimated_soil_moisture' in df.columns:
                    return df[['estimated_soil_moisture']].rename(
                        columns={'estimated_soil_moisture': 'soil_moisture'}
                    )
        
        # 如果没有找到历史数据，返回None
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ 获取历史数据失败: {e}")
        return None

# API端点

@router.post("/validate", response_model=ValidationResponse)
async def validate_prediction_quality(request: ValidationRequest):
    """
    验证预测结果质量
    
    - **predictions**: 预测结果数据列表
    - **variable_type**: 变量类型 (soil_moisture, snow_water_equivalent, runoff, temperature, precipitation)
    - **source_name**: 数据源名称
    - **prediction_id**: 预测ID（可选）
    - **include_historical_validation**: 是否包含历史数据验证
    """
    try:
        logger.info(f"🔍 开始验证预测质量: {request.variable_type} from {request.source_name}")
        
        # 检查验证器是否初始化
        if prediction_validator is None:
            initialize_validators()
        
        # 转换数据格式
        predictions_df = convert_to_dataframe(request.predictions)
        
        # 获取历史数据
        historical_data = None
        if request.include_historical_validation:
            historical_data = get_historical_data(request.variable_type)
        
        # 执行验证
        validation_result = prediction_validator.validate_prediction_quality(
            predictions=predictions_df,
            variable_type=request.variable_type,
            historical_data=historical_data,
            source_name=request.source_name
        )
        
        # 转换为可序列化的格式
        result_dict = {
            'is_valid': validation_result.is_valid,
            'confidence_score': validation_result.confidence_score,
            'validation_details': validation_result.validation_details,
            'warnings': validation_result.warnings,
            'errors': validation_result.errors,
            'recommendations': validation_result.recommendations,
            'timestamp': validation_result.timestamp.isoformat()
        }
        
        logger.info(f"✅ 预测质量验证完成: 分数 {validation_result.confidence_score:.2%}")
        
        return ValidationResponse(
            success=True,
            message="预测质量验证完成",
            validation_result=result_dict,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"❌ 预测质量验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")

@router.post("/validate-multi-source", response_model=ValidationResponse)
async def validate_multi_source_consistency(request: MultiSourceValidationRequest):
    """
    验证多数据源预测结果的一致性
    
    - **predictions**: 多数据源预测结果字典
    - **variable_type**: 变量类型
    """
    try:
        logger.info(f"🔍 开始多源一致性验证: {request.variable_type}")
        
        # 检查验证器是否初始化
        if prediction_validator is None:
            initialize_validators()
        
        # 转换数据格式
        predictions_dict = {}
        for source_name, data_list in request.predictions.items():
            predictions_dict[source_name] = convert_to_dataframe(data_list)
        
        # 执行验证
        validation_result = prediction_validator.validate_prediction_quality(
            predictions=predictions_dict,
            variable_type=request.variable_type,
            source_name="multi_source"
        )
        
        # 转换为可序列化的格式
        result_dict = {
            'is_valid': validation_result.is_valid,
            'confidence_score': validation_result.confidence_score,
            'validation_details': validation_result.validation_details,
            'warnings': validation_result.warnings,
            'errors': validation_result.errors,
            'recommendations': validation_result.recommendations,
            'timestamp': validation_result.timestamp.isoformat()
        }
        
        logger.info(f"✅ 多源一致性验证完成: 分数 {validation_result.confidence_score:.2%}")
        
        return ValidationResponse(
            success=True,
            message="多源一致性验证完成",
            validation_result=result_dict,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"❌ 多源一致性验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")

@router.post("/real-time/validate", response_model=RealTimeValidationResponse)
async def add_real_time_validation_task(request: RealTimeValidationRequest):
    """
    添加实时验证任务
    
    - **predictions**: 预测结果数据列表
    - **variable_type**: 变量类型
    - **source_name**: 数据源名称
    - **prediction_id**: 预测ID（可选）
    """
    try:
        logger.info(f"🔍 添加实时验证任务: {request.variable_type} from {request.source_name}")
        
        # 检查验证器是否初始化
        if real_time_validator is None:
            initialize_validators()
        
        # 转换数据格式
        predictions_df = convert_to_dataframe(request.predictions)
        
        # 添加验证任务
        real_time_validator.add_validation_task(
            predictions=predictions_df,
            variable_type=request.variable_type,
            source_name=request.source_name,
            prediction_id=request.prediction_id
        )
        
        return RealTimeValidationResponse(
            success=True,
            message="实时验证任务已添加",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"❌ 添加实时验证任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加任务失败: {str(e)}")

@router.get("/real-time/status", response_model=ValidationStatusResponse)
async def get_real_time_validation_status():
    """获取实时验证状态"""
    try:
        if real_time_validator is None:
            raise HTTPException(status_code=503, detail="实时验证器未初始化")
        
        status = real_time_validator.get_validation_status()
        
        return ValidationStatusResponse(
            success=True,
            status=status,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"❌ 获取实时验证状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.get("/real-time/results")
async def get_recent_real_time_results(count: int = Query(10, ge=1, le=100)):
    """获取最近的实时验证结果"""
    try:
        if real_time_validator is None:
            raise HTTPException(status_code=503, detail="实时验证器未初始化")
        
        results = real_time_validator.get_recent_results(count)
        
        # 转换为可序列化的格式
        results_list = []
        for result in results:
            results_list.append({
                'timestamp': result.timestamp.isoformat(),
                'prediction_id': result.prediction_id,
                'is_valid': result.is_valid,
                'quality_score': result.quality_score,
                'alerts': result.alerts,
                'metrics': result.metrics,
                'recommendations': result.recommendations
            })
        
        return {
            'success': True,
            'results': results_list,
            'count': len(results_list),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ 获取实时验证结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")

@router.post("/real-time/initialize-reference")
async def initialize_reference_distribution(
    variable_type: str = Query(..., description="变量类型"),
    source_name: str = Query(..., description="数据源名称")
):
    """初始化实时验证器的参考分布"""
    try:
        if real_time_validator is None:
            initialize_validators()
        
        # 获取历史数据
        historical_data = get_historical_data(variable_type)
        if historical_data is None:
            raise HTTPException(status_code=400, detail="无法获取历史数据")
        
        # 初始化参考分布
        real_time_validator.initialize_reference_distribution(historical_data)
        
        return {
            'success': True,
            'message': f"参考分布初始化完成: {variable_type} from {source_name}",
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ 初始化参考分布失败: {e}")
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")

@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        prediction_validator_ok = prediction_validator is not None
        real_time_validator_ok = real_time_validator is not None
        
        return {
            'status': 'healthy',
            'prediction_validator': 'ok' if prediction_validator_ok else 'not_initialized',
            'real_time_validator': 'ok' if real_time_validator_ok else 'not_initialized',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ 健康检查失败: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@router.get("/metrics")
async def get_validation_metrics():
    """获取验证指标统计"""
    try:
        metrics = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_confidence_score': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # 统计预测验证器指标
        if prediction_validator is not None:
            # 这里可以添加更多统计信息
            pass
        
        # 统计实时验证器指标
        if real_time_validator is not None:
            status = real_time_validator.get_validation_status()
            metrics['total_validations'] = status.get('total_validations', 0)
            
            # 计算平均置信度分数
            recent_results = real_time_validator.get_recent_results(100)
            if recent_results:
                scores = [result.quality_score for result in recent_results]
                metrics['average_confidence_score'] = sum(scores) / len(scores)
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ 获取验证指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")

# 后台任务
@router.post("/batch-validate")
async def batch_validate_predictions(
    background_tasks: BackgroundTasks,
    requests: List[ValidationRequest]
):
    """批量验证预测结果（后台任务）"""
    try:
        logger.info(f"🔍 开始批量验证: {len(requests)} 个任务")
        
        # 添加后台任务
        background_tasks.add_task(process_batch_validation, requests)
        
        return {
            'success': True,
            'message': f"批量验证任务已启动，共 {len(requests)} 个任务",
            'task_id': f"batch_{int(datetime.now().timestamp())}",
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ 启动批量验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动批量验证失败: {str(e)}")

async def process_batch_validation(requests: List[ValidationRequest]):
    """处理批量验证任务"""
    try:
        results = []
        
        for i, request in enumerate(requests):
            try:
                logger.info(f"处理批量验证任务 {i+1}/{len(requests)}")
                
                # 转换数据格式
                predictions_df = convert_to_dataframe(request.predictions)
                
                # 获取历史数据
                historical_data = None
                if request.include_historical_validation:
                    historical_data = get_historical_data(request.variable_type)
                
                # 执行验证
                validation_result = prediction_validator.validate_prediction_quality(
                    predictions=predictions_df,
                    variable_type=request.variable_type,
                    historical_data=historical_data,
                    source_name=request.source_name
                )
                
                results.append({
                    'request_index': i,
                    'success': True,
                    'result': {
                        'is_valid': validation_result.is_valid,
                        'confidence_score': validation_result.confidence_score,
                        'warnings': validation_result.warnings,
                        'errors': validation_result.errors
                    }
                })
                
            except Exception as e:
                logger.error(f"批量验证任务 {i+1} 失败: {e}")
                results.append({
                    'request_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        # 保存批量验证结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"validation_results/batch_validation_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 批量验证完成，结果已保存: {results_file}")
        
    except Exception as e:
        logger.error(f"❌ 批量验证处理失败: {e}")

# 启动时确保目录存在
os.makedirs("validation_results", exist_ok=True)
os.makedirs("real_time_validation", exist_ok=True)
