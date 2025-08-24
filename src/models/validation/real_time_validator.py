#!/usr/bin/env python3
"""
HydrAI-SWE 实时监控验证器
在线监控预测结果质量，实时检测异常和性能下降
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RealTimeValidationResult:
    """实时验证结果数据类"""
    timestamp: datetime
    prediction_id: str
    is_valid: bool
    quality_score: float
    alerts: List[str]
    metrics: Dict[str, float]
    recommendations: List[str]

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, window_size: int = 100, alert_threshold: float = 0.8):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.performance_history = deque(maxlen=window_size)
        self.alert_history = deque(maxlen=100)
        
    def add_performance_metric(self, metric_name: str, value: float, timestamp: datetime):
        """添加性能指标"""
        self.performance_history.append({
            'timestamp': timestamp,
            'metric_name': metric_name,
            'value': value
        })
    
    def get_performance_trend(self, metric_name: str, window: int = None) -> Dict[str, Any]:
        """获取性能趋势"""
        if window is None:
            window = self.window_size
        
        # 过滤指定指标
        metric_data = [
            item for item in self.performance_history 
            if item['metric_name'] == metric_name
        ][-window:]
        
        if len(metric_data) < 2:
            return {
                'trend': 'insufficient_data',
                'slope': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'alert': False
            }
        
        values = [item['value'] for item in metric_data]
        timestamps = [item['timestamp'] for item in metric_data]
        
        # 计算趋势
        x = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])
        y = np.array(values)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            mean = np.mean(values)
            std = np.std(values)
            
            # 判断趋势
            if slope > 0.01:
                trend = 'improving'
            elif slope < -0.01:
                trend = 'declining'
            else:
                trend = 'stable'
            
            # 判断是否需要告警
            alert = mean < self.alert_threshold or (trend == 'declining' and slope < -0.05)
            
            return {
                'trend': trend,
                'slope': slope,
                'mean': mean,
                'std': std,
                'alert': alert,
                'data_points': len(values)
            }
        
        return {
            'trend': 'insufficient_data',
            'slope': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'alert': False
        }
    
    def add_alert(self, alert_type: str, message: str, severity: str = 'warning'):
        """添加告警"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        self.alert_history.append(alert)
        logger.warning(f"🚨 告警 [{severity.upper()}]: {message}")

class DriftDetector:
    """数据漂移检测器"""
    
    def __init__(self, reference_window: int = 1000, detection_threshold: float = 0.1):
        self.reference_window = reference_window
        self.detection_threshold = detection_threshold
        self.reference_distribution = None
        self.is_initialized = False
        
    def initialize_reference(self, reference_data: pd.DataFrame):
        """初始化参考分布"""
        logger.info("🔧 初始化数据漂移检测参考分布...")
        
        # 计算参考分布的统计特征
        self.reference_distribution = {
            'mean': reference_data.mean().to_dict(),
            'std': reference_data.std().to_dict(),
            'quantiles': reference_data.quantile([0.25, 0.5, 0.75]).to_dict()
        }
        self.is_initialized = True
        
        logger.info("✅ 参考分布初始化完成")
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """检测数据漂移"""
        if not self.is_initialized:
            raise RuntimeError("数据漂移检测器尚未初始化，请先调用initialize_reference()")
        
        logger.info("🔍 开始数据漂移检测...")
        
        drift_results = {}
        total_features = len(current_data.columns)
        drifted_features = 0
        
        for column in current_data.columns:
            if column in self.reference_distribution['mean']:
                ref_mean = self.reference_distribution['mean'][column]
                ref_std = self.reference_distribution['std'][column]
                
                current_mean = current_data[column].mean()
                current_std = current_data[column].std()
                
                # 计算分布差异
                mean_diff = abs(current_mean - ref_mean) / (abs(ref_mean) + 1e-8)
                std_diff = abs(current_std - ref_std) / (abs(ref_std) + 1e-8)
                
                # 判断是否漂移
                is_drifted = mean_diff > self.detection_threshold or std_diff > self.detection_threshold
                
                drift_results[column] = {
                    'is_drifted': is_drifted,
                    'mean_difference': mean_diff,
                    'std_difference': std_diff,
                    'reference_mean': ref_mean,
                    'current_mean': current_mean,
                    'reference_std': ref_std,
                    'current_std': current_std
                }
                
                if is_drifted:
                    drifted_features += 1
        
        # 计算整体漂移分数
        drift_score = drifted_features / total_features if total_features > 0 else 0.0
        
        result = {
            'overall_drift_score': drift_score,
            'total_features': total_features,
            'drifted_features': drifted_features,
            'feature_details': drift_results,
            'is_drifted': drift_score > 0.2,  # 超过20%的特征漂移认为整体漂移
            'timestamp': datetime.now()
        }
        
        logger.info(f"✅ 数据漂移检测完成: 漂移分数 {drift_score:.2%}")
        return result

class RealTimeValidator:
    """实时验证器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # 初始化组件
        self.performance_monitor = PerformanceMonitor(
            window_size=self.config['performance_window_size'],
            alert_threshold=self.config['alert_threshold']
        )
        self.drift_detector = DriftDetector(
            reference_window=self.config['reference_window_size'],
            detection_threshold=self.config['drift_threshold']
        )
        
        # 验证队列和结果存储
        self.validation_queue = queue.Queue()
        self.validation_results = deque(maxlen=1000)
        self.alert_history = deque(maxlen=100)
        
        # 创建目录
        os.makedirs("real_time_validation", exist_ok=True)
        os.makedirs("real_time_validation/results", exist_ok=True)
        os.makedirs("real_time_validation/alerts", exist_ok=True)
        
        # 启动监控线程
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🚀 实时验证器启动完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'performance_window_size': 100,
            'alert_threshold': 0.8,
            'reference_window_size': 1000,
            'drift_threshold': 0.1,
            'validation_interval': 1.0,  # 秒
            'max_queue_size': 1000,
            'save_interval': 60  # 秒
        }
    
    def _monitoring_loop(self):
        """监控循环"""
        last_save_time = time.time()
        
        while self.monitoring_active:
            try:
                # 处理验证队列
                while not self.validation_queue.empty():
                    validation_task = self.validation_queue.get_nowait()
                    self._process_validation_task(validation_task)
                
                # 定期保存结果
                current_time = time.time()
                if current_time - last_save_time > self.config['save_interval']:
                    self._save_monitoring_data()
                    last_save_time = current_time
                
                time.sleep(self.config['validation_interval'])
                
            except Exception as e:
                logger.error(f"❌ 监控循环错误: {e}")
                time.sleep(5)  # 错误后等待5秒再继续
    
    def _process_validation_task(self, task: Dict[str, Any]):
        """处理验证任务"""
        try:
            prediction_id = task.get('prediction_id', f"pred_{int(time.time())}")
            predictions = task['predictions']
            variable_type = task['variable_type']
            source_name = task.get('source_name', 'unknown')
            
            # 执行验证
            validation_result = self._validate_single_prediction(
                predictions, variable_type, source_name, prediction_id
            )
            
            # 存储结果
            self.validation_results.append(validation_result)
            
            # 更新性能监控
            self.performance_monitor.add_performance_metric(
                'quality_score', validation_result.quality_score, validation_result.timestamp
            )
            
            # 检查告警
            if validation_result.quality_score < self.config['alert_threshold']:
                alert_message = f"预测质量下降: {validation_result.quality_score:.2%}"
                self.performance_monitor.add_alert('quality_decline', alert_message, 'warning')
            
            logger.info(f"✅ 验证任务完成: {prediction_id}, 质量分数: {validation_result.quality_score:.2%}")
            
        except Exception as e:
            logger.error(f"❌ 处理验证任务失败: {e}")
    
    def _validate_single_prediction(self, predictions: pd.DataFrame, 
                                  variable_type: str, source_name: str,
                                  prediction_id: str) -> RealTimeValidationResult:
        """验证单个预测结果"""
        timestamp = datetime.now()
        alerts = []
        recommendations = []
        
        # 基础质量检查
        quality_score = self._calculate_basic_quality(predictions, variable_type)
        
        # 数据漂移检测
        if self.drift_detector.is_initialized:
            try:
                drift_result = self.drift_detector.detect_drift(predictions)
                if drift_result['is_drifted']:
                    alerts.append(f"检测到数据漂移: {drift_result['overall_drift_score']:.2%}")
                    recommendations.append("建议重新训练模型或更新参考分布")
            except Exception as e:
                logger.warning(f"数据漂移检测失败: {e}")
        
        # 性能趋势分析
        trend_result = self.performance_monitor.get_performance_trend('quality_score', 20)
        if trend_result['alert']:
            alerts.append(f"性能趋势下降: {trend_result['trend']}")
            recommendations.append("建议检查模型状态和数据质量")
        
        # 计算综合质量分数
        final_quality_score = quality_score * 0.7 + (1 - trend_result['slope']) * 0.3
        final_quality_score = max(0.0, min(1.0, final_quality_score))
        
        # 判断有效性
        is_valid = final_quality_score > self.config['alert_threshold'] and len(alerts) < 3
        
        # 生成建议
        if final_quality_score < 0.7:
            recommendations.append("预测质量较低，建议检查模型")
        elif final_quality_score < 0.9:
            recommendations.append("预测质量中等，建议优化")
        else:
            recommendations.append("预测质量良好")
        
        # 计算指标
        metrics = {
            'quality_score': final_quality_score,
            'prediction_count': len(predictions),
            'alert_count': len(alerts),
            'trend_slope': trend_result.get('slope', 0.0)
        }
        
        return RealTimeValidationResult(
            timestamp=timestamp,
            prediction_id=prediction_id,
            is_valid=is_valid,
            quality_score=final_quality_score,
            alerts=alerts,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _calculate_basic_quality(self, predictions: pd.DataFrame, variable_type: str) -> float:
        """计算基础质量分数"""
        if predictions.empty:
            return 0.0
        
        # 检查缺失值
        missing_ratio = predictions.isnull().sum().sum() / (len(predictions) * len(predictions.columns))
        
        # 检查数值范围
        if variable_type == 'soil_moisture':
            valid_range = (0.0, 1.0)
        elif variable_type == 'snow_water_equivalent':
            valid_range = (0.0, 2000.0)
        elif variable_type == 'runoff':
            valid_range = (0.0, 10000.0)
        else:
            valid_range = (float('-inf'), float('inf'))
        
        in_range_ratio = np.mean(
            (predictions >= valid_range[0]) & (predictions <= valid_range[1])
        )
        
        # 综合质量分数
        quality_score = (1 - missing_ratio) * 0.5 + in_range_ratio * 0.5
        return quality_score
    
    def add_validation_task(self, predictions: pd.DataFrame, variable_type: str, 
                           source_name: str = "unknown", prediction_id: str = None):
        """添加验证任务到队列"""
        if prediction_id is None:
            prediction_id = f"pred_{int(time.time())}"
        
        task = {
            'prediction_id': prediction_id,
            'predictions': predictions,
            'variable_type': variable_type,
            'source_name': source_name,
            'timestamp': datetime.now()
        }
        
        try:
            self.validation_queue.put_nowait(task)
            logger.info(f"✅ 验证任务已添加到队列: {prediction_id}")
        except queue.Full:
            logger.warning(f"⚠️ 验证队列已满，丢弃任务: {prediction_id}")
    
    def initialize_reference_distribution(self, reference_data: pd.DataFrame):
        """初始化参考分布（用于漂移检测）"""
        self.drift_detector.initialize_reference(reference_data)
        logger.info("✅ 参考分布初始化完成")
    
    def get_validation_status(self) -> Dict[str, Any]:
        """获取验证状态"""
        return {
            'queue_size': self.validation_queue.qsize(),
            'total_validations': len(self.validation_results),
            'active_monitoring': self.monitoring_active,
            'last_validation_time': self.validation_results[-1].timestamp if self.validation_results else None,
            'performance_trend': self.performance_monitor.get_performance_trend('quality_score', 50)
        }
    
    def get_recent_results(self, count: int = 10) -> List[RealTimeValidationResult]:
        """获取最近的验证结果"""
        return list(self.validation_results)[-count:]
    
    def _save_monitoring_data(self):
        """保存监控数据"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存验证结果
            results_file = f"real_time_validation/results/validation_results_{timestamp}.json"
            recent_results = self.get_recent_results(100)
            
            results_data = []
            for result in recent_results:
                results_data.append({
                    'timestamp': result.timestamp.isoformat(),
                    'prediction_id': result.prediction_id,
                    'is_valid': result.is_valid,
                    'quality_score': result.quality_score,
                    'alerts': result.alerts,
                    'metrics': result.metrics,
                    'recommendations': result.recommendations
                })
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            # 保存告警历史
            alerts_file = f"real_time_validation/alerts/alerts_{timestamp}.json"
            alerts_data = []
            for alert in self.performance_monitor.alert_history:
                alerts_data.append({
                    'timestamp': alert['timestamp'].isoformat(),
                    'type': alert['type'],
                    'message': alert['message'],
                    'severity': alert['severity']
                })
            
            with open(alerts_file, 'w', encoding='utf-8') as f:
                json.dump(alerts_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 监控数据已保存: {results_file}, {alerts_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存监控数据失败: {e}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 实时监控已停止")

def main():
    """主函数：演示实时验证器使用"""
    logger.info("🚀 启动实时验证器演示")
    
    # 创建实时验证器
    validator = RealTimeValidator()
    
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    
    # 初始化参考分布
    reference_data = pd.DataFrame({
        'soil_moisture': np.random.uniform(0.1, 0.8, 1000)
    })
    validator.initialize_reference_distribution(reference_data)
    
    # 模拟实时预测验证
    for i in range(10):
        # 生成预测数据
        predictions = pd.DataFrame({
            'soil_moisture': np.random.uniform(0.1, 0.8, 10)
        }, index=dates[i*10:(i+1)*10])
        
        # 添加验证任务
        validator.add_validation_task(
            predictions, 'soil_moisture', 'demo_model', f"demo_pred_{i}"
        )
        
        # 等待处理
        time.sleep(2)
    
    # 等待所有任务处理完成
    time.sleep(5)
    
    # 获取状态和结果
    status = validator.get_validation_status()
    recent_results = validator.get_recent_results(5)
    
    logger.info(f"验证状态: {status}")
    logger.info(f"最近结果数量: {len(recent_results)}")
    
    # 停止监控
    validator.stop_monitoring()
    
    logger.info("✅ 实时验证器演示完成")

if __name__ == "__main__":
    main()
