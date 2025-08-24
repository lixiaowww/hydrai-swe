#!/usr/bin/env python3
"""
HydrAI-SWE 预测结果验证器
确保生产环境中预测结果的质量和可信度
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    confidence_score: float
    validation_details: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    timestamp: datetime

class PhysicalConstraintValidator:
    """物理约束验证器"""
    
    def __init__(self):
        # 定义物理约束范围
        self.constraints = {
            'soil_moisture': {
                'min': 0.0,
                'max': 1.0,
                'unit': 'm³/m³',
                'description': '土壤湿度应在0-1之间'
            },
            'snow_water_equivalent': {
                'min': 0.0,
                'max': 2000.0,
                'unit': 'mm',
                'description': '积雪水当量应在0-2000mm之间'
            },
            'runoff': {
                'min': 0.0,
                'max': 10000.0,
                'unit': 'm³/s',
                'description': '径流应在0-10000m³/s之间'
            },
            'temperature': {
                'min': -50.0,
                'max': 50.0,
                'unit': '°C',
                'description': '温度应在-50到50°C之间'
            },
            'precipitation': {
                'min': 0.0,
                'max': 500.0,
                'unit': 'mm/day',
                'description': '日降水量应在0-500mm之间'
            }
        }
    
    def validate_physical_constraints(self, predictions: pd.DataFrame, 
                                   variable_type: str) -> Dict[str, Any]:
        """
        验证预测结果的物理合理性
        
        Args:
            predictions: 预测结果DataFrame
            variable_type: 变量类型 (soil_moisture, snow_water_equivalent, runoff, temperature, precipitation)
        
        Returns:
            验证结果字典
        """
        logger.info(f"🔍 开始物理约束验证: {variable_type}")
        
        if variable_type not in self.constraints:
            raise ValueError(f"未知的变量类型: {variable_type}")
        
        constraint = self.constraints[variable_type]
        min_val = constraint['min']
        max_val = constraint['max']
        
        # 检查数值范围
        out_of_range = predictions[
            (predictions < min_val) | (predictions > max_val)
        ]
        
        # 检查异常跳跃
        if len(predictions) > 1:
            diff = predictions.diff().abs()
            jump_threshold = (max_val - min_val) * 0.5  # 50%的合理范围作为跳跃阈值
            large_jumps = diff[diff > jump_threshold]
        else:
            large_jumps = pd.Series(dtype=float)
        
        # 计算验证分数
        total_points = len(predictions)
        valid_points = total_points - len(out_of_range)
        physical_score = valid_points / total_points if total_points > 0 else 0.0
        
        result = {
            'is_valid': len(out_of_range) == 0,
            'physical_score': physical_score,
            'total_points': total_points,
            'valid_points': valid_points,
            'out_of_range_count': len(out_of_range),
            'out_of_range_values': out_of_range.to_dict() if len(out_of_range) > 0 else {},
            'large_jumps_count': len(large_jumps),
            'large_jumps_values': large_jumps.to_dict() if len(large_jumps) > 0 else {},
            'constraint': constraint,
            'warnings': [],
            'errors': []
        }
        
        # 生成警告和错误信息
        if len(out_of_range) > 0:
            result['errors'].append(
                f"发现 {len(out_of_range)} 个超出物理范围的预测值 "
                f"({min_val} - {max_val} {constraint['unit']})"
            )
        
        if len(large_jumps) > 0:
            result['warnings'].append(
                f"发现 {len(large_jumps)} 个异常跳跃，可能表示预测不稳定"
            )
        
        if physical_score < 0.95:
            result['warnings'].append(
                f"物理合理性分数较低: {physical_score:.2%}，建议检查模型"
            )
        
        logger.info(f"✅ 物理约束验证完成: 分数 {physical_score:.2%}")
        return result

class StatisticalAnomalyDetector:
    """统计异常检测器"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, historical_data: pd.DataFrame):
        """训练异常检测模型"""
        logger.info("🔧 训练统计异常检测模型...")
        
        # 标准化数据
        scaled_data = self.scaler.fit_transform(historical_data)
        
        # 训练隔离森林
        self.isolation_forest.fit(scaled_data)
        self.is_fitted = True
        
        logger.info("✅ 统计异常检测模型训练完成")
    
    def detect_anomalies(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        检测预测结果中的统计异常
        
        Args:
            predictions: 预测结果DataFrame
        
        Returns:
            异常检测结果字典
        """
        if not self.is_fitted:
            raise RuntimeError("异常检测模型尚未训练，请先调用fit()方法")
        
        logger.info("🔍 开始统计异常检测...")
        
        # 标准化预测数据
        scaled_predictions = self.scaler.transform(predictions)
        
        # 检测异常
        anomaly_labels = self.isolation_forest.predict(scaled_predictions)
        anomaly_scores = self.isolation_forest.decision_function(scaled_predictions)
        
        # 统计异常
        normal_count = np.sum(anomaly_labels == 1)
        anomaly_count = np.sum(anomaly_labels == -1)
        total_count = len(anomaly_labels)
        
        # 计算异常分数
        anomaly_score = anomaly_count / total_count if total_count > 0 else 0.0
        
        # 识别异常点
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        anomaly_values = predictions.iloc[anomaly_indices] if len(anomaly_indices) > 0 else pd.DataFrame()
        
        result = {
            'is_valid': anomaly_score < self.contamination,
            'anomaly_score': anomaly_score,
            'total_count': total_count,
            'normal_count': normal_count,
            'anomaly_count': anomaly_count,
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_values': anomaly_values.to_dict() if len(anomaly_values) > 0 else {},
            'anomaly_scores': anomaly_scores.tolist(),
            'warnings': [],
            'errors': []
        }
        
        # 生成警告和错误信息
        if anomaly_score > self.contamination:
            result['warnings'].append(
                f"异常检测分数较高: {anomaly_score:.2%}，超过阈值 {self.contamination:.2%}"
            )
        
        if anomaly_count > 0:
            result['warnings'].append(
                f"发现 {anomaly_count} 个统计异常点，建议进一步分析"
            )
        
        logger.info(f"✅ 统计异常检测完成: 异常分数 {anomaly_score:.2%}")
        return result

class MultiSourceConsistencyValidator:
    """多源一致性验证器"""
    
    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance
    
    def validate_consistency(self, predictions: Dict[str, pd.DataFrame], 
                           variable_type: str) -> Dict[str, Any]:
        """
        验证多个数据源预测结果的一致性
        
        Args:
            predictions: 不同数据源的预测结果字典
            variable_type: 变量类型
        
        Returns:
            一致性验证结果字典
        """
        logger.info(f"🔍 开始多源一致性验证: {variable_type}")
        
        if len(predictions) < 2:
            return {
                'is_valid': True,
                'consistency_score': 1.0,
                'message': '只有一个数据源，无法进行一致性验证'
            }
        
        # 获取所有数据源的时间索引
        all_indices = set()
        for source, data in predictions.items():
            all_indices.update(data.index)
        
        # 找到共同的时间索引
        common_indices = all_indices
        for source, data in predictions.items():
            common_indices = common_indices.intersection(data.index)
        
        if len(common_indices) == 0:
            return {
                'is_valid': False,
                'consistency_score': 0.0,
                'message': '没有共同的时间索引，无法进行一致性验证'
            }
        
        # 计算一致性指标
        consistency_scores = []
        source_pairs = list(predictions.keys())
        
        for i in range(len(source_pairs)):
            for j in range(i + 1, len(source_pairs)):
                source1, source2 = source_pairs[i], source_pairs[j]
                
                # 获取共同时间的数据
                data1 = predictions[source1].loc[common_indices]
                data2 = predictions[source2].loc[common_indices]
                
                # 计算相关系数
                correlation = data1.corr(data2)
                if pd.isna(correlation):
                    correlation = 0.0
                
                # 计算相对误差
                relative_error = np.mean(np.abs(data1 - data2) / (np.abs(data1) + 1e-8))
                
                # 计算一致性分数
                pair_score = (correlation + (1 - relative_error)) / 2
                consistency_scores.append(pair_score)
        
        # 计算总体一致性分数
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        result = {
            'is_valid': overall_consistency > (1 - self.tolerance),
            'consistency_score': overall_consistency,
            'total_sources': len(predictions),
            'common_time_points': len(common_indices),
            'pairwise_scores': consistency_scores,
            'tolerance': self.tolerance,
            'warnings': [],
            'errors': []
        }
        
        # 生成警告和错误信息
        if overall_consistency < (1 - self.tolerance):
            result['warnings'].append(
                f"多源一致性分数较低: {overall_consistency:.2%}，低于阈值 {(1 - self.tolerance):.2%}"
            )
        
        if len(common_indices) < 10:
            result['warnings'].append(
                f"共同时间点较少: {len(common_indices)}，可能影响一致性验证的可靠性"
            )
        
        logger.info(f"✅ 多源一致性验证完成: 分数 {overall_consistency:.2%}")
        return result

class PredictionQualityValidator:
    """预测质量综合验证器"""
    
    def __init__(self):
        self.physical_validator = PhysicalConstraintValidator()
        self.anomaly_detector = StatisticalAnomalyDetector()
        self.consistency_validator = MultiSourceConsistencyValidator()
        
        # 创建结果目录
        os.makedirs("validation_results", exist_ok=True)
        os.makedirs("validation_logs", exist_ok=True)
    
    def validate_prediction_quality(self, 
                                  predictions: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                  variable_type: str,
                                  historical_data: Optional[pd.DataFrame] = None,
                                  source_name: str = "unknown") -> ValidationResult:
        """
        综合验证预测结果质量
        
        Args:
            predictions: 预测结果（单个DataFrame或多个数据源的字典）
            variable_type: 变量类型
            historical_data: 历史数据（用于异常检测）
            source_name: 数据源名称
        
        Returns:
            综合验证结果
        """
        logger.info(f"🚀 开始综合预测质量验证: {variable_type} from {source_name}")
        
        start_time = datetime.now()
        validation_details = {}
        warnings = []
        errors = []
        recommendations = []
        
        try:
            # 1. 物理约束验证
            if isinstance(predictions, pd.DataFrame):
                physical_result = self.physical_validator.validate_physical_constraints(
                    predictions, variable_type
                )
                validation_details['physical_constraints'] = physical_result
                
                if not physical_result['is_valid']:
                    errors.extend(physical_result['errors'])
                warnings.extend(physical_result['warnings'])
                
                # 2. 统计异常检测（如果有历史数据）
                if historical_data is not None:
                    try:
                        # 训练异常检测模型
                        self.anomaly_detector.fit(historical_data)
                        
                        # 检测异常
                        anomaly_result = self.anomaly_detector.detect_anomalies(predictions)
                        validation_details['statistical_anomalies'] = anomaly_result
                        
                        if not anomaly_result['is_valid']:
                            errors.extend(anomaly_result['errors'])
                        warnings.extend(anomaly_result['warnings'])
                        
                    except Exception as e:
                        logger.warning(f"统计异常检测失败: {e}")
                        warnings.append(f"统计异常检测失败: {e}")
                
                # 3. 计算综合质量分数
                physical_score = physical_result.get('physical_score', 0.0)
                anomaly_score = validation_details.get('statistical_anomalies', {}).get('anomaly_score', 0.0)
                
                # 综合分数：物理约束权重70%，异常检测权重30%
                overall_score = physical_score * 0.7 + (1 - anomaly_score) * 0.3
                
            else:
                # 多数据源验证
                consistency_result = self.consistency_validator.validate_consistency(
                    predictions, variable_type
                )
                validation_details['multi_source_consistency'] = consistency_result
                
                overall_score = consistency_result.get('consistency_score', 0.0)
                
                if not consistency_result['is_valid']:
                    warnings.extend(consistency_result['warnings'])
            
            # 4. 生成建议
            if overall_score < 0.7:
                recommendations.append("预测质量较低，建议检查模型训练数据和参数")
            elif overall_score < 0.9:
                recommendations.append("预测质量中等，建议优化模型或增加训练数据")
            else:
                recommendations.append("预测质量良好，可以投入生产使用")
            
            # 5. 判断整体有效性
            is_valid = overall_score > 0.7 and len(errors) == 0
            
            # 6. 保存验证结果
            self._save_validation_result(
                source_name, variable_type, validation_details, 
                overall_score, warnings, errors, recommendations
            )
            
            validation_result = ValidationResult(
                is_valid=is_valid,
                confidence_score=overall_score,
                validation_details=validation_details,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations,
                timestamp=start_time
            )
            
            logger.info(f"✅ 预测质量验证完成: 分数 {overall_score:.2%}, 有效: {is_valid}")
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ 预测质量验证失败: {e}")
            errors.append(f"验证过程发生错误: {e}")
            
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_details={'error': str(e)},
                warnings=warnings,
                errors=errors,
                recommendations=["验证过程失败，请检查数据和系统状态"],
                timestamp=start_time
            )
    
    def _save_validation_result(self, source_name: str, variable_type: str,
                               validation_details: Dict, overall_score: float,
                               warnings: List[str], errors: List[str],
                               recommendations: List[str]):
        """保存验证结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_results/{source_name}_{variable_type}_{timestamp}.json"
        
        result_data = {
            'source_name': source_name,
            'variable_type': variable_type,
            'timestamp': timestamp,
            'overall_score': overall_score,
            'validation_details': validation_details,
            'warnings': warnings,
            'errors': errors,
            'recommendations': recommendations
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"✅ 验证结果已保存: {filename}")
        except Exception as e:
            logger.error(f"❌ 保存验证结果失败: {e}")
    
    def generate_validation_report(self, validation_result: ValidationResult) -> str:
        """生成验证报告"""
        report = f"""
# 预测质量验证报告

## 基本信息
- 验证时间: {validation_result.timestamp}
- 整体有效性: {'✅ 有效' if validation_result.is_valid else '❌ 无效'}
- 置信度分数: {validation_result.confidence_score:.2%}

## 验证详情
"""
        
        for category, details in validation_result.validation_details.items():
            report += f"\n### {category.replace('_', ' ').title()}\n"
            if isinstance(details, dict):
                for key, value in details.items():
                    if key not in ['warnings', 'errors']:
                        report += f"- {key}: {value}\n"
            else:
                report += f"- {details}\n"
        
        if validation_result.warnings:
            report += "\n## ⚠️ 警告\n"
            for warning in validation_result.warnings:
                report += f"- {warning}\n"
        
        if validation_result.errors:
            report += "\n## ❌ 错误\n"
            for error in validation_result.errors:
                report += f"- {error}\n"
        
        if validation_result.recommendations:
            report += "\n## 💡 建议\n"
            for rec in validation_result.recommendations:
                report += f"- {rec}\n"
        
        return report

def main():
    """主函数：演示验证器使用"""
    logger.info("🚀 启动预测结果验证器演示")
    
    # 创建验证器
    validator = PredictionQualityValidator()
    
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # 示例1：正常的土壤湿度预测
    normal_predictions = pd.DataFrame({
        'soil_moisture': np.random.uniform(0.1, 0.8, 100)
    }, index=dates)
    
    # 示例2：异常的土壤湿度预测（包含超出范围的值）
    abnormal_predictions = pd.DataFrame({
        'soil_moisture': np.random.uniform(-0.1, 1.2, 100)  # 包含负值和超过1的值
    }, index=dates)
    
    # 示例3：历史数据
    historical_data = pd.DataFrame({
        'soil_moisture': np.random.uniform(0.1, 0.8, 200)
    }, index=pd.date_range('2023-01-01', periods=200, freq='D'))
    
    # 验证正常预测
    logger.info("\n" + "="*50)
    logger.info("验证正常预测结果")
    normal_result = validator.validate_prediction_quality(
        normal_predictions, 'soil_moisture', historical_data, 'normal_model'
    )
    
    # 验证异常预测
    logger.info("\n" + "="*50)
    logger.info("验证异常预测结果")
    abnormal_result = validator.validate_prediction_quality(
        abnormal_predictions, 'soil_moisture', historical_data, 'abnormal_model'
    )
    
    # 生成报告
    logger.info("\n" + "="*50)
    logger.info("生成验证报告")
    
    normal_report = validator.generate_validation_report(normal_result)
    abnormal_report = validator.generate_validation_report(abnormal_result)
    
    # 保存报告
    with open("validation_logs/normal_validation_report.md", "w", encoding="utf-8") as f:
        f.write(normal_report)
    
    with open("validation_logs/abnormal_validation_report.md", "w", encoding="utf-8") as f:
        f.write(abnormal_report)
    
    logger.info("✅ 验证器演示完成，报告已保存到 validation_logs/ 目录")

if __name__ == "__main__":
    main()
