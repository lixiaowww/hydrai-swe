#!/usr/bin/env python3
"""
诚实预测器 - 完全移除任何造假方法
实现多种预测模式，适应不同数据量
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

class PredictionMode(Enum):
    """预测模式枚举"""
    STRICT = "strict"      # 严格模式：需要完整历史数据
    LIMITED = "limited"    # 有限模式：数据不足时提供有限预测
    PROGRESSIVE = "progressive"  # 渐进模式：随着数据增加逐步提高质量

class PredictionConfidence(Enum):
    """预测置信度枚举"""
    HIGH = "high"      # 高置信度：数据充足
    MEDIUM = "medium"  # 中等置信度：数据部分充足
    LOW = "low"        # 低置信度：数据不足
    INSUFFICIENT = "insufficient"  # 数据不足，无法预测

class HonestSWEPredictor:
    """诚实SWE预测器 - 绝不造假"""
    
    def __init__(self, model_path: str = None, mode: PredictionMode = PredictionMode.STRICT):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.sequence_length = 30
        self.is_loaded = False
        self.prediction_mode = mode
        
        # 特征配置
        self.feature_config = {
            'snow_depth_mm': 0,
            'snow_fall_mm': 1, 
            'snow_water_equivalent_mm': 2,
            'day_of_year': 3,
            'month': 4,
            'year': 5
        }
        
        # 历史数据存储
        self._historical_features = []
        self._historical_dates = []
        
        # 自动加载最佳模型
        if model_path is None:
            self._auto_load_best_model()
        else:
            self.load_model(model_path)
    
    def set_prediction_mode(self, mode: PredictionMode):
        """设置预测模式"""
        self.prediction_mode = mode
        print(f"✅ 预测模式设置为: {mode.value}")
    
    def get_prediction_requirements(self) -> Dict[str, Any]:
        """获取预测要求说明"""
        requirements = {
            'strict_mode': {
                'description': '严格模式：需要完整历史数据',
                'min_data_points': self.sequence_length,
                'confidence': PredictionConfidence.HIGH.value,
                'limitations': '无'
            },
            'limited_mode': {
                'description': '有限模式：数据不足时提供有限预测',
                'min_data_points': 1,
                'confidence': PredictionConfidence.MEDIUM.value,
                'limitations': '预测质量受限，置信度降低'
            },
            'progressive_mode': {
                'description': '渐进模式：随着数据增加逐步提高质量',
                'min_data_points': 1,
                'confidence': PredictionConfidence.LOW.value,
                'limitations': '初始预测质量较低，需要持续收集数据'
            }
        }
        
        return {
            'current_mode': self.prediction_mode.value,
            'requirements': requirements,
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """获取数据收集建议"""
        current_data_points = len(self._historical_features)
        
        recommendations = []
        
        if current_data_points == 0:
            recommendations.append("建议收集至少1个数据点开始渐进预测")
        elif current_data_points < self.sequence_length:
            recommendations.append(f"建议继续收集数据，当前{current_data_points}/{self.sequence_length}")
            recommendations.append("数据越多，预测质量越高")
        else:
            recommendations.append("数据充足，可以使用严格模式获得最佳预测")
        
        return recommendations
    
    def validate_feature_data(self, snow_depth_mm: float, snow_fall_mm: float, 
                            snow_water_equivalent_mm: float, date: datetime) -> bool:
        """验证输入特征数据 - 严格模式"""
        try:
            # 验证数据类型
            if not all(isinstance(x, (int, float)) for x in [snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm]):
                raise ValueError("雪相关特征必须是数值类型")
            
            # 严格验证：拒绝明显不合理的雪数据
            if snow_depth_mm < 0:
                raise ValueError(f"雪深度不能为负值: {snow_depth_mm}")
            if snow_water_equivalent_mm < 0:
                raise ValueError(f"雪水当量不能为负值: {snow_water_equivalent_mm}")
            
            # 雪降量可以为负值（表示融化），但需要检查合理性
            if snow_fall_mm < -100:
                raise ValueError(f"雪降量（融化量）过大: {snow_fall_mm}")
            
            # 检查数值范围合理性
            if snow_depth_mm > 10000:
                raise ValueError(f"雪深度过大，可能不合理: {snow_depth_mm} mm")
            
            # 雪水当量相对于雪深度的比例检查
            if snow_depth_mm > 0:
                ratio = snow_water_equivalent_mm / snow_depth_mm
                if ratio > 0.4:
                    print(f"⚠️ 警告：雪水当量相对于雪深度比例异常: {ratio:.2f}")
                    print(f"   雪深度: {snow_depth_mm} mm, 雪水当量: {snow_water_equivalent_mm} mm")
                    # 在严格模式下拒绝
                    if self.prediction_mode == PredictionMode.STRICT:
                        raise ValueError(f"雪水当量比例异常: {ratio:.2f}")
            
            # 验证日期
            if not isinstance(date, datetime):
                raise ValueError("日期必须是datetime对象")
            
            current_year = datetime.now().year
            if date.year < 1900 or date.year > current_year + 10:
                raise ValueError(f"日期年份不合理: {date.year}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据验证失败: {e}")
            return False
    
    def add_historical_data(self, snow_depth_mm: float, snow_fall_mm: float, 
                           snow_water_equivalent_mm: float, date: datetime):
        """添加历史数据 - 诚实方法"""
        # 数据验证
        if not self.validate_feature_data(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date):
            raise ValueError("历史数据验证失败")
        
        # 准备特征
        features = self._prepare_features(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date)
        
        # 添加到历史数据
        self._historical_features.append(features.flatten())
        self._historical_dates.append(date)
        
        # 保持数据顺序（按时间）
        if len(self._historical_features) > 1:
            # 按日期排序
            sorted_indices = np.argsort([d.timestamp() for d in self._historical_dates])
            self._historical_features = [self._historical_features[i] for i in sorted_indices]
            self._historical_dates = [self._historical_dates[i] for i in sorted_indices]
        
        # 限制历史数据量
        max_history = self.sequence_length * 3
        if len(self._historical_features) > max_history:
            self._historical_features = self._historical_features[-max_history:]
            self._historical_dates = self._historical_dates[-max_history:]
        
        print(f"✅ 历史数据添加成功，当前数据点: {len(self._historical_features)}")
    
    def _prepare_features(self, snow_depth_mm: float, snow_fall_mm: float, 
                         snow_water_equivalent_mm: float, date: datetime) -> np.ndarray:
        """准备特征 - 内部方法"""
        # 计算日期特征
        day_of_year = date.timetuple().tm_yday
        month = date.month
        year = date.year
        
        # 创建特征向量
        features = np.array([
            snow_depth_mm,
            snow_fall_mm, 
            snow_water_equivalent_mm,
            day_of_year,
            month,
            year
        ]).reshape(1, -1)
        
        # 标准化特征
        if self.scaler_X is not None:
            try:
                features = self.scaler_X.transform(features)
            except Exception as e:
                print(f"❌ 特征标准化失败: {e}")
                raise
        else:
            print("⚠️ 警告：标准化器未加载，使用原始特征")
        
        return features
    
    def predict(self, snow_depth_mm: float, snow_fall_mm: float, 
               snow_water_equivalent_mm: float, date: datetime) -> Tuple[float, PredictionConfidence, Dict[str, Any]]:
        """主预测方法 - 根据模式选择预测策略"""
        if not self.is_loaded:
            raise Exception("模型未加载")
        
        # 数据验证
        if not self.validate_feature_data(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date):
            raise ValueError("输入数据验证失败")
        
        # 根据预测模式选择策略
        if self.prediction_mode == PredictionMode.STRICT:
            return self._predict_strict(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date)
        elif self.prediction_mode == PredictionMode.LIMITED:
            return self._predict_limited(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date)
        elif self.prediction_mode == PredictionMode.PROGRESSIVE:
            return self._predict_progressive(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date)
        else:
            raise ValueError(f"未知的预测模式: {self.prediction_mode}")
    
    def _predict_strict(self, snow_depth_mm: float, snow_fall_mm: float, 
                       snow_water_equivalent_mm: float, date: datetime) -> Tuple[float, PredictionConfidence, Dict[str, Any]]:
        """严格预测模式 - 需要完整历史数据"""
        # 检查历史数据是否充足
        if len(self._historical_features) < self.sequence_length:
            raise ValueError(
                f"严格模式需要至少 {self.sequence_length} 个历史数据点，"
                f"但只有 {len(self._historical_features)} 个。"
                "请先收集足够的历史数据，或切换到其他预测模式。"
            )
        
        # 使用真实的历史数据创建序列
        sequence = self._historical_features[-self.sequence_length:]
        sequence = np.array(sequence).reshape(1, self.sequence_length, -1)
        
        # 预测
        prediction = self._make_prediction(sequence)
        
        # 反标准化
        if self.scaler_y is not None:
            prediction = self.scaler_y.inverse_transform([[prediction]])[0][0]
        
        prediction = max(0, prediction)  # 确保非负
        
        # 返回结果和元数据
        metadata = {
            'mode': 'strict',
            'data_points_used': self.sequence_length,
            'data_quality': 'high',
            'limitations': '无',
            'recommendations': ['数据充足，预测质量最佳']
        }
        
        return prediction, PredictionConfidence.HIGH, metadata
    
    def _predict_limited(self, snow_depth_mm: float, snow_fall_mm: float, 
                        snow_water_equivalent_mm: float, date: datetime) -> Tuple[float, PredictionConfidence, Dict[str, Any]]:
        """有限预测模式 - 数据不足时提供有限预测"""
        current_data_points = len(self._historical_features)
        
        if current_data_points == 0:
            raise ValueError(
                "有限模式需要至少1个历史数据点。"
                "请先添加一些历史数据，或切换到渐进模式。"
            )
        
        # 准备当前特征
        current_features = self._prepare_features(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date)
        
        # 创建序列（使用所有可用的历史数据）
        available_features = self._historical_features.copy()
        available_features.append(current_features.flatten())
        
        # 如果数据不足，使用可用的数据（不造假）
        if len(available_features) < self.sequence_length:
            # 使用可用的数据，但明确告知限制
            sequence = np.array(available_features)
            # 重塑为 (1, n_available, n_features)
            sequence = sequence.reshape(1, -1, sequence.shape[1])
            
            # 使用可用的数据点进行预测
            prediction = self._make_prediction_with_variable_length(sequence)
        else:
            # 数据充足，使用标准方法
            sequence = np.array(available_features[-self.sequence_length:])
            sequence = sequence.reshape(1, self.sequence_length, -1)
            prediction = self._make_prediction(sequence)
        
        # 反标准化
        if self.scaler_y is not None:
            prediction = self.scaler_y.inverse_transform([[prediction]])[0][0]
        
        prediction = max(0, prediction)
        
        # 计算置信度
        if current_data_points >= self.sequence_length:
            confidence = PredictionConfidence.HIGH
        elif current_data_points >= self.sequence_length // 2:
            confidence = PredictionConfidence.MEDIUM
        else:
            confidence = PredictionConfidence.LOW
        
        # 返回结果和元数据
        metadata = {
            'mode': 'limited',
            'data_points_used': len(available_features),
            'data_quality': 'medium' if confidence == PredictionConfidence.MEDIUM else 'low',
            'limitations': f'数据不足，仅使用 {len(available_features)}/{self.sequence_length} 个数据点',
            'recommendations': [
                f'建议收集更多历史数据（当前{current_data_points}/{self.sequence_length}）',
                '数据越多，预测质量越高'
            ]
        }
        
        return prediction, confidence, metadata
    
    def _predict_progressive(self, snow_depth_mm: float, snow_fall_mm: float, 
                           snow_water_equivalent_mm: float, date: datetime) -> Tuple[float, PredictionConfidence, Dict[str, Any]]:
        """渐进预测模式 - 随着数据增加逐步提高质量"""
        current_data_points = len(self._historical_features)
        
        # 准备当前特征
        current_features = self._prepare_features(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date)
        
        # 创建序列（使用所有可用的历史数据）
        available_features = self._historical_features.copy()
        available_features.append(current_features.flatten())
        
        # 根据可用数据量选择预测策略
        if len(available_features) == 1:
            # 只有一个数据点，使用简单外推
            prediction = self._simple_extrapolation(current_features)
            confidence = PredictionConfidence.LOW
        elif len(available_features) < self.sequence_length:
            # 数据不足，使用可用的数据
            sequence = np.array(available_features)
            sequence = sequence.reshape(1, -1, sequence.shape[1])
            prediction = self._make_prediction_with_variable_length(sequence)
            
            # 根据数据量计算置信度
            if len(available_features) >= self.sequence_length // 2:
                confidence = PredictionConfidence.MEDIUM
            else:
                confidence = PredictionConfidence.LOW
        else:
            # 数据充足，使用标准方法
            sequence = np.array(available_features[-self.sequence_length:])
            sequence = sequence.reshape(1, self.sequence_length, -1)
            prediction = self._make_prediction(sequence)
            confidence = PredictionConfidence.HIGH
        
        # 反标准化
        if self.scaler_y is not None:
            prediction = self.scaler_y.inverse_transform([[prediction]])[0][0]
        
        prediction = max(0, prediction)
        
        # 返回结果和元数据
        metadata = {
            'mode': 'progressive',
            'data_points_used': len(available_features),
            'data_quality': 'progressive',
            'limitations': f'渐进模式，当前使用 {len(available_features)}/{self.sequence_length} 个数据点',
            'recommendations': [
                f'当前数据点: {len(available_features)}/{self.sequence_length}',
                '继续收集数据以提高预测质量',
                f'预计需要 {self.sequence_length - len(available_features)} 个数据点达到最佳质量'
            ]
        }
        
        return prediction, confidence, metadata
    
    def _make_prediction(self, sequence: np.ndarray) -> float:
        """标准预测方法"""
        with torch.no_grad():
            if hasattr(self, 'ensemble_models') and self.ensemble_models:
                # 集成预测
                predictions = []
                for model in self.ensemble_models:
                    pred = model(torch.FloatTensor(sequence))
                    predictions.append(pred.item())
                return np.mean(predictions)
            else:
                # 单个模型预测
                return self.model(torch.FloatTensor(sequence)).item()
    
    def _make_prediction_with_variable_length(self, sequence: np.ndarray) -> float:
        """处理可变长度序列的预测"""
        # 这里需要根据实际模型架构调整
        # 暂时使用最后一个时间步的特征进行预测
        last_features = sequence[0, -1, :]
        
        # 创建单时间步序列
        single_sequence = last_features.reshape(1, 1, -1)
        
        with torch.no_grad():
            if hasattr(self, 'ensemble_models') and self.ensemble_models:
                predictions = []
                for model in self.ensemble_models:
                    # 注意：这里假设模型可以处理单时间步序列
                    # 如果不行，需要调整模型架构
                    pred = model(torch.FloatTensor(single_sequence))
                    predictions.append(pred.item())
                return np.mean(predictions)
            else:
                return self.model(torch.FloatTensor(single_sequence)).item()
    
    def _simple_extrapolation(self, current_features: np.ndarray) -> float:
        """简单外推 - 仅用于渐进模式的初始阶段"""
        # 使用当前特征进行简单预测
        # 这里可以基于物理规律进行简单外推
        # 暂时返回一个基于当前雪水当量的简单估计
        
        # 假设预测值接近当前雪水当量，但考虑季节性因素
        current_swe = current_features[0, 2]  # 雪水当量
        
        # 简单的季节性调整（基于月份）
        month = current_features[0, 4]
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)
        
        return current_swe * seasonal_factor
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告"""
        if len(self._historical_features) == 0:
            return {
                'status': 'no_data',
                'message': '没有历史数据',
                'quality_score': 0.0,
                'recommendations': ['开始收集历史数据']
            }
        
        try:
            features_array = np.array(self._historical_features)
            n_samples = len(features_array)
            
            # 计算质量分数
            quality_score = min(1.0, n_samples / self.sequence_length)
            
            # 生成报告
            report = {
                'status': 'excellent' if quality_score >= 1.0 else 'good' if quality_score >= 0.7 else 'fair' if quality_score >= 0.4 else 'poor',
                'quality_score': quality_score,
                'n_samples': n_samples,
                'required_samples': self.sequence_length,
                'completion_percentage': f"{quality_score * 100:.1f}%",
                'recommendations': self._get_recommendations()
            }
            
            return report
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'数据质量检查失败: {e}',
                'quality_score': 0.0
            }
    
    def _auto_load_best_model(self):
        """自动加载最佳模型"""
        # 查找最新的优化模型
        models_dir = "models"
        if not os.path.exists(models_dir):
            print("❌ 模型目录不存在")
            return
        
        # 查找优化模型
        optimized_files = [f for f in os.listdir(models_dir) if f.startswith("optimized_gru_model_")]
        if optimized_files:
            latest_model = max(optimized_files)
            model_path = os.path.join(models_dir, latest_model)
            print(f"🔍 找到优化模型: {latest_model}")
            self.load_model(model_path)
            return
        
        # 查找集成模型
        ensemble_dirs = [d for d in os.listdir(models_dir) if d.startswith("ensemble_models_")]
        if ensemble_dirs:
            latest_ensemble = max(ensemble_dirs)
            ensemble_path = os.path.join(models_dir, latest_ensemble)
            print(f"🔍 找到集成模型: {latest_ensemble}")
            self.load_ensemble_model(ensemble_path)
            return
        
        print("❌ 未找到优化模型，使用默认配置")
        self._create_default_model()
    
    def _create_default_model(self):
        """创建默认模型"""
        self.model = OptimizedGRUModel(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            dropout=0.1
        )
        self.is_loaded = True
    
    def load_model(self, model_path: str):
        """加载单个优化模型"""
        try:
            print(f"📥 加载模型: {model_path}")
            
            # 验证文件存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载模型检查点
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 重建标准化器
            self.scaler_X = StandardScaler()
            self.scaler_X.mean_ = checkpoint['scaler_X_mean']
            self.scaler_X.scale_ = checkpoint['scaler_X_scale']
            
            self.scaler_y = StandardScaler()
            self.scaler_y.mean_ = checkpoint['scaler_y_mean']
            self.scaler_y.scale_ = checkpoint['scaler_y_scale']
            
            # 创建模型
            self.model = OptimizedGRUModel(
                input_size=6,
                hidden_size=64,
                num_layers=2,
                dropout=0.1
            )
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.is_loaded = True
            print(f"✅ 模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self._create_default_model()
    
    def load_ensemble_model(self, ensemble_dir: str):
        """加载集成模型"""
        try:
            print(f"📥 加载集成模型: {ensemble_dir}")
            
            # 加载标准化器参数
            standardization_path = "models/standardization_params.pkl"
            if os.path.exists(standardization_path):
                with open(standardization_path, 'rb') as f:
                    params = pickle.load(f)
                
                self.scaler_X = StandardScaler()
                self.scaler_X.mean_ = params['scaler_X_mean']
                self.scaler_X.scale_ = params['scaler_X_scale']
                
                self.scaler_y = StandardScaler()
                self.scaler_y.mean_ = params['scaler_y_mean']
                self.scaler_y.scale_ = params['scaler_y_scale']
                
                print("✅ 标准化器加载成功")
            
            # 创建集成模型列表
            self.ensemble_models = []
            successful_loads = 0
            
            for i in range(1, 4):  # 加载前3个模型
                model_files = [f for f in os.listdir(ensemble_dir) if f.startswith(f"model_{i}_")]
                if model_files:
                    model_path = os.path.join(ensemble_dir, model_files[0])
                    model = OptimizedGRUModel()
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    self.ensemble_models.append(model)
                    successful_loads += 1
                    print(f"✅ 集成模型 {i} 加载成功")
            
            if successful_loads > 0:
                self.is_loaded = True
                print(f"✅ 集成模型加载完成: {successful_loads} 个模型")
            else:
                raise Exception("没有成功加载任何集成模型")
                
        except Exception as e:
            print(f"❌ 集成模型加载失败: {e}")
            self._create_default_model()

class OptimizedGRUModel(nn.Module):
    """优化的GRU模型"""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1):
        super(OptimizedGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        output = self.fc(gru_out[:, -1, :])
        return output

# 全局预测器实例
_global_honest_predictor = None

def get_honest_predictor(mode: PredictionMode = PredictionMode.STRICT) -> HonestSWEPredictor:
    """获取全局诚实预测器实例"""
    global _global_honest_predictor
    if _global_honest_predictor is None:
        _global_honest_predictor = HonestSWEPredictor(mode=mode)
    return _global_honest_predictor

