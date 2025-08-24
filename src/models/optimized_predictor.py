#!/usr/bin/env python3
"""
优化的SWE预测服务
使用我们训练好的最佳超参数模型、集成模型和数据增强技术
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os
import pickle
from typing import Dict, List, Any, Optional

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

class OptimizedSWEPredictor:
    """优化的SWE预测器"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.sequence_length = 30
        self.is_loaded = False
        
        # 新增：特征列配置和验证
        self.feature_config = {
            'snow_depth_mm': 0,
            'snow_fall_mm': 1, 
            'snow_water_equivalent_mm': 2,
            'day_of_year': 3,
            'month': 4,
            'year': 5
        }
        
        # 新增：数据验证标志
        self._data_validation_enabled = True
        
        # 自动加载最佳模型
        if model_path is None:
            self._auto_load_best_model()
        else:
            self.load_model(model_path)
    
    def validate_feature_data(self, snow_depth_mm: float, snow_fall_mm: float, 
                            snow_water_equivalent_mm: float, date: datetime) -> bool:
        """验证输入特征数据 - 加强：拒绝不合理数据"""
        if not self._data_validation_enabled:
            return True
        
        try:
            # 验证数据类型
            if not all(isinstance(x, (int, float)) for x in [snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm]):
                raise ValueError("雪相关特征必须是数值类型")
            
            # 加强验证：拒绝明显不合理的雪数据
            # 雪深度和雪水当量不能为负值
            if snow_depth_mm < 0:
                raise ValueError(f"雪深度不能为负值: {snow_depth_mm}")
            if snow_water_equivalent_mm < 0:
                raise ValueError(f"雪水当量不能为负值: {snow_water_equivalent_mm}")
            
            # 雪降量可以为负值（表示融化），但需要检查合理性
            if snow_fall_mm < -100:  # 融化量过大
                raise ValueError(f"雪降量（融化量）过大: {snow_fall_mm}")
            
            # 检查数值范围合理性
            # 雪深度通常不会超过10米（10000mm）
            if snow_depth_mm > 10000:
                raise ValueError(f"雪深度过大，可能不合理: {snow_depth_mm} mm")
            
            # 雪水当量通常不会超过雪深度的1/3
            if snow_water_equivalent_mm > snow_depth_mm * 0.4:
                print(f"⚠️ 警告：雪水当量相对于雪深度可能过大")
                print(f"   雪深度: {snow_depth_mm} mm, 雪水当量: {snow_water_equivalent_mm} mm")
                print(f"   比例: {snow_water_equivalent_mm/snow_depth_mm:.2f}")
            
            # 验证日期
            if not isinstance(date, datetime):
                raise ValueError("日期必须是datetime对象")
            
            # 验证日期合理性
            current_year = datetime.now().year
            if date.year < 1900 or date.year > current_year + 10:
                raise ValueError(f"日期年份不合理: {date.year}")
            
            # 验证日期是否在未来（如果是预测）
            if date > datetime.now() + timedelta(days=365):
                print(f"⚠️ 警告：预测日期较远: {date}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据验证失败: {e}")
            return False
    
    def validate_historical_data_quality(self) -> Dict[str, Any]:
        """验证历史数据质量 - 新增方法"""
        if not hasattr(self, '_historical_features') or len(self._historical_features) == 0:
            return {
                'status': 'no_data',
                'message': '没有历史数据',
                'quality_score': 0.0
            }
        
        try:
            features_array = np.array(self._historical_features)
            n_samples = len(features_array)
            
            # 检查数据连续性
            continuity_score = 1.0
            if n_samples > 1:
                # 检查相邻样本之间的变化是否合理
                diffs = np.diff(features_array, axis=0)
                
                # 雪相关特征的变化应该相对平滑
                snow_diffs = diffs[:, :3]  # 前3列是雪相关特征
                max_snow_change = np.max(np.abs(snow_diffs))
                
                if max_snow_change > 1000:  # 变化过大
                    continuity_score = 0.5
                    print(f"⚠️ 警告：雪相关特征变化过大: {max_snow_change}")
                
                # 日期特征应该递增
                date_diffs = diffs[:, 3:]  # 后3列是日期特征
                if not all(date_diffs[:, 1] >= 0):  # 月份应该递增
                    continuity_score = 0.3
                    print("⚠️ 警告：日期特征不连续")
            
            # 检查数据范围合理性
            range_score = 1.0
            for i, feature_name in enumerate(['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm']):
                values = features_array[:, i]
                
                # 检查负值
                if np.any(values < 0):
                    if i == 1:  # 雪降量可以为负
                        if np.any(values < -100):
                            range_score *= 0.8
                            print(f"⚠️ 警告：{feature_name} 包含过大的负值")
                    else:  # 雪深度和雪水当量不能为负
                        range_score *= 0.5
                        print(f"❌ 错误：{feature_name} 包含负值")
                
                # 检查异常大的值
                if np.any(values > 10000):
                    range_score *= 0.7
                    print(f"⚠️ 警告：{feature_name} 包含异常大的值")
            
            # 计算总体质量分数
            quality_score = (continuity_score + range_score) / 2
            
            # 生成质量报告
            quality_report = {
                'status': 'valid' if quality_score > 0.7 else 'warning' if quality_score > 0.4 else 'error',
                'quality_score': quality_score,
                'n_samples': n_samples,
                'continuity_score': continuity_score,
                'range_score': range_score,
                'recommendations': []
            }
            
            # 添加建议
            if quality_score < 0.7:
                quality_report['recommendations'].append("建议检查数据源和预处理步骤")
            if continuity_score < 0.7:
                quality_report['recommendations'].append("建议检查数据时间顺序和缺失值")
            if range_score < 0.7:
                quality_report['recommendations'].append("建议检查数据范围和异常值")
            
            return quality_report
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'数据质量检查失败: {e}',
                'quality_score': 0.0
            }
    
    def get_data_requirements(self) -> Dict[str, Any]:
        """获取数据要求说明 - 新增方法"""
        return {
            'minimum_historical_data': self.sequence_length,
            'feature_requirements': {
                'snow_depth_mm': {
                    'type': 'float',
                    'range': '[0, 10000]',
                    'unit': 'mm',
                    'description': '雪深度，不能为负值'
                },
                'snow_fall_mm': {
                    'type': 'float',
                    'range': '[-100, 10000]',
                    'unit': 'mm',
                    'description': '雪降量，负值表示融化'
                },
                'snow_water_equivalent_mm': {
                    'type': 'float',
                    'range': '[0, 10000]',
                    'unit': 'mm',
                    'description': '雪水当量，不能为负值'
                },
                'day_of_year': {
                    'type': 'int',
                    'range': '[1, 366]',
                    'description': '一年中的第几天'
                },
                'month': {
                    'type': 'int',
                    'range': '[1, 12]',
                    'description': '月份'
                },
                'year': {
                    'type': 'int',
                    'range': '[1900, 2030]',
                    'description': '年份'
                }
            },
            'data_quality_requirements': {
                'continuity': '数据应该连续，相邻样本变化合理',
                'range': '数值在合理范围内',
                'completeness': '没有缺失值或异常值'
            }
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
        """加载单个优化模型 - 增强：更好的错误处理"""
        try:
            print(f"📥 加载模型: {model_path}")
            
            # 验证文件存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 验证文件大小
            file_size = os.path.getsize(model_path)
            if file_size < 1000:  # 小于1KB可能是损坏文件
                raise ValueError(f"模型文件可能损坏，大小异常: {file_size} bytes")
            
            # 加载模型检查点
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
            except Exception as e:
                raise RuntimeError(f"模型文件加载失败: {e}")
            
            # 验证检查点结构
            required_keys = ['model_state_dict', 'scaler_X_mean', 'scaler_X_scale', 
                           'scaler_y_mean', 'scaler_y_scale']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise ValueError(f"模型检查点缺少必要键: {missing_keys}")
            
            # 重建标准化器
            try:
                self.scaler_X = StandardScaler()
                self.scaler_X.mean_ = checkpoint['scaler_X_mean']
                self.scaler_X.scale_ = checkpoint['scaler_X_scale']
                
                self.scaler_y = StandardScaler()
                self.scaler_y.mean_ = checkpoint['scaler_y_mean']
                self.scaler_y.scale_ = checkpoint['scaler_y_scale']
            except Exception as e:
                raise RuntimeError(f"标准化器重建失败: {e}")
            
            # 创建模型
            try:
                self.model = OptimizedGRUModel(
                    input_size=6,
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.1
                )
            except Exception as e:
                raise RuntimeError(f"模型创建失败: {e}")
            
            # 加载模型权重
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
            except Exception as e:
                raise RuntimeError(f"模型权重加载失败: {e}")
            
            self.is_loaded = True
            print(f"✅ 模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("🔄 回退到默认模型")
            self._create_default_model()
    
    def load_ensemble_model(self, ensemble_dir: str):
        """加载集成模型 - 增强：更好的错误处理"""
        try:
            print(f"📥 加载集成模型: {ensemble_dir}")
            
            # 验证目录存在
            if not os.path.exists(ensemble_dir):
                raise FileNotFoundError(f"集成模型目录不存在: {ensemble_dir}")
            
            # 验证目录结构
            if not os.path.isdir(ensemble_dir):
                raise ValueError(f"路径不是目录: {ensemble_dir}")
            
            # 加载集成配置
            config_path = os.path.join(ensemble_dir, "ensemble_config.json")
            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    print(f"✅ 集成配置加载成功: {config['n_models']} 个模型")
                except Exception as e:
                    print(f"⚠️ 集成配置加载失败: {e}")
                    config = {'n_models': 3}  # 默认配置
            else:
                print("⚠️ 未找到集成配置文件，使用默认配置")
                config = {'n_models': 3}
            
            # 加载标准化器参数
            standardization_path = "models/standardization_params.pkl"
            if os.path.exists(standardization_path):
                try:
                    with open(standardization_path, 'rb') as f:
                        params = pickle.load(f)
                    
                    self.scaler_X = StandardScaler()
                    self.scaler_X.mean_ = params['scaler_X_mean']
                    self.scaler_X.scale_ = params['scaler_X_scale']
                    
                    self.scaler_y = StandardScaler()
                    self.scaler_y.mean_ = params['scaler_y_mean']
                    self.scaler_y.scale_ = params['scaler_y_scale']
                    
                    print("✅ 标准化器加载成功")
                except Exception as e:
                    print(f"⚠️ 标准化器加载失败: {e}")
                    # 继续尝试加载模型，可能模型文件中有标准化器信息
            
            # 创建集成模型列表
            self.ensemble_models = []
            successful_loads = 0
            
            for i in range(1, config['n_models'] + 1):
                try:
                    # 查找模型文件
                    model_files = [f for f in os.listdir(ensemble_dir) if f.startswith(f"model_{i}_")]
                    if not model_files:
                        print(f"⚠️ 未找到模型 {i} 的文件")
                        continue
                    
                    model_path = os.path.join(ensemble_dir, model_files[0])
                    
                    # 验证模型文件
                    if not os.path.exists(model_path):
                        print(f"⚠️ 模型文件不存在: {model_path}")
                        continue
                    
                    # 创建模型
                    model = OptimizedGRUModel()
                    
                    # 加载检查点
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()
                        
                        self.ensemble_models.append(model)
                        successful_loads += 1
                        print(f"✅ 集成模型 {i} 加载成功")
                        
                    except Exception as e:
                        print(f"⚠️ 模型 {i} 加载失败: {e}")
                        continue
                        
                except Exception as e:
                    print(f"⚠️ 处理模型 {i} 时出错: {e}")
                    continue
            
            if successful_loads > 0:
                self.is_loaded = True
                print(f"✅ 集成模型加载完成: {successful_loads}/{config['n_models']} 个模型")
            else:
                raise Exception("没有成功加载任何集成模型")
                
        except Exception as e:
            print(f"❌ 集成模型加载失败: {e}")
            print("🔄 回退到默认模型")
            self._create_default_model()
    
    def prepare_input_features(self, snow_depth_mm: float, snow_fall_mm: float, 
                             snow_water_equivalent_mm: float, date: datetime) -> np.ndarray:
        """准备输入特征 - 增强：添加数据验证"""
        # 数据验证
        if not self.validate_feature_data(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date):
            raise ValueError("输入数据验证失败")
        
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
        
        # 验证特征维度
        if features.shape[1] != len(self.feature_config):
            raise ValueError(f"特征维度不匹配: 期望{len(self.feature_config)}, 实际{features.shape[1]}")
        
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
    
    def create_sequence(self, features_list: List[np.ndarray]) -> np.ndarray:
        """创建序列数据"""
        if len(features_list) < self.sequence_length:
            # 如果数据不足，用零填充
            padding = [np.zeros_like(features_list[0]) for _ in range(self.sequence_length - len(features_list))]
            features_list = padding + features_list
        
        # 取最后sequence_length个特征
        sequence = features_list[-self.sequence_length:]
        return np.array(sequence).reshape(1, self.sequence_length, -1)
    
    def predict_single(self, snow_depth_mm: float, snow_fall_mm: float, 
                      snow_water_equivalent_mm: float, date: datetime) -> float:
        """单次预测 - 彻底修复：拒绝虚假序列"""
        if not self.is_loaded:
            raise Exception("模型未加载")
        
        # 准备特征
        features = self.prepare_input_features(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date)
        
        # 彻底修复：检查是否有真实的历史数据
        if not hasattr(self, '_historical_features') or len(self._historical_features) < self.sequence_length:
            # 没有足够的历史数据，拒绝预测
            raise ValueError(
                f"无法进行预测：需要至少 {self.sequence_length} 个历史数据点，"
                f"但只有 {len(self._historical_features) if hasattr(self, '_historical_features') else 0} 个。"
                "请先使用 update_historical_features() 方法添加足够的历史数据。"
            )
        
        # 使用真实的历史数据创建序列
        sequence = self._historical_features[-self.sequence_length:]
        sequence = np.array(sequence).reshape(1, self.sequence_length, -1)
        
        # 预测
        with torch.no_grad():
            if hasattr(self, 'ensemble_models') and self.ensemble_models:
                # 集成预测
                predictions = []
                for model in self.ensemble_models:
                    pred = model(torch.FloatTensor(sequence))
                    predictions.append(pred.item())
                
                # 平均集成
                prediction = np.mean(predictions)
            else:
                # 单个模型预测
                prediction = self.model(torch.FloatTensor(sequence)).item()
        
        # 反标准化
        if self.scaler_y is not None:
            prediction = self.scaler_y.inverse_transform([[prediction]])[0][0]
        
        return max(0, prediction)  # 确保非负
    
    def predict_with_minimal_data(self, snow_depth_mm: float, snow_fall_mm: float, 
                                snow_water_equivalent_mm: float, date: datetime) -> float:
        """使用最小数据要求进行预测 - 新增方法"""
        if not self.is_loaded:
            raise Exception("模型未加载")
        
        # 准备特征
        features = self.prepare_input_features(snow_depth_mm, snow_fall_mm, snow_water_equivalent_mm, date)
        
        # 如果历史数据不足，使用滑动窗口方法
        if hasattr(self, '_historical_features') and len(self._historical_features) > 0:
            # 使用所有可用的历史数据
            available_features = self._historical_features.copy()
            available_features.append(features.flatten())
            
            # 如果数据仍然不足，使用重复填充（但给出明确警告）
            if len(available_features) < self.sequence_length:
                print(f"⚠️ 警告：历史数据不足，使用重复填充。这可能导致预测不准确。")
                print(f"   需要 {self.sequence_length} 个数据点，实际只有 {len(available_features)} 个")
                
                # 重复最后一个特征直到达到序列长度
                while len(available_features) < self.sequence_length:
                    available_features.append(available_features[-1])
            
            # 取最后sequence_length个特征
            sequence = available_features[-self.sequence_length:]
        else:
            # 完全没有历史数据，拒绝预测
            raise ValueError(
                "无法进行预测：完全没有历史数据。"
                "请先使用 update_historical_features() 方法添加历史数据，"
                "或使用 predict_with_minimal_data() 方法进行有限预测。"
            )
        
        sequence = np.array(sequence).reshape(1, self.sequence_length, -1)
        
        # 预测
        with torch.no_grad():
            if hasattr(self, 'ensemble_models') and self.ensemble_models:
                predictions = []
                for model in self.ensemble_models:
                    pred = model(torch.FloatTensor(sequence))
                    predictions.append(pred.item())
                prediction = np.mean(predictions)
            else:
                prediction = self.model(torch.FloatTensor(sequence)).item()
        
        # 反标准化
        if self.scaler_y is not None:
            prediction = self.scaler_y.inverse_transform([[prediction]])[0][0]
        
        return max(0, prediction)
    
    def initialize_with_real_data(self, real_features: np.ndarray):
        """使用真实数据初始化历史特征 - 禁止合成数据"""
        print("✅ 使用真实数据初始化历史特征")
        
        if not hasattr(self, '_historical_features'):
            self._historical_features = []
        
        # 只接受真实观测数据
        if real_features is not None and len(real_features) > 0:
            self._historical_features.append(real_features.flatten())
            print(f"✅ 已添加真实历史特征数据")
        else:
            print("⚠️ 警告：没有提供真实数据，历史特征保持为空")
            print("⚠️ 注意：系统禁止使用合成数据，请提供真实的观测数据")
    
    def update_historical_features(self, features: np.ndarray):
        """更新历史特征数据 - 新增方法"""
        if not hasattr(self, '_historical_features'):
            self._historical_features = []
        
        self._historical_features.append(features.flatten())
        
        # 保持最近的历史数据
        if len(self._historical_features) > self.sequence_length * 2:
            self._historical_features = self._historical_features[-self.sequence_length * 2:]
    
    def predict_series(self, start_date: str, end_date: str, 
                      snow_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """预测时间序列"""
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # 生成日期范围
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
            
            predictions = []
            for date in date_range:
                # 查找对应的雪数据
                date_str = date.strftime("%Y-%m-%d")
                if date_str in snow_data.index:
                    row = snow_data.loc[date_str]
                    snow_depth = row.get('snow_depth_mm', 0)
                    snow_fall = row.get('snow_fall_mm', 0)
                    snow_we = row.get('snow_water_equivalent_mm', 0)
                else:
                    # 如果没有数据，使用默认值
                    snow_depth = 0
                    snow_fall = 0
                    snow_we = 0
                
                # 预测
                prediction = self.predict_single(snow_depth, snow_fall, snow_we, date)
                
                predictions.append({
                    'date': date_str,
                    'snow_depth_mm': snow_depth,
                    'snow_fall_mm': snow_fall,
                    'snow_water_equivalent_mm': snow_we,
                    'predicted_swe_mm': round(prediction, 2),
                    'confidence': 'high' if self.is_loaded else 'low'
                })
            
            return predictions
            
        except Exception as e:
            print(f"❌ 序列预测失败: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_type': 'ensemble' if hasattr(self, 'ensemble_models') else 'single',
            'is_loaded': self.is_loaded,
            'sequence_length': self.sequence_length,
            'features': ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 'day_of_year', 'month', 'year']
        }
        
        if hasattr(self, 'ensemble_models'):
            info['n_models'] = len(self.ensemble_models)
            info['ensemble_method'] = 'simple_average'
        
        if self.scaler_X is not None:
            info['scaling'] = 'standardized'
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy' if self.is_loaded else 'unhealthy',
            'model_loaded': self.is_loaded,
            'timestamp': datetime.now().isoformat(),
            'model_info': self.get_model_info()
        }

# 全局预测器实例
_global_predictor = None

def get_predictor() -> OptimizedSWEPredictor:
    """获取全局预测器实例"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = OptimizedSWEPredictor()
    return _global_predictor
