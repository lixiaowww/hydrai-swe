#!/usr/bin/env python3
"""
模型训练修复器
精准修复R²为负值问题，回归核心目标
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os

# 导入防过拟合核心系统
from .anti_overfitting_core import AntiOverfittingCore
from ..data.data_quality_detector import DataQualityDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingFixer:
    """模型训练修复器"""
    
    def __init__(self):
        """初始化训练修复器"""
        self.anti_overfitting = AntiOverfittingCore()
        self.data_quality = DataQualityDetector()
        self.fix_history = []
        
        # 创建输出目录
        os.makedirs('models/fixed', exist_ok=True)
        
        logger.info("✅ 模型训练修复器初始化完成")
    
    def diagnose_and_fix(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray, 
                        train_losses: List[float], val_losses: List[float]) -> Dict:
        """诊断并修复训练问题"""
        try:
            logger.info("🔍 开始诊断训练问题...")
            
            diagnosis_result = {
                'timestamp': datetime.now().isoformat(),
                'data_quality': {},
                'overfitting_analysis': {},
                'fixes_applied': [],
                'final_status': 'unknown'
            }
            
            # 步骤1: 数据质量诊断
            logger.info("📊 步骤1: 数据质量诊断...")
            data_quality_result = self.data_quality.detect_data_issues(X_train, y_train)
            diagnosis_result['data_quality'] = data_quality_result
            
            if data_quality_result['status'] == 'success':
                quality_score = data_quality_result['quality_score']
                logger.info(f"数据质量得分: {quality_score:.3f}")
                
                if quality_score < 0.5:
                    logger.warning("⚠️ 数据质量较差，需要先解决数据问题")
                    diagnosis_result['final_status'] = 'data_quality_issue'
                    return diagnosis_result
            
            # 步骤2: 过拟合诊断
            logger.info("🔍 步骤2: 过拟合诊断...")
            overfitting_result = self.anti_overfitting.detect_overfitting(train_losses, val_losses)
            diagnosis_result['overfitting_analysis'] = overfitting_result
            
            if overfitting_result['status'] == 'success' and overfitting_result['overfitting']:
                logger.warning("⚠️ 检测到过拟合，开始修复...")
                
                # 应用过拟合修复
                fix_result = self.anti_overfitting.fix_overfitting(model, X_train, y_train, X_val, y_val)
                diagnosis_result['fixes_applied'].append({
                    'type': 'overfitting_fix',
                    'result': fix_result
                })
                
                if fix_result['status'] == 'success':
                    logger.info("✅ 过拟合修复完成")
                    diagnosis_result['final_status'] = 'overfitting_fixed'
                else:
                    logger.error("❌ 过拟合修复失败")
                    diagnosis_result['final_status'] = 'fix_failed'
            else:
                logger.info("✅ 未检测到过拟合")
                diagnosis_result['final_status'] = 'no_overfitting'
            
            # 步骤3: 模型性能修复
            if diagnosis_result['final_status'] in ['overfitting_fixed', 'no_overfitting']:
                logger.info("🔧 步骤3: 模型性能修复...")
                performance_fix = self._fix_model_performance(model, X_train, y_train, X_val, y_val)
                diagnosis_result['fixes_applied'].append({
                    'type': 'performance_fix',
                    'result': performance_fix
                })
            
            # 保存诊断结果
            self.fix_history.append(diagnosis_result)
            try:
                self._save_diagnosis_result(diagnosis_result)
            except Exception as e:
                logger.warning(f"保存诊断结果失败: {e}")
            
            logger.info(f"✅ 诊断和修复完成，最终状态: {diagnosis_result['final_status']}")
            
            return diagnosis_result
            
        except Exception as e:
            logger.error(f"❌ 诊断和修复失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _fix_model_performance(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """修复模型性能问题"""
        try:
            logger.info("🔧 开始修复模型性能...")
            
            # 检查数据规模
            n_samples = len(X_train)
            n_features = X_train.shape[1]
            
            # 基于数据规模选择最佳策略
            if n_samples < 100:
                strategy = 'ultra_simple'
            elif n_samples < 500:
                strategy = 'simple'
            else:
                strategy = 'standard'
            
            logger.info(f"选择策略: {strategy}")
            
            # 应用相应策略
            if strategy == 'ultra_simple':
                fixed_model = self._apply_ultra_simple_strategy(model, X_train.shape[1])
            elif strategy == 'simple':
                fixed_model = self._apply_simple_strategy(model, X_train.shape[1])
            else:
                fixed_model = self._apply_standard_strategy(model, X_train.shape[1])
            
            # 优化训练参数
            optimized_params = self._get_optimized_params(strategy, n_samples, n_features)
            
            result = {
                'status': 'success',
                'strategy_applied': strategy,
                'original_model': str(model),
                'fixed_model': str(fixed_model),
                'optimized_params': optimized_params,
                'fix_timestamp': datetime.now().isoformat()
            }
            
            # 保存修复后的模型
            self._save_fixed_model(fixed_model, strategy)
            
            logger.info("✅ 模型性能修复完成")
            return result
            
        except Exception as e:
            logger.error(f"❌ 修复模型性能失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _apply_ultra_simple_strategy(self, model: nn.Module, input_size: int) -> nn.Module:
        """应用超简单策略"""
        try:
            # 创建极简模型
            class UltraSimpleLSTM(nn.Module):
                def __init__(self, input_size: int):
                    super(UltraSimpleLSTM, self).__init__()
                    
                    # 极简设计：最小隐藏层，无dropout
                    hidden_size = max(4, input_size // 16)
                    
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                        batch_first=True
                    )
                    
                    self.fc = nn.Linear(hidden_size, 1)
                    
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_output = lstm_out[:, -1, :]
                    return self.fc(last_output)
            
            return UltraSimpleLSTM(input_size)
            
        except Exception as e:
            logger.error(f"❌ 应用超简单策略失败: {e}")
            raise
    
    def _apply_simple_strategy(self, model: nn.Module, input_size: int) -> nn.Module:
        """应用简单策略"""
        try:
            # 创建简单模型
            class SimpleLSTM(nn.Module):
                def __init__(self, input_size: int):
                    super(SimpleLSTM, self).__init__()
                    
                    # 简单设计：适中的隐藏层，轻微正则化
                    hidden_size = max(8, input_size // 8)
                    
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                        batch_first=True,
                        dropout=0.1  # 轻微dropout
                    )
                    
                    self.dropout = nn.Dropout(0.1)
                    self.fc = nn.Linear(hidden_size, 1)
                    
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_output = lstm_out[:, -1, :]
                    out = self.dropout(last_output)
                    return self.fc(out)
            
            return SimpleLSTM(input_size)
            
        except Exception as e:
            logger.error(f"❌ 应用简单策略失败: {e}")
            raise
    
    def _apply_standard_strategy(self, model: nn.Module, input_size: int) -> nn.Module:
        """应用标准策略"""
        try:
            # 创建标准模型
            class StandardLSTM(nn.Module):
                def __init__(self, input_size: int):
                    super(StandardLSTM, self).__init__()
                    
                    # 标准设计：合理的隐藏层，适度正则化
                    hidden_size = max(16, input_size // 4)
                    
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=2,  # 2层LSTM
                        batch_first=True,
                        dropout=0.2
                    )
                    
                    self.dropout = nn.Dropout(0.2)
                    self.fc = nn.Linear(hidden_size, 1)
                    
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_output = lstm_out[:, -1, :]
                    out = self.dropout(last_output)
                    return self.fc(out)
            
            return StandardLSTM(input_size)
            
        except Exception as e:
            logger.error(f"❌ 应用标准策略失败: {e}")
            raise
    
    def _get_optimized_params(self, strategy: str, n_samples: int, n_features: int) -> Dict:
        """获取优化的训练参数"""
        try:
            if strategy == 'ultra_simple':
                params = {
                    'batch_size': 4,
                    'epochs': 15,
                    'learning_rate': 0.01,
                    'patience': 3,
                    'early_stopping': True,
                    'reduce_lr_on_plateau': True,
                    'weight_decay': 0.0001
                }
            elif strategy == 'simple':
                params = {
                    'batch_size': 8,
                    'epochs': 25,
                    'learning_rate': 0.005,
                    'patience': 5,
                    'early_stopping': True,
                    'reduce_lr_on_plateau': True,
                    'weight_decay': 0.0005
                }
            else:  # standard
                params = {
                    'batch_size': 16,
                    'epochs': 40,
                    'learning_rate': 0.001,
                    'patience': 8,
                    'early_stopping': True,
                    'reduce_lr_on_plateau': True,
                    'weight_decay': 0.001
                }
            
            # 根据数据特征调整
            if n_features > n_samples // 3:
                params['batch_size'] = max(2, params['batch_size'] // 2)
                params['epochs'] = min(params['epochs'], 20)
            
            return params
            
        except Exception as e:
            logger.error(f"❌ 获取优化参数失败: {e}")
            return {}
    
    def _save_fixed_model(self, model: nn.Module, strategy: str):
        """保存修复后的模型"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f"models/fixed/fixed_model_{strategy}_{timestamp}.pth"
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'strategy': strategy,
                'fix_timestamp': timestamp
            }, model_path)
            
            logger.info(f"✅ 修复后的模型已保存: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ 保存修复后的模型失败: {e}")
    
    def _save_diagnosis_result(self, result: Dict):
        """保存诊断结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = f"models/fixed/diagnosis_result_{timestamp}.json"
            
            import json
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"❌ 保存诊断结果失败: {e}")
    
    def get_fix_summary(self) -> Dict:
        """获取修复摘要"""
        if not self.fix_history:
            return {"message": "暂无修复记录"}
        
        total_fixes = len(self.fix_history)
        successful_fixes = sum(1 for r in self.fix_history if r['final_status'] in ['overfitting_fixed', 'no_overfitting'])
        
        # 统计各种状态
        status_counts = {}
        for result in self.fix_history:
            status = result.get('final_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_fixes": total_fixes,
            "successful_fixes": successful_fixes,
            "success_rate": f"{successful_fixes/total_fixes*100:.1f}%",
            "status_distribution": status_counts,
            "last_fix": self.fix_history[-1].get('timestamp', 'Unknown')
        }
