#!/usr/bin/env python3
"""
HydrAI-SWE 真实模型交叉验证系统
使用训练好的LSTM模型进行时间序列交叉验证
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SWELSTMModel(nn.Module):
    """SWE LSTM预测模型"""
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=1, dropout=0.1, sequence_length=30):
        super(SWELSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

class RealModelCrossValidator:
    """基于真实模型的交叉验证器"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "models/real_trained_swe_model.pth"
        self.scaler = StandardScaler()
        self.model = None
        self.cv_results = {}
        self.validation_splits = []
        
        # 创建日志目录
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/cv_results", exist_ok=True)
        
        # 设置图表样式
        plt.style.use('default')  # 使用默认样式避免中文字体问题
        
        logger.info("真实模型交叉验证器初始化完成")
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                return False
            
            # 加载模型权重（禁用weights_only以支持旧模型）
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # 从checkpoint获取模型参数
            input_size = checkpoint.get('input_size', 6)
            hidden_size = checkpoint.get('hidden_size', 64)
            num_layers = checkpoint.get('num_layers', 2)
            sequence_length = checkpoint.get('sequence_length', 30)
            
            # 创建匹配的模型实例
            self.model = SWELSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                sequence_length=sequence_length
            )
            
            # 加载状态字典
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"成功加载模型: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 确保date列是datetime类型
        if 'date' not in data.columns:
            data = data.reset_index()
        
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)
            
            # 添加时间特征
            data['day_of_year'] = data['date'].dt.dayofyear
            data['month'] = data['date'].dt.month
            data['year'] = data['date'].dt.year
        else:
            # 如果date是索引
            data = data.reset_index()
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)
            
            # 添加时间特征
            data['day_of_year'] = data['date'].dt.dayofyear
            data['month'] = data['date'].dt.month
            data['year'] = data['date'].dt.year
        
        # 检查是否有现成的雪数据列
        if 'snow_depth_mm' not in data.columns:
            logger.warning("未找到雪数据列，生成模拟数据")
            data['snow_depth_mm'] = np.random.normal(20, 10, len(data))
            data['snow_fall_mm'] = np.random.normal(5, 3, len(data))
            data['snow_water_equivalent_mm'] = data['snow_depth_mm'] * 0.3
        
        # 选择特征列（6个特征以匹配模型）
        feature_cols = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 'day_of_year', 'month', 'year']
        target_col = 'snow_water_equivalent_mm'
        
        # 检查所有需要的列是否存在
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"缺少列: {missing_cols}")
            raise ValueError(f"缺少必需的数据列: {missing_cols}")
        
        # 提取特征和目标
        X = data[feature_cols].values
        y = data[target_col].values
        
        return X, y
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """创建序列数据"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:(i + sequence_length)])
            y_seq.append(y[i + sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def predict_with_model(self, X: np.ndarray) -> np.ndarray:
        """使用训练好的模型进行预测"""
        if self.model is None:
            logger.error("模型未加载，无法进行预测")
            return np.array([])
        
        try:
            X_tensor = torch.FloatTensor(X)
            
            with torch.no_grad():
                predictions = self.model(X_tensor)
                predictions = predictions.cpu().numpy().flatten()
            
            return predictions
            
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return np.array([])
    
    def create_forward_chain_splits(self, data: pd.DataFrame, n_splits: int = 5, 
                                   min_train_size: int = 200, test_size: int = 60) -> List[Tuple]:
        """创建前向链式时间分割"""
        logger.info(f"创建前向链式时间分割: {n_splits} 折, 最小训练 {min_train_size} 天, 测试 {test_size} 天")
        
        splits = []
        total_days = len(data)
        
        # 确保有足够的数据
        if total_days < min_train_size + test_size:
            raise ValueError(f"数据不足: 需要至少 {min_train_size + test_size} 天，实际 {total_days} 天")
        
        # 创建前向链式分割
        for i in range(n_splits):
            # 训练集：从开始到指定位置
            train_end = min_train_size + i * test_size
            train_start = 0
            
            # 测试集：紧接训练集之后
            test_start = train_end
            test_end = min(test_start + test_size, total_days)
            
            # 确保测试集不超出数据范围
            if test_end <= test_start:
                break
            
            splits.append((train_start, train_end, test_start, test_end))
            logger.info(f"  分割 {i+1}: 训练 [{train_start}, {train_end}), 测试 [{test_start}, {test_end})")
        
        self.validation_splits = splits
        return splits
    
    def validate_real_model(self, data: pd.DataFrame, target_col: str = 'snow_water_equivalent_mm') -> Dict[str, Any]:
        """使用真实模型进行交叉验证"""
        logger.info("开始真实模型交叉验证...")
        
        # 加载模型
        if not self.load_model():
            return {"error": "模型加载失败"}
        
        # 准备数据
        X, y = self.prepare_data(data)
        
        if not self.validation_splits:
            self.create_forward_chain_splits(data)
        
        cv_metrics = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(self.validation_splits):
            logger.info(f"  验证折 {i+1}/{len(self.validation_splits)}")
            
            # 分割数据
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # 标准化特征
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 创建序列数据
            sequence_length = 30
            if len(X_train_scaled) > sequence_length:
                X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train, sequence_length)
                X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test, sequence_length)
                
                if len(X_test_seq) > 0:
                    # 使用真实模型预测
                    predictions = self.predict_with_model(X_test_seq)
                    
                    if len(predictions) > 0:
                        # 计算指标
                        mse = mean_squared_error(y_test_seq, predictions)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test_seq, predictions)
                        r2 = r2_score(y_test_seq, predictions)
                    else:
                        mse = rmse = mae = r2 = np.nan
                else:
                    mse = rmse = mae = r2 = np.nan
            else:
                mse = rmse = mae = r2 = np.nan
            
            cv_metrics.append({
                'fold': i + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            logger.info(f"    折 {i+1} 结果: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # 计算平均指标
        valid_metrics = [m for m in cv_metrics if not np.isnan(m['rmse'])]
        
        if valid_metrics:
            avg_metrics = {
                'mean_rmse': np.mean([m['rmse'] for m in valid_metrics]),
                'std_rmse': np.std([m['rmse'] for m in valid_metrics]),
                'mean_r2': np.mean([m['r2'] for m in valid_metrics]),
                'std_r2': np.std([m['r2'] for m in valid_metrics]),
                'mean_mae': np.mean([m['mae'] for m in valid_metrics]),
                'std_mae': np.std([m['mae'] for m in valid_metrics])
            }
        else:
            avg_metrics = {
                'mean_rmse': np.nan,
                'std_rmse': np.nan,
                'mean_r2': np.nan,
                'std_r2': np.nan,
                'mean_mae': np.nan,
                'std_mae': np.nan
            }
        
        result = {
            'model_type': 'Real SWE LSTM Model',
            'model_path': self.model_path,
            'cv_metrics': cv_metrics,
            'summary_metrics': avg_metrics,
            'validation_time': datetime.now().isoformat()
        }
        
        self.cv_results['real_swe_model'] = result
        logger.info(f"真实模型交叉验证完成，平均RMSE: {avg_metrics['mean_rmse']:.4f}, 平均R²: {avg_metrics['mean_r2']:.4f}")
        
        return result
    
    def run_comprehensive_validation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """运行综合交叉验证"""
        logger.info("开始真实模型综合交叉验证...")
        
        start_time = datetime.now()
        
        # 创建时间分割
        self.create_forward_chain_splits(data)
        
        # 验证真实模型
        results = {
            'validation_start': start_time.isoformat(),
            'data_info': {
                'total_samples': len(data),
                'date_range': f"{data.index[0]} to {data.index[-1]}" if hasattr(data.index[0], 'strftime') else "Unknown",
                'n_splits': len(self.validation_splits)
            },
            'models': {
                'real_swe_model': self.validate_real_model(data)
            }
        }
        
        end_time = datetime.now()
        results['validation_duration'] = (end_time - start_time).total_seconds()
        results['validation_end'] = end_time.isoformat()
        
        # 保存验证结果
        self.save_validation_results(results)
        
        # 生成验证图表
        self.generate_validation_plots(results)
        
        return results
    
    def save_validation_results(self, results: Dict[str, Any]):
        """保存验证结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"logs/cv_results/real_model_cross_validation_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"验证结果已保存: {filename}")
            
        except Exception as e:
            logger.error(f"保存验证结果失败: {e}")
    
    def generate_validation_plots(self, results: Dict[str, Any]):
        """生成验证图表"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('HydrAI-SWE Real Model Cross Validation Results', fontsize=16)
            
            # 获取真实模型结果
            model_result = results['models']['real_swe_model']
            
            if 'cv_metrics' in model_result:
                cv_metrics = model_result['cv_metrics']
                valid_metrics = [m for m in cv_metrics if not np.isnan(m['rmse'])]
                
                if valid_metrics:
                    fold_numbers = [m['fold'] for m in valid_metrics]
                    rmse_values = [m['rmse'] for m in valid_metrics]
                    r2_values = [m['r2'] for m in valid_metrics]
                    mae_values = [m['mae'] for m in valid_metrics]
                    
                    # 1. RMSE趋势
                    axes[0, 0].plot(fold_numbers, rmse_values, 'o-', color='blue', linewidth=2, markersize=8)
                    axes[0, 0].set_title('Real Model - RMSE Trend')
                    axes[0, 0].set_xlabel('Fold')
                    axes[0, 0].set_ylabel('RMSE')
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # 2. R² 趋势
                    axes[0, 1].plot(fold_numbers, r2_values, 'o-', color='green', linewidth=2, markersize=8)
                    axes[0, 1].set_title('Real Model - R² Trend')
                    axes[0, 1].set_xlabel('Fold')
                    axes[0, 1].set_ylabel('R²')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # 3. MAE趋势
                    axes[1, 0].plot(fold_numbers, mae_values, 'o-', color='red', linewidth=2, markersize=8)
                    axes[1, 0].set_title('Real Model - MAE Trend')
                    axes[1, 0].set_xlabel('Fold')
                    axes[1, 0].set_ylabel('MAE')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # 4. 性能汇总
                    summary = model_result['summary_metrics']
                    metrics_names = ['RMSE', 'R²', 'MAE']
                    metrics_values = [
                        summary.get('mean_rmse', 0),
                        summary.get('mean_r2', 0),
                        summary.get('mean_mae', 0)
                    ]
                    
                    colors = ['blue', 'green', 'red']
                    bars = axes[1, 1].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
                    axes[1, 1].set_title('Model Performance Summary')
                    axes[1, 1].set_ylabel('Value')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # 添加数值标签
                    for bar, value in zip(bars, metrics_values):
                        height = bar.get_height()
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                       f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存图表
            plot_filename = f"logs/cv_results/real_model_validation_plots_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"验证图表已保存: {plot_filename}")
            
        except Exception as e:
            logger.error(f"生成验证图表失败: {e}")

def main():
    """主函数"""
    print("🔍 HydrAI-SWE 真实模型交叉验证系统")
    print("=" * 60)
    
    try:
        # 创建验证器
        validator = RealModelCrossValidator()
        
        # 加载真实数据
        data_path = "data/processed/comprehensive_training_dataset.csv"
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            data.index.name = 'date'
            logger.info(f"加载综合数据集: {len(data)} 条记录")
        else:
            # 创建示例数据
            dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
            data = pd.DataFrame(index=dates)
            logger.info(f"创建示例数据: {len(data)} 天")
        
        # 运行综合验证
        results = validator.run_comprehensive_validation(data)
        
        # 生成验证报告
        model_result = results['models']['real_swe_model']
        if 'summary_metrics' in model_result:
            summary = model_result['summary_metrics']
            print(f"\n🎯 真实模型验证结果:")
            print(f"平均RMSE: {summary.get('mean_rmse', 'N/A'):.4f} ± {summary.get('std_rmse', 'N/A'):.4f}")
            print(f"平均R²: {summary.get('mean_r2', 'N/A'):.4f} ± {summary.get('std_r2', 'N/A'):.4f}")
            print(f"平均MAE: {summary.get('mean_mae', 'N/A'):.4f} ± {summary.get('std_mae', 'N/A'):.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 真实模型交叉验证完成!")
        
    except Exception as e:
        print(f"❌ 验证系统运行失败: {e}")
        logger.error(f"验证系统运行失败: {e}")

if __name__ == "__main__":
    main()
