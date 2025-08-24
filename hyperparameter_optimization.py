#!/usr/bin/env python3
"""
超参数优化脚本
使用Optuna对最佳模型进行深度优化
"""

import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime
import time

class OptimizedGRUModel(nn.Module):
    """可优化的GRU模型"""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1, 
                 use_batch_norm=True, use_residual=True, activation='relu'):
        super(OptimizedGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, 
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        
        # 批归一化
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            self.activation,
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)
        
        # GRU处理
        gru_out, _ = self.gru(x)
        
        # 取最后一个时间步
        last_output = gru_out[:, -1, :]
        
        # 批归一化
        if self.use_batch_norm:
            last_output = self.bn(last_output)
        
        # 残差连接
        if self.use_residual and self.num_layers > 1:
            residual = self.input_projection(x[:, -1, :])
            last_output = last_output + residual
        
        # 输出层
        output = self.output_layers(last_output)
        return output

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, n_trials=100):
        self.n_trials = n_trials
        self.best_trial = None
        self.optimization_history = []
        self.scaler_X = None
        self.scaler_y = None
        self.sequence_length = 30
        
    def load_data_and_scalers(self):
        """加载数据和标准化器"""
        print("📊 加载数据和标准化器...")
        
        try:
            # 加载标准化数据
            data_path = "data/processed/standardized_training_dataset.csv"
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"✅ 加载数据: {len(data)} 条记录")
            
            # 加载标准化器参数
            import pickle
            with open('models/standardization_params.pkl', 'rb') as f:
                params = pickle.load(f)
            
            # 重建标准化器
            self.scaler_X = StandardScaler()
            self.scaler_X.mean_ = params['scaler_X_mean']
            self.scaler_X.scale_ = params['scaler_X_scale']
            
            self.scaler_y = StandardScaler()
            self.scaler_y.mean_ = params['scaler_y_mean']
            self.scaler_y.scale_ = params['scaler_y_scale']
            
            print("✅ 标准化器加载成功")
            return data
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return None
    
    def prepare_sequences(self, data):
        """准备序列数据"""
        print("🔄 准备序列数据...")
        
        feature_cols = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                       'day_of_year', 'month', 'year']
        target_col = 'snow_water_equivalent_mm'
        
        X = data[feature_cols].values
        y = data[target_col].values
        
        # 标准化
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
        
        # 创建序列
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:(i + self.sequence_length)])
            y_seq.append(y_scaled[i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"✅ 序列数据准备完成: {X_seq.shape}, {y_seq.shape}")
        return X_seq, y_seq
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """分割数据"""
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_loaders(self, train_data, val_data, test_data, batch_size):
        """创建数据加载器"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def objective(self, trial):
        """优化目标函数"""
        # 超参数采样
        params = {
            'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
            'activation': trial.suggest_categorical('activation', ['relu', 'swish', 'gelu']),
            'sequence_length': trial.suggest_categorical('sequence_length', [15, 30, 45, 60])
        }
        
        try:
            # 准备数据
            data = self.load_data_and_scalers()
            if data is None:
                return float('inf')
            
            # 使用新的序列长度
            self.sequence_length = params['sequence_length']
            X, y = self.prepare_sequences(data)
            
            # 分割数据
            train_data, val_data, test_data = self.split_data(X, y)
            
            # 创建数据加载器
            train_loader, val_loader, test_loader = self.create_data_loaders(
                train_data, val_data, test_data, params['batch_size']
            )
            
            # 创建模型
            model = OptimizedGRUModel(
                input_size=6,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                use_batch_norm=params['use_batch_norm'],
                use_residual=params['use_residual'],
                activation=params['activation']
            )
            
            # 训练模型
            val_loss = self.train_and_evaluate(
                model, train_loader, val_loader, params
            )
            
            # 记录优化历史
            self.optimization_history.append({
                'trial': trial.number,
                'params': params,
                'val_loss': val_loss
            })
            
            return val_loss
            
        except Exception as e:
            print(f"❌ 试验 {trial.number} 失败: {e}")
            return float('inf')
    
    def train_and_evaluate(self, model, train_loader, val_loader, params):
        """训练和评估模型"""
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        # 训练参数
        epochs = 30
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 8
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 学习率调度
            scheduler.step()
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        return best_val_loss
    
    def run_optimization(self):
        """运行超参数优化"""
        print("🚀 开始超参数优化...")
        
        # 创建研究
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # 运行优化
        study.optimize(self.objective, n_trials=self.n_trials, timeout=3600)
        
        # 保存最佳结果
        self.best_trial = study.best_trial
        
        print(f"\n🏆 优化完成!")
        print(f"最佳验证损失: {self.best_trial.value:.6f}")
        print(f"最佳参数:")
        for key, value in self.best_trial.params.items():
            print(f"  {key}: {value}")
        
        # 保存优化历史
        self.save_optimization_results(study)
        
        return study
    
    def save_optimization_results(self, study):
        """保存优化结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存最佳参数
        best_params_path = f"logs/best_hyperparameters_{timestamp}.json"
        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
        
        import json
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_value': study.best_trial.value,
                'best_params': study.best_trial.params,
                'n_trials': len(study.trials),
                'optimization_time': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 最佳参数已保存: {best_params_path}")
        
        # 保存优化历史
        history_path = f"logs/optimization_history_{timestamp}.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.optimization_history, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 优化历史已保存: {history_path}")
        
        # 生成优化报告
        self.generate_optimization_report(study, timestamp)
    
    def generate_optimization_report(self, study, timestamp):
        """生成优化报告"""
        print("📝 生成优化报告...")
        
        report_path = f"logs/hyperparameter_optimization_report_{timestamp}.md"
        
        report_content = f"""# 超参数优化报告

## 优化时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 优化配置
- **试验次数**: {self.n_trials}
- **优化方向**: 最小化验证损失
- **采样器**: TPE (Tree-structured Parzen Estimator)
- **剪枝器**: Median Pruner

## 最佳结果
🏆 **最佳验证损失**: {study.best_trial.value:.6f}

### 最佳超参数
"""
        
        for key, value in study.best_trial.params.items():
            report_content += f"- **{key}**: {value}\n"
        
        report_content += f"""

## 优化统计
- **总试验数**: {len(study.trials)}
- **成功试验数**: {len([t for t in study.trials if t.value != float('inf')])}
- **失败试验数**: {len([t for t in study.trials if t.value == float('inf')])}

## 参数重要性分析
"""
        
        # 计算参数重要性
        try:
            importance = optuna.importance.get_param_importances(study)
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                report_content += f"- **{param}**: {imp:.4f}\n"
        except:
            report_content += "- 无法计算参数重要性\n"
        
        report_content += f"""

## 优化建议
1. **重点关注**: 根据参数重要性，优先调整重要参数
2. **进一步探索**: 在最佳参数附近进行精细搜索
3. **模型集成**: 考虑集成多个优秀试验的结果
4. **数据增强**: 结合最佳超参数，尝试数据增强技术

## 下一步行动
- 使用最佳超参数重新训练完整模型
- 在测试集上评估最终性能
- 考虑模型集成和部署策略
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 优化报告已保存: {report_path}")

def main():
    """主函数"""
    print("🚀 HydrAI-SWE 超参数优化")
    print("=" * 60)
    
    try:
        # 创建优化器
        optimizer = HyperparameterOptimizer(n_trials=50)  # 减少试验次数以节省时间
        
        # 运行优化
        study = optimizer.run_optimization()
        
        print("\n" + "=" * 60)
        print("🎉 超参数优化完成!")
        print(f"✅ 最佳验证损失: {study.best_trial.value:.6f}")
        print("✅ 优化结果已保存")
        print("✅ 优化报告已生成")
        
        # 显示优化建议
        print(f"\n💡 优化建议:")
        print(f"  1. 使用最佳超参数重新训练模型")
        print(f"  2. 尝试数据增强技术")
        print(f"  3. 考虑模型集成")
        print(f"  4. 进一步探索参数空间")
        
    except Exception as e:
        print(f"❌ 超参数优化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
