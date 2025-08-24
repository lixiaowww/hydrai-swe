#!/usr/bin/env python3
"""
快速超参数优化脚本
使用轻量级策略快速找到好的参数组合
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import time

class QuickGRUModel(nn.Module):
    """快速优化的GRU模型"""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1):
        super(QuickGRUModel, self).__init__()
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

class QuickHyperparameterOptimizer:
    """快速超参数优化器"""
    
    def __init__(self):
        self.scaler_X = None
        self.scaler_y = None
        self.sequence_length = 30
        self.optimization_results = []
        
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
    
    def split_data(self, X, y, train_ratio=0.8, val_ratio=0.2):
        """快速数据分割"""
        n = len(X)
        train_end = int(n * train_ratio)
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:]
        y_val = y[train_end:]
        
        return (X_train, y_train), (X_val, y_val)
    
    def create_data_loaders(self, train_data, val_data, batch_size):
        """创建数据加载器"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def quick_train_and_evaluate(self, model, train_loader, val_loader, params):
        """快速训练和评估"""
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # 快速训练参数
        epochs = 15  # 减少训练轮数
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5  # 减少耐心值
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
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
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        return best_val_loss
    
    def run_quick_optimization(self):
        """运行快速优化"""
        print("🚀 开始快速超参数优化...")
        
        # 加载数据
        data = self.load_data_and_scalers()
        if data is None:
            return
        
        # 准备序列数据
        X, y = self.prepare_sequences(data)
        
        # 分割数据
        train_data, val_data = self.split_data(X, y)
        
        # 定义要测试的参数组合（减少组合数量）
        param_combinations = [
            # 基础配置
            {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 32},
            {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 32},
            {'hidden_size': 64, 'num_layers': 3, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 32},
            
            # 学习率变化
            {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.0005, 'batch_size': 32},
            {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.002, 'batch_size': 32},
            
            # Dropout变化
            {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.05, 'learning_rate': 0.001, 'batch_size': 32},
            {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32},
            
            # 批大小变化
            {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 16},
            {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 64},
            
            # 组合优化
            {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.15, 'learning_rate': 0.0008, 'batch_size': 48},
            {'hidden_size': 96, 'num_layers': 2, 'dropout': 0.12, 'learning_rate': 0.0012, 'batch_size': 40},
        ]
        
        print(f"🎯 测试 {len(param_combinations)} 种参数组合...")
        
        best_result = None
        best_val_loss = float('inf')
        
        for i, params in enumerate(param_combinations):
            print(f"\n{'='*50}")
            print(f"🔍 试验 {i+1}/{len(param_combinations)}")
            print(f"参数: {params}")
            print(f"{'='*50}")
            
            try:
                # 创建数据加载器
                train_loader, val_loader = self.create_data_loaders(
                    train_data, val_data, params['batch_size']
                )
                
                # 创建模型
                model = QuickGRUModel(
                    input_size=6,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout']
                )
                
                # 快速训练和评估
                start_time = time.time()
                val_loss = self.quick_train_and_evaluate(model, train_loader, val_loader, params)
                training_time = time.time() - start_time
                
                # 记录结果
                result = {
                    'trial': i + 1,
                    'params': params,
                    'val_loss': val_loss,
                    'training_time': training_time
                }
                
                self.optimization_results.append(result)
                
                print(f"✅ 试验 {i+1} 完成:")
                print(f"   验证损失: {val_loss:.6f}")
                print(f"   训练时间: {training_time:.2f} 秒")
                
                # 更新最佳结果
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_result = result
                    print(f"🏆 新的最佳结果!")
                
            except Exception as e:
                print(f"❌ 试验 {i+1} 失败: {e}")
                continue
        
        # 保存结果
        self.save_quick_optimization_results(best_result)
        
        return best_result
    
    def save_quick_optimization_results(self, best_result):
        """保存快速优化结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存最佳参数
        best_params_path = f"logs/quick_best_hyperparameters_{timestamp}.json"
        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
        
        import json
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_value': best_result['val_loss'],
                'best_params': best_result['params'],
                'n_trials': len(self.optimization_results),
                'optimization_time': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 最佳参数已保存: {best_params_path}")
        
        # 保存所有结果
        all_results_path = f"logs/quick_optimization_all_results_{timestamp}.json"
        with open(all_results_path, 'w', encoding='utf-8') as f:
            json.dump(self.optimization_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 所有结果已保存: {all_results_path}")
        
        # 生成快速优化报告
        self.generate_quick_optimization_report(best_result, timestamp)
    
    def generate_quick_optimization_report(self, best_result, timestamp):
        """生成快速优化报告"""
        print("📝 生成快速优化报告...")
        
        report_path = f"logs/quick_optimization_report_{timestamp}.md"
        
        # 按验证损失排序
        sorted_results = sorted(self.optimization_results, key=lambda x: x['val_loss'])
        
        report_content = f"""# 快速超参数优化报告

## 优化时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 优化策略
- **快速训练**: 每个试验最多15个epoch
- **早停机制**: 5个epoch无改善即停止
- **参数组合**: 预定义的10种参数组合
- **总耗时**: 约 {sum(r['training_time'] for r in self.optimization_results):.1f} 秒

## 最佳结果
🏆 **最佳验证损失**: {best_result['val_loss']:.6f}

### 最佳超参数
"""
        
        for key, value in best_result['params'].items():
            report_content += f"- **{key}**: {value}\n"
        
        report_content += f"""

## 所有试验结果排名

| 排名 | 试验 | 隐藏大小 | 层数 | Dropout | 学习率 | 批大小 | 验证损失 | 训练时间(s) |
|------|------|----------|------|---------|--------|---------|----------|-------------|
"""
        
        for i, result in enumerate(sorted_results):
            params = result['params']
            report_content += f"| {i+1} | {result['trial']} | {params['hidden_size']} | {params['num_layers']} | {params['dropout']} | {params['learning_rate']} | {params['batch_size']} | {result['val_loss']:.6f} | {result['training_time']:.2f} |\n"
        
        report_content += f"""

## 关键发现
1. **最佳配置**: {best_result['params']['hidden_size']}隐藏单元, {best_result['params']['num_layers']}层, {best_result['params']['dropout']}dropout
2. **训练效率**: 平均每次试验 {sum(r['training_time'] for r in self.optimization_results)/len(self.optimization_results):.2f} 秒
3. **性能提升**: 相比默认参数，验证损失从 {max(r['val_loss'] for r in self.optimization_results):.6f} 降至 {best_result['val_loss']:.6f}

## 下一步行动
1. **使用最佳参数**: 用最佳配置重新训练完整模型
2. **精细调优**: 在最佳参数附近进行更精细的搜索
3. **模型集成**: 考虑集成前3-5个最佳配置
4. **数据增强**: 结合最佳超参数尝试数据增强
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 快速优化报告已保存: {report_path}")

def main():
    """主函数"""
    print("⚡ HydrAI-SWE 快速超参数优化")
    print("=" * 60)
    
    try:
        # 创建快速优化器
        optimizer = QuickHyperparameterOptimizer()
        
        # 运行快速优化
        best_result = optimizer.run_quick_optimization()
        
        if best_result:
            print("\n" + "=" * 60)
            print("🎉 快速超参数优化完成!")
            print(f"✅ 最佳验证损失: {best_result['val_loss']:.6f}")
            print(f"✅ 总耗时: {sum(r['training_time'] for r in optimizer.optimization_results):.1f} 秒")
            print("✅ 优化结果已保存")
            print("✅ 快速优化报告已生成")
            
            # 显示优化建议
            print(f"\n💡 立即行动建议:")
            print(f"  1. 使用最佳参数重新训练完整模型")
            print(f"  2. 在最佳参数附近进行精细搜索")
            print(f"  3. 考虑集成前3个最佳配置")
            print(f"  4. 尝试数据增强技术")
        else:
            print("❌ 快速优化失败")
        
    except Exception as e:
        print(f"❌ 快速超参数优化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
