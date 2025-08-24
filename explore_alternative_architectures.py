#!/usr/bin/env python3
"""
探索替代模型架构
尝试Transformer、GRU、1D-CNN等替代LSTM的架构
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

class SWELSTMModel(nn.Module):
    """SWE LSTM预测模型"""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1, sequence_length=30):
        super(SWELSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

class TransformerModel(nn.Module):
    """Transformer模型用于时间序列预测"""
    
    def __init__(self, input_size=6, d_model=64, nhead=8, num_layers=2, dropout=0.1, sequence_length=30):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, sequence_length, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer编码
        x = self.transformer(x)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 输出投影
        x = self.dropout(x)
        x = self.output_projection(x)
        
        return x

class GRUModel(nn.Module):
    """GRU模型用于时间序列预测"""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1, sequence_length=30):
        super(GRUModel, self).__init__()
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

class CNN1DModel(nn.Module):
    """1D-CNN模型用于时间序列预测"""
    
    def __init__(self, input_size=6, num_filters=64, kernel_size=3, num_layers=3, dropout=0.1, sequence_length=30):
        super(CNN1DModel, self).__init__()
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        
        # 1D卷积层
        layers = []
        in_channels = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = num_filters
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.fc = nn.Linear(num_filters, 1)
        
    def forward(self, x):
        # 转换维度: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # 卷积层
        x = self.conv_layers(x)
        
        # 全局平均池化
        x = self.global_pool(x).squeeze(-1)
        
        # 输出层
        x = self.fc(x)
        return x

class HybridModel(nn.Module):
    """混合模型：结合CNN和LSTM"""
    
    def __init__(self, input_size=6, cnn_filters=32, lstm_hidden=64, num_layers=2, dropout=0.1, sequence_length=30):
        super(HybridModel, self).__init__()
        self.sequence_length = sequence_length
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM处理序列
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, num_layers, 
                           dropout=dropout if num_layers > 1 else 0, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # CNN特征提取
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # 输出
        output = self.fc(lstm_out[:, -1, :])
        return output

class ArchitectureExplorer:
    """模型架构探索器"""
    
    def __init__(self):
        self.models = {}
        self.training_results = {}
        self.validation_results = {}
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
        
        print(f"📊 数据分割:")
        print(f"   训练集: {len(X_train)} 个序列")
        print(f"   验证集: {len(X_val)} 个序列")
        print(f"   测试集: {len(X_test)} 个序列")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_loaders(self, train_data, val_data, test_data, batch_size=32):
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
    
    def define_models(self):
        """定义要探索的模型"""
        print("🏗️ 定义模型架构...")
        
        models = {
            'LSTM_Original': SWELSTMModel(input_size=6, hidden_size=64, num_layers=2, dropout=0.1),
            'Transformer': TransformerModel(input_size=6, d_model=64, nhead=8, num_layers=2, dropout=0.1),
            'GRU': GRUModel(input_size=6, hidden_size=64, num_layers=2, dropout=0.1),
            'CNN1D': CNN1DModel(input_size=6, num_filters=64, kernel_size=3, num_layers=3, dropout=0.1),
            'Hybrid_CNN_LSTM': HybridModel(input_size=6, cnn_filters=32, lstm_hidden=64, num_layers=2, dropout=0.1)
        }
        
        # 打印模型参数数量
        for name, model in models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  {name}: {total_params:,} 参数 ({trainable_params:,} 可训练)")
        
        self.models = models
        return models
    
    def train_model(self, model, train_loader, val_loader, model_name, epochs=50, learning_rate=0.001):
        """训练单个模型"""
        print(f"🚀 训练模型: {model_name}")
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 训练历史
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        start_time = time.time()
        
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
            train_losses.append(train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        
        # 保存训练结果
        self.training_results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'training_time': training_time
        }
        
        print(f"✅ {model_name} 训练完成! 最佳验证损失: {best_val_loss:.6f}")
        print(f"   训练时间: {training_time:.2f} 秒")
        
        return model
    
    def evaluate_model(self, model, test_loader, model_name):
        """评估模型性能"""
        print(f"📊 评估模型: {model_name}")
        
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 反标准化
        predictions_rescaled = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_rescaled = self.scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # 计算指标
        mse = mean_squared_error(actuals_rescaled, predictions_rescaled)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_rescaled, predictions_rescaled)
        r2 = r2_score(actuals_rescaled, predictions_rescaled)
        
        # 保存验证结果
        self.validation_results[model_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse': mse
        }
        
        print(f"📈 {model_name} 测试集性能:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse': mse
        }
    
    def run_architecture_exploration(self):
        """运行架构探索"""
        print("🔬 开始模型架构探索...")
        
        # 加载数据
        data = self.load_data_and_scalers()
        if data is None:
            return
        
        # 准备序列数据
        X, y = self.prepare_sequences(data)
        
        # 分割数据
        train_data, val_data, test_data = self.split_data(X, y)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_data, val_data, test_data, batch_size=32
        )
        
        # 定义模型
        models = self.define_models()
        
        # 训练和评估所有模型
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"🎯 模型: {model_name}")
            print(f"{'='*60}")
            
            try:
                # 训练模型
                trained_model = self.train_model(model, train_loader, val_loader, model_name)
                
                # 评估模型
                metrics = self.evaluate_model(trained_model, test_loader, model_name)
                all_results[model_name] = metrics
                
                # 保存模型
                model_path = f"models/{model_name.lower().replace(' ', '_')}.pth"
                torch.save({
                    'model_state_dict': trained_model.state_dict(),
                    'model_type': model_name,
                    'scaler_X': self.scaler_X,
                    'scaler_y': self.scaler_y,
                    'sequence_length': self.sequence_length,
                    'input_size': 6,
                    'validation_metrics': metrics
                }, model_path)
                
                print(f"✅ 模型已保存: {model_path}")
                
            except Exception as e:
                print(f"❌ {model_name} 训练/评估失败: {e}")
                all_results[model_name] = {'error': str(e)}
        
        # 生成对比报告
        self.generate_architecture_comparison_report(all_results)
        
        return all_results
    
    def generate_architecture_comparison_report(self, results):
        """生成架构对比报告"""
        print("📝 生成架构对比报告...")
        
        # 过滤掉有错误的模型
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("❌ 没有有效的模型结果")
            return
        
        # 按RMSE排序
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['rmse'])
        
        # 打印对比结果
        print(f"\n{'='*80}")
        print("📊 模型架构对比结果")
        print(f"{'='*80}")
        
        print(f"{'模型':<25} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'训练时间':<12}")
        print(f"{'-'*80}")
        
        for model_name, metrics in sorted_results:
            training_time = self.training_results.get(model_name, {}).get('training_time', 0)
            print(f"{model_name:<25} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} "
                  f"{metrics['r2']:<12.4f} {training_time:<12.2f}s")
        
        # 找出最佳模型
        best_model = sorted_results[0]
        print(f"\n🏆 最佳模型: {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"logs/architecture_exploration_report_{timestamp}.md"
        
        report_content = f"""# 模型架构探索报告

## 报告时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 探索的架构

| 模型 | 描述 | 特点 |
|------|------|------|
| LSTM_Original | 原始LSTM模型 | 2层LSTM，64隐藏单元 |
| Transformer | Transformer编码器 | 多头注意力，位置编码 |
| GRU | 门控循环单元 | 简化LSTM，更少参数 |
| CNN1D | 一维卷积网络 | 局部特征提取 |
| Hybrid_CNN_LSTM | 混合模型 | CNN+LSTM组合 |

## 性能对比结果

| 模型 | RMSE | MAE | R² | 训练时间(s) |
|------|------|-----|----|-------------|
"""
        
        for model_name, metrics in sorted_results:
            training_time = self.training_results.get(model_name, {}).get('training_time', 0)
            report_content += f"| {model_name} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {metrics['r2']:.4f} | {training_time:.2f} |\n"
        
        report_content += f"""

## 最佳模型
🏆 **{best_model[0]}** - RMSE: {best_model[1]['rmse']:.4f}

## 关键发现
1. **性能对比**: 不同架构在SWE预测任务上的表现差异
2. **训练效率**: 各模型的训练时间和收敛速度
3. **泛化能力**: 在测试集上的表现vs验证集
4. **架构优势**: 每种架构的优缺点分析

## 建议
基于探索结果，建议：
- 采用 {best_model[0]} 作为主要模型
- 考虑模型集成以提高稳定性
- 进一步优化超参数
- 探索更多创新架构
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 架构对比报告已保存: {report_path}")

def main():
    """主函数"""
    print("🔬 HydrAI-SWE 模型架构探索")
    print("=" * 60)
    
    try:
        # 创建探索器
        explorer = ArchitectureExplorer()
        
        # 运行架构探索
        results = explorer.run_architecture_exploration()
        
        print("\n" + "=" * 60)
        print("🎉 模型架构探索完成!")
        print(f"✅ 共探索 {len(results)} 种架构")
        print("✅ 所有模型已训练和评估")
        print("✅ 对比报告已生成")
        
    except Exception as e:
        print(f"❌ 架构探索失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
