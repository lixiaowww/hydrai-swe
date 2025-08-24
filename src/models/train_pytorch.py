#!/usr/bin/env python3
"""
使用PyTorch直接训练LSTM模型
避免NeuralHydrology的复杂性，直接实现训练流程
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class SnowRunoffDataset(Dataset):
    """积雪-径流数据集"""
    
    def __init__(self, data, sequence_length=30):
        self.data = data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        # 获取输入序列
        x = self.data[idx:idx + self.sequence_length, :-1]  # 除了最后一列（目标变量）
        y = self.data[idx + self.sequence_length, -1]  # 目标变量
        
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMRegressor(nn.Module):
    """LSTM回归模型"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

def create_training_data():
    """创建训练数据"""
    print("Creating training data...")
    
    # 创建时间序列
    dates = pd.date_range('1979-01-01', '1998-12-31', freq='D')
    
    # 创建模拟数据
    np.random.seed(42)  # 确保可重复性
    
    # 积雪深度：季节性变化 + 随机噪声
    seasonal_snow = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + 50
    snow_depth = np.maximum(0, seasonal_snow + np.random.normal(0, 20, len(dates)))
    
    # 降雪量：冬季较高
    snow_fall = np.where(dates.month.isin([12, 1, 2, 3]), 
                         np.random.exponential(10, len(dates)), 
                         np.random.exponential(2, len(dates)))
    
    # 雪水当量：积雪深度的30%
    snow_water_equivalent = snow_depth * 0.3
    
    # 径流：基于积雪融化的简化模型
    streamflow = 1000 + snow_depth * 0.1 + np.random.normal(0, 50, len(dates))
    streamflow = np.maximum(500, streamflow)  # 最小径流
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': dates,
        'snow_depth_mm': snow_depth,
        'snow_fall_mm': snow_fall,
        'snow_water_equivalent_mm': snow_water_equivalent,
        'day_of_year': dates.dayofyear,
        'month': dates.month,
        'year': dates.year,
        'streamflow_m3s': streamflow
    })
    
    print(f"Created {len(data)} records")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    return data

def prepare_data_for_training(data, sequence_length=30):
    """准备训练数据"""
    print("Preparing data for training...")
    
    # 选择特征列
    feature_columns = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                      'day_of_year', 'month', 'year', 'streamflow_m3s']
    
    # 提取特征和目标
    features = data[feature_columns].values
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(features_scaled))
    train_data = features_scaled[:train_size]
    val_data = features_scaled[train_size:]
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    
    return train_data, val_data, scaler

def train_model(train_data, val_data, model_params):
    """训练模型"""
    print("Starting model training...")
    
    # 创建数据加载器
    train_dataset = SnowRunoffDataset(train_data, model_params['sequence_length'])
    val_dataset = SnowRunoffDataset(val_data, model_params['sequence_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'], shuffle=False)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTMRegressor(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout']
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
    
    # 训练循环
    train_losses = []
    val_losses = []
    
    for epoch in range(model_params['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 计算R²分数
        val_r2 = r2_score(val_targets, val_predictions)
        
        print(f"Epoch {epoch+1}/{model_params['epochs']}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val R²: {val_r2:.4f}")
    
    # 注意：这里需要从外部传入scaler，因为它在prepare_data_for_training函数中定义
    return model, train_losses, val_losses

def save_model_and_results(model, train_losses, val_losses, scaler, model_params):
    """保存模型和结果"""
    print("Saving model and results...")
    
    # 创建输出目录
    output_dir = Path("models/pytorch_lstm")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = output_dir / "snow_runoff_lstm.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': model_params,
        'scaler': scaler
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_params': model_params
    }
    
    history_path = output_dir / "training_history.json"
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Training history saved to: {history_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = output_dir / "training_loss.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to: {plot_path}")

def main():
    """主函数"""
    print("🚀 开始积雪-径流LSTM模型训练")
    print("=" * 50)
    
    # 模型参数
    model_params = {
        'input_size': 6,  # 特征数量
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'sequence_length': 30,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001
    }
    
    print("Model parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    try:
        # 创建数据
        data = create_training_data()
        
        # 准备训练数据
        train_data, val_data, scaler = prepare_data_for_training(data, model_params['sequence_length'])
        
        # 训练模型
        model, train_losses, val_losses = train_model(train_data, val_data, model_params)
        
        # 保存结果
        save_model_and_results(model, train_losses, val_losses, scaler, model_params)
        
        print("\n🎉 训练完成！")
        print("模型和结果已保存到 models/pytorch_lstm/ 目录")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
