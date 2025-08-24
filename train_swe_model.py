#!/usr/bin/env python3
"""
HydrAI-SWE 模型训练脚本
使用PyTorch直接训练LSTM模型进行SWE预测
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import os

class SWELSTMModel(nn.Module):
    """SWE预测LSTM模型"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SWELSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        output = self.fc(self.dropout(last_output))
        return output

def load_and_prepare_data(data_path, sequence_length=30):
    """加载和准备训练数据"""
    print("📊 加载训练数据...")
    
    # 加载数据
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # 选择特征列
    feature_columns = [
        'snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm',
        'day_of_year', 'month', 'year'
    ]
    target_column = 'streamflow_m3s'
    
    # 检查数据质量
    print(f"数据形状: {df.shape}")
    print(f"特征列: {feature_columns}")
    print(f"目标列: {target_column}")
    
    # 处理缺失值
    df = df.fillna(method='ffill').fillna(0)
    
    # 创建序列数据
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df[feature_columns].iloc[i-sequence_length:i].values)
        y.append(df[target_column].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"序列数据形状: X={X.shape}, y={y.shape}")
    
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # 重塑X进行标准化
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler_X.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def train_model(X_train, y_train, X_val, y_val, input_size, hidden_size=64, 
                num_layers=2, epochs=100, learning_rate=0.001):
    """训练模型"""
    print("🤖 开始训练SWE预测模型...")
    
    # 创建模型
    model = SWELSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1
    )
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练历史
    train_losses = []
    val_losses = []
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        
        # 前向传播
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs.squeeze(), torch.FloatTensor(y_train))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_outputs.squeeze(), torch.FloatTensor(y_val))
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    print("✅ 模型训练完成!")
    return model, train_losses, val_losses

def evaluate_model(model, X_test, y_test, scaler_y):
    """评估模型性能"""
    print("📈 评估模型性能...")
    
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_test))
        predictions = predictions.squeeze().numpy()
    
    # 反标准化预测结果
    predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # 计算指标
    mse = mean_squared_error(y_test_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)
    
    print(f"📊 模型性能指标:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return predictions_original, y_test_original, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def plot_training_history(train_losses, val_losses, save_path=None):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', alpha=0.7)
    plt.plot(val_losses, label='验证损失', alpha=0.7)
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    plt.title('SWE模型训练历史')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 训练历史图保存到: {save_path}")
    
    plt.show()

def plot_predictions(predictions, actual, save_path=None):
    """绘制预测结果"""
    plt.figure(figsize=(12, 6))
    
    # 选择前100个点进行可视化
    n_points = min(100, len(predictions))
    x = range(n_points)
    
    plt.plot(x, actual[:n_points], label='实际值', alpha=0.7, linewidth=2)
    plt.plot(x, predictions[:n_points], label='预测值', alpha=0.7, linewidth=2)
    plt.xlabel('时间步')
    plt.ylabel('流量 (m³/s)')
    plt.title('SWE模型预测结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 预测结果图保存到: {save_path}")
    
    plt.show()

def main():
    """主训练流程"""
    print("🚀 开始HydrAI-SWE模型训练")
    print("=" * 50)
    
    # 数据路径
    data_path = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行数据准备脚本")
        return
    
    # 训练参数
    sequence_length = 30
    hidden_size = 64
    num_layers = 2
    epochs = 100
    learning_rate = 0.001
    
    print(f"🔧 训练参数:")
    print(f"  序列长度: {sequence_length}")
    print(f"  隐藏层大小: {hidden_size}")
    print(f"  LSTM层数: {num_layers}")
    print(f"  训练轮数: {epochs}")
    print(f"  学习率: {learning_rate}")
    
    # 加载和准备数据
    X, y, scaler_X, scaler_y = load_and_prepare_data(data_path, sequence_length)
    
    # 划分训练/验证/测试集
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"📊 数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 训练模型
    input_size = X.shape[-1]
    model, train_losses, val_losses = train_model(
        X_train, y_train, X_val, y_val,
        input_size, hidden_size, num_layers, epochs, learning_rate
    )
    
    # 评估模型
    predictions, actual, metrics = evaluate_model(model, X_test, y_test, scaler_y)
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"models/swe_lstm_model_{timestamp}.pth"
    os.makedirs("models", exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'sequence_length': sequence_length,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'metrics': metrics
    }, model_save_path)
    
    print(f"💾 模型保存到: {model_save_path}")
    
    # 绘制结果
    plot_training_history(train_losses, val_losses, f"models/training_history_{timestamp}.png")
    plot_predictions(predictions, actual, f"models/predictions_{timestamp}.png")
    
    print("\n🎉 模型训练完成!")
    print(f"📊 最终性能: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

if __name__ == "__main__":
    main()
