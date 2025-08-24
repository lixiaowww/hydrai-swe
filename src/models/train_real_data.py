#!/usr/bin/env python3
"""
使用真实数据训练LSTM模型
使用已经预处理好的真实积雪和径流数据
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class SnowRunoffDataset(Dataset):
    """积雪-径流数据集"""
    
    def __init__(self, data, target_col, sequence_length=30):
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.target_col = target_col
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        # 获取输入序列（不包括目标变量）
        x = self.data[idx:idx + self.sequence_length, :-1]  
        # 获取目标值
        y = self.data[idx + self.sequence_length, self.target_col]
        
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

def load_real_data():
    """加载真实数据"""
    print("Loading real training data...")
    
    # 加载训练数据
    train_path = "data/processed/ready_for_training/train_data.csv"
    test_path = "data/processed/ready_for_training/test_data.csv"
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found: {train_path}")
        return None, None
    
    train_data = pd.read_csv(train_path)
    
    # 如果测试数据存在就加载，否则从训练数据中分割
    if os.path.exists(test_path):
        test_data = pd.read_csv(test_path)
        print(f"✅ Loaded train data: {len(train_data)} records")
        print(f"✅ Loaded test data: {len(test_data)} records")
    else:
        # 从训练数据中分割出测试集
        split_idx = int(0.8 * len(train_data))
        test_data = train_data.iloc[split_idx:].copy()
        train_data = train_data.iloc[:split_idx].copy()
        print(f"✅ Split data - train: {len(train_data)}, test: {len(test_data)} records")
    
    # 检查数据范围
    print(f"\n📊 Data Summary:")
    print(f"  Training period: {train_data['date'].min()} to {train_data['date'].max()}")
    print(f"  Test period: {test_data['date'].min()} to {test_data['date'].max()}")
    
    # 显示特征列
    feature_columns = [col for col in train_data.columns if col not in ['date']]
    print(f"  Features ({len(feature_columns)}): {feature_columns[:5]}...")
    
    return train_data, test_data

def prepare_features_and_target(data):
    """准备特征和目标变量"""
    print("Preparing features and target...")
    
    # 排除非特征列
    exclude_cols = ['date']
    feature_columns = [col for col in data.columns if col not in exclude_cols]
    
    # 检查是否有径流相关的目标变量
    # 寻找可能的目标变量
    target_candidates = ['streamflow_m3s', 'streamflow', 'flow', 'discharge']
    target_col = None
    target_col_idx = None
    
    for candidate in target_candidates:
        if candidate in data.columns:
            target_col = candidate
            target_col_idx = feature_columns.index(candidate)
            break
    
    if target_col is None:
        # 如果没有明确的径流变量，使用第一个数值列作为目标
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        available_cols = [col for col in numeric_cols if col not in exclude_cols]
        if available_cols:
            target_col = available_cols[0]
            target_col_idx = feature_columns.index(target_col)
            print(f"⚠️ No streamflow column found, using '{target_col}' as target")
        else:
            print("❌ No suitable target column found")
            return None, None, None, None
    
    print(f"✅ Using '{target_col}' as target variable (index: {target_col_idx})")
    
    # 提取特征和目标
    features = data[feature_columns].values
    
    # 检查缺失值
    try:
        # 首先确保数据是数值类型
        features_df = pd.DataFrame(features, columns=feature_columns)
        features_numeric = features_df.select_dtypes(include=[np.number])
        
        if len(features_numeric.columns) < len(feature_columns):
            # 有非数值列，需要转换或移除
            print("⚠️ Found non-numeric columns, converting to numeric...")
            for col in feature_columns:
                if col not in features_numeric.columns:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        features = features_df.values
        
        # 检查缺失值
        if pd.isna(features).any():
            print("⚠️ Found NaN values, filling with column means...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            features = imputer.fit_transform(features)
    except Exception as e:
        print(f"⚠️ Error in data preprocessing: {e}")
        # 强制转换为数值类型
        features = pd.DataFrame(features).apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Target column: {target_col}")
    print(f"  Target range: {features[:, target_col_idx].min():.2f} - {features[:, target_col_idx].max():.2f}")
    
    return features, target_col, target_col_idx, feature_columns

def train_model(train_data, val_data, target_col_idx, model_params):
    """训练模型"""
    print("Starting model training...")
    
    # 创建数据加载器
    train_dataset = SnowRunoffDataset(train_data, target_col_idx, model_params['sequence_length'])
    val_dataset = SnowRunoffDataset(val_data, target_col_idx, model_params['sequence_length'])
    
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
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
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
        
        # 计算评估指标
        val_predictions = np.array(val_predictions).flatten()
        val_targets = np.array(val_targets).flatten()
        val_r2 = r2_score(val_targets, val_predictions)
        val_mae = mean_absolute_error(val_targets, val_predictions)
        
        print(f"Epoch {epoch+1}/{model_params['epochs']}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val R²: {val_r2:.4f}, "
              f"Val MAE: {val_mae:.4f}")
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def save_model_and_results(model, train_losses, val_losses, scaler, model_params, feature_columns, target_col):
    """保存模型和结果"""
    print("Saving model and results...")
    
    # 创建输出目录
    output_dir = Path("models/real_data_lstm")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = output_dir / "real_data_lstm.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': model_params,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'target_column': target_col
    }, model_path, weights_only=False)
    print(f"Model saved to: {model_path}")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_params': model_params,
        'feature_columns': feature_columns,
        'target_column': target_col,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': min(val_losses)
    }
    
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Training history saved to: {history_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Real Data Training: Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    
    plot_path = output_dir / "training_loss.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to: {plot_path}")

def main():
    """主函数"""
    print("🚀 开始使用真实数据训练LSTM模型")
    print("=" * 50)
    
    try:
        # 加载真实数据
        train_data, test_data = load_real_data()
        if train_data is None:
            return
        
        # 准备特征和目标变量
        train_features, target_col, target_col_idx, feature_columns = prepare_features_and_target(train_data)
        test_features, _, _, _ = prepare_features_and_target(test_data)
        
        if train_features is None:
            return
        
        # 标准化特征
        print("Standardizing features...")
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # 分割训练集和验证集
        val_split = 0.2
        val_size = int(val_split * len(train_features_scaled))
        val_data = train_features_scaled[-val_size:]
        train_data_final = train_features_scaled[:-val_size]
        
        print(f"Final data split - train: {len(train_data_final)}, val: {len(val_data)}, test: {len(test_features_scaled)}")
        
        # 模型参数 - 使用实际的特征数量
        actual_input_size = train_features_scaled.shape[1] - 1  # 减去目标变量
        model_params = {
            'input_size': actual_input_size,
            'hidden_size': 128,  # 增加隐藏单元数量
            'num_layers': 3,     # 增加层数
            'dropout': 0.3,
            'sequence_length': 30,
            'batch_size': 64,    # 增加批次大小
            'epochs': 100,       # 增加训练轮数
            'learning_rate': 0.001
        }
        
        print("\nModel parameters:")
        for key, value in model_params.items():
            print(f"  {key}: {value}")
        
        # 训练模型
        model, train_losses, val_losses = train_model(train_data_final, val_data, target_col_idx, model_params)
        
        # 保存结果
        save_model_and_results(model, train_losses, val_losses, scaler, model_params, feature_columns, target_col)
        
        print(f"\n🎉 训练完成！")
        print(f"最佳验证损失: {min(val_losses):.4f}")
        print("模型和结果已保存到 models/real_data_lstm/ 目录")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
