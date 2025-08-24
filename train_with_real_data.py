#!/usr/bin/env python3
"""
使用真实数据重新训练SWE LSTM模型
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime

class SWELSTMModel(nn.Module):
    """SWE LSTM预测模型"""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1, sequence_length=30):
        super(SWELSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

class RealDataTrainer:
    """真实数据训练器"""
    
    def __init__(self, data_path="data/processed/comprehensive_training_dataset.csv"):
        self.data_path = data_path
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.sequence_length = 30
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("📊 加载训练数据...")
            
            # 加载数据
        data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        print(f"✅ 加载数据: {len(data)} 条记录")
        print(f"   时间范围: {data.index.min()} 到 {data.index.max()}")
        
        # 准备特征和目标
        feature_cols = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                       'day_of_year', 'month', 'year']
        target_col = 'snow_water_equivalent_mm'
        
        X = data[feature_cols].values
        y = data[target_col].values.reshape(-1, 1)
        
        # 标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # 创建序列数据
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled.flatten())
        
        print(f"✅ 创建序列数据: {len(X_seq)} 个序列")
        
        return X_seq, y_seq
    
    def create_sequences(self, X, y):
        """创建序列数据"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
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
    
    def train_model(self, train_loader, val_loader, epochs=100, learning_rate=0.001):
        """训练模型"""
        print(f"🚀 开始训练模型 (epochs={epochs})...")
        
        # 创建模型
        self.model = SWELSTMModel(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            sequence_length=self.sequence_length
        )
            
            # 损失函数和优化器
            criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # 训练历史
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
        patience = 15
            
            for epoch in range(epochs):
                # 训练阶段
                self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
            train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # 验证阶段
                self.model.eval()
            val_loss = 0.0
                with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
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
                    # 保存最佳模型
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'scaler_X': self.scaler_X,
                    'scaler_y': self.scaler_y,
                    'sequence_length': self.sequence_length,
                    'input_size': 6,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, 'models/real_trained_swe_model.pth')
                else:
                    patience_counter += 1
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"✅ 训练完成! 最佳验证损失: {best_val_loss:.6f}")
        
        # 绘制训练历史
        self.plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def evaluate_model(self, test_loader):
        """评估模型"""
        print("📊 评估模型性能...")
            
            # 加载最佳模型
        checkpoint = torch.load('models/real_trained_swe_model.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
            self.model.eval()
        predictions = []
        actuals = []
            
            with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
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
        mae = np.mean(np.abs(actuals_rescaled - predictions_rescaled))
        r2 = r2_score(actuals_rescaled, predictions_rescaled)
        
        print(f"📈 测试集性能:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")
        
        # 绘制预测结果
        self.plot_predictions(actuals_rescaled, predictions_rescaled)
        
        return {
            'rmse': rmse,
                'mae': mae,
            'r2': r2,
            'mse': mse
        }
    
    def plot_training_history(self, train_losses, val_losses):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'models/real_training_history_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 训练历史图表已保存: {filename}")
    
    def plot_predictions(self, actuals, predictions):
        """绘制预测结果"""
        plt.figure(figsize=(12, 8))
        
        # 子图1: 散点图
        plt.subplot(2, 2, 1)
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.xlabel('Actual SWE (mm)')
        plt.ylabel('Predicted SWE (mm)')
        plt.title('Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # 子图2: 时间序列对比
        plt.subplot(2, 2, 2)
        sample_size = min(500, len(actuals))
        plt.plot(actuals[:sample_size], label='Actual', alpha=0.7)
        plt.plot(predictions[:sample_size], label='Predicted', alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('SWE (mm)')
        plt.title('Time Series Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图3: 残差图
        plt.subplot(2, 2, 3)
        residuals = actuals - predictions
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted SWE (mm)')
        plt.ylabel('Residuals (mm)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # 子图4: 残差分布
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residuals (mm)')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'models/real_predictions_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 预测结果图表已保存: {filename}")

def main():
    """主函数"""
    print("🚀 HydrAI-SWE 真实数据训练系统")
    print("=" * 60)
    
    # 创建模型目录
    os.makedirs("models", exist_ok=True)
    
    try:
        # 创建训练器
        trainer = RealDataTrainer()
        
        # 加载和准备数据
        X, y = trainer.load_and_prepare_data()
        
        # 分割数据
        train_data, val_data, test_data = trainer.split_data(X, y)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = trainer.create_data_loaders(
            train_data, val_data, test_data, batch_size=32
        )
        
        # 训练模型
        train_losses, val_losses = trainer.train_model(
            train_loader, val_loader, epochs=100, learning_rate=0.001
        )
        
        # 评估模型
        metrics = trainer.evaluate_model(test_loader)
        
        print("\n" + "=" * 60)
        print("🎉 真实数据训练完成!")
        print(f"📊 最终性能: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()