#!/usr/bin/env python3
"""
HydrAI-SWE 农业模块 - 土壤水分预测器
基于GitHub项目 SoilWeatherPredictor 集成
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SoilMoistureLSTM(nn.Module):
    """土壤水分预测LSTM模型 - 基于SoilWeatherPredictor架构"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(SoilMoistureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        x = self.dropout(last_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        
        return output

class AgricultureDataProcessor:
    """农业数据处理器"""
    
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.config = {
            'input_size': None,
            'sequence_length': 30
        }
        
    def prepare_soil_moisture_data(self, data_path, sequence_length=30):
        """
        准备土壤水分预测数据
        
        Args:
            data_path (str): 数据文件路径
            sequence_length (int): 序列长度
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, scalers)
        """
        print("📊 准备土壤水分预测数据...")
        
        # 加载数据
        df = pd.read_csv(data_path, parse_dates=['date'])
        df.set_index('date', inplace=True)
        
        # 选择特征列
        feature_columns = [
            'snow_depth_mm', 'snow_water_equivalent_mm',
            'day_of_year', 'month', 'year'
        ]
        
        # 确保所有特征列存在
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"可用特征: {available_features}")
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(0)
        
        # 如果没有土壤水分列，使用真实数据或标记为不可用
        if 'soil_moisture' not in df.columns:
            print("⚠️ 警告：没有土壤水分数据列")
            print("❌ 系统禁止使用合成数据，请提供真实的土壤水分观测数据")
            print("💡 建议：联系数据提供方获取真实的土壤水分传感器数据")
            raise ValueError("Missing soil moisture data. Synthetic data generation is prohibited.")
        
        # 创建序列数据
        X, y = [], []
        for i in range(sequence_length, len(df)):
            X.append(df[available_features].iloc[i-sequence_length:i].values)
            y.append(df['soil_moisture'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"序列数据形状: X={X.shape}, y={y.shape}")
        
        # 数据标准化
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler_X.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 划分数据集
        train_size = int(0.7 * len(X_scaled))
        val_size = int(0.15 * len(X_scaled))
        
        X_train = X_scaled[:train_size]
        y_train = y_scaled[:train_size]
        X_val = X_scaled[train_size:train_size+val_size]
        y_val = y_scaled[train_size:train_size+val_size]
        X_test = X_scaled[train_size+val_size:]
        y_test = y_scaled[train_size+val_size:]
        
        print(f"数据集划分: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")
        
        # 动态设置input_size
        self.config['input_size'] = len(available_features)
        print(f"🔧 动态设置input_size: {self.config['input_size']}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, (self.scaler_X, self.scaler_y)
    
    def process_real_soil_data(self, weather_data, soil_measurements):
        """
        处理真实土壤水分观测数据
        
        Args:
            weather_data (pd.DataFrame): 天气数据
            soil_measurements (pd.Series): 真实土壤水分观测数据
            
        Returns:
            pd.Series: 处理后的土壤水分数据
        """
        print("🌱 处理真实土壤水分观测数据...")
        
        if soil_measurements is None or soil_measurements.empty:
            print("⚠️ 警告：没有提供真实土壤水分数据")
            print("⚠️ 注意：系统禁止使用合成数据，请提供真实的观测数据")
            return pd.Series(dtype=float)
        
        # 数据质量检查
        missing_rate = soil_measurements.isnull().sum() / len(soil_measurements) * 100
        if missing_rate > 50:
            print(f"⚠️ 警告：土壤水分数据缺失率过高 ({missing_rate:.1f}%)")
            print("⚠️ 建议：检查数据源或联系数据提供方")
        
        # 简单的数据清理（不生成新数据）
        cleaned_data = soil_measurements.copy()
        
        # 只对少量缺失值进行插值，大量缺失则标记为不可用
        if missing_rate <= 20:
            cleaned_data = cleaned_data.interpolate(method='linear', limit=3)
            print(f"✅ 已清理土壤水分数据，缺失率从 {missing_rate:.1f}% 降至 {cleaned_data.isnull().sum() / len(cleaned_data) * 100:.1f}%")
        else:
            print(f"⚠️ 数据缺失率过高，无法进行有效插值")
        
        return cleaned_data

class SoilMoisturePredictor:
    """土壤水分预测器主类"""
    
    def __init__(self, config=None):
        """
        初始化土壤水分预测器
        
        Args:
            config (dict): 配置参数
        """
        self.config = config or self._default_config()
        self.model = None
        self.data_processor = AgricultureDataProcessor()
        self.training_history = {}
        
    def _default_config(self):
        """默认配置"""
        return {
            'input_size': None,  # 动态设置，匹配实际特征数量
            'hidden_size': 64,    # 减少隐藏层大小，避免过拟合
            'num_layers': 1,      # 减少层数，简化模型
            'dropout': 0.1,       # 减少dropout，提高训练稳定性
            'learning_rate': 0.0005,  # 降低学习率，提高训练稳定性
            'batch_size': 64,     # 增加batch size，提高训练稳定性
            'epochs': 100,
            'sequence_length': 30,
            'patience': 15,       # 早停耐心值
            'min_delta': 0.0001   # 最小改善阈值
        }
    
    def build_model(self):
        """构建模型"""
        print("🏗️ 构建土壤水分预测模型...")
        
        # 确保input_size已设置
        if self.config['input_size'] is None:
            raise ValueError("input_size未设置，请先准备数据")
        
        self.model = SoilMoistureLSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        print(f"✅ 模型构建完成: {self.model}")
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        训练模型
        
        Args:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            
        Returns:
            dict: 训练历史
        """
        print("🚀 开始训练土壤水分预测模型...")
        
        if self.model is None:
            self.build_model()
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 训练历史
        train_losses = []
        val_losses = []
        learning_rates = []
        
        # 早停机制
        patience = self.config['patience']
        min_delta = self.config['min_delta']
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            self.model.train()
            
            # 前向传播
            outputs = self.model(torch.FloatTensor(X_train))
            loss = criterion(outputs.squeeze(), torch.FloatTensor(y_train))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(torch.FloatTensor(X_val))
                val_loss = criterion(val_outputs.squeeze(), torch.FloatTensor(y_val))
            
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            
            # 更新学习率
            scheduler.step(val_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # 早停检查
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"早停机制触发，验证损失在 {patience} 轮后没有改善。")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.config['epochs']}], "
                      f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        print("✅ 模型训练完成!")
        
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates
        }
        
        return self.training_history
    
    def predict(self, X, scaler_y):
        """
        进行预测
        
        Args:
            X (np.array): 输入数据
            scaler_y (StandardScaler): 目标变量标准化器
            
        Returns:
            np.array: 预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(X))
            predictions = predictions.squeeze().numpy()
        
        # 反标准化
        predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions_original
    
    def evaluate_model(self, X_test, y_test, scaler_y):
        """
        评估模型性能
        
        Args:
            X_test, y_test: 测试数据
            scaler_y: 目标变量标准化器
            
        Returns:
            dict: 评估指标
        """
        print("📈 评估模型性能...")
        
        predictions = self.predict(X_test, scaler_y)
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # 计算指标
        mse = mean_squared_error(y_test_original, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, predictions)
        r2 = r2_score(y_test_original, predictions)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"📊 模型性能指标:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return predictions, y_test_original, metrics
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有模型可保存")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, filepath)
        
        print(f"💾 模型保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath)
        
        self.config = checkpoint['config']
        self.training_history = checkpoint.get('training_history', {})
        
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"📥 模型从 {filepath} 加载完成")
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        if not self.training_history:
            print("❌ 没有训练历史数据")
            return
        
        plt.figure(figsize=(12, 5))
        
        # 训练损失
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_losses'], label='训练损失', alpha=0.7)
        plt.plot(self.training_history['val_losses'], label='验证损失', alpha=0.7)
        plt.xlabel('训练轮数')
        plt.ylabel('损失值')
        plt.title('土壤水分预测模型训练历史')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 损失对比
        plt.subplot(1, 2, 2)
        final_train_loss = self.training_history['train_losses'][-1]
        final_val_loss = self.training_history['val_losses'][-1]
        
        plt.bar(['训练损失', '验证损失'], [final_train_loss, final_val_loss], 
                color=['skyblue', 'lightcoral'])
        plt.ylabel('损失值')
        plt.title('最终损失对比')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练历史图保存到: {save_path}")
        
        plt.show()

def main():
    """主函数 - 示例用法"""
    print("🚀 HydrAI-SWE 农业模块 - 土壤水分预测器")
    print("=" * 60)
    
    # 创建预测器
    predictor = SoilMoisturePredictor()
    
    # 准备数据
    data_path = "../../neuralhydrology/data/red_river_basin/timeseries.csv"
    
    try:
        # 准备数据
        X_train, y_train, X_val, y_val, X_test, y_test, scalers = \
            predictor.data_processor.prepare_soil_moisture_data(data_path)
        
        # 训练模型
        training_history = predictor.train_model(X_train, y_train, X_val, y_val)
        
        # 评估模型
        predictions, actual, metrics = predictor.evaluate_model(X_test, y_test, scalers[1])
        
        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"soil_moisture_model_{timestamp}.pth"
        predictor.save_model(model_save_path)
        
        # 绘制训练历史
        predictor.plot_training_history(f"training_history_{timestamp}.png")
        
        print("\n✅ 土壤水分预测器训练完成!")
        print(f"📊 最终性能: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("请检查数据文件路径和格式")

if __name__ == "__main__":
    main()
