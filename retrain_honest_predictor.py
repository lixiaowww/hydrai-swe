#!/usr/bin/env python3
"""
重新训练诚实预测器
使用扩展后的数据集训练诚实预测器，验证数据扩展的效果
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class HonestPredictorTrainer:
    """诚实预测器训练器"""
    
    def __init__(self):
        self.data_dir = "data/processed/ready_for_training"
        self.models_dir = "models"
        self.results_dir = "training_results"
        
        # 创建目录
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 训练参数
        self.sequence_length = 30
        self.batch_size = 64
        self.epochs = 100
        self.learning_rate = 0.001
        self.patience = 10
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
    
    def load_processed_data(self) -> tuple:
        """加载处理后的数据"""
        print("📥 加载处理后的数据")
        
        try:
            # 加载训练数据
            train_path = os.path.join(self.data_dir, 'train_data_scaled.csv')
            test_path = os.path.join(self.data_dir, 'test_data_scaled.csv')
            
            train_data = pd.read_csv(train_path, index_col=0)
            test_data = pd.read_csv(test_path, index_col=0)
            
            print(f"✅ 训练数据: {train_data.shape}")
            print(f"✅ 测试数据: {test_data.shape}")
            
            return train_data, test_data
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None, None
    
    def create_sequences(self, data: pd.DataFrame, target_col: str = 'snow_water_equivalent_mm') -> tuple:
        """创建时间序列数据"""
        print(f"🔄 创建时间序列数据 (序列长度: {self.sequence_length})")
        
        sequences = []
        targets = []
        
        # 选择特征列（排除目标列）
        feature_cols = [col for col in data.columns if col != target_col]
        
        for i in range(self.sequence_length, len(data)):
            # 创建序列
            seq = data[feature_cols].iloc[i-self.sequence_length:i].values
            target = data[target_col].iloc[i]
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        print(f"   序列形状: {sequences.shape}")
        print(f"   目标形状: {targets.shape}")
        print(f"   特征数: {sequences.shape[2]}")
        
        return sequences, targets
    
    def create_data_loaders(self, train_sequences: np.ndarray, train_targets: np.ndarray,
                           test_sequences: np.ndarray, test_targets: np.ndarray) -> tuple:
        """创建数据加载器"""
        print("🔄 创建数据加载器")
        
        # 转换为PyTorch张量
        train_sequences_tensor = torch.FloatTensor(train_sequences).to(self.device)
        train_targets_tensor = torch.FloatTensor(train_targets).to(self.device)
        test_sequences_tensor = torch.FloatTensor(test_sequences).to(self.device)
        test_targets_tensor = torch.FloatTensor(test_targets).to(self.device)
        
        # 创建数据集
        train_dataset = TensorDataset(train_sequences_tensor, train_targets_tensor)
        test_dataset = TensorDataset(test_sequences_tensor, test_targets_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"   训练批次: {len(train_loader)}")
        print(f"   测试批次: {len(test_loader)}")
        
        return train_loader, test_loader
    
    def create_model(self, input_size: int) -> nn.Module:
        """创建模型"""
        print(f"🏗️ 创建模型 (输入特征: {input_size})")
        
        model = nn.Sequential(
            # LSTM层
            nn.LSTM(input_size, 128, num_layers=2, batch_first=True, dropout=0.2),
            # 全连接层
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        # 自定义前向传播
        class CustomLSTMModel(nn.Module):
            def __init__(self, lstm, fc_layers):
                super().__init__()
                self.lstm = lstm
                self.fc_layers = fc_layers
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # 取最后一个时间步的输出
                last_output = lstm_out[:, -1, :]
                # 通过全连接层
                for layer in self.fc_layers:
                    last_output = layer(last_output)
                return last_output
        
        lstm_layer = model[0]
        fc_layers = model[1:]
        
        custom_model = CustomLSTMModel(lstm_layer, fc_layers)
        custom_model.to(self.device)
        
        return custom_model
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   test_loader: DataLoader) -> dict:
        """训练模型"""
        print("🚀 开始训练模型")
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # 训练历史
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        patience_counter = 0
        
        print(f"   训练轮数: {self.epochs}")
        print(f"   学习率: {self.learning_rate}")
        print(f"   批次大小: {self.batch_size}")
        
        for epoch in range(self.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for sequences, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 测试阶段
            model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for sequences, targets in test_loader:
                    outputs = model(sequences)
                    loss = criterion(outputs.squeeze(), targets)
                    test_loss += loss.item()
            
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            
            # 学习率调度
            scheduler.step(test_loss)
            
            # 早停检查
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_path = os.path.join(self.models_dir, 'best_honest_predictor.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'best_test_loss': best_test_loss
                }, best_model_path)
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1:3d}/{self.epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Test Loss: {test_loss:.6f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停
            if patience_counter >= self.patience:
                print(f"   🛑 早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 训练完成
        print(f"✅ 训练完成！最佳测试损失: {best_test_loss:.6f}")
        
        # 保存训练历史
        training_history = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_epoch': epoch - patience_counter + 1,
            'best_test_loss': best_test_loss,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
        
        return training_history
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> dict:
        """评估模型"""
        print("📊 评估模型性能")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                outputs = model(sequences)
                predictions = outputs.squeeze().cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets_np)
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # 计算评估指标
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        # 计算相对误差
        mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': all_predictions.tolist(),
            'targets': all_targets.tolist()
        }
        
        print(f"   评估指标:")
        print(f"     MSE: {mse:.6f}")
        print(f"     RMSE: {rmse:.6f}")
        print(f"     MAE: {mae:.6f}")
        print(f"     R²: {r2:.6f}")
        print(f"     MAPE: {mape:.2f}%")
        
        return metrics
    
    def plot_training_results(self, training_history: dict, evaluation_metrics: dict):
        """绘制训练结果"""
        print("📈 绘制训练结果")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 训练损失曲线
        axes[0, 0].plot(training_history['train_losses'], label='训练损失', color='blue')
        axes[0, 0].plot(training_history['test_losses'], label='测试损失', color='red')
        axes[0, 0].set_title('训练和测试损失')
        axes[0, 0].set_xlabel('轮数')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 预测vs实际值散点图
        predictions = np.array(evaluation_metrics['predictions'])
        targets = np.array(evaluation_metrics['targets'])
        
        axes[0, 1].scatter(targets, predictions, alpha=0.6, color='green')
        axes[0, 1].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        axes[0, 1].set_title('预测值 vs 实际值')
        axes[0, 1].set_xlabel('实际值')
        axes[0, 1].set_ylabel('预测值')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差图
        residuals = predictions - targets
        axes[1, 0].scatter(predictions, residuals, alpha=0.6, color='orange')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('残差图')
        axes[1, 0].set_xlabel('预测值')
        axes[1, 0].set_ylabel('残差')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 误差分布直方图
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('误差分布')
        axes[1, 1].set_xlabel('误差')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.results_dir, 'training_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   训练结果图表已保存: {plot_path}")
        
        plt.show()
    
    def save_results(self, training_history: dict, evaluation_metrics: dict):
        """保存训练结果"""
        print("💾 保存训练结果")
        
        # 保存训练历史
        history_path = os.path.join(self.results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2, default=str)
        
        # 保存评估指标
        metrics_path = os.path.join(self.results_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(evaluation_metrics, f, indent=2, default=str)
        
        # 生成综合报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'sequence_length': self.sequence_length,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'device': str(self.device)
            },
            'training_summary': {
                'best_epoch': training_history['best_epoch'],
                'best_test_loss': training_history['best_test_loss'],
                'final_train_loss': training_history['final_train_loss'],
                'final_test_loss': training_history['final_test_loss']
            },
            'evaluation_summary': {
                'mse': evaluation_metrics['mse'],
                'rmse': evaluation_metrics['rmse'],
                'mae': evaluation_metrics['mae'],
                'r2': evaluation_metrics['r2'],
                'mape': evaluation_metrics['mape']
            }
        }
        
        report_path = os.path.join(self.results_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   训练历史: {history_path}")
        print(f"   评估指标: {metrics_path}")
        print(f"   综合报告: {report_path}")

def main():
    """主函数"""
    print("🚀 开始重新训练诚实预测器")
    print("=" * 50)
    
    trainer = HonestPredictorTrainer()
    
    # 1. 加载数据
    train_data, test_data = trainer.load_processed_data()
    if train_data is None or test_data is None:
        print("❌ 无法加载数据，退出")
        return False
    
    # 2. 创建序列数据
    train_sequences, train_targets = trainer.create_sequences(train_data)
    test_sequences, test_targets = trainer.create_sequences(test_data)
    
    # 3. 创建数据加载器
    train_loader, test_loader = trainer.create_data_loaders(
        train_sequences, train_targets, test_sequences, test_targets
    )
    
    # 4. 创建模型
    input_size = train_sequences.shape[2]
    model = trainer.create_model(input_size)
    
    # 5. 训练模型
    training_history = trainer.train_model(model, train_loader, test_loader)
    
    # 6. 评估模型
    evaluation_metrics = trainer.evaluate_model(model, test_loader)
    
    # 7. 绘制结果
    trainer.plot_training_results(training_history, evaluation_metrics)
    
    # 8. 保存结果
    trainer.save_results(training_history, evaluation_metrics)
    
    print(f"\n🎉 诚实预测器重新训练完成！")
    print(f"   最佳R²分数: {evaluation_metrics['r2']:.4f}")
    print(f"   最佳测试损失: {training_history['best_test_loss']:.6f}")
    print(f"   训练轮数: {training_history['best_epoch']}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
