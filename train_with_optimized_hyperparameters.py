#!/usr/bin/env python3
"""
使用最佳超参数训练完整模型
基于快速优化的最佳参数配置
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

class OptimizedGRUModel(nn.Module):
    """使用最佳超参数的GRU模型"""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1):
        super(OptimizedGRUModel, self).__init__()
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

class OptimizedModelTrainer:
    """使用最佳超参数的模型训练器"""
    
    def __init__(self):
        self.scaler_X = None
        self.scaler_y = None
        self.sequence_length = 30
        self.best_params = {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 16
        }
        
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
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """数据分割：训练70%，验证15%，测试15%"""
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
    
    def create_data_loaders(self, train_data, val_data, test_data):
        """创建数据加载器"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=self.best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.best_params['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.best_params['batch_size'], shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_full_model(self, train_loader, val_loader):
        """训练完整模型"""
        print("🚀 开始训练完整模型...")
        
        # 创建模型
        model = OptimizedGRUModel(
            input_size=6,
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            dropout=self.best_params['dropout']
        )
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.best_params['learning_rate'])
        
        # 训练参数
        epochs = 100
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        train_losses = []
        val_losses = []
        
        print(f"🎯 训练配置:")
        print(f"   - 隐藏大小: {self.best_params['hidden_size']}")
        print(f"   - 层数: {self.best_params['num_layers']}")
        print(f"   - Dropout: {self.best_params['dropout']}")
        print(f"   - 学习率: {self.best_params['learning_rate']}")
        print(f"   - 批大小: {self.best_params['batch_size']}")
        print(f"   - 最大轮数: {epochs}")
        print(f"   - 早停耐心: {patience}")
        
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
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"🛑 早停触发，在epoch {epoch+1}停止训练")
                break
        
        training_time = time.time() - start_time
        
        # 恢复最佳模型
        model.load_state_dict(best_model_state)
        
        print(f"✅ 训练完成!")
        print(f"   - 最佳验证损失: {best_val_loss:.6f}")
        print(f"   - 训练轮数: {epoch+1}")
        print(f"   - 总耗时: {training_time:.2f} 秒")
        
        return model, train_losses, val_losses, training_time
    
    def evaluate_model(self, model, test_loader):
        """评估模型"""
        print("🔍 评估模型性能...")
        
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        # 反标准化预测值和实际值
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        
        predictions_original = self.scaler_y.inverse_transform(predictions).flatten()
        actuals_original = self.scaler_y.inverse_transform(actuals).flatten()
        
        # 计算指标
        mse = mean_squared_error(actuals_original, predictions_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_original, predictions_original)
        r2 = r2_score(actuals_original, predictions_original)
        
        print(f"✅ 测试集性能:")
        print(f"   - RMSE: {rmse:.4f}")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - R²: {r2:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions_original,
            'actuals': actuals_original
        }
    
    def save_model_and_results(self, model, train_losses, val_losses, test_results, training_time):
        """保存模型和结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存模型
        model_path = f"models/optimized_gru_model_{timestamp}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_params': self.best_params,
            'sequence_length': self.sequence_length,
            'scaler_X_mean': self.scaler_X.mean_,
            'scaler_X_scale': self.scaler_X.scale_,
            'scaler_y_mean': self.scaler_y.mean_,
            'scaler_y_scale': self.scaler_y.scale_,
            'test_results': test_results,
            'training_time': training_time
        }, model_path)
        
        print(f"✅ 模型已保存: {model_path}")
        
        # 保存训练历史
        history_path = f"logs/optimized_training_history_{timestamp}.json"
        import json
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': self.best_params,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_results': {
                    'rmse': test_results['rmse'],
                    'mae': test_results['mae'],
                    'r2': test_results['r2']
                },
                'training_time': training_time,
                'training_date': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 训练历史已保存: {history_path}")
        
        # 生成训练报告
        self.generate_training_report(test_results, training_time, timestamp)
        
        return model_path
    
    def generate_training_report(self, test_results, training_time, timestamp):
        """生成训练报告"""
        print("📝 生成训练报告...")
        
        report_path = f"logs/optimized_training_report_{timestamp}.md"
        
        report_content = f"""# 最佳超参数模型训练报告

## 训练时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 最佳超参数配置
基于快速优化的最佳参数:
- **隐藏大小**: {self.best_params['hidden_size']}
- **层数**: {self.best_params['num_layers']}
- **Dropout**: {self.best_params['dropout']}
- **学习率**: {self.best_params['learning_rate']}
- **批大小**: {self.best_params['batch_size']}

## 训练结果
- **训练轮数**: 根据早停机制自动确定
- **训练耗时**: {training_time:.2f} 秒
- **早停耐心**: 15个epoch

## 测试集性能
🏆 **最终模型性能**:
- **RMSE**: {test_results['rmse']:.4f}
- **MAE**: {test_results['mae']:.4f}
- **R²**: {test_results['r2']:.4f}

## 关键改进
1. **超参数优化**: 从快速优化中选择了最佳配置
2. **完整训练**: 使用最佳参数进行完整模型训练
3. **早停机制**: 防止过拟合，自动选择最佳训练轮数
4. **性能验证**: 在独立测试集上验证最终性能

## 下一步行动
1. **精细调优**: 在最佳参数附近进行更精细的搜索
2. **模型集成**: 考虑集成前3个最佳配置
3. **数据增强**: 结合最佳超参数尝试数据增强
4. **部署准备**: 准备模型部署和监控

## 模型文件
- **模型文件**: `models/optimized_gru_model_{timestamp}.pth`
- **训练历史**: `logs/optimized_training_history_{timestamp}.json`
- **本报告**: `logs/optimized_training_report_{timestamp}.md`
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 训练报告已保存: {report_path}")
    
    def plot_training_history(self, train_losses, val_losses):
        """绘制训练历史"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='训练损失', color='blue')
        plt.plot(val_losses, label='验证损失', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='训练损失', color='blue', alpha=0.7)
        plt.plot(val_losses, label='验证损失', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('损失 (对数尺度)')
        plt.title('训练和验证损失 (对数尺度)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f"logs/optimized_training_plots_{timestamp}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练历史图表已保存: {plot_path}")
    
    def run_full_training(self):
        """运行完整训练流程"""
        print("🎯 开始完整模型训练流程...")
        
        # 1. 加载数据
        data = self.load_data_and_scalers()
        if data is None:
            return
        
        # 2. 准备序列数据
        X, y = self.prepare_sequences(data)
        
        # 3. 分割数据
        train_data, val_data, test_data = self.split_data(X, y)
        print(f"📊 数据分割:")
        print(f"   - 训练集: {len(train_data[0])} 样本")
        print(f"   - 验证集: {len(val_data[0])} 样本")
        print(f"   - 测试集: {len(test_data[0])} 样本")
        
        # 4. 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders(train_data, val_data, test_data)
        
        # 5. 训练模型
        model, train_losses, val_losses, training_time = self.train_full_model(train_loader, val_loader)
        
        # 6. 评估模型
        test_results = self.evaluate_model(model, test_loader)
        
        # 7. 保存结果
        model_path = self.save_model_and_results(model, train_losses, val_losses, test_results, training_time)
        
        # 8. 绘制训练历史
        self.plot_training_history(train_losses, val_losses)
        
        print("\n" + "=" * 60)
        print("🎉 完整模型训练完成!")
        print(f"✅ 最佳超参数模型已保存: {model_path}")
        print(f"✅ 测试集R²: {test_results['r2']:.4f}")
        print(f"✅ 总训练时间: {training_time:.2f} 秒")
        print("✅ 所有结果已保存")
        
        return model_path, test_results

def main():
    """主函数"""
    print("🚀 HydrAI-SWE 最佳超参数完整模型训练")
    print("=" * 60)
    
    try:
        # 创建训练器
        trainer = OptimizedModelTrainer()
        
        # 运行完整训练
        model_path, test_results = trainer.run_full_training()
        
        if model_path and test_results:
            print("\n💡 下一步建议:")
            print("  1. 在最佳参数附近进行精细搜索")
            print("  2. 考虑集成前3个最佳配置")
            print("  3. 尝试数据增强技术")
            print("  4. 准备模型部署")
        else:
            print("❌ 完整训练失败")
        
    except Exception as e:
        print(f"❌ 完整模型训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
