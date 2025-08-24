#!/usr/bin/env python3
"""
ERA5土壤湿度预测模型
基于ERA5替代数据的LSTM模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ERA5SoilMoistureLSTM(nn.Module):
    """ERA5土壤湿度LSTM模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(ERA5SoilMoistureLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        logger.info(f"✅ ERA5土壤湿度LSTM模型创建完成: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
    
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # 使用sigmoid确保输出在0-1之间 (土壤湿度范围)
        x = self.sigmoid(x)
        
        return x

class ERA5SoilMoisturePredictor:
    """ERA5土壤湿度预测器"""
    
    def __init__(self, model_dir: str = "models/era5_soil_moisture"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型配置
        self.config = {
            'input_size': None,  # 动态设置
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 100,
            'sequence_length': 30,
            'patience': 15,
            'min_delta': 0.0001
        }
        
        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        logger.info(f"✅ ERA5土壤湿度预测器初始化完成，设备: {self.device}")
    
    def build_model(self, input_size: int) -> None:
        """构建模型"""
        try:
            logger.info(f"🔧 构建模型，输入特征数: {input_size}")
            
            self.config['input_size'] = input_size
            
            # 创建模型
            self.model = ERA5SoilMoistureLSTM(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(self.device)
            
            # 创建优化器
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config['learning_rate']
            )
            
            # 创建损失函数
            self.criterion = nn.MSELoss()
            
            # 学习率调度器
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=10
            )
            
            logger.info("✅ 模型构建完成")
            
        except Exception as e:
            logger.error(f"❌ 模型构建失败: {e}")
            raise
    
    def load_data(self, data_dir: str = "data/processed/era5") -> Dict:
        """加载处理后的数据"""
        try:
            logger.info(f"📥 加载数据: {data_dir}")
            
            # 检查数据文件
            required_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']
            for file in required_files:
                file_path = os.path.join(data_dir, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"数据文件不存在: {file_path}")
            
            # 加载数据
            data = {}
            for split in ['train', 'val', 'test']:
                X_file = os.path.join(data_dir, f'X_{split}.npy')
                y_file = os.path.join(data_dir, f'y_{split}.npy')
                
                data[f'X_{split}'] = np.load(X_file)
                data[f'y_{split}'] = np.load(y_file)
                
                logger.info(f"  📊 {split}: X={data[f'X_{split}'].shape}, y={data[f'y_{split}'].shape}")
            
            # 加载配置
            config_file = os.path.join(data_dir, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    data_config = json.load(f)
                    self.config['sequence_length'] = data_config.get('sequence_length', 30)
            
            logger.info("✅ 数据加载完成")
            return data
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise
    
    def create_data_loaders(self, data: Dict, batch_size: int = None) -> Dict:
        """创建数据加载器"""
        try:
            logger.info("🔧 创建数据加载器...")
            
            if batch_size is None:
                batch_size = self.config['batch_size']
            
            loaders = {}
            
            for split in ['train', 'val', 'test']:
                X = torch.FloatTensor(data[f'X_{split}']).to(self.device)
                y = torch.FloatTensor(data[f'y_{split}']).to(self.device)
                
                dataset = TensorDataset(X, y)
                loader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=(split == 'train')
                )
                
                loaders[split] = loader
                logger.info(f"  📊 {split}: {len(loader)} 批次")
            
            logger.info("✅ 数据加载器创建完成")
            return loaders
            
        except Exception as e:
            logger.error(f"❌ 创建数据加载器失败: {e}")
            raise
    
    def train_model(self, data_loaders: Dict) -> Dict:
        """训练模型"""
        try:
            logger.info("🚀 开始训练模型...")
            
            if self.model is None:
                raise ValueError("模型未构建，请先调用build_model()")
            
            # 训练参数
            epochs = self.config['epochs']
            patience = self.config['patience']
            min_delta = self.config['min_delta']
            
            # 早停变量
            best_val_loss = float('inf')
            patience_counter = 0
            
            # 训练循环
            for epoch in range(epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0.0
                train_mae = 0.0
                train_batches = 0
                
                for batch_X, batch_y in data_loaders['train']:
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    
                    # 反向传播
                    loss.backward()
                    self.optimizer.step()
                    
                    # 统计
                    train_loss += loss.item()
                    train_mae += mean_absolute_error(
                        batch_y.cpu().numpy(), 
                        outputs.detach().cpu().numpy().squeeze()
                    )
                    train_batches += 1
                
                # 验证阶段
                self.model.eval()
                val_loss = 0.0
                val_mae = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in data_loaders['val']:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs.squeeze(), batch_y)
                        
                        val_loss += loss.item()
                        val_mae += mean_absolute_error(
                            batch_y.cpu().numpy(), 
                            outputs.cpu().numpy().squeeze()
                        )
                        val_batches += 1
                
                # 计算平均损失
                avg_train_loss = train_loss / train_batches
                avg_val_loss = val_loss / val_batches
                avg_train_mae = train_mae / train_batches
                avg_val_mae = val_mae / val_batches
                
                # 更新学习率
                self.scheduler.step(avg_val_loss)
                
                # 记录历史
                self.training_history['train_loss'].append(avg_train_loss)
                self.training_history['val_loss'].append(avg_val_loss)
                self.training_history['train_mae'].append(avg_train_mae)
                self.training_history['val_mae'].append(avg_val_mae)
                
                # 打印进度
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: "
                              f"Train Loss: {avg_train_loss:.6f}, "
                              f"Val Loss: {avg_val_loss:.6f}, "
                              f"Train MAE: {avg_train_mae:.6f}, "
                              f"Val MAE: {avg_val_mae:.6f}")
                
                # 早停检查
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    self.save_model('best_model.pth')
                else:
                    patience_counter += 1
                
                # 早停
                if patience_counter >= patience:
                    logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
            
            logger.info("✅ 模型训练完成")
            
            return {
                'status': 'success',
                'epochs_trained': epoch + 1,
                'best_val_loss': best_val_loss,
                'final_train_loss': avg_train_loss,
                'final_val_loss': avg_val_loss,
                'training_history': self.training_history
            }
            
        except Exception as e:
            logger.error(f"❌ 模型训练失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def evaluate_model(self, data_loaders: Dict) -> Dict:
        """评估模型"""
        try:
            logger.info("📊 评估模型...")
            
            if self.model is None:
                raise ValueError("模型未构建")
            
            self.model.eval()
            results = {}
            
            with torch.no_grad():
                for split in ['train', 'val', 'test']:
                    all_predictions = []
                    all_targets = []
                    
                    for batch_X, batch_y in data_loaders[split]:
                        outputs = self.model(batch_X)
                        predictions = outputs.cpu().numpy().squeeze()
                        targets = batch_y.cpu().numpy()
                        
                        all_predictions.extend(predictions)
                        all_targets.extend(targets)
                    
                    # 计算指标
                    mse = mean_squared_error(all_targets, all_predictions)
                    mae = mean_absolute_error(all_targets, all_predictions)
                    r2 = r2_score(all_targets, all_predictions)
                    
                    results[split] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'rmse': np.sqrt(mse),
                        'predictions': all_predictions,
                        'targets': all_targets
                    }
                    
                    logger.info(f"  📊 {split}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
            
            logger.info("✅ 模型评估完成")
            return results
            
        except Exception as e:
            logger.error(f"❌ 模型评估失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def save_model(self, filename: str) -> None:
        """保存模型"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, filename)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'training_history': self.training_history
            }, model_path)
            
            logger.info(f"✅ 模型保存完成: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ 模型保存失败: {e}")
            raise
    
    def load_model(self, filename: str) -> None:
        """加载模型"""
        try:
            model_path = os.path.join(self.model_dir, filename)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 构建模型
            if self.model is None:
                input_size = checkpoint['config'].get('input_size')
                if input_size is None:
                    # 如果没有input_size，使用默认值
                    input_size = 35
                self.build_model(input_size)
            
            # 加载状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config = checkpoint['config']
            self.training_history = checkpoint['training_history']
            
            logger.info(f"✅ 模型加载完成: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """进行预测"""
        try:
            if self.model is None:
                raise ValueError("模型未构建")
            
            self.model.eval()
            
            # 转换为tensor
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor)
                return predictions.cpu().numpy().squeeze()
                
        except Exception as e:
            logger.error(f"❌ 预测失败: {e}")
            raise
    
    def plot_training_history(self, save_path: str = None) -> None:
        """绘制训练历史"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 损失曲线
            ax1.plot(self.training_history['train_loss'], label='Train Loss')
            ax1.plot(self.training_history['val_loss'], label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # MAE曲线
            ax2.plot(self.training_history['train_mae'], label='Train MAE')
            ax2.plot(self.training_history['val_mae'], label='Validation MAE')
            ax2.set_title('Training and Validation MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ 训练历史图保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ 绘制训练历史失败: {e}")

def main():
    """主函数"""
    print("🚀 ERA5土壤湿度预测模型测试")
    print("=" * 60)
    
    try:
        # 创建预测器
        predictor = ERA5SoilMoisturePredictor()
        
        # 加载数据
        print("\n📥 加载数据...")
        data = predictor.load_data()
        
        # 构建模型
        print("\n🔧 构建模型...")
        input_size = data['X_train'].shape[2]  # 特征数量
        predictor.build_model(input_size)
        
        # 创建数据加载器
        print("\n🔧 创建数据加载器...")
        data_loaders = predictor.create_data_loaders(data)
        
        # 训练模型
        print("\n🚀 训练模型...")
        training_result = predictor.train_model(data_loaders)
        
        if training_result['status'] == 'success':
            print(f"✅ 训练完成!")
            print(f"📊 训练轮数: {training_result['epochs_trained']}")
            print(f"📊 最佳验证损失: {training_result['best_val_loss']:.6f}")
            
            # 评估模型
            print("\n📊 评估模型...")
            evaluation_results = predictor.evaluate_model(data_loaders)
            
            # 保存模型
            print("\n💾 保存模型...")
            predictor.save_model('era5_soil_moisture_model.pth')
            
            # 绘制训练历史
            print("\n📈 绘制训练历史...")
            plot_path = os.path.join(predictor.model_dir, 'training_history.png')
            predictor.plot_training_history(plot_path)
            
        else:
            print(f"❌ 训练失败: {training_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()
