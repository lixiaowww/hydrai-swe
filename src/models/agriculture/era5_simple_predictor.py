#!/usr/bin/env python3
"""
ERA5简化土壤湿度预测模型
专门针对小数据集设计，防止过拟合
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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSoilMoistureLSTM(nn.Module):
    """简化的土壤湿度LSTM模型 - 防过拟合设计"""
    
    def __init__(self, input_size: int, hidden_size: int = 16, num_layers: int = 1, dropout: float = 0.3):
        super(SimpleSoilMoistureLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入层标准化
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # 简化的LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0  # 单层LSTM不使用内置dropout
        )
        
        # 简化的全连接层
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 1)
        
        # 更强的Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # 权重初始化
        self._init_weights()
        
        logger.info(f"✅ 简化LSTM模型创建: input_size={input_size}, hidden_size={hidden_size}, dropout={dropout}")
    
    def _init_weights(self):
        """权重初始化"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # 输入标准化 (对每个时间步)
        x_reshaped = x.view(-1, features)
        x_norm = self.input_norm(x_reshaped)
        x = x_norm.view(batch_size, seq_len, features)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 添加强dropout
        last_output = self.dropout(last_output)
        
        # 简化的全连接层
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # 确保输出在0-1之间
        x = self.sigmoid(x)
        
        return x

class ERA5SimplePredictor:
    """ERA5简化预测器 - 防过拟合版本"""
    
    def __init__(self, model_dir: str = "models/era5_simple"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 简化的模型配置
        self.config = {
            'input_size': None,
            'hidden_size': 16,  # 大幅减少
            'num_layers': 1,    # 减少层数
            'dropout': 0.4,     # 增加dropout
            'learning_rate': 0.01,  # 增加学习率
            'batch_size': 8,    # 减少批次大小
            'epochs': 50,       # 减少训练轮数
            'sequence_length': 5,  # 减少序列长度
            'patience': 8,      # 减少耐心值
            'min_delta': 0.001, # 增加最小改进
            'weight_decay': 0.01,  # L2正则化
            'feature_selection': True,  # 启用特征选择
            'k_best_features': 10  # 只选择最重要的10个特征
        }
        
        # 特征选择器和标准化器
        self.feature_selector = None
        self.scaler = None
        
        # 模型组件
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rates': []
        }
        
        logger.info(f"✅ ERA5简化预测器初始化完成，设备: {self.device}")
    
    def select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """特征选择"""
        try:
            logger.info("🔍 执行特征选择...")
            
            if not self.config['feature_selection']:
                return X
            
            # 将序列数据重塑为2D
            n_samples, seq_len, n_features = X.shape
            X_2d = X.reshape(-1, n_features)
            y_expanded = np.repeat(y, seq_len)
            
            # 特征选择
            if self.feature_selector is None:
                self.feature_selector = SelectKBest(
                    score_func=f_regression,
                    k=self.config['k_best_features']
                )
                X_selected_2d = self.feature_selector.fit_transform(X_2d, y_expanded)
            else:
                X_selected_2d = self.feature_selector.transform(X_2d)
            
            # 重塑回序列形状
            X_selected = X_selected_2d.reshape(n_samples, seq_len, -1)
            
            # 更新输入大小
            self.config['input_size'] = X_selected.shape[2]
            
            # 获取选中的特征名称
            selected_features = self.feature_selector.get_support(indices=True)
            logger.info(f"✅ 特征选择完成: {len(selected_features)} 个特征被选中")
            logger.info(f"📊 选中的特征索引: {selected_features[:10]}...")  # 只显示前10个
            
            return X_selected
            
        except Exception as e:
            logger.error(f"❌ 特征选择失败: {e}")
            return X
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """特征标准化"""
        try:
            logger.info("📏 执行特征标准化...")
            
            # 将序列数据重塑为2D
            n_samples, seq_len, n_features = X.shape
            X_2d = X.reshape(-1, n_features)
            
            # 标准化
            if fit:
                if self.scaler is None:
                    self.scaler = StandardScaler()
                X_scaled_2d = self.scaler.fit_transform(X_2d)
            else:
                X_scaled_2d = self.scaler.transform(X_2d)
            
            # 重塑回序列形状
            X_scaled = X_scaled_2d.reshape(n_samples, seq_len, n_features)
            
            logger.info("✅ 特征标准化完成")
            return X_scaled
            
        except Exception as e:
            logger.error(f"❌ 特征标准化失败: {e}")
            return X
    
    def build_model(self, input_size: int) -> None:
        """构建简化模型"""
        try:
            logger.info(f"🔧 构建简化模型，输入特征数: {input_size}")
            
            self.config['input_size'] = input_size
            
            # 创建简化模型
            self.model = SimpleSoilMoistureLSTM(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(self.device)
            
            # 创建优化器 (添加L2正则化)
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            
            # 创建损失函数
            self.criterion = nn.MSELoss()
            
            # 学习率调度器 (更激进的衰减)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.3,  # 更大的衰减因子
                patience=5   # 更小的耐心值
            )
            
            logger.info("✅ 简化模型构建完成")
            
        except Exception as e:
            logger.error(f"❌ 模型构建失败: {e}")
            raise
    
    def load_and_preprocess_data(self, data_dir: str = "data/processed/era5") -> Dict:
        """加载并预处理数据"""
        try:
            logger.info(f"📥 加载并预处理数据: {data_dir}")
            
            # 检查数据文件
            required_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']
            for file in required_files:
                file_path = os.path.join(data_dir, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"数据文件不存在: {file_path}")
            
            # 加载原始数据
            data = {}
            for split in ['train', 'val', 'test']:
                X_file = os.path.join(data_dir, f'X_{split}.npy')
                y_file = os.path.join(data_dir, f'y_{split}.npy')
                
                data[f'X_{split}_raw'] = np.load(X_file)
                data[f'y_{split}'] = np.load(y_file)
            
            # 特征选择 (只在训练集上fit)
            X_train_selected = self.select_features(data['X_train_raw'], data['y_train'])
            X_val_selected = self.select_features(data['X_val_raw'], data['y_val']) if self.feature_selector else data['X_val_raw']
            X_test_selected = self.select_features(data['X_test_raw'], data['y_test']) if self.feature_selector else data['X_test_raw']
            
            # 特征标准化
            data['X_train'] = self.scale_features(X_train_selected, fit=True)
            data['X_val'] = self.scale_features(X_val_selected, fit=False)
            data['X_test'] = self.scale_features(X_test_selected, fit=False)
            
            # 记录数据信息
            for split in ['train', 'val', 'test']:
                logger.info(f"  📊 {split}: X={data[f'X_{split}'].shape}, y={data[f'y_{split}'].shape}")
            
            logger.info("✅ 数据预处理完成")
            return data
            
        except Exception as e:
            logger.error(f"❌ 数据预处理失败: {e}")
            raise
    
    def create_data_loaders(self, data: Dict) -> Dict:
        """创建数据加载器"""
        try:
            logger.info("🔧 创建数据加载器...")
            
            loaders = {}
            batch_size = self.config['batch_size']
            
            for split in ['train', 'val', 'test']:
                X = torch.FloatTensor(data[f'X_{split}']).to(self.device)
                y = torch.FloatTensor(data[f'y_{split}']).to(self.device)
                
                dataset = TensorDataset(X, y)
                loader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=(split == 'train'),
                    drop_last=False  # 保留最后一个不完整的批次
                )
                
                loaders[split] = loader
                logger.info(f"  📊 {split}: {len(loader)} 批次")
            
            logger.info("✅ 数据加载器创建完成")
            return loaders
            
        except Exception as e:
            logger.error(f"❌ 创建数据加载器失败: {e}")
            raise
    
    def train_with_regularization(self, data_loaders: Dict) -> Dict:
        """带正则化的训练"""
        try:
            logger.info("🚀 开始正则化训练...")
            
            if self.model is None:
                raise ValueError("模型未构建")
            
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
                    
                    # 梯度裁剪 (防止梯度爆炸)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
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
                avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                avg_train_mae = train_mae / train_batches if train_batches > 0 else 0
                avg_val_mae = val_mae / val_batches if val_batches > 0 else 0
                
                # 更新学习率
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(avg_val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                
                # 记录历史
                self.training_history['train_loss'].append(avg_train_loss)
                self.training_history['val_loss'].append(avg_val_loss)
                self.training_history['train_mae'].append(avg_train_mae)
                self.training_history['val_mae'].append(avg_val_mae)
                self.training_history['learning_rates'].append(new_lr)
                
                # 打印进度
                if (epoch + 1) % 5 == 0:
                    lr_change = " (LR降低)" if new_lr < old_lr else ""
                    logger.info(f"Epoch {epoch+1}/{epochs}: "
                              f"Train Loss: {avg_train_loss:.6f}, "
                              f"Val Loss: {avg_val_loss:.6f}, "
                              f"Train MAE: {avg_train_mae:.6f}, "
                              f"Val MAE: {avg_val_mae:.6f}, "
                              f"LR: {new_lr:.6f}{lr_change}")
                
                # 早停检查
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    self.save_model('best_simple_model.pth')
                else:
                    patience_counter += 1
                
                # 早停
                if patience_counter >= patience:
                    logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
            
            logger.info("✅ 正则化训练完成")
            
            return {
                'status': 'success',
                'epochs_trained': epoch + 1,
                'best_val_loss': best_val_loss,
                'final_train_loss': avg_train_loss,
                'final_val_loss': avg_val_loss,
                'final_lr': new_lr
            }
            
        except Exception as e:
            logger.error(f"❌ 训练失败: {e}")
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
                        
                        # 处理单个样本的情况
                        if predictions.ndim == 0:
                            predictions = np.array([predictions])
                        if targets.ndim == 0:
                            targets = np.array([targets])
                        
                        all_predictions.extend(predictions)
                        all_targets.extend(targets)
                    
                    # 计算指标
                    if len(all_predictions) > 0 and len(all_targets) > 0:
                        mse = mean_squared_error(all_targets, all_predictions)
                        mae = mean_absolute_error(all_targets, all_predictions)
                        r2 = r2_score(all_targets, all_predictions)
                        
                        results[split] = {
                            'mse': mse,
                            'mae': mae,
                            'r2': r2,
                            'rmse': np.sqrt(mse),
                            'predictions': all_predictions,
                            'targets': all_targets,
                            'n_samples': len(all_predictions)
                        }
                        
                        logger.info(f"  📊 {split}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, n={len(all_predictions)}")
                    else:
                        logger.warning(f"  ⚠️ {split}: 无有效预测结果")
            
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
                'training_history': self.training_history,
                'feature_selector': self.feature_selector,
                'scaler': self.scaler
            }, model_path)
            
            logger.info(f"✅ 模型保存完成: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ 模型保存失败: {e}")
            raise
    
    def plot_training_analysis(self, save_path: str = None) -> None:
        """绘制训练分析图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 损失曲线
            ax1.plot(self.training_history['train_loss'], label='Train Loss', linewidth=2)
            ax1.plot(self.training_history['val_loss'], label='Validation Loss', linewidth=2)
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # MAE曲线
            ax2.plot(self.training_history['train_mae'], label='Train MAE', linewidth=2)
            ax2.plot(self.training_history['val_mae'], label='Validation MAE', linewidth=2)
            ax2.set_title('Training and Validation MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 学习率变化
            ax3.plot(self.training_history['learning_rates'], label='Learning Rate', linewidth=2, color='orange')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 训练/验证损失比率
            if len(self.training_history['train_loss']) > 0 and len(self.training_history['val_loss']) > 0:
                ratios = [v/t if t > 0 else 1 for t, v in zip(self.training_history['train_loss'], self.training_history['val_loss'])]
                ax4.plot(ratios, label='Val Loss / Train Loss', linewidth=2, color='red')
                ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Perfect Ratio')
                ax4.set_title('Overfitting Indicator')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Loss Ratio')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ 训练分析图保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ 绘制训练分析失败: {e}")

def main():
    """主函数"""
    print("🛡️ ERA5简化土壤湿度预测模型 - 防过拟合版本")
    print("=" * 60)
    
    try:
        # 创建简化预测器
        predictor = ERA5SimplePredictor()
        
        # 加载并预处理数据
        print("\n📥 加载并预处理数据...")
        data = predictor.load_and_preprocess_data()
        
        # 构建模型
        print("\n🔧 构建简化模型...")
        input_size = data['X_train'].shape[2]
        predictor.build_model(input_size)
        
        # 创建数据加载器
        print("\n🔧 创建数据加载器...")
        data_loaders = predictor.create_data_loaders(data)
        
        # 正则化训练
        print("\n🚀 开始正则化训练...")
        training_result = predictor.train_with_regularization(data_loaders)
        
        if training_result['status'] == 'success':
            print(f"✅ 训练完成!")
            print(f"📊 训练轮数: {training_result['epochs_trained']}")
            print(f"📊 最佳验证损失: {training_result['best_val_loss']:.6f}")
            print(f"📊 最终学习率: {training_result['final_lr']:.6f}")
            
            # 评估模型
            print("\n📊 评估模型...")
            evaluation_results = predictor.evaluate_model(data_loaders)
            
            # 保存模型
            print("\n💾 保存模型...")
            predictor.save_model('era5_simple_final.pth')
            
            # 绘制训练分析
            print("\n📈 绘制训练分析...")
            plot_path = os.path.join(predictor.model_dir, 'training_analysis.png')
            predictor.plot_training_analysis(plot_path)
            
        else:
            print(f"❌ 训练失败: {training_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()
