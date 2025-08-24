#!/usr/bin/env python3
"""
使用NOAA真实数据训练土壤湿度预测模型
替代合成数据，使用真实观测数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NOAASoilMoisturePredictor:
    """使用NOAA数据的土壤湿度预测器"""
    
    def __init__(self):
        """初始化"""
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"✅ NOAA土壤湿度预测器初始化完成，设备: {self.device}")
    
    def load_and_prepare_data(self) -> Dict:
        """加载和准备NOAA数据"""
        try:
            logger.info("📥 加载NOAA处理后的数据...")
            
            # 加载每日数据
            daily_file = "data/processed/noaa_daily/noaa_daily_processed_20250821_192255.csv"
            if os.path.exists(daily_file):
                daily_data = pd.read_csv(daily_file)
                logger.info(f"📊 每日数据: {daily_data.shape}")
            else:
                logger.warning("⚠️ 每日数据文件不存在")
                daily_data = pd.DataFrame()
            
            # 加载小时数据
            hourly_file = "data/processed/noaa_hourly/noaa_hourly_processed_20250821_192255.csv"
            if os.path.exists(hourly_file):
                hourly_data = pd.read_csv(hourly_file)
                logger.info(f"📊 小时数据: {hourly_data.shape}")
            else:
                logger.warning("⚠️ 小时数据文件不存在")
                hourly_data = pd.DataFrame()
            
            # 合并数据
            if not daily_data.empty and not hourly_data.empty:
                # 将每日数据扩展到小时级别
                daily_expanded = daily_data.copy()
                daily_expanded['hour'] = 12  # 假设每日数据代表中午12点
                
                # 合并数据
                combined_data = pd.concat([daily_expanded, hourly_data], ignore_index=True)
                logger.info(f"📊 合并后数据: {combined_data.shape}")
            elif not daily_data.empty:
                combined_data = daily_data
                logger.info("📊 使用每日数据")
            elif not hourly_data.empty:
                combined_data = hourly_data
                logger.info("📊 使用小时数据")
            else:
                raise ValueError("没有可用的NOAA数据")
            
            # 特征工程
            features, target = self._engineer_features(combined_data)
            
            # 数据分割
            X_train, X_temp, y_train, y_temp = train_test_split(
                features, target, test_size=0.3, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
            
            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 转换为张量
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_scaled), 
                torch.FloatTensor(y_train.values)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_scaled), 
                torch.FloatTensor(y_val.values)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_scaled), 
                torch.FloatTensor(y_test.values)
            )
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
            
            data_loaders = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
            
            logger.info(f"✅ 数据准备完成:")
            logger.info(f"  训练集: {X_train.shape}")
            logger.info(f"  验证集: {X_val.shape}")
            logger.info(f"  测试集: {X_test.shape}")
            logger.info(f"  特征数: {X_train.shape[1]}")
            
            return {
                'status': 'success',
                'data_loaders': data_loaders,
                'data_shapes': {
                    'train': X_train.shape,
                    'val': X_val.shape,
                    'test': X_test.shape
                },
                'feature_names': features.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _engineer_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """特征工程"""
        try:
            # 选择数值特征
            feature_cols = [
                'temperature', 'temp_squared', 'max_temp', 'min_temp', 'temp_range',
                'precipitation', 'precip_log', 'snow_depth', 'wind_speed', 'pressure',
                'humidity', 'wind_direction', 'wind_direction_sin', 'wind_direction_cos',
                'wind_speed_squared', 'dewpoint',
                'year', 'month', 'day', 'hour', 'day_of_year', 'day_of_week',
                'is_winter', 'is_spring', 'is_summer', 'is_fall',
                'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'
            ]
            
            # 只保留存在的列
            available_cols = [col for col in feature_cols if col in data.columns]
            features = data[available_cols].copy()
            
            # 处理缺失值
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            # 目标变量
            target = data['estimated_soil_moisture']
            
            # 移除包含NaN的行
            valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
            features = features[valid_indices]
            target = target[valid_indices]
            
            logger.info(f"✅ 特征工程完成: {features.shape[1]} 个特征")
            return features, target
            
        except Exception as e:
            logger.error(f"❌ 特征工程失败: {e}")
            return pd.DataFrame(), pd.Series()
    
    def build_model(self, input_size: int) -> nn.Module:
        """构建模型"""
        try:
            model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.1),
                
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            
            self.model = model.to(self.device)
            logger.info(f"✅ 模型构建完成: 输入特征数 {input_size}")
            return model
            
        except Exception as e:
            logger.error(f"❌ 模型构建失败: {e}")
            return None
    
    def train_model(self, data_loaders: Dict, epochs: int = 150) -> Dict:
        """训练模型"""
        try:
            if self.model is None:
                raise ValueError("模型未构建，请先调用build_model()")
            
            logger.info("🚀 开始训练模型...")
            
            # 损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
            
            # 训练历史
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0
                for batch_X, batch_y in data_loaders['train']:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(data_loaders['train'])
                train_losses.append(train_loss)
                
                # 验证阶段
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in data_loaders['val']:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(data_loaders['val'])
                val_losses.append(val_loss)
                
                # 学习率调度
                scheduler.step(val_loss)
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_noaa_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= 25:
                    logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            logger.info("✅ 模型训练完成")
            
            return {
                'status': 'success',
                'epochs_trained': epoch + 1,
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
        except Exception as e:
            logger.error(f"❌ 模型训练失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def evaluate_model(self, data_loaders: Dict) -> Dict:
        """评估模型"""
        try:
            if self.model is None:
                raise ValueError("模型未构建，请先调用build_model()")
            
            logger.info("📊 开始模型评估...")
            
            # 加载最佳模型
            self.model.load_state_dict(torch.load('best_noaa_model.pth'))
            self.model.eval()
            
            # 测试集评估
            test_predictions = []
            test_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in data_loaders['test']:
                    batch_X = batch_X.to(self.device)
                    outputs = self.model(batch_X).squeeze()
                    test_predictions.extend(outputs.cpu().numpy())
                    test_targets.extend(batch_y.numpy())
            
            # 计算性能指标
            test_predictions = np.array(test_predictions)
            test_targets = np.array(test_targets)
            
            r2 = r2_score(test_targets, test_predictions)
            mae = mean_absolute_error(test_targets, test_predictions)
            rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
            
            performance = {
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'status': 'overfitting' if r2 < 0 else 'normal',
                'test_samples': len(test_targets)
            }
            
            logger.info(f"📊 模型性能评估完成:")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  状态: {'过拟合' if r2 < 0 else '正常'}")
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ 模型评估失败: {e}")
            return {'status': 'error', 'error': str(e)}

def main():
    """主函数"""
    try:
        logger.info("🚀 启动使用NOAA真实数据训练土壤湿度预测模型...")
        
        # 创建预测器
        predictor = NOAASoilMoisturePredictor()
        
        # 加载和准备数据
        data_result = predictor.load_and_prepare_data()
        
        if data_result['status'] != 'success':
            logger.error(f"❌ 数据准备失败: {data_result}")
            return
        
        # 构建模型
        input_size = data_result['data_shapes']['train'][1]
        model = predictor.build_model(input_size)
        
        if model is None:
            logger.error("❌ 模型构建失败")
            return
        
        # 训练模型
        training_result = predictor.train_model(data_result['data_loaders'])
        
        if training_result['status'] != 'success':
            logger.error(f"❌ 模型训练失败: {training_result}")
            return
        
        # 评估模型
        performance = predictor.evaluate_model(data_result['data_loaders'])
        
        if 'status' not in performance:
            logger.info("🎉 使用NOAA真实数据训练模型成功！")
            
            # 生成训练报告
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'total_samples': sum(data_result['data_shapes'].values()),
                    'features': input_size,
                    'feature_names': data_result['feature_names']
                },
                'training_summary': training_result,
                'model_performance': performance
            }
            
            # 保存报告
            report_file = f"noaa_data_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ 训练报告已保存: {report_file}")
            
            # 显示关键结果
            if performance['r2_score'] > 0:
                logger.info("🎯 成功！R²已转为正值，NOAA真实数据解决了过拟合问题！")
                logger.info(f"🏆 最终R²: {performance['r2_score']:.4f}")
                logger.info(f"📊 使用真实数据: {performance['test_samples']} 个测试样本")
            else:
                logger.info("⚠️ R²仍为负值，需要进一步分析")
            
            return report
        else:
            logger.error(f"❌ 模型评估失败: {performance}")
            return None
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return None

if __name__ == "__main__":
    main()
