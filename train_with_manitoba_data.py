#!/usr/bin/env python3
"""
使用曼尼托巴省本土数据训练土壤湿度预测模型
替代挪威数据，使用曼省实际气候特征
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

class ManitobaSoilMoisturePredictor:
    """使用曼省数据的土壤湿度预测器"""
    
    def __init__(self):
        """初始化"""
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"✅ 曼省土壤湿度预测器初始化完成，设备: {self.device}")
    
    def load_and_prepare_data(self) -> Dict:
        """加载和准备曼省数据"""
        try:
            logger.info("📥 加载曼省农业数据...")
            
            # 加载曼省农业数据
            manitoba_file = "data/real/manitoba/agriculture/manitoba_agriculture_20250821_193550.csv"
            if os.path.exists(manitoba_file):
                manitoba_data = pd.read_csv(manitoba_file)
                logger.info(f"📊 曼省农业数据: {manitoba_data.shape}")
            else:
                raise ValueError("曼省农业数据文件不存在")
            
            # 特征工程
            features, target = self._engineer_manitoba_features(manitoba_data)
            
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
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
            
            data_loaders = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
            
            logger.info(f"✅ 曼省数据准备完成:")
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
                'feature_names': features.columns.tolist(),
                'region': 'Manitoba, Canada'
            }
            
        except Exception as e:
            logger.error(f"❌ 曼省数据准备失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _engineer_manitoba_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """曼省数据特征工程"""
        try:
            # 基础时间特征
            features = data[['year', 'month', 'day', 'day_of_year']].copy()
            
            # 数值特征
            features['temperature'] = data['temperature']
            features['temp_squared'] = data['temperature'] ** 2
            features['temp_cubed'] = data['temperature'] ** 3
            
            features['precipitation'] = data['precipitation']
            features['precip_log'] = np.log1p(data['precipitation'])
            features['precip_squared'] = data['precipitation'] ** 2
            
            # 作物生长状态
            features['crop_growth_status'] = data['crop_growth_status']
            
            # 季节性特征
            features['is_winter'] = (data['month'].isin([12, 1, 2])).astype(int)
            features['is_spring'] = (data['month'].isin([3, 4, 5])).astype(int)
            features['is_summer'] = (data['month'].isin([6, 7, 8])).astype(int)
            features['is_fall'] = (data['month'].isin([9, 10, 11])).astype(int)
            
            # 周期性特征
            features['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            features['day_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
            features['day_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)
            
            # 曼省特有特征
            features['growing_season'] = (data['month'].isin([5, 6, 7, 8, 9])).astype(int)
            features['freezing_season'] = (data['month'].isin([11, 12, 1, 2, 3])).astype(int)
            
            # 温度-降水交互特征
            features['temp_precip_interaction'] = data['temperature'] * data['precipitation']
            features['temp_crop_interaction'] = data['temperature'] * data['crop_growth_status']
            
            # 目标变量
            target = data['estimated_soil_moisture']
            
            # 移除包含NaN的行
            valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
            features = features[valid_indices]
            target = target[valid_indices]
            
            logger.info(f"✅ 曼省特征工程完成: {features.shape[1]} 个特征")
            logger.info(f"📊 特征列表: {list(features.columns)}")
            
            return features, target
            
        except Exception as e:
            logger.error(f"❌ 曼省特征工程失败: {e}")
            return pd.DataFrame(), pd.Series()
    
    def build_model(self, input_size: int) -> nn.Module:
        """构建模型"""
        try:
            model = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.2),
                
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                nn.Dropout(0.1),
                
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
            
            self.model = model.to(self.device)
            logger.info(f"✅ 曼省模型构建完成: 输入特征数 {input_size}")
            return model
            
        except Exception as e:
            logger.error(f"❌ 曼省模型构建失败: {e}")
            return None
    
    def train_model(self, data_loaders: Dict, epochs: int = 100) -> Dict:
        """训练模型"""
        try:
            if self.model is None:
                raise ValueError("模型未构建，请先调用build_model()")
            
            logger.info("🚀 开始训练曼省土壤湿度预测模型...")
            
            # 损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
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
                    torch.save(self.model.state_dict(), 'best_manitoba_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= 20:
                    logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            logger.info("✅ 曼省模型训练完成")
            
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
            logger.error(f"❌ 曼省模型训练失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def evaluate_model(self, data_loaders: Dict) -> Dict:
        """评估模型"""
        try:
            if self.model is None:
                raise ValueError("模型未构建，请先调用build_model()")
            
            logger.info("📊 开始曼省模型评估...")
            
            # 加载最佳模型
            self.model.load_state_dict(torch.load('best_manitoba_model.pth'))
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
                'test_samples': len(test_targets),
                'region': 'Manitoba, Canada'
            }
            
            logger.info(f"📊 曼省模型性能评估完成:")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  状态: {'过拟合' if r2 < 0 else '正常'}")
            logger.info(f"  地区: 曼尼托巴省, 加拿大")
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ 曼省模型评估失败: {e}")
            return {'status': 'error', 'error': str(e)}

def main():
    """主函数"""
    try:
        logger.info("🚀 启动使用曼尼托巴省本土数据训练土壤湿度预测模型...")
        
        # 创建预测器
        predictor = ManitobaSoilMoisturePredictor()
        
        # 加载和准备曼省数据
        data_result = predictor.load_and_prepare_data()
        
        if data_result['status'] != 'success':
            logger.error(f"❌ 曼省数据准备失败: {data_result}")
            return
        
        # 构建模型
        input_size = data_result['data_shapes']['train'][1]
        model = predictor.build_model(input_size)
        
        if model is None:
            logger.error("❌ 曼省模型构建失败")
            return
        
        # 训练模型
        training_result = predictor.train_model(data_result['data_loaders'])
        
        if training_result['status'] != 'success':
            logger.error(f"❌ 曼省模型训练失败: {training_result}")
            return
        
        # 评估模型
        performance = predictor.evaluate_model(data_result['data_loaders'])
        
        if 'status' not in performance:
            logger.info("🎉 使用曼尼托巴省本土数据训练模型成功！")
            
            # 生成训练报告
            report = {
                'timestamp': datetime.now().isoformat(),
                'region': 'Manitoba, Canada',
                'data_info': {
                    'total_samples': sum(data_result['data_shapes'].values()),
                    'features': input_size,
                    'feature_names': data_result['feature_names']
                },
                'training_summary': training_result,
                'model_performance': performance,
                'advantages': [
                    "使用曼省本土数据，气候特征更准确",
                    "大陆性气候，四季分明，适合农业应用",
                    "温度范围: -40°C to +35°C，符合曼省实际",
                    "降水模式: 夏季多雨，冬季少雨",
                    "作物生长季节: 5-9月"
                ]
            }
            
            # 保存报告
            report_file = f"manitoba_data_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ 曼省训练报告已保存: {report_file}")
            
            # 显示关键结果
            if performance['r2_score'] > 0:
                logger.info("🎯 成功！曼省本土数据解决了过拟合问题！")
                logger.info(f"🏆 最终R²: {performance['r2_score']:.4f}")
                logger.info(f"📊 使用曼省数据: {performance['test_samples']} 个测试样本")
                logger.info(f"🌍 地区: {performance['region']}")
                logger.info("💡 优势: 曼省数据比挪威数据更适合本地应用")
            else:
                logger.info("⚠️ R²仍为负值，需要进一步分析")
            
            return report
        else:
            logger.error(f"❌ 曼省模型评估失败: {performance}")
            return None
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return None

if __name__ == "__main__":
    main()
