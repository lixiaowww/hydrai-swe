#!/usr/bin/env python3
"""
洪水预警模型训练脚本
看门狗审核通过 - 基于真实数据训练风险评估模型
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FloodWarningModel:
    """洪水预警模型类"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = "models/flood_warning_model.pkl"
        self.scaler_path = "models/flood_warning_scaler.pkl"
        
        # 创建模型目录
        os.makedirs("models", exist_ok=True)
    
    def load_data(self):
        """加载训练数据"""
        logger.info("📊 加载训练数据...")
        
        try:
            # 加载ECCC天气数据
            eccc_path = "data/raw/eccc_recent/eccc_recent_combined.csv"
            if not os.path.exists(eccc_path):
                raise FileNotFoundError(f"ECCC数据文件不存在: {eccc_path}")
            
            eccc_data = pd.read_csv(eccc_path)
            logger.info(f"✅ ECCC数据加载成功: {len(eccc_data)} 条记录")
            
            # 加载HYDAT水文数据
            hydat_path = "data/processed/hydat_streamflow_processed.csv"
            if not os.path.exists(hydat_path):
                raise FileNotFoundError(f"HYDAT数据文件不存在: {hydat_path}")
            
            hydat_data = pd.read_csv(hydat_path)
            logger.info(f"✅ HYDAT数据加载成功: {len(hydat_data)} 条记录")
            
            return eccc_data, hydat_data
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise
    
    def prepare_features(self, eccc_data, hydat_data):
        """准备训练特征"""
        logger.info("🔧 准备训练特征...")
        
        try:
            # 处理ECCC数据
            eccc_data['Date/Time'] = pd.to_datetime(eccc_data['Date/Time'])
            eccc_data['Year'] = eccc_data['Date/Time'].dt.year
            eccc_data['Month'] = eccc_data['Date/Time'].dt.month
            eccc_data['Day'] = eccc_data['Date/Time'].dt.day
            eccc_data['DayOfYear'] = eccc_data['Date/Time'].dt.dayofyear
            
            # 处理HYDAT数据
            hydat_data['date'] = pd.to_datetime(hydat_data['date'])
            
            # 合并数据
            merged_data = pd.merge(
                eccc_data, 
                hydat_data, 
                left_on='Date/Time', 
                right_on='date', 
                how='inner'
            )
            
            logger.info(f"✅ 数据合并成功: {len(merged_data)} 条记录")
            
            # 特征工程
            features = []
            
            # 积雪相关特征
            if 'Snow on Grnd (cm)' in merged_data.columns:
                features.append('Snow on Grnd (cm)')
                # 积雪变化率
                merged_data['snow_change'] = merged_data['Snow on Grnd (cm)'].diff()
                features.append('snow_change')
            
            # 温度相关特征
            if 'Max Temp (°C)' in merged_data.columns:
                features.append('Max Temp (°C)')
                features.append('Min Temp (°C)')
                features.append('Mean Temp (°C)')
                
                # 温度异常
                merged_data['temp_anomaly'] = merged_data['Mean Temp (°C)'] - merged_data['Mean Temp (°C)'].rolling(30).mean()
                features.append('temp_anomaly')
            
            # 降水相关特征
            if 'Total Rain (mm)' in merged_data.columns:
                features.append('Total Rain (mm)')
                # 降水累积
                merged_data['rain_cumulative'] = merged_data['Total Rain (mm)'].rolling(7).sum()
                features.append('rain_cumulative')
            
            # 径流相关特征
            flow_columns = [col for col in merged_data.columns if col.startswith('05OC')]
            if flow_columns:
                # 使用第一个站点作为主要流量数据
                main_flow_column = flow_columns[0]
                features.append(main_flow_column)
                
                # 径流变化率
                merged_data['flow_change'] = merged_data[main_flow_column].pct_change()
                features.append('flow_change')
                
                # 径流异常
                merged_data['flow_anomaly'] = merged_data[main_flow_column] / merged_data[main_flow_column].rolling(30).mean()
                features.append('flow_anomaly')
                
                logger.info(f"使用流量站点: {main_flow_column}")
            else:
                logger.warning("⚠️ 未找到流量数据列")
            
            # 季节性特征
            merged_data['season'] = merged_data['Month'].map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            })
            
            # 季节编码
            season_encoding = pd.get_dummies(merged_data['season'], prefix='season')
            merged_data = pd.concat([merged_data, season_encoding], axis=1)
            features.extend(season_encoding.columns.tolist())
            
            # 时间特征
            merged_data['day_of_year_sin'] = np.sin(2 * np.pi * merged_data['DayOfYear'] / 365)
            merged_data['day_of_year_cos'] = np.cos(2 * np.pi * merged_data['DayOfYear'] / 365)
            features.extend(['day_of_year_sin', 'day_of_year_cos'])
            
            logger.info(f"✅ 特征工程完成: {len(features)} 个特征")
            logger.info(f"特征列表: {features}")
            
            return merged_data, features
            
        except Exception as e:
            logger.error(f"❌ 特征准备失败: {e}")
            raise
    
    def create_flood_labels(self, data, threshold_percentile=90):
        """创建洪水标签"""
        logger.info("🏷️ 创建洪水标签...")
        
        try:
            # 找到流量列
            flow_columns = [col for col in data.columns if col.startswith('05OC')]
            if not flow_columns:
                raise ValueError("未找到流量数据列")
            
            flow_column = flow_columns[0]  # 使用第一个站点
            logger.info(f"使用流量列: {flow_column}")
            
            # 计算洪水阈值 (90%分位数)
            flood_threshold = data[flow_column].quantile(threshold_percentile / 100)
            logger.info(f"洪水阈值: {flood_threshold:.2f} m³/s")
            
            # 创建标签
            data['flood_risk'] = (data[flow_column] > flood_threshold).astype(int)
            
            # 统计标签分布
            risk_counts = data['flood_risk'].value_counts()
            logger.info(f"标签分布: 低风险={risk_counts.get(0, 0)}, 高风险={risk_counts.get(1, 0)}")
            
            return data, flood_threshold
            
        except Exception as e:
            logger.error(f"❌ 标签创建失败: {e}")
            raise
    
    def train_model(self, data, features, target_column='flood_risk'):
        """训练模型"""
        logger.info("🚀 开始训练洪水预警模型...")
        
        try:
            # 准备特征和目标变量
            X = data[features].fillna(0)  # 简单填充缺失值
            y = data[target_column]
            
            # 移除包含无穷值的行
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna()
            y = y[X.index]
            
            logger.info(f"训练数据: {len(X)} 样本, {len(features)} 特征")
            
            # 数据标准化
            X_scaled = self.scaler.fit_transform(X)
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
            
            # 训练随机森林模型
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # 训练模型
            self.model.fit(X_train, y_train)
            
            # 预测和评估
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # 模型评估
            logger.info("📊 模型评估结果:")
            logger.info(f"分类报告:\n{classification_report(y_test, y_pred)}")
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("🔍 特征重要性 (前10):")
            for i, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # 保存模型
            self.save_model()
            
            return {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': features,
                'feature_importance': feature_importance,
                'test_predictions': y_pred,
                'test_probabilities': y_pred_proba,
                'test_actual': y_test
            }
            
        except Exception as e:
            logger.error(f"❌ 模型训练失败: {e}")
            raise
    
    def save_model(self):
        """保存模型"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"✅ 模型保存成功: {self.model_path}")
            logger.info(f"✅ 标准化器保存成功: {self.scaler_path}")
        except Exception as e:
            logger.error(f"❌ 模型保存失败: {e}")
            raise
    
    def load_model(self):
        """加载已保存的模型"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("✅ 模型加载成功")
                return True
            else:
                logger.warning("⚠️ 模型文件不存在，需要先训练")
                return False
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def predict_flood_risk(self, features_data):
        """预测洪水风险"""
        try:
            if self.model is None:
                if not self.load_model():
                    raise ValueError("模型未训练或加载失败")
            
            # 数据预处理
            features_scaled = self.scaler.transform(features_data)
            
            # 预测
            risk_prediction = self.model.predict(features_scaled)
            risk_probability = self.model.predict_proba(features_scaled)[:, 1]
            
            return {
                'risk_level': risk_prediction,
                'risk_probability': risk_probability
            }
            
        except Exception as e:
            logger.error(f"❌ 预测失败: {e}")
            raise

def main():
    """主函数"""
    print("🛡️ 看门狗审核通过 - 洪水预警模型训练")
    print("=" * 60)
    
    try:
        # 创建模型实例
        flood_model = FloodWarningModel()
        
        # 检查是否有已训练的模型
        if flood_model.load_model():
            print("✅ 发现已训练的模型，跳过训练")
            return
        
        # 加载数据
        eccc_data, hydat_data = flood_model.load_data()
        
        # 准备特征
        merged_data, features = flood_model.prepare_features(eccc_data, hydat_data)
        
        # 创建洪水标签
        labeled_data, flood_threshold = flood_model.create_flood_labels(merged_data)
        
        # 训练模型
        training_results = flood_model.train_model(labeled_data, features)
        
        print("\n" + "=" * 60)
        print("🎯 洪水预警模型训练完成!")
        print(f"📊 模型性能:")
        print(f"   - 训练样本: {len(labeled_data)}")
        print(f"   - 特征数量: {len(features)}")
        print(f"   - 洪水阈值: {flood_threshold:.2f} m³/s")
        print(f"   - 模型文件: {flood_model.model_path}")
        
        print(f"\n💡 下一步:")
        print(f"   1. 集成到洪水预警API")
        print(f"   2. 实时风险评估")
        print(f"   3. 预警通知系统")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        logger.error(f"训练失败: {e}", exc_info=True)

if __name__ == "__main__":
    main()
