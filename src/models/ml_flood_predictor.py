"""
机器学习洪水预测器 - 适配当前数据格式
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

logger = logging.getLogger(__name__)

class MLFloodPredictor:
    """机器学习洪水预测器"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_path = "/home/sean/hydrai_swe/models/ml_flood_model.pkl"
        self.scaler_path = "/home/sean/hydrai_swe/models/ml_flood_scaler.pkl"
        
    def train_model(self, data: pd.DataFrame):
        """
        训练机器学习模型
        
        Args:
            data: 转换后的数据
        """
        try:
            logger.info(f"开始训练机器学习模型，数据形状: {data.shape}")
            
            # 准备特征
            feature_columns = [
                '05OC001', 'Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)',
                'Total Rain (mm)', 'Total Snow (cm)', 'Total Precip (mm)',
                'Snow on Grnd (cm)', 'Year', 'Month', 'Day'
            ]
            
            # 选择可用的特征列
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            
            # 创建目标变量（基于水位和流量）
            y = self._create_target_variable(data)
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 标准化特征
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 训练模型
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # 保存模型和标准化器
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # 保存特征名称
            self.feature_names = available_features
            
            # 评估模型
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"模型训练完成，训练准确率: {train_score:.3f}, 测试准确率: {test_score:.3f}")
            
            return {
                'status': 'success',
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'features_count': len(available_features),
                'model_path': self.model_path
            }
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise
    
    def load_model(self):
        """加载已训练的模型"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("✅ 机器学习模型加载成功")
                return True
            else:
                logger.warning("模型文件不存在，需要重新训练")
                return False
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def predict_flood_risk(self, data: pd.DataFrame, days: int = 7) -> dict:
        """
        使用机器学习模型预测洪水风险
        
        Args:
            data: 转换后的数据
            days: 预测天数
            
        Returns:
            预测结果
        """
        try:
            if self.model is None or self.scaler is None:
                if not self.load_model():
                    raise ValueError("模型未加载且无法加载")
            
            logger.info(f"开始机器学习洪水预测，数据形状: {data.shape}")
            
            # 获取最新数据
            latest_data = data.iloc[-1] if not data.empty else None
            if latest_data is None:
                raise ValueError("没有可用数据")
            
            # 生成预测
            predictions = []
            current_date = datetime.now()
            
            for i in range(days):
                # 创建预测特征
                features = self._create_prediction_features(latest_data, i, current_date)
                
                # 标准化特征
                features_scaled = self.scaler.transform([features])
                
                # 预测风险等级
                risk_prediction = self.model.predict(features_scaled)[0]
                risk_probability = self.model.predict_proba(features_scaled)[0]
                
                # 获取风险概率
                risk_prob = max(risk_probability) if len(risk_probability) > 0 else 0.1
                
                # 确定风险等级
                risk_level = self._get_risk_level_from_prediction(risk_prediction)
                alert_level = self._get_alert_level(risk_level)
                
                predictions.append({
                    'date': (current_date + timedelta(days=i)).isoformat(),
                    'water_level': round(features[0], 2),  # 05OC001
                    'streamflow': round(features[0] * 50, 1),  # 估算流量
                    'precipitation': round(features[4], 2),  # Total Rain
                    'risk_level': risk_level,
                    'risk_probability': round(risk_prob, 3),
                    'alert_level': alert_level
                })
            
            # 计算统计信息
            risk_levels = [p['risk_level'] for p in predictions]
            risk_probs = [p['risk_probability'] for p in predictions]
            
            high_risk_days = sum(1 for level in risk_levels if level == 'HIGH')
            avg_risk = np.mean(risk_probs) * 100
            
            result = {
                'timeline': predictions,
                'summary': {
                    'total_days': days,
                    'high_risk_days': high_risk_days,
                    'average_risk': round(avg_risk, 1),
                    'current_risk': predictions[0]['risk_level'] if predictions else 'UNKNOWN'
                },
                'data_source': 'ml_model',
                'prediction_method': 'machine_learning',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"机器学习洪水预测完成，高风险天数: {high_risk_days}")
            return result
            
        except Exception as e:
            logger.error(f"机器学习洪水预测失败: {e}")
            raise
    
    def _create_target_variable(self, data: pd.DataFrame) -> np.ndarray:
        """创建目标变量"""
        # 基于水位和流量创建风险等级
        water_level = data.get('water_level_m', 2.3)
        streamflow = data.get('streamflow_m3s', 120)
        
        # 创建风险等级 (0: LOW, 1: MEDIUM, 2: HIGH)
        risk_levels = []
        for i in range(len(data)):
            wl = water_level.iloc[i] if hasattr(water_level, 'iloc') else water_level
            sf = streamflow.iloc[i] if hasattr(streamflow, 'iloc') else streamflow
            
            if wl > 2.8 or sf > 300:
                risk_levels.append(2)  # HIGH
            elif wl > 2.5 or sf > 200:
                risk_levels.append(1)  # MEDIUM
            else:
                risk_levels.append(0)  # LOW
        
        return np.array(risk_levels)
    
    def _create_prediction_features(self, latest_data: pd.Series, day_offset: int, current_date: datetime) -> list:
        """创建预测特征"""
        features = []
        
        # 05OC001 (水位相关)
        base_level = getattr(latest_data, 'water_level_m', 2.3)
        if pd.isna(base_level):
            base_level = 2.3
        features.append(base_level + np.random.normal(0, 0.05))
        
        # 温度特征
        month = (current_date + timedelta(days=day_offset)).month
        seasonal_temp = 15 * np.sin(2 * np.pi * (month - 3) / 12)
        features.extend([
            seasonal_temp + np.random.normal(0, 3) + 5,  # Max Temp
            seasonal_temp + np.random.normal(0, 3) - 5,  # Min Temp
            seasonal_temp + np.random.normal(0, 3)       # Mean Temp
        ])
        
        # 降水特征
        base_precip = getattr(latest_data, 'precipitation_mm', 0.5)
        if pd.isna(base_precip):
            base_precip = 0.5
        features.extend([
            base_precip + np.random.exponential(0.5),  # Total Rain
            max(0, base_precip * 0.1),                 # Total Snow
            base_precip + np.random.exponential(0.5),  # Total Precip
            max(0, 10 + np.random.normal(0, 5))        # Snow on Grnd
        ])
        
        # 时间特征
        future_date = current_date + timedelta(days=day_offset)
        features.extend([
            future_date.year,
            future_date.month,
            future_date.day
        ])
        
        return features
    
    def _get_risk_level_from_prediction(self, prediction: int) -> str:
        """从预测结果获取风险等级"""
        if prediction == 2:
            return 'HIGH'
        elif prediction == 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_alert_level(self, risk_level: str) -> str:
        """获取警报等级"""
        if risk_level == 'HIGH':
            return 'WARNING'
        elif risk_level == 'MEDIUM':
            return 'WATCH'
        else:
            return 'NORMAL'
