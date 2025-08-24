#!/usr/bin/env python3
"""
使用真实数据改进的高级洪水预警模型训练脚本
基于现有真实数据，改进特征工程和模型训练
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_analyze_real_data():
    """加载和分析真实数据"""
    try:
        logger.info("📊 加载真实洪水预警数据...")
        
        # 尝试加载多个数据源
        data_sources = [
            "data/processed/flood_warning/flood_warning_optimized.csv",
            "data/real_flood_data/hydat_streamflow_realistic.csv"
        ]
        
        main_data = None
        for source in data_sources:
            if os.path.exists(source):
                logger.info(f"找到数据源: {source}")
                data = pd.read_csv(source)
                logger.info(f"数据形状: {data.shape}")
                logger.info(f"列名: {list(data.columns)}")
                
                if main_data is None:
                    main_data = data
                else:
                    # 合并数据
                    if 'Date/Time' in main_data.columns and 'Date' in data.columns:
                        main_data['Date/Time'] = pd.to_datetime(main_data['Date/Time'])
                        data['Date'] = pd.to_datetime(data['Date'])
                        main_data = pd.merge(main_data, data, 
                                           left_on='Date/Time', right_on='Date', 
                                           how='left')
                        logger.info(f"合并后数据形状: {main_data.shape}")
        
        if main_data is None:
            raise FileNotFoundError("没有找到可用的数据源")
        
        return main_data
        
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise

def create_realistic_flood_target(data: pd.DataFrame) -> pd.DataFrame:
    """创建真实的洪水目标变量"""
    try:
        logger.info("🎯 创建真实的洪水目标变量...")
        
        # 检查径流数据列
        flow_columns = [col for col in data.columns if col.startswith('05OC')]
        logger.info(f"找到径流列: {flow_columns}")
        
        if not flow_columns:
            logger.warning("没有找到径流数据列，使用合成目标变量")
            np.random.seed(42)
            data['flood_risk'] = np.random.choice([0, 1], size=len(data), p=[0.85, 0.15])
            return data
        
        # 使用主要径流列
        main_flow_col = flow_columns[0]
        flow_data = data[main_flow_col].fillna(0)
        
        # 计算真实的洪水风险阈值
        # 使用多种方法确定阈值
        
        # 方法1: 统计阈值（90%分位数）
        quantile_threshold = flow_data.quantile(0.9)
        
        # 方法2: 均值 + 2倍标准差
        mean_threshold = flow_data.mean() + 2 * flow_data.std()
        
        # 方法3: 基于历史极值的阈值
        historical_max = flow_data.max()
        historical_threshold = historical_max * 0.7
        
        # 选择最合适的阈值
        thresholds = [quantile_threshold, mean_threshold, historical_threshold]
        valid_thresholds = [t for t in thresholds if t > 0 and not np.isnan(t)]
        
        if valid_thresholds:
            # 选择中等阈值，确保有足够的高风险样本
            selected_threshold = np.median(valid_thresholds)
            logger.info(f"选择的洪水阈值: {selected_threshold:.2f}")
        else:
            # 如果没有有效阈值，使用分位数
            selected_threshold = flow_data.quantile(0.85)
            logger.info(f"使用分位数阈值: {selected_threshold:.2f}")
        
        # 创建目标变量
        data['flood_risk'] = (flow_data > selected_threshold).astype(int)
        
        # 检查目标变量分布
        risk_distribution = data['flood_risk'].value_counts()
        logger.info(f"洪水风险分布: {risk_distribution.to_dict()}")
        
        # 如果高风险样本太少，调整阈值
        high_risk_count = risk_distribution.get(1, 0)
        if high_risk_count < 50:
            logger.warning(f"高风险样本太少 ({high_risk_count})，调整阈值...")
            # 降低阈值到75%分位数
            adjusted_threshold = flow_data.quantile(0.75)
            data['flood_risk'] = (flow_data > adjusted_threshold).astype(int)
            logger.info(f"调整后阈值: {adjusted_threshold:.2f}")
            logger.info(f"调整后风险分布: {data['flood_risk'].value_counts().to_dict()}")
        
        return data
        
    except Exception as e:
        logger.error(f"创建洪水目标变量失败: {e}")
        raise

def engineer_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """工程化高级特征"""
    try:
        logger.info("🔧 工程化高级特征...")
        
        # 确保日期列是datetime类型
        if 'Date/Time' in data.columns:
            data['Date/Time'] = pd.to_datetime(data['Date/Time'])
        elif 'Date' in data.columns:
            data['Date/Time'] = pd.to_datetime(data['Date'])
        
        # 时间特征
        data['Year'] = data['Date/Time'].dt.year
        data['Month'] = data['Date/Time'].dt.month
        data['Day'] = data['Date/Time'].dt.day
        data['DayOfYear'] = data['Date/Time'].dt.dayofyear
        data['WeekOfYear'] = data['Date/Time'].dt.isocalendar().week
        
        # 季节性特征
        data['Season'] = data['Month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # 季节性编码
        season_dummies = pd.get_dummies(data['Season'], prefix='season')
        data = pd.concat([data, season_dummies], axis=1)
        
        # 时间周期性特征
        data['day_of_year_sin'] = np.sin(2 * np.pi * data['DayOfYear'] / 365)
        data['day_of_year_cos'] = np.cos(2 * np.pi * data['DayOfYear'] / 365)
        data['month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
        
        # 气象特征工程
        if 'Max Temp (°C)' in data.columns:
            data['temp_range'] = data['Max Temp (°C)'] - data['Min Temp (°C)']
            data['temp_anomaly'] = data['Mean Temp (°C)'] - data['Mean Temp (°C)'].rolling(30).mean()
            data['temp_trend'] = data['Mean Temp (°C)'].rolling(7).mean()
            
            # 温度变化率
            data['temp_change'] = data['Mean Temp (°C)'].diff()
            data['temp_acceleration'] = data['temp_change'].diff()
        
        if 'Total Rain (mm)' in data.columns:
            # 降水特征
            data['rain_cumulative_3d'] = data['Total Rain (mm)'].rolling(3).sum()
            data['rain_cumulative_7d'] = data['Total Rain (mm)'].rolling(7).sum()
            data['rain_intensity'] = data['Total Rain (mm)'].rolling(3).max()
            data['rain_frequency'] = (data['Total Rain (mm)'] > 0).rolling(7).sum()
            
            # 降水变化
            data['rain_change'] = data['Total Rain (mm)'].diff()
            data['rain_trend'] = data['Total Rain (mm)'].rolling(7).mean()
        
        if 'Snow on Grnd (cm)' in data.columns:
            # 积雪特征
            data['snow_change'] = data['Snow on Grnd (cm)'].diff()
            data['snow_trend'] = data['Snow on Grnd (cm)'].rolling(7).mean()
            data['snow_accumulation'] = data['Snow on Grnd (cm)'].rolling(30).sum()
            
            # 积雪融化率
            data['snow_melt_rate'] = -data['snow_change']  # 负值表示融化
            data['snow_melt_rate'] = data['snow_melt_rate'].clip(lower=0)  # 只保留融化
        
        # 径流特征工程
        flow_columns = [col for col in data.columns if col.startswith('05OC')]
        if flow_columns:
            main_flow_col = flow_columns[0]
            
            # 径流变化特征
            data['flow_change'] = data[main_flow_col].pct_change()
            data['flow_acceleration'] = data['flow_change'].diff()
            data['flow_trend'] = data[main_flow_col].rolling(7).mean()
            data['flow_volatility'] = data[main_flow_col].rolling(7).std()
            
            # 径流异常
            data['flow_anomaly'] = data[main_flow_col] / data[main_flow_col].rolling(30).mean()
            data['flow_anomaly'] = data['flow_anomaly'].fillna(1.0)
            
            # 径流峰值
            data['flow_peak'] = data[main_flow_col].rolling(7).max()
            data['flow_peak_ratio'] = data[main_flow_col] / data['flow_peak']
            
            # 多站点径流相关性
            if len(flow_columns) > 1:
                for i, col1 in enumerate(flow_columns):
                    for j, col2 in enumerate(flow_columns[i+1:], i+1):
                        col_name = f'flow_corr_{i}_{j}'
                        data[col_name] = data[col1].rolling(30).corr(data[col2])
        
        # 交互特征
        if 'Mean Temp (°C)' in data.columns and 'Total Rain (mm)' in data.columns:
            data['temp_rain_interaction'] = data['Mean Temp (°C)'] * data['Total Rain (mm)']
        
        if 'Snow on Grnd (cm)' in data.columns and 'Mean Temp (°C)' in data.columns:
            data['snow_temp_interaction'] = data['Snow on Grnd (cm)'] * data['Mean Temp (°C)']
        
        # 滞后特征
        if '05OC001' in data.columns:
            for lag in [1, 3, 7]:
                data[f'flow_lag_{lag}'] = data['05OC001'].shift(lag)
        
        if 'Total Rain (mm)' in data.columns:
            for lag in [1, 3, 7]:
                data[f'rain_lag_{lag}'] = data['Total Rain (mm)'].shift(lag)
        
        # 检查NaN值情况
        initial_rows = len(data)
        nan_counts = data.isnull().sum()
        logger.info(f"NaN值统计:")
        for col, count in nan_counts[nan_counts > 0].items():
            logger.info(f"  {col}: {count} NaN值")
        
        # 使用更智能的NaN处理策略
        # 1. 对于数值列，用0填充
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)
        
        # 2. 对于分类列，用前向填充
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[categorical_cols] = data[categorical_cols].fillna(method='ffill')
        
        # 3. 检查是否还有NaN值
        remaining_nans = data.isnull().sum().sum()
        logger.info(f"处理后剩余NaN值: {remaining_nans}")
        
        if remaining_nans > 0:
            # 如果还有NaN值，用0填充
            data = data.fillna(0)
            logger.info("使用0填充剩余NaN值")
        
        logger.info(f"特征工程后数据: {len(data)} 行")
        
        return data
        
    except Exception as e:
        logger.error(f"特征工程失败: {e}")
        raise

def select_best_features(data: pd.DataFrame, target_col: str = 'flood_risk', k: int = 20):
    """选择最佳特征"""
    try:
        logger.info(f"🔍 选择最佳 {k} 个特征...")
        
        # 分离特征和目标
        feature_cols = [col for col in data.columns if col != target_col and col not in ['Date/Time', 'Date']]
        X = data[feature_cols]
        y = data[target_col]
        
        # 只选择数值列
        numeric_feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"数值特征列数量: {len(numeric_feature_cols)}")
        
        if len(numeric_feature_cols) < k:
            logger.warning(f"数值特征数量 ({len(numeric_feature_cols)}) 少于请求的 {k} 个")
            k = len(numeric_feature_cols)
        
        X_numeric = X[numeric_feature_cols]
        
        # 处理无穷值
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(0)
        
        # 使用F检验选择特征
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X_numeric, y)
        
        # 获取选中的特征
        selected_features = X_numeric.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_[selector.get_support()]
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': selected_features,
            'score': feature_scores
        }).sort_values('score', ascending=False)
        
        logger.info("前10个最佳特征:")
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
            logger.info(f"  {i+1:2d}. {row['feature']}: {row['score']:.4f}")
        
        # 返回选中的特征数据
        selected_data = data[selected_features + [target_col, 'Date/Time']]
        
        return selected_data, selected_features
        
    except Exception as e:
        logger.error(f"特征选择失败: {e}")
        raise

def train_improved_model(data: pd.DataFrame, selected_features: list):
    """训练改进的模型"""
    try:
        logger.info("🎯 训练改进的洪水预警模型...")
        
        # 准备训练数据
        X = data[selected_features]
        y = data['flood_risk']
        
        # 处理无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        logger.info(f"特征矩阵形状: {X.shape}")
        logger.info(f"目标变量分布: {y.value_counts().to_dict()}")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"训练集: {X_train.shape[0]} 样本")
        logger.info(f"测试集: {X_test.shape[0]} 样本")
        
        # 特征标准化
        scaler = RobustScaler()  # 使用RobustScaler处理异常值
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练聚类模型
        logger.info("🎯 训练聚类模型...")
        n_clusters = min(5, len(X_train) // 10)  # 动态确定聚类数
        cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = cluster_model.fit_predict(X_train_scaled)
        
        # 训练主模型
        logger.info("🎯 训练主分类模型...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # 处理类别不平衡
        )
        
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        # 模型评估
        logger.info("📊 模型评估...")
        
        # 训练集性能
        y_train_pred = model.predict(X_train_scaled)
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        train_accuracy = (y_train_pred == y_train).mean()
        
        # 测试集性能
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_accuracy = (y_test_pred == y_test).mean()
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        # ROC AUC
        train_roc_auc = roc_auc_score(y_train, y_train_proba)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        
        logger.info(f"训练集准确率: {train_accuracy:.4f}")
        logger.info(f"测试集准确率: {test_accuracy:.4f}")
        logger.info(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        logger.info(f"训练集ROC AUC: {train_roc_auc:.4f}")
        logger.info(f"测试集ROC AUC: {test_roc_auc:.4f}")
        
        # 分类报告
        logger.info("分类报告:")
        logger.info(classification_report(y_test, y_test_pred))
        
        # 特征重要性
        feature_importance = dict(zip(selected_features, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("前15个重要特征:")
        for i, (feature, importance) in enumerate(top_features[:15]):
            logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
        return model, scaler, cluster_model, {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'train_roc_auc': train_roc_auc,
            'test_roc_auc': test_roc_auc,
            'feature_importance': feature_importance
        }
        
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise

def save_improved_model(model, scaler, cluster_model, performance_metrics, selected_features):
    """保存改进的模型"""
    try:
        logger.info("💾 保存改进的模型...")
        
        # 创建模型目录
        os.makedirs("models", exist_ok=True)
        
        # 保存主模型
        model_path = "models/advanced_flood_warning_model_improved.pkl"
        joblib.dump(model, model_path)
        logger.info(f"主模型已保存: {model_path}")
        
        # 保存标准化器
        scaler_path = "models/advanced_flood_warning_scaler_improved.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"标准化器已保存: {scaler_path}")
        
        # 保存聚类模型
        cluster_path = "models/advanced_flood_cluster_model_improved.pkl"
        joblib.dump(cluster_model, cluster_path)
        logger.info(f"聚类模型已保存: {cluster_path}")
        
        # 保存特征列表
        features_path = "models/advanced_flood_features_improved.json"
        import json
        with open(features_path, 'w') as f:
            json.dump(selected_features, f, indent=2)
        logger.info(f"特征列表已保存: {features_path}")
        
        # 保存性能指标
        metrics_path = "models/advanced_flood_performance_improved.json"
        with open(metrics_path, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        logger.info(f"性能指标已保存: {metrics_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"保存模型失败: {e}")
        return False

def main():
    """主函数"""
    try:
        logger.info("🚀 开始使用真实数据训练改进的洪水预警模型...")
        
        # 1. 加载真实数据
        data = load_and_analyze_real_data()
        
        # 2. 创建真实的洪水目标变量
        data = create_realistic_flood_target(data)
        
        # 3. 工程化高级特征
        data = engineer_advanced_features(data)
        
        # 4. 选择最佳特征
        selected_data, selected_features = select_best_features(data)
        
        # 5. 训练改进的模型
        model, scaler, cluster_model, performance_metrics = train_improved_model(
            selected_data, selected_features
        )
        
        # 6. 保存模型
        success = save_improved_model(model, scaler, cluster_model, performance_metrics, selected_features)
        
        if success:
            logger.info("✅ 改进的洪水预警模型训练完成！")
            logger.info(f"测试集准确率: {performance_metrics['test_accuracy']:.4f}")
            logger.info(f"测试集ROC AUC: {performance_metrics['test_roc_auc']:.4f}")
            logger.info(f"使用特征数量: {len(selected_features)}")
            
            return True
        else:
            logger.error("❌ 模型保存失败")
            return False
            
    except Exception as e:
        logger.error(f"训练流程失败: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        
        if success:
            print("\n🎉 改进的洪水预警模型训练成功！")
            print("模型文件已保存到 models/ 目录")
        else:
            print("\n❌ 模型训练失败")
            exit(1)
            
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        exit(1)
