#!/usr/bin/env python3
"""
ERA5数据处理器
专门处理ERA5替代数据，用于农业模块
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ERA5DataProcessor:
    """ERA5数据处理器"""
    
    def __init__(self, data_path: str = "data/raw/era5_alternative"):
        self.data_path = data_path
        self.csv_file = os.path.join(data_path, "era5_soil_moisture_data.csv")
        self.json_file = os.path.join(data_path, "era5_soil_moisture_data.json")
        
        # 数据配置
        self.config = {
            'input_size': None,  # 动态设置
            'sequence_length': 7,  # 改为7天，适合小数据集
            'target_variable': 'soil_moisture',
            'feature_variables': ['temperature', 'precipitation'],
            'time_variables': ['day_of_year', 'month', 'season']
        }
        
        logger.info(f"✅ ERA5数据处理器初始化完成，数据路径: {data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """加载ERA5数据"""
        try:
            logger.info("📥 加载ERA5数据...")
            
            if not os.path.exists(self.csv_file):
                raise FileNotFoundError(f"ERA5数据文件不存在: {self.csv_file}")
            
            # 读取CSV数据
            df = pd.read_csv(self.csv_file)
            
            # 转换日期列
            df['date'] = pd.to_datetime(df['date'])
            
            # 添加时间特征
            df = self._add_time_features(df)
            
            # 添加工程特征
            df = self._add_engineered_features(df)
            
            # 设置输入大小
            self.config['input_size'] = len(df.columns) - 2  # 减去date和target_variable
            
            logger.info(f"✅ 数据加载完成: {len(df)} 行, {len(df.columns)} 列")
            logger.info(f"📊 输入特征数量: {self.config['input_size']}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        try:
            logger.info("⏰ 添加时间特征...")
            
            # 日期的年积日
            df['day_of_year'] = df['date'].dt.dayofyear
            
            # 月份
            df['month'] = df['date'].dt.month
            
            # 季节 (1=春季, 2=夏季, 3=秋季, 4=冬季)
            df['season'] = df['date'].dt.month.map({
                3: 1, 4: 1, 5: 1,      # 春季
                6: 2, 7: 2, 8: 2,      # 夏季
                9: 3, 10: 3, 11: 3,    # 秋季
                12: 4, 1: 4, 2: 4      # 冬季
            })
            
            # 周几
            df['day_of_week'] = df['date'].dt.dayofweek
            
            # 是否周末
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            logger.info("✅ 时间特征添加完成")
            return df
            
        except Exception as e:
            logger.error(f"❌ 添加时间特征失败: {e}")
            return df
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加工程特征"""
        try:
            logger.info("🔧 添加工程特征...")
            
            # 温度相关特征
            df['temperature_squared'] = df['temperature'] ** 2
            df['temperature_cubed'] = df['temperature'] ** 3
            
            # 降水相关特征
            df['precipitation_squared'] = df['precipitation'] ** 2
            df['precipitation_log'] = np.log1p(df['precipitation'])  # log(1+x)避免log(0)
            
            # 交互特征
            df['temp_precip_interaction'] = df['temperature'] * df['precipitation']
            
            # 滞后特征 (前1天、前3天、前7天)
            for lag in [1, 3, 7]:
                df[f'soil_moisture_lag_{lag}'] = df['soil_moisture'].shift(lag)
                df[f'temperature_lag_{lag}'] = df['temperature'].shift(lag)
                df[f'precipitation_lag_{lag}'] = df['precipitation'].shift(lag)
            
            # 移动平均特征
            for window in [3, 7, 14]:
                df[f'soil_moisture_ma_{window}'] = df['soil_moisture'].rolling(window=window, min_periods=1).mean()
                df[f'temperature_ma_{window}'] = df['temperature'].rolling(window=window, min_periods=1).mean()
                df[f'precipitation_ma_{window}'] = df['precipitation'].rolling(window=window, min_periods=1).mean()
            
            # 趋势特征
            df['soil_moisture_trend'] = df['soil_moisture'].diff()
            df['temperature_trend'] = df['temperature'].diff()
            df['precipitation_trend'] = df['precipitation'].diff()
            
            logger.info("✅ 工程特征添加完成")
            return df
            
        except Exception as e:
            logger.error(f"❌ 添加工程特征失败: {e}")
            return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        try:
            logger.info("🔧 准备训练数据...")
            
            # 选择特征列 (排除date和target_variable)
            feature_columns = [col for col in df.columns 
                             if col not in ['date', self.config['target_variable']]]
            
            # 处理缺失值
            df_clean = df[feature_columns + [self.config['target_variable']]].dropna()
            
            if len(df_clean) == 0:
                raise ValueError("清理后没有可用数据")
            
            # 分离特征和目标
            X = df_clean[feature_columns].values
            y = df_clean[self.config['target_variable']].values
            
            # 创建序列数据
            X_sequences, y_sequences = self._create_sequences(X, y)
            
            logger.info(f"✅ 训练数据准备完成: {X_sequences.shape} -> {y_sequences.shape}")
            
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"❌ 准备训练数据失败: {e}")
            raise
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建序列数据"""
        try:
            sequence_length = self.config['sequence_length']
            
            X_sequences = []
            y_sequences = []
            
            for i in range(len(X) - sequence_length):
                X_sequences.append(X[i:(i + sequence_length)])
                y_sequences.append(y[i + sequence_length])
            
            return np.array(X_sequences), np.array(y_sequences)
            
        except Exception as e:
            logger.error(f"❌ 创建序列数据失败: {e}")
            raise
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """分割数据为训练、验证和测试集"""
        try:
            logger.info("✂️ 分割数据...")
            
            total_samples = len(X)
            train_size = int(total_samples * train_ratio)
            val_size = int(total_samples * val_ratio)
            
            # 时间序列分割 (保持时间顺序)
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            split_info = {
                'train': {'X': X_train, 'y': y_train, 'size': len(X_train)},
                'validation': {'X': X_val, 'y': y_val, 'size': len(X_val)},
                'test': {'X': X_test, 'y': y_test, 'size': len(X_test)},
                'total_samples': total_samples,
                'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': 1 - train_ratio - val_ratio}
            }
            
            logger.info(f"✅ 数据分割完成:")
            logger.info(f"  📊 训练集: {len(X_train)} 样本")
            logger.info(f"  📊 验证集: {len(X_val)} 样本")
            logger.info(f"  📊 测试集: {len(X_test)} 样本")
            
            return split_info
            
        except Exception as e:
            logger.error(f"❌ 数据分割失败: {e}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        try:
            if not hasattr(self, '_feature_names'):
                # 加载数据获取特征名称
                df = self.load_data()
                feature_columns = [col for col in df.columns 
                                 if col not in ['date', self.config['target_variable']]]
                self._feature_names = feature_columns
            
            return self._feature_names
            
        except Exception as e:
            logger.error(f"❌ 获取特征名称失败: {e}")
            return []
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要"""
        try:
            df = self.load_data()
            
            summary = {
                'data_source': 'ERA5_Alternative',
                'total_records': len(df),
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d')
                },
                'features': {
                    'total': len(df.columns),
                    'input_features': self.config['input_size'],
                    'target_variable': self.config['target_variable']
                },
                'variables': {
                    'soil_moisture': {
                        'mean': float(df['soil_moisture'].mean()),
                        'std': float(df['soil_moisture'].std()),
                        'min': float(df['soil_moisture'].min()),
                        'max': float(df['soil_moisture'].max())
                    },
                    'temperature': {
                        'mean': float(df['temperature'].mean()),
                        'std': float(df['temperature'].std()),
                        'min': float(df['temperature'].min()),
                        'max': float(df['temperature'].max())
                    },
                    'precipitation': {
                        'mean': float(df['precipitation'].mean()),
                        'std': float(df['precipitation'].std()),
                        'min': float(df['precipitation'].min()),
                        'max': float(df['precipitation'].max())
                    }
                },
                'config': self.config
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 获取数据摘要失败: {e}")
            return {'error': str(e)}
    
    def save_processed_data(self, output_dir: str = "data/processed/era5") -> Dict:
        """保存处理后的数据"""
        try:
            logger.info("💾 保存处理后的数据...")
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 加载和准备数据
            df = self.load_data()
            X, y = self.prepare_training_data(df)
            split_info = self.split_data(X, y)
            
            # 保存分割后的数据
            np.save(os.path.join(output_dir, 'X_train.npy'), split_info['train']['X'])
            np.save(os.path.join(output_dir, 'y_train.npy'), split_info['train']['y'])
            np.save(os.path.join(output_dir, 'X_val.npy'), split_info['validation']['X'])
            np.save(os.path.join(output_dir, 'y_val.npy'), split_info['validation']['y'])
            np.save(os.path.join(output_dir, 'X_test.npy'), split_info['test']['X'])
            np.save(os.path.join(output_dir, 'y_test.npy'), split_info['test']['y'])
            
            # 保存特征名称
            feature_names = self.get_feature_names()
            with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
                json.dump(feature_names, f)
            
            # 保存配置
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # 保存数据摘要
            summary = self.get_data_summary()
            with open(os.path.join(output_dir, 'data_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✅ 处理后数据保存完成: {output_dir}")
            
            return {
                'status': 'success',
                'output_dir': output_dir,
                'files_saved': [
                    'X_train.npy', 'y_train.npy',
                    'X_val.npy', 'y_val.npy',
                    'X_test.npy', 'y_test.npy',
                    'feature_names.json', 'config.json', 'data_summary.json'
                ],
                'data_summary': summary
            }
            
        except Exception as e:
            logger.error(f"❌ 保存处理后数据失败: {e}")
            return {'status': 'error', 'error': str(e)}

def main():
    """主函数"""
    print("🚀 ERA5数据处理器测试")
    print("=" * 60)
    
    try:
        # 创建数据处理器
        processor = ERA5DataProcessor()
        
        # 获取数据摘要
        print("\n📊 数据摘要:")
        summary = processor.get_data_summary()
        print(f"📁 数据源: {summary.get('data_source', 'Unknown')}")
        print(f"📊 总记录数: {summary.get('total_records', 0)}")
        print(f"📅 日期范围: {summary.get('date_range', {}).get('start', 'Unknown')} 到 {summary.get('date_range', {}).get('end', 'Unknown')}")
        print(f"🔧 特征数量: {summary.get('features', {}).get('input_features', 0)}")
        
        # 保存处理后的数据
        print("\n💾 保存处理后的数据...")
        save_result = processor.save_processed_data()
        
        if save_result['status'] == 'success':
            print(f"✅ 数据保存成功!")
            print(f"📁 输出目录: {save_result['output_dir']}")
            print(f"📄 保存的文件: {len(save_result['files_saved'])} 个")
        else:
            print(f"❌ 数据保存失败: {save_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()
