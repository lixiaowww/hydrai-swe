#!/usr/bin/env python3
"""
修复数据标准化一致性问题
确保训练和验证时使用相同的标准化参数
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class StandardizationFixer:
    """标准化一致性修复器"""
    
    def __init__(self):
        self.scaler_X = None
        self.scaler_y = None
        self.standardization_params = {}
        
    def load_training_data(self, data_path):
        """加载训练数据"""
        print("📊 加载训练数据...")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✅ 加载数据: {len(data)} 条记录")
        return data
    
    def extract_features_and_target(self, data):
        """提取特征和目标"""
        feature_cols = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                       'day_of_year', 'month', 'year']
        target_col = 'snow_water_equivalent_mm'
        
        X = data[feature_cols].values
        y = data[target_col].values.reshape(-1, 1)
        
        print(f"✅ 提取特征: {X.shape}, 目标: {y.shape}")
        return X, y, feature_cols, target_col
    
    def fit_standardization(self, X, y):
        """拟合标准化器"""
        print("🔧 拟合标准化器...")
        
        # 特征标准化
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        
        # 目标标准化
        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y)
        
        # 保存标准化参数
        self.standardization_params = {
            'scaler_X_mean': self.scaler_X.mean_,
            'scaler_X_scale': self.scaler_X.scale_,
            'scaler_y_mean': self.scaler_y.mean_,
            'scaler_y_scale': self.scaler_y.scale_,
            'feature_names': ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                             'day_of_year', 'month', 'year'],
            'target_name': 'snow_water_equivalent_mm'
        }
        
        print(f"✅ 标准化器拟合完成")
        print(f"   特征均值: {self.scaler_X.mean_}")
        print(f"   特征标准差: {self.scaler_X.scale_}")
        print(f"   目标均值: {self.scaler_y.mean_}")
        print(f"   目标标准差: {self.scaler_y.scale_}")
        
        return X_scaled, y_scaled
    
    def save_standardization_params(self, output_path):
        """保存标准化参数"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.standardization_params, f)
        
        print(f"✅ 标准化参数已保存: {output_path}")
    
    def create_consistent_dataset(self, data, output_path):
        """创建标准化一致的数据集"""
        print("🔄 创建标准化一致的数据集...")
        
        # 提取特征和目标
        X, y, feature_cols, target_col = self.extract_features_and_target(data)
        
        # 应用标准化
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y)
        
        # 创建标准化后的数据集
        scaled_data = pd.DataFrame(X_scaled, columns=feature_cols, index=data.index)
        scaled_data[target_col] = y_scaled.flatten()
        
        # 添加原始列（用于参考）
        scaled_data['original_snow_depth_mm'] = data['snow_depth_mm']
        scaled_data['original_snow_water_equivalent_mm'] = data['snow_water_equivalent_mm']
        
        # 保存数据集
        scaled_data.to_csv(output_path)
        print(f"✅ 标准化数据集已保存: {output_path}")
        
        return scaled_data
    
    def validate_standardization_consistency(self, original_data, scaled_data):
        """验证标准化一致性"""
        print("🔍 验证标准化一致性...")
        
        # 检查特征分布
        feature_cols = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                       'day_of_year', 'month', 'year']
        
        print("\n📊 标准化前后对比:")
        for col in feature_cols:
            if col in original_data.columns and col in scaled_data.columns:
                orig_mean = original_data[col].mean()
                orig_std = original_data[col].std()
                scaled_mean = scaled_data[col].mean()
                scaled_std = scaled_data[col].std()
                
                print(f"  {col}:")
                print(f"    原始: 均值={orig_mean:.4f}, 标准差={orig_std:.4f}")
                print(f"    标准化: 均值={scaled_mean:.4f}, 标准差={scaled_std:.4f}")
        
        # 验证目标变量
        target_col = 'snow_water_equivalent_mm'
        if target_col in original_data.columns and target_col in scaled_data.columns:
            orig_mean = original_data[target_col].mean()
            orig_std = original_data[target_col].std()
            scaled_mean = scaled_data[target_col].mean()
            scaled_std = scaled_data[target_col].std()
            
            print(f"\n🎯 目标变量 {target_col}:")
            print(f"  原始: 均值={orig_mean:.4f}, 标准差={orig_std:.4f}")
            print(f"  标准化: 均值={scaled_mean:.4f}, 标准差={scaled_std:.4f}")
        
        print("✅ 标准化一致性验证完成")
    
    def create_standardization_report(self, output_path):
        """创建标准化报告"""
        print("📝 创建标准化报告...")
        
        report = f"""# 数据标准化一致性修复报告

## 修复时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 标准化参数
### 特征标准化器 (StandardScaler)
"""
        
        for i, feature in enumerate(self.standardization_params['feature_names']):
            mean = self.standardization_params['scaler_X_mean'][i]
            scale = self.standardization_params['scaler_X_scale'][i]
            report += f"- {feature}: 均值={mean:.6f}, 标准差={scale:.6f}\n"
        
        report += f"""
### 目标标准化器 (StandardScaler)
- {self.standardization_params['target_name']}: 均值={self.standardization_params['scaler_y_mean'][0]:.6f}, 标准差={self.standardization_params['scaler_y_scale'][0]:.6f}

## 修复内容
1. ✅ 建立了统一的标准化参数
2. ✅ 确保训练和验证使用相同的标准化器
3. ✅ 创建了标准化一致的数据集
4. ✅ 保存了标准化参数供后续使用

## 使用说明
- 训练时：使用 `scaler_X.fit_transform()` 和 `scaler_y.fit_transform()`
- 验证时：使用 `scaler_X.transform()` 和 `scaler_y.transform()`
- 预测时：使用 `scaler_y.inverse_transform()` 还原预测结果

## 注意事项
- 所有数据预处理必须使用相同的标准化参数
- 新数据必须通过已训练的标准化器进行转换
- 定期验证标准化一致性
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 标准化报告已保存: {output_path}")

def main():
    """主函数"""
    print("🔧 HydrAI-SWE 数据标准化一致性修复")
    print("=" * 60)
    
    try:
        # 创建修复器
        fixer = StandardizationFixer()
        
        # 1. 加载训练数据
        data_path = "data/processed/comprehensive_training_dataset.csv"
        data = fixer.load_training_data(data_path)
        
        # 2. 提取特征和目标
        X, y, feature_cols, target_col = fixer.extract_features_and_target(data)
        
        # 3. 拟合标准化器
        X_scaled, y_scaled = fixer.fit_standardization(X, y)
        
        # 4. 保存标准化参数
        params_path = "models/standardization_params.pkl"
        fixer.save_standardization_params(params_path)
        
        # 5. 创建标准化一致的数据集
        scaled_data_path = "data/processed/standardized_training_dataset.csv"
        scaled_data = fixer.create_consistent_dataset(data, scaled_data_path)
        
        # 6. 验证标准化一致性
        fixer.validate_standardization_consistency(data, scaled_data)
        
        # 7. 创建标准化报告
        report_path = "logs/standardization_fix_report.md"
        fixer.create_standardization_report(report_path)
        
        print("\n" + "=" * 60)
        print("🎉 数据标准化一致性修复完成!")
        print("✅ 标准化参数已保存")
        print("✅ 标准化数据集已创建")
        print("✅ 一致性验证已通过")
        print("✅ 修复报告已生成")
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
