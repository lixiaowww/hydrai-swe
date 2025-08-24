#!/usr/bin/env python3
"""
使用训练好的PyTorch LSTM模型进行预测
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.train_pytorch import LSTMRegressor

def load_trained_model(model_path):
    """加载训练好的模型"""
    print(f"Loading model from: {model_path}")
    
    # 加载模型状态和参数，设置weights_only=False以支持sklearn对象
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_params = checkpoint['model_params']
    scaler = checkpoint['scaler']
    
    # 创建模型
    model = LSTMRegressor(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout']
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    print(f"Model parameters: {model_params}")
    
    return model, scaler, model_params

def create_test_data():
    """创建测试数据"""
    print("Creating test data...")
    
    # 创建测试时间序列（1999年的数据）
    dates = pd.date_range('1999-01-01', '1999-12-31', freq='D')
    
    # 创建模拟测试数据
    np.random.seed(123)  # 不同的随机种子
    
    # 积雪深度：季节性变化 + 随机噪声
    seasonal_snow = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + 50
    snow_depth = np.maximum(0, seasonal_snow + np.random.normal(0, 20, len(dates)))
    
    # 降雪量：冬季较高
    snow_fall = np.where(dates.month.isin([12, 1, 2, 3]), 
                         np.random.exponential(10, len(dates)), 
                         np.random.exponential(2, len(dates)))
    
    # 雪水当量：积雪深度的30%
    snow_water_equivalent = snow_depth * 0.3
    
    # 创建DataFrame
    test_data = pd.DataFrame({
        'date': dates,
        'snow_depth_mm': snow_depth,
        'snow_fall_mm': snow_fall,
        'snow_water_equivalent_mm': snow_water_equivalent,
        'day_of_year': dates.dayofyear,
        'month': dates.month,
        'year': dates.year
    })
    
    print(f"Created test data: {len(test_data)} records")
    print(f"Date range: {test_data['date'].min()} to {test_data['date'].max()}")
    
    return test_data

def prepare_input_data(test_data, scaler, sequence_length):
    """准备模型输入数据"""
    print("Preparing input data...")
    
    # 选择特征列（不包括目标变量）
    feature_columns = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                      'day_of_year', 'month', 'year']
    
    # 提取特征
    features = test_data[feature_columns].values
    
    # 标准化特征
    # 注意：scaler是在训练时用7个特征训练的，但预测时我们只有6个特征
    # 我们需要只使用scaler的前6个特征的标准化参数
    features_scaled = features.copy()
    for i in range(features.shape[1]):
        features_scaled[:, i] = (features[:, i] - scaler.mean_[i]) / scaler.scale_[i]
    
    # 创建序列数据
    sequences = []
    for i in range(len(features_scaled) - sequence_length + 1):
        sequence = features_scaled[i:i + sequence_length]
        sequences.append(sequence)
    
    sequences = np.array(sequences)
    
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    
    return sequences

def predict_streamflow(model, input_sequences, scaler):
    """使用模型预测径流"""
    print("Making predictions...")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequence in input_sequences:
            # 转换为tensor
            x = torch.FloatTensor(sequence).unsqueeze(0)  # 添加batch维度
            
            # 预测
            output = model(x)
            prediction = output.item()
            
            # 反标准化预测结果
            # 注意：这里需要根据scaler的具体实现来调整
            # 假设scaler是StandardScaler，我们需要手动反标准化
            prediction_denorm = prediction * scaler.scale_[-1] + scaler.mean_[-1]
            
            predictions.append(prediction_denorm)
    
    return np.array(predictions)

def evaluate_predictions(predictions, test_data, sequence_length):
    """评估预测结果"""
    print("Evaluating predictions...")
    
    # 获取对应的实际日期
    prediction_dates = test_data['date'].iloc[sequence_length-1:].values
    
    # 创建预测结果DataFrame
    results_df = pd.DataFrame({
        'date': prediction_dates,
        'predicted_streamflow_m3s': predictions
    })
    
    # 添加输入特征
    for col in ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm']:
        results_df[col] = test_data[col].iloc[sequence_length-1:].values
    
    # 计算统计信息
    print("\n📊 Prediction Statistics:")
    print(f"  Number of predictions: {len(predictions)}")
    print(f"  Mean predicted streamflow: {predictions.mean():.2f} m³/s")
    print(f"  Min predicted streamflow: {predictions.min():.2f} m³/s")
    print(f"  Max predicted streamflow: {predictions.max():.2f} m³/s")
    print(f"  Std predicted streamflow: {predictions.std():.2f} m³/s")
    
    return results_df

def plot_results(results_df):
    """绘制预测结果"""
    print("Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制积雪深度
    ax1.plot(results_df['date'], results_df['snow_depth_mm'], 'b-', alpha=0.7, label='Snow Depth')
    ax1.set_ylabel('Snow Depth (mm)')
    ax1.set_title('Snow Depth and Predicted Streamflow')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制预测的径流
    ax2.plot(results_df['date'], results_df['predicted_streamflow_m3s'], 'r-', alpha=0.7, label='Predicted Streamflow')
    ax2.set_ylabel('Streamflow (m³/s)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path("models/pytorch_lstm")
    plot_path = output_dir / "prediction_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_path}")

def save_predictions(results_df):
    """保存预测结果"""
    print("Saving predictions...")
    
    output_dir = Path("models/pytorch_lstm")
    predictions_path = output_dir / "predictions_1999.csv"
    
    results_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")

def main():
    """主函数"""
    print("🚀 开始使用训练好的LSTM模型进行预测")
    print("=" * 50)
    
    try:
        # 加载训练好的模型
        model_path = "models/pytorch_lstm/snow_runoff_lstm.pth"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("Please train the model first using train_pytorch.py")
            return
        
        model, scaler, model_params = load_trained_model(model_path)
        
        # 创建测试数据
        test_data = create_test_data()
        
        # 准备输入数据
        input_sequences = prepare_input_data(test_data, scaler, model_params['sequence_length'])
        
        # 进行预测
        predictions = predict_streamflow(model, input_sequences, scaler)
        
        # 评估结果
        results_df = evaluate_predictions(predictions, test_data, model_params['sequence_length'])
        
        # 绘制结果
        plot_results(results_df)
        
        # 保存预测结果
        save_predictions(results_df)
        
        print("\n🎉 预测完成！")
        print("结果已保存到 models/pytorch_lstm/ 目录")
        
    except Exception as e:
        print(f"\n❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
