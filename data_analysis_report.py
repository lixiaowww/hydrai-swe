#!/usr/bin/env python3
"""
HydrAI-SWE 数据分析报告
Data Analysis Report for HydrAI-SWE Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def analyze_data_availability():
    """分析数据可用性"""
    
    print("🔍 HydrAI-SWE 数据可用性分析")
    print("=" * 60)
    
    # 检查数据文件
    data_files = {
        "ECCC积雪数据": "data/processed/eccc_manitoba_snow_processed.csv",
        "HYDAT径流数据": "data/processed/hydat_streamflow_processed.csv",
        "HYDAT数据库": "data/raw/Hydat_with_snow.sqlite3",
        "NASA MODIS": "data/raw/nasa_modis_snow/",
        "ECCC天气": "data/raw/eccc_grib/",
        "ECCC近期": "data/raw/eccc_recent/"
    }
    
    print("\n📊 数据文件状态:")
    print("-" * 40)
    
    available_data = {}
    for name, path in data_files.items():
        if os.path.exists(path):
            if os.path.isfile(path):
                size = os.path.getsize(path) / 1024  # KB
                print(f"✅ {name}: {path} ({size:.1f} KB)")
                available_data[name] = path
            else:
                files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                print(f"✅ {name}: {path} ({files} 文件)")
                available_data[name] = path
        else:
            print(f"❌ {name}: {path} (不存在)")
    
    return available_data

def analyze_training_data_volume():
    """分析训练数据量"""
    
    print("\n📈 训练数据量分析")
    print("=" * 60)
    
    # ECCC积雪数据
    eccc_file = "data/processed/eccc_manitoba_snow_processed.csv"
    if os.path.exists(eccc_file):
        df_eccc = pd.read_csv(eccc_file)
        print(f"\n🌨️ ECCC积雪数据:")
        print(f"   - 总记录数: {len(df_eccc):,}")
        print(f"   - 时间范围: {df_eccc['date'].min()} 到 {df_eccc['date'].max()}")
        print(f"   - 站点数量: {df_eccc['station_name'].nunique()}")
        
        # 计算年数
        df_eccc['date'] = pd.to_datetime(df_eccc['date'])
        years = df_eccc['date'].dt.year.unique()
        print(f"   - 覆盖年份: {len(years)} 年 ({min(years)}-{max(years)})")
        
        # 计算每日记录数
        daily_records = df_eccc.groupby('date').size()
        print(f"   - 平均每日记录: {daily_records.mean():.1f}")
        print(f"   - 数据完整性: {len(daily_records)} 天")
    
    # HYDAT径流数据
    hydat_file = "data/processed/hydat_streamflow_processed.csv"
    if os.path.exists(hydat_file):
        df_hydat = pd.read_csv(hydat_file)
        print(f"\n🌊 HYDAT径流数据:")
        print(f"   - 总记录数: {len(df_hydat):,}")
        print(f"   - 时间范围: {df_hydat['date'].min()} 到 {df_hydat['date'].max()}")
        print(f"   - 站点数量: {len(df_hydat.columns) - 1}")  # 减去date列
        
        # 计算年数
        df_hydat['date'] = pd.to_datetime(df_hydat['date'])
        years = df_hydat['date'].dt.year.unique()
        print(f"   - 覆盖年份: {len(years)} 年 ({min(years)}-{max(years)})")

def calculate_training_capacity():
    """计算训练能力"""
    
    print("\n🤖 模型训练能力分析")
    print("=" * 60)
    
    # 基于NeuralHydrology配置
    config_info = {
        "训练期": "1979-1995 (17年)",
        "验证期": "1996-1997 (1.5年)", 
        "测试期": "1997-1998 (1.5年)",
        "序列长度": "30天",
        "批次大小": "16",
        "训练轮数": "30",
        "隐藏层大小": "64"
    }
    
    print("📋 NeuralHydrology配置:")
    for key, value in config_info.items():
        print(f"   - {key}: {value}")
    
    # 计算实际可用训练数据
    eccc_file = "data/processed/eccc_manitoba_snow_processed.csv"
    if os.path.exists(eccc_file):
        df_eccc = pd.read_csv(eccc_file)
        df_eccc['date'] = pd.to_datetime(df_eccc['date'])
        
        # 1979-1995年数据
        train_data = df_eccc[(df_eccc['date'].dt.year >= 1979) & (df_eccc['date'].dt.year <= 1995)]
        train_days = len(train_data['date'].dt.date.unique())
        
        print(f"\n📊 实际训练数据:")
        print(f"   - 训练期天数: {train_days:,}")
        print(f"   - 可用序列数: {max(0, train_days - 30):,}")  # 减去序列长度
        print(f"   - 训练批次: {max(0, (train_days - 30) // 16):,}")
        
        # 验证数据
        val_data = df_eccc[(df_eccc['date'].dt.year >= 1996) & (df_eccc['date'].dt.year <= 1997)]
        val_days = len(val_data['date'].dt.date.unique())
        print(f"   - 验证期天数: {val_days:,}")
        
        # 测试数据
        test_data = df_eccc[(df_eccc['date'].dt.year >= 1997) & (df_eccc['date'].dt.year <= 1998)]
        test_days = len(test_data['date'].dt.date.unique())
        print(f"   - 测试期天数: {test_days:,}")

def analyze_data_source_complementarity():
    """分析数据源互补性"""
    
    print("\n🔄 数据源互补性分析")
    print("=" * 60)
    
    data_sources = {
        "NASA MODIS": {
            "数据类型": "卫星遥感积雪",
            "空间分辨率": "500m",
            "时间分辨率": "每日",
            "覆盖范围": "全球",
            "优势": "大范围覆盖、连续观测",
            "劣势": "云层遮挡、地面验证需求",
            "互补性": "提供大尺度积雪分布"
        },
        "ECCC积雪": {
            "数据类型": "地面观测积雪",
            "空间分辨率": "站点级别",
            "时间分辨率": "每日",
            "覆盖范围": "加拿大",
            "优势": "高精度、连续记录",
            "劣势": "空间覆盖有限",
            "互补性": "提供地面真值验证"
        },
        "ECCC天气": {
            "数据类型": "数值天气预报",
            "空间分辨率": "15km",
            "时间分辨率": "3小时",
            "覆盖范围": "加拿大",
            "优势": "未来预测、多变量",
            "劣势": "预测不确定性",
            "互补性": "提供未来天气驱动"
        },
        "HYDAT": {
            "数据类型": "水文观测",
            "空间分辨率": "站点级别",
            "时间分辨率": "每日",
            "覆盖范围": "加拿大",
            "优势": "长期记录、高精度",
            "劣势": "站点稀疏",
            "互补性": "提供径流目标变量"
        }
    }
    
    print("📋 数据源特性:")
    for source, info in data_sources.items():
        print(f"\n🔹 {source}:")
        for key, value in info.items():
            print(f"   - {key}: {value}")
    
    print("\n🎯 互补性分析:")
    print("   - NASA MODIS + ECCC积雪: 空间覆盖 + 地面验证")
    print("   - ECCC积雪 + ECCC天气: 积雪状态 + 融化驱动")
    print("   - 所有数据源: 完整的水文循环建模")

def analyze_local_training_feasibility():
    """分析本地训练可行性"""
    
    print("\n💻 本地训练可行性分析")
    print("=" * 60)
    
    # 数据量估算
    data_volume = {
        "ECCC积雪": "7306条记录 × 6列 × 8字节 ≈ 350KB",
        "HYDAT径流": "326条记录 × 4列 × 8字节 ≈ 10KB",
        "总数据量": "约 360KB"
    }
    
    print("📊 数据量估算:")
    for item, volume in data_volume.items():
        print(f"   - {item}: {volume}")
    
    # 模型复杂度
    model_complexity = {
        "LSTM隐藏层": "64个神经元",
        "输入特征": "12个变量",
        "序列长度": "30天",
        "参数数量": "约 50K-100K"
    }
    
    print("\n🤖 模型复杂度:")
    for item, complexity in model_complexity.items():
        print(f"   - {item}: {complexity}")
    
    # 训练资源需求
    resource_requirements = {
        "内存需求": "低 (< 2GB)",
        "存储需求": "低 (< 100MB)",
        "计算需求": "中等 (CPU训练可行)",
        "训练时间": "估计 1-4小时 (CPU)"
    }
    
    print("\n💾 资源需求:")
    for item, requirement in resource_requirements.items():
        print(f"   - {item}: {requirement}")
    
    print("\n✅ 结论: 本地训练完全可行！")
    print("   - 数据量小，适合本地处理")
    print("   - 模型复杂度适中，CPU训练可行")
    print("   - 资源需求低，无需云服务")

def provide_training_recommendations():
    """提供训练建议"""
    
    print("\n💡 训练建议")
    print("=" * 60)
    
    recommendations = [
        {
            "阶段": "第一阶段",
            "目标": "验证数据流程",
            "建议": "使用红河流域数据，训练简单模型",
            "预期结果": "确认数据质量和模型框架"
        },
        {
            "阶段": "第二阶段", 
            "目标": "优化模型性能",
            "建议": "增加特征工程，调整超参数",
            "预期结果": "提高预测精度"
        },
        {
            "阶段": "第三阶段",
            "目标": "扩展应用范围",
            "建议": "扩展到其他区域，集成更多数据源",
            "预期结果": "建立完整的预测系统"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n🎯 阶段 {i}: {rec['阶段']}")
        print(f"   目标: {rec['目标']}")
        print(f"   建议: {rec['建议']}")
        print(f"   预期: {rec['预期结果']}")
    
    print("\n🚀 立即行动建议:")
    print("   1. 运行数据验证: python debug_data_sources.py")
    print("   2. 启动训练流程: python run_full_training.py")
    print("   3. 监控训练进度: 检查 runs/ 目录")
    print("   4. 评估模型性能: 使用交叉验证")

def main():
    """主函数"""
    
    print("🚀 HydrAI-SWE 项目数据分析报告")
    print("=" * 60)
    
    # 执行各项分析
    available_data = analyze_data_availability()
    analyze_training_data_volume()
    calculate_training_capacity()
    analyze_data_source_complementarity()
    analyze_local_training_feasibility()
    provide_training_recommendations()
    
    print("\n" + "=" * 60)
    print("✅ 数据分析报告完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
