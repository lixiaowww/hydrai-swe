#!/usr/bin/env python3
"""
Convert CSV data to NetCDF format for NeuralHydrology
将CSV数据转换为NeuralHydrology期望的netCDF格式
"""

import pandas as pd
import xarray as xr
import os
from pathlib import Path

def convert_csv_to_netcdf():
    """
    将CSV数据转换为netCDF格式
    """
    print("🔄 将CSV数据转换为netCDF格式...")
    
    # 读取CSV数据（与prepare_data输出路径保持一致）
    csv_file = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
    df = pd.read_csv(csv_file)
    
    print(f"✅ 读取CSV数据: {len(df)} 条记录")
    print(f"列名: {df.columns.tolist()}")
    
    # 转换日期列
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 创建时间序列数据
    time_series_data = {}
    
    # 添加强制变量 (forcings)
    for col in ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 'day_of_year', 'month', 'year']:
        if col in df.columns:
            time_series_data[col] = df[col].values
    
    # 添加目标变量 (targets)
    if 'streamflow_m3s' in df.columns:
        time_series_data['streamflow_m3s'] = df['streamflow_m3s'].values
    
    # 创建数据变量
    data_vars = {}
    for var_name, values in time_series_data.items():
        if var_name == 'streamflow_m3s':
            # 目标变量
            data_vars[var_name] = xr.DataArray(
                values.reshape(-1, 1),  # (time, basin)
                dims=['time', 'basin'],
                attrs={'long_name': 'Streamflow', 'units': 'm³/s'}
            )
        else:
            # 强制变量
            data_vars[var_name] = xr.DataArray(
                values.reshape(-1, 1),  # (time, basin)
                dims=['time', 'basin'],
                attrs={'long_name': var_name, 'units': 'mm' if 'mm' in var_name else 'unitless'}
            )
    
    # 创建数据集
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'time': df.index, 'basin': ['red_river_basin']},
        attrs={
            'title': 'Red River Basin Snow and Streamflow Data',
            'description': 'Daily snow and streamflow data for Red River Basin',
            'source': 'ECCC Manitoba and HYDAT',
            'time_coverage_start': str(df.index.min()),
            'time_coverage_end': str(df.index.max())
        }
    )
    
    print(f"✅ 创建数据集: {ds.dims}")
    
    # 创建输出目录
    output_dir = Path("data/time_series")
    output_dir.mkdir(exist_ok=True)
    
    # 保存为netCDF文件
    output_file = output_dir / "red_river_basin.nc"
    ds.to_netcdf(output_file)
    
    print(f"✅ 保存netCDF文件: {output_file}")
    print(f"文件大小: {output_file.stat().st_size / 1024:.1f} KB")
    
    # 验证文件
    print("\n🔍 验证netCDF文件...")
    ds_loaded = xr.open_dataset(output_file)
    print(f"加载成功: {ds_loaded.dims}")
    print(f"变量: {list(ds_loaded.data_vars.keys())}")
    print(f"时间范围: {ds_loaded.time.min()} 到 {ds_loaded.time.max()}")
    
    ds_loaded.close()
    
    return output_file

def create_basin_metadata():
    """
    创建流域元数据文件
    """
    print("\n📝 创建流域元数据...")
    
    # 创建流域信息文件
    basin_info = {
        'red_river_basin': {
            'name': 'Red River Basin',
            'area': 116000,  # km²
            'location': 'Manitoba, Canada',
            'coordinates': [-97.5, 49.0, -96.5, 50.5],
            'description': 'Red River Basin snow and streamflow modeling region'
        }
    }
    
    # 保存为YAML文件
    import yaml
    output_file = Path("data/basin_info.yml")
    with open(output_file, 'w') as f:
        yaml.dump(basin_info, f, default_flow_style=False)
    
    print(f"✅ 流域元数据保存: {output_file}")
    
    return basin_info

def main():
    """主函数"""
    print("🚀 开始数据格式转换...")
    print("=" * 50)
    
    try:
        # 转换CSV到netCDF
        netcdf_file = convert_csv_to_netcdf()
        
        # 创建流域元数据
        basin_info = create_basin_metadata()
        
        print("\n🎉 数据格式转换完成！")
        print("下一步:")
        print("1. 测试NeuralHydrology训练")
        print("2. 开始模型训练")
        print(f"\nnetCDF文件: {netcdf_file}")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
