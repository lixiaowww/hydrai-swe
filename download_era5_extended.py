#!/usr/bin/env python3
"""
ERA5再分析数据扩展下载脚本
获取更多气象和雪相关数据，扩展数据来源
"""

import os
import sys
import cdsapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class ERA5ExtendedDownloader:
    """ERA5再分析数据扩展下载器"""
    
    def __init__(self):
        self.data_dir = "data/raw/era5_extended"
        self.processed_dir = "data/processed/era5_extended"
        
        # 创建目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # 目标区域（Manitoba附近）
        self.target_area = [60.0, -102.0, 49.0, -88.0]  # [北, 西, 南, 东]
        
        # 时间范围
        self.start_year = 2000
        self.end_year = 2024
        
        # 初始化CDS API客户端
        try:
            self.c = cdsapi.Client()
            print("✅ ERA5 CDS API客户端初始化成功")
        except Exception as e:
            print(f"❌ ERA5 CDS API客户端初始化失败: {e}")
            print("⚠️ 请确保已安装cdsapi: pip install cdsapi")
            print("⚠️ 请确保已配置CDS API密钥")
            self.c = None
    
    def download_snow_parameters(self, year: int) -> Optional[str]:
        """下载雪相关参数"""
        if not self.c:
            print("❌ CDS API客户端未初始化")
            return None
        
        try:
            print(f"📥 下载ERA5雪参数数据: {year}")
            
            # 雪相关参数
            variables = [
                'snow_density',           # 雪密度
                'snow_depth',             # 雪深度
                'snow_depth_water_equivalent',  # 雪水当量
                'snow_evaporation',       # 雪蒸发
                'snowfall',               # 降雪量
                'snowmelt',               # 融雪量
            ]
            
            # 构建请求参数
            request_params = {
                'product_type': 'reanalysis',
                'variable': variables,
                'year': str(year),
                'month': [f"{i:02d}" for i in range(1, 13)],
                'day': [f"{i:02d}" for i in range(1, 32)],
                'time': [f"{i:02d}:00" for i in range(0, 24, 6)],  # 6小时间隔
                'area': self.target_area,
                'format': 'netcdf'
            }
            
            # 生成文件名
            filename = f"era5_snow_{year}.nc"
            filepath = os.path.join(self.data_dir, filename)
            
            print(f"   参数: {variables}")
            print(f"   区域: {self.target_area}")
            print(f"   保存到: {filepath}")
            
            # 下载数据
            self.c.retrieve('reanalysis-era5-single-levels', request_params, filepath)
            
            print(f"✅ 下载完成: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None
    
    def download_meteorological_parameters(self, year: int) -> Optional[str]:
        """下载气象参数"""
        if not self.c:
            print("❌ CDS API客户端未初始化")
            return None
        
        try:
            print(f"📥 下载ERA5气象参数数据: {year}")
            
            # 气象相关参数
            variables = [
                '2m_temperature',         # 2米温度
                '2m_relative_humidity',   # 2米相对湿度
                'total_precipitation',    # 总降水量
                'surface_pressure',       # 表面气压
                '10m_u_component_of_wind',  # 10米风速U分量
                '10m_v_component_of_wind',  # 10米风速V分量
                'surface_solar_radiation_downwards_hourly',  # 表面太阳辐射
                'surface_thermal_radiation_downwards_hourly',  # 表面热辐射
            ]
            
            # 构建请求参数
            request_params = {
                'product_type': 'reanalysis',
                'variable': variables,
                'year': str(year),
                'month': [f"{i:02d}" for i in range(1, 13)],
                'day': [f"{i:02d}" for i in range(1, 32)],
                'time': [f"{i:02d}:00" for i in range(0, 24, 6)],  # 6小时间隔
                'area': self.target_area,
                'format': 'netcdf'
            }
            
            # 生成文件名
            filename = f"era5_meteo_{year}.nc"
            filepath = os.path.join(self.data_dir, filename)
            
            print(f"   参数: {variables}")
            print(f"   区域: {self.target_area}")
            print(f"   保存到: {filepath}")
            
            # 下载数据
            self.c.retrieve('reanalysis-era5-single-levels', request_params, filepath)
            
            print(f"✅ 下载完成: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None
    
    def download_soil_parameters(self, year: int) -> Optional[str]:
        """下载土壤参数"""
        if not self.c:
            print("❌ CDS API客户端未初始化")
            return None
        
        try:
            print(f"📥 下载ERA5土壤参数数据: {year}")
            
            # 土壤相关参数
            variables = [
                'volumetric_soil_water_layer_1',  # 第1层土壤体积含水量
                'volumetric_soil_water_layer_2',  # 第2层土壤体积含水量
                'volumetric_soil_water_layer_3',  # 第3层土壤体积含水量
                'volumetric_soil_water_layer_4',  # 第4层土壤体积含水量
                'soil_temperature_level_1',       # 第1层土壤温度
                'soil_temperature_level_2',       # 第2层土壤温度
                'soil_temperature_level_3',       # 第3层土壤温度
                'soil_temperature_level_4',       # 第4层土壤温度
            ]
            
            # 构建请求参数
            request_params = {
                'product_type': 'reanalysis',
                'variable': variables,
                'year': str(year),
                'month': [f"{i:02d}" for i in range(1, 13)],
                'day': [f"{i:02d}" for i in range(1, 32)],
                'time': [f"{i:02d}:00" for i in range(0, 24, 6)],  # 6小时间隔
                'area': self.target_area,
                'format': 'netcdf'
            }
            
            # 生成文件名
            filename = f"era5_soil_{year}.nc"
            filepath = os.path.join(self.data_dir, filename)
            
            print(f"   参数: {variables}")
            print(f"   区域: {self.target_area}")
            print(f"   保存到: {filepath}")
            
            # 下载数据
            self.c.retrieve('reanalysis-era5-single-levels', request_params, filepath)
            
            print(f"✅ 下载完成: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None
    
    def download_all_parameters(self, year: int) -> List[str]:
        """下载所有参数"""
        print(f"🚀 开始下载ERA5数据: {year}")
        print("=" * 50)
        
        downloaded_files = []
        
        # 下载雪参数
        snow_file = self.download_snow_parameters(year)
        if snow_file:
            downloaded_files.append(snow_file)
        
        # 下载气象参数
        meteo_file = self.download_meteorological_parameters(year)
        if meteo_file:
            downloaded_files.append(meteo_file)
        
        # 下载土壤参数
        soil_file = self.download_soil_parameters(year)
        if soil_file:
            downloaded_files.append(soil_file)
        
        print(f"\n📊 {year}年下载完成: {len(downloaded_files)}/{3} 个文件")
        return downloaded_files
    
    def process_netcdf_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """处理NetCDF文件"""
        try:
            import netCDF4 as nc
            import xarray as xr
            
            print(f"🔄 处理NetCDF文件: {filepath}")
            
            # 使用xarray读取数据
            ds = xr.open_dataset(filepath)
            
            print(f"   变量: {list(ds.variables.keys())}")
            print(f"   维度: {list(ds.dims.keys())}")
            
            # 转换为DataFrame
            df = ds.to_dataframe()
            
            # 重置索引
            df = df.reset_index()
            
            # 处理时间列
            if 'time' in df.columns:
                df['date'] = pd.to_datetime(df['time'])
                df = df.drop('time', axis=1)
            
            # 处理坐标列
            if 'latitude' in df.columns and 'longitude' in df.columns:
                # 选择中心点数据（简化处理）
                center_lat = (self.target_area[0] + self.target_area[2]) / 2
                center_lon = (self.target_area[1] + self.target_area[3]) / 2
                
                # 找到最接近的坐标点
                df = df[
                    (df['latitude'].between(center_lat - 0.5, center_lat + 0.5)) &
                    (df['longitude'].between(center_lon - 0.5, center_lon + 0.5))
                ]
                
                # 删除坐标列
                df = df.drop(['latitude', 'longitude'], axis=1)
            
            print(f"   处理后数据形状: {df.shape}")
            return df
            
        except ImportError:
            print("⚠️ 需要安装netCDF4和xarray: pip install netCDF4 xarray")
            return None
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return None
    
    def merge_era5_data(self) -> pd.DataFrame:
        """合并所有ERA5数据"""
        print("🔄 合并ERA5数据")
        
        all_data = []
        
        # 处理所有下载的NetCDF文件
        netcdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.nc')]
        
        for netcdf_file in netcdf_files:
            filepath = os.path.join(self.data_dir, netcdf_file)
            print(f"\n📊 处理文件: {netcdf_file}")
            
            df = self.process_netcdf_file(filepath)
            if df is not None and not df.empty:
                # 添加数据源标识
                df['data_source'] = 'ERA5'
                
                # 根据文件名添加数据类型标识
                if 'snow' in netcdf_file:
                    df['data_type'] = 'snow'
                elif 'meteo' in netcdf_file:
                    df['data_type'] = 'meteorological'
                elif 'soil' in netcdf_file:
                    df['data_type'] = 'soil'
                else:
                    df['data_type'] = 'unknown'
                
                all_data.append(df)
                print(f"✅ 数据加载成功: {len(df)} 条记录")
            else:
                print(f"⚠️ 数据加载失败")
        
        if not all_data:
            print("❌ 没有成功加载任何数据")
            return pd.DataFrame()
        
        # 合并所有数据
        merged_data = pd.concat(all_data, ignore_index=True)
        
        # 去重和排序
        if 'date' in merged_data.columns:
            merged_data = merged_data.drop_duplicates(subset=['date']).sort_values('date')
        
        print(f"✅ ERA5数据合并完成: {len(merged_data)} 条记录")
        
        # 保存合并后的数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"era5_extended_data_{timestamp}.csv"
        output_path = os.path.join(self.processed_dir, output_file)
        
        merged_data.to_csv(output_path, index=False)
        print(f"✅ 合并数据已保存: {output_path}")
        
        return merged_data
    
    def generate_download_report(self, downloaded_files: List[str]) -> Dict[str, Any]:
        """生成下载报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(downloaded_files),
            'successful_downloads': len([f for f in downloaded_files if f]),
            'failed_downloads': len([f for f in downloaded_files if not f]),
            'file_details': [],
            'target_area': self.target_area,
            'time_range': f"{self.start_year}-{self.end_year}"
        }
        
        for filepath in downloaded_files:
            if filepath and os.path.exists(filepath):
                file_info = {
                    'filename': os.path.basename(filepath),
                    'size_mb': os.path.getsize(filepath) / 1024 / 1024,
                    'download_time': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                }
                report['file_details'].append(file_info)
        
        return report

def main():
    """主函数"""
    print("🚀 ERA5再分析数据扩展下载器启动")
    print("=" * 50)
    
    downloader = ERA5ExtendedDownloader()
    
    if not downloader.c:
        print("❌ 无法继续，CDS API客户端未初始化")
        return
    
    # 下载所有年份的数据
    all_downloaded_files = []
    
    for year in range(downloader.start_year, downloader.end_year + 1):
        print(f"\n🎯 开始下载 {year} 年数据")
        downloaded_files = downloader.download_all_parameters(year)
        all_downloaded_files.extend(downloaded_files)
        
        # 添加延迟避免请求过快
        time.sleep(5)
    
    # 生成下载报告
    report = downloader.generate_download_report(all_downloaded_files)
    
    # 保存报告
    report_path = os.path.join(downloader.data_dir, f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 下载报告已保存: {report_path}")
    print(f"   总文件数: {report['total_files']}")
    print(f"   成功下载: {report['successful_downloads']}")
    print(f"   失败下载: {report['failed_downloads']}")
    
    # 合并数据
    print(f"\n🔄 开始合并ERA5数据...")
    merged_data = downloader.merge_era5_data()
    
    if not merged_data.empty:
        print(f"✅ ERA5扩展数据集创建完成，总记录数: {len(merged_data)}")
    else:
        print("❌ ERA5数据合并失败")

if __name__ == "__main__":
    main()

