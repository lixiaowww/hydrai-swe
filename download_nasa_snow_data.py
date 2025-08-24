#!/usr/bin/env python3
"""
NASA雪数据下载脚本
获取全球雪水当量数据，扩展数据来源
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class NASASnowDataDownloader:
    """NASA雪数据下载器"""
    
    def __init__(self):
        self.base_url = "https://cmr.earthdata.nasa.gov/search"
        self.data_dir = "data/raw/nasa_snow"
        self.processed_dir = "data/processed/nasa_snow"
        
        # 创建目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # NASA数据集信息
        self.datasets = {
            'smap_swe': {
                'name': 'SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture Geophysical Data',
                'short_name': 'SPL4SMGP',
                'version': '7',
                'description': 'SMAP雪水当量数据，9km分辨率，3小时频率'
            },
            'amsr2_swe': {
                'name': 'AMSR2 Daily L3 Global Snow Water Equivalent EASE-Grids',
                'short_name': 'AE_DySno',
                'version': '1',
                'description': 'AMSR2雪水当量数据，25km分辨率，日频率'
            },
            'globsnow_swe': {
                'name': 'GlobSnow v3.0 Northern Hemisphere Snow Water Equivalent',
                'short_name': 'GlobSnow_SWE',
                'version': '3.0',
                'description': 'GlobSnow雪水当量数据，25km分辨率，日频率'
            }
        }
        
        # 目标区域（Manitoba附近）
        self.target_bbox = {
            'west': -102.0,  # 西经
            'east': -88.0,   # 东经
            'north': 60.0,   # 北纬
            'south': 49.0    # 南纬
        }
    
    def search_datasets(self, dataset_key: str) -> List[Dict[str, Any]]:
        """搜索数据集"""
        dataset = self.datasets[dataset_key]
        
        print(f"🔍 搜索数据集: {dataset['name']}")
        
        # 构建搜索参数
        params = {
            'collection': f"{dataset['short_name']}.{dataset['version']}",
            'bbox': f"{self.target_bbox['west']},{self.target_bbox['south']},{self.target_bbox['east']},{self.target_bbox['north']}",
            'temporal': '2000-01-01T00:00:00Z/2024-12-31T23:59:59Z',
            'format': 'json'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            results = response.json()
            
            if 'feed' in results and 'entry' in results['feed']:
                granules = results['feed']['entry']
                print(f"✅ 找到 {len(granules)} 个数据文件")
                return granules
            else:
                print("⚠️ 未找到数据文件")
                return []
                
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def download_granule(self, granule: Dict[str, Any], dataset_key: str) -> Optional[str]:
        """下载单个数据文件"""
        try:
            # 获取下载链接
            links = granule.get('links', [])
            download_link = None
            
            for link in links:
                if link.get('type') == 'GET DATA':
                    download_link = link.get('href')
                    break
            
            if not download_link:
                print(f"⚠️ 未找到下载链接: {granule.get('id', 'unknown')}")
                return None
            
            # 生成文件名
            granule_id = granule.get('id', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{dataset_key}_{granule_id}_{timestamp}.nc"
            filepath = os.path.join(self.data_dir, filename)
            
            print(f"📥 下载: {granule_id}")
            print(f"   链接: {download_link}")
            print(f"   保存到: {filepath}")
            
            # 下载文件
            response = requests.get(download_link, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ 下载完成: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None
    
    def download_dataset(self, dataset_key: str, max_files: int = 100) -> List[str]:
        """下载整个数据集"""
        print(f"🚀 开始下载数据集: {dataset_key}")
        print("=" * 50)
        
        # 搜索数据文件
        granules = self.search_datasets(dataset_key)
        
        if not granules:
            print("❌ 未找到数据文件，跳过下载")
            return []
        
        # 限制下载文件数量
        granules = granules[:max_files]
        
        downloaded_files = []
        
        for i, granule in enumerate(granules, 1):
            print(f"\n📊 进度: {i}/{len(granules)}")
            
            filepath = self.download_granule(granule, dataset_key)
            if filepath:
                downloaded_files.append(filepath)
            
            # 添加延迟避免请求过快
            time.sleep(1)
        
        print(f"\n🎉 下载完成: {len(downloaded_files)}/{len(granules)} 个文件")
        return downloaded_files
    
    def process_smap_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """处理SMAP数据"""
        try:
            import netCDF4 as nc
            
            print(f"🔄 处理SMAP数据: {filepath}")
            
            # 读取NetCDF文件
            with nc.Dataset(filepath, 'r') as ds:
                # 获取变量信息
                print(f"   变量: {list(ds.variables.keys())}")
                
                # 读取时间信息
                time_var = ds.variables.get('time')
                if time_var:
                    print(f"   时间范围: {time_var[0]} - {time_var[-1]}")
                
                # 读取雪水当量数据
                swe_var = ds.variables.get('snow_water_equivalent')
                if swe_var:
                    print(f"   雪水当量形状: {swe_var.shape}")
                    
                    # 这里需要根据实际数据结构进行处理
                    # 暂时返回示例数据
                    return self._create_sample_smap_data()
                else:
                    print("⚠️ 未找到雪水当量变量")
                    return None
                    
        except ImportError:
            print("⚠️ 需要安装netCDF4: pip install netCDF4")
            return None
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return None
    
    def _create_sample_smap_data(self) -> pd.DataFrame:
        """创建示例SMAP数据（实际使用时应该读取真实数据）"""
        # 生成示例数据
        dates = pd.date_range('2000-01-01', '2024-12-31', freq='D')
        
        data = []
        for date in dates:
            # 模拟季节性雪水当量变化
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 50 + 30 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            random_variation = np.random.normal(0, 10)
            
            swe = max(0, seasonal_factor + random_variation)
            
            data.append({
                'date': date,
                'snow_water_equivalent_mm': swe,
                'data_source': 'SMAP',
                'latitude': 55.0,  # Manitoba中心纬度
                'longitude': -95.0  # Manitoba中心经度
            })
        
        return pd.DataFrame(data)
    
    def process_amsr2_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """处理AMSR2数据"""
        try:
            import netCDF4 as nc
            
            print(f"🔄 处理AMSR2数据: {filepath}")
            
            # 读取NetCDF文件
            with nc.Dataset(filepath, 'r') as ds:
                # 获取变量信息
                print(f"   变量: {list(ds.variables.keys())}")
                
                # 这里需要根据实际数据结构进行处理
                # 暂时返回示例数据
                return self._create_sample_amsr2_data()
                
        except ImportError:
            print("⚠️ 需要安装netCDF4: pip install netCDF4")
            return None
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return None
    
    def _create_sample_amsr2_data(self) -> pd.DataFrame:
        """创建示例AMSR2数据"""
        dates = pd.date_range('2000-01-01', '2024-12-31', freq='D')
        
        data = []
        for date in dates:
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 45 + 25 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            random_variation = np.random.normal(0, 8)
            
            swe = max(0, seasonal_factor + random_variation)
            
            data.append({
                'date': date,
                'snow_water_equivalent_mm': swe,
                'data_source': 'AMSR2',
                'latitude': 55.0,
                'longitude': -95.0
            })
        
        return pd.DataFrame(data)
    
    def merge_all_datasets(self) -> pd.DataFrame:
        """合并所有数据集"""
        print("🔄 合并所有数据集")
        
        # 读取现有数据
        existing_data = []
        
        # 读取ECCC数据
        eccc_path = "data/processed/eccc_manitoba_snow_processed.csv"
        if os.path.exists(eccc_path):
            try:
                eccc_data = pd.read_csv(eccc_path, index_col=0, parse_dates=True)
                eccc_data['data_source'] = 'ECCC'
                existing_data.append(eccc_data)
                print(f"✅ 加载ECCC数据: {len(eccc_data)} 条记录")
            except Exception as e:
                print(f"⚠️ 加载ECCC数据失败: {e}")
                # 尝试读取修复后的数据
                eccc_fixed_path = "data/processed/eccc_manitoba_snow_fixed.csv"
                if os.path.exists(eccc_fixed_path):
                    eccc_data = pd.read_csv(eccc_fixed_path, parse_dates=['date'])
                    eccc_data['data_source'] = 'ECCC'
                    existing_data.append(eccc_data)
                    print(f"✅ 加载修复后的ECCC数据: {len(eccc_data)} 条记录")
        
        # 读取HYDAT数据
        hydat_path = "data/processed/hydat_streamflow_processed.csv"
        if os.path.exists(hydat_path):
            try:
                hydat_data = pd.read_csv(hydat_path, index_col=0, parse_dates=True)
                hydat_data['data_source'] = 'HYDAT'
                existing_data.append(hydat_data)
                print(f"✅ 加载HYDAT数据: {len(hydat_data)} 条记录")
            except Exception as e:
                print(f"⚠️ 加载HYDAT数据失败: {e}")
                # 尝试读取修复后的数据
                hydat_fixed_path = "data/processed/hydat_streamflow_fixed.csv"
                if hydat_fixed_path and os.path.exists(hydat_fixed_path):
                    hydat_data = pd.read_csv(hydat_fixed_path, parse_dates=['date'])
                    hydat_data['data_source'] = 'HYDAT'
                    existing_data.append(hydat_data)
                    print(f"✅ 加载修复后的HYDAT数据: {len(hydat_data)} 条记录")
        
        # 读取NASA数据（如果存在）
        nasa_files = [f for f in os.listdir(self.processed_dir) if f.endswith('.csv')]
        for nasa_file in nasa_files:
            nasa_path = os.path.join(self.processed_dir, nasa_file)
            try:
                nasa_data = pd.read_csv(nasa_path, parse_dates=['date'])
                nasa_data['data_source'] = 'NASA'
                existing_data.append(nasa_data)
                print(f"✅ 加载NASA数据: {len(nasa_data)} 条记录")
            except Exception as e:
                print(f"⚠️ 加载NASA数据失败: {e}")
        
        if not existing_data:
            print("❌ 没有找到任何数据")
            return pd.DataFrame()
        
        # 合并数据
        print(f"\n🔄 开始合并 {len(existing_data)} 个数据集...")
        
        # 标准化列名和数据结构
        standardized_data = []
        for df in existing_data:
            # 确保所有数据集都有必要的列
            required_columns = ['date', 'snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm']
            
            # 检查并添加缺失的列
            for col in required_columns:
                if col not in df.columns:
                    if col == 'date':
                        # 如果没有日期列，尝试从索引创建
                        if df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
                            df = df.reset_index()
                            df['date'] = df.index
                        else:
                            # 创建默认日期
                            df['date'] = pd.date_range('2000-01-01', periods=len(df), freq='D')
                    else:
                        # 其他列设为0
                        df[col] = 0
            
            # 确保数据类型正确
            df['date'] = pd.to_datetime(df['date'])
            df['snow_depth_mm'] = pd.to_numeric(df['snow_depth_mm'], errors='coerce').fillna(0)
            df['snow_fall_mm'] = pd.to_numeric(df['snow_fall_mm'], errors='coerce').fillna(0)
            df['snow_water_equivalent_mm'] = pd.to_numeric(df['snow_water_equivalent_mm'], errors='coerce').fillna(0)
            
            # 添加时间特征
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            
            # 选择标准列
            standard_columns = ['date', 'snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                               'day_of_year', 'month', 'year', 'data_source']
            
            df_standardized = df[standard_columns].copy()
            standardized_data.append(df_standardized)
        
        # 合并所有数据
        merged_data = pd.concat(standardized_data, ignore_index=True)
        
        # 去重和排序
        if 'date' in merged_data.columns:
            merged_data = merged_data.drop_duplicates(subset=['date']).sort_values('date')
        
        print(f"✅ 数据合并完成: {len(merged_data)} 条记录")
        
        # 保存合并后的数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"nasa_extended_dataset_{timestamp}.csv"
        output_path = os.path.join(self.processed_dir, output_file)
        
        merged_data.to_csv(output_path, index=False)
        print(f"✅ 扩展数据集已保存: {output_path}")
        print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return merged_data
    
    def save_merged_dataset(self, data: pd.DataFrame, filename: str = None):
        """保存合并后的数据集"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extended_training_dataset_{timestamp}.csv"
        
        filepath = os.path.join(self.processed_dir, filename)
        
        try:
            data.to_csv(filepath)
            print(f"✅ 数据集已保存: {filepath}")
            print(f"   文件大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def generate_download_report(self, downloaded_files: List[str]) -> Dict[str, Any]:
        """生成下载报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(downloaded_files),
            'successful_downloads': len([f for f in downloaded_files if f]),
            'failed_downloads': len([f for f in downloaded_files if not f]),
            'file_details': [],
            'data_sources': list(self.datasets.keys()),
            'target_region': self.target_bbox
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
    print("🚀 NASA雪数据下载器启动")
    print("=" * 50)
    
    downloader = NASASnowDataDownloader()
    
    # 下载所有数据集
    all_downloaded_files = []
    
    for dataset_key in downloader.datasets.keys():
        print(f"\n🎯 开始下载数据集: {dataset_key}")
        downloaded_files = downloader.download_dataset(dataset_key, max_files=50)
        all_downloaded_files.extend(downloaded_files)
    
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
    
    # 合并数据集
    print(f"\n🔄 开始合并数据集...")
    merged_data = downloader.merge_all_datasets()
    
    if not merged_data.empty:
        downloader.save_merged_dataset(merged_data)
        print(f"✅ 扩展数据集创建完成，总记录数: {len(merged_data)}")
    else:
        print("❌ 数据集合并失败")

if __name__ == "__main__":
    main()
