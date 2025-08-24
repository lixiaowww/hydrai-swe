#!/usr/bin/env python3
"""
数据扩展主脚本
协调所有数据源的下载和合并，解决数据不足问题
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class DataSourceExpander:
    """数据源扩展器"""
    
    def __init__(self):
        self.data_dir = "data"
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.extended_dir = os.path.join(self.processed_dir, "extended")
        
        # 创建目录
        os.makedirs(self.extended_dir, exist_ok=True)
        
        # 数据源配置
        self.data_sources = {
            'nasa': {
                'name': 'NASA雪数据',
                'script': 'download_nasa_snow_data.py',
                'description': 'SMAP、AMSR2、GlobSnow等卫星雪水当量数据',
                'priority': 'high'
            },
            'era5': {
                'name': 'ERA5再分析数据',
                'script': 'download_era5_extended.py',
                'description': 'ECMWF再分析气象、雪、土壤数据',
                'priority': 'high'
            },
            'noaa': {
                'name': 'NOAA气象数据',
                'script': 'download_noaa_extended.py',
                'description': '美国国家海洋和大气管理局气象数据',
                'priority': 'medium'
            },
            'canada': {
                'name': '加拿大环境数据',
                'script': 'download_canada_environment.py',
                'description': '加拿大环境部雪、气象、水文数据',
                'priority': 'high'
            }
        }
        
        # 目标区域（Manitoba附近）
        self.target_region = {
            'name': 'Manitoba Region',
            'bbox': [-102.0, 49.0, -88.0, 60.0],  # [西, 南, 东, 北]
            'center': [54.5, -95.0],  # [纬度, 经度]
            'area_km2': 647797  # Manitoba面积
        }
    
    def analyze_current_data(self) -> Dict[str, Any]:
        """分析当前数据状况"""
        print("🔍 分析当前数据状况")
        print("=" * 50)
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'current_data_sources': [],
            'data_volume': {},
            'time_coverage': {},
            'quality_issues': [],
            'recommendations': []
        }
        
        # 检查现有数据
        existing_datasets = [
            'eccc_manitoba_snow_processed.csv',
            'hydat_streamflow_processed.csv',
            'comprehensive_training_dataset.csv',
            'standardized_training_dataset.csv'
        ]
        
        total_records = 0
        total_size_mb = 0
        
        for dataset in existing_datasets:
            filepath = os.path.join(self.processed_dir, dataset)
            if os.path.exists(filepath):
                try:
                    # 读取数据
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    records = len(df)
                    size_mb = os.path.getsize(filepath) / 1024 / 1024
                    
                    # 时间范围
                    if 'date' in df.columns:
                        date_col = 'date'
                    else:
                        date_col = df.index.name if df.index.name else df.index[0]
                    
                    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                        time_range = f"{df[date_col].min()} - {df[date_col].max()}"
                        days_covered = (df[date_col].max() - df[date_col].min()).days
                    else:
                        time_range = "Unknown"
                        days_covered = 0
                    
                    dataset_info = {
                        'name': dataset,
                        'records': records,
                        'size_mb': size_mb,
                        'time_range': time_range,
                        'days_covered': days_covered
                    }
                    
                    analysis['current_data_sources'].append(dataset_info)
                    total_records += records
                    total_size_mb += size_mb
                    
                    print(f"✅ {dataset}: {records:,} 条记录, {size_mb:.2f} MB, {time_range}")
                    
                except Exception as e:
                    print(f"❌ 读取 {dataset} 失败: {e}")
                    analysis['quality_issues'].append(f"无法读取 {dataset}: {e}")
        
        # 总体统计
        analysis['data_volume'] = {
            'total_records': total_records,
            'total_size_mb': total_size_mb,
            'datasets_count': len(analysis['current_data_sources'])
        }
        
        # 数据质量评估
        if total_records < 10000:
            analysis['quality_issues'].append("数据量不足：总记录数少于10,000条")
            analysis['recommendations'].append("需要下载更多数据源")
        
        if total_records < 50000:
            analysis['quality_issues'].append("数据量有限：总记录数少于50,000条")
            analysis['recommendations'].append("建议扩展到更多年份和地区")
        
        # 时间覆盖评估
        if analysis['current_data_sources']:
            max_days = max([ds['days_covered'] for ds in analysis['current_data_sources']])
            if max_days < 365 * 10:  # 少于10年
                analysis['quality_issues'].append("时间覆盖不足：少于10年的数据")
                analysis['recommendations'].append("需要更长期的历史数据")
        
        print(f"\n📊 总体统计:")
        print(f"   总记录数: {total_records:,}")
        print(f"   总大小: {total_size_mb:.2f} MB")
        print(f"   数据集数量: {len(analysis['current_data_sources'])}")
        
        return analysis
    
    def check_data_source_availability(self) -> Dict[str, Any]:
        """检查数据源可用性"""
        print("\n🔍 检查数据源可用性")
        print("=" * 50)
        
        availability = {
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        for source_key, source_info in self.data_sources.items():
            print(f"\n🎯 检查数据源: {source_info['name']}")
            
            # 检查脚本是否存在
            script_path = source_info['script']
            script_exists = os.path.exists(script_path)
            
            # 检查依赖
            dependencies = self._check_dependencies(source_key)
            
            # 检查配置
            config_status = self._check_configuration(source_key)
            
            source_status = {
                'name': source_info['name'],
                'priority': source_info['priority'],
                'script_exists': script_exists,
                'dependencies_met': dependencies['all_met'],
                'config_ready': config_status['ready'],
                'dependencies': dependencies,
                'config': config_status,
                'status': 'ready' if script_exists and dependencies['all_met'] and config_status['ready'] else 'not_ready'
            }
            
            availability['sources'][source_key] = source_status
            
            # 显示状态
            status_emoji = "✅" if source_status['status'] == 'ready' else "❌"
            print(f"   {status_emoji} 状态: {source_status['status']}")
            print(f"   脚本: {'✅' if script_exists else '❌'} {script_path}")
            print(f"   依赖: {'✅' if dependencies['all_met'] else '❌'} {dependencies['missing'] if dependencies['missing'] else '全部满足'}")
            print(f"   配置: {'✅' if config_status['ready'] else '❌'} {config_status['issues'] if config_status['issues'] else '配置正确'}")
        
        return availability
    
    def _check_dependencies(self, source_key: str) -> Dict[str, Any]:
        """检查数据源依赖"""
        dependencies = {
            'all_met': True,
            'missing': [],
            'details': {}
        }
        
        if source_key == 'nasa':
            # NASA数据源依赖
            required_packages = ['requests', 'netCDF4', 'xarray']
            for package in required_packages:
                try:
                    __import__(package)
                    dependencies['details'][package] = 'installed'
                except ImportError:
                    dependencies['details'][package] = 'missing'
                    dependencies['missing'].append(package)
                    dependencies['all_met'] = False
        
        elif source_key == 'era5':
            # ERA5数据源依赖
            required_packages = ['cdsapi', 'netCDF4', 'xarray']
            for package in required_packages:
                try:
                    __import__(package)
                    dependencies['details'][package] = 'installed'
                except ImportError:
                    dependencies['details'][package] = 'missing'
                    dependencies['missing'].append(package)
                    dependencies['all_met'] = False
        
        return dependencies
    
    def _check_configuration(self, source_key: str) -> Dict[str, Any]:
        """检查数据源配置"""
        config = {
            'ready': False,
            'issues': [],
            'details': {}
        }
        
        if source_key == 'nasa':
            # NASA数据源配置检查
            config['ready'] = True  # 暂时不需要特殊配置
            config['details']['api_key'] = 'not_required'
        
        elif source_key == 'era5':
            # ERA5数据源配置检查
            cds_config_path = os.path.expanduser("~/.cdsapirc")
            if os.path.exists(cds_config_path):
                config['ready'] = True
                config['details']['cds_config'] = 'found'
            else:
                config['ready'] = False
                config['issues'].append("缺少CDS API配置文件")
                config['details']['cds_config'] = 'missing'
        
        return config
    
    def download_data_source(self, source_key: str) -> bool:
        """下载指定数据源"""
        if source_key not in self.data_sources:
            print(f"❌ 未知数据源: {source_key}")
            return False
        
        source_info = self.data_sources[source_key]
        script_path = source_info['script']
        
        if not os.path.exists(script_path):
            print(f"❌ 脚本不存在: {script_path}")
            return False
        
        print(f"🚀 开始下载数据源: {source_info['name']}")
        print("=" * 50)
        
        try:
            # 执行下载脚本
            import subprocess
            result = subprocess.run([sys.executable, script_path], 
                                 capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"✅ 数据源下载成功: {source_info['name']}")
                print(f"   输出: {result.stdout}")
                return True
            else:
                print(f"❌ 数据源下载失败: {source_info['name']}")
                print(f"   错误: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ 数据源下载超时: {source_info['name']}")
            return False
        except Exception as e:
            print(f"❌ 数据源下载异常: {source_info['name']}: {e}")
            return False
    
    def merge_all_extended_data(self) -> pd.DataFrame:
        """合并所有扩展数据"""
        print("\n🔄 合并所有扩展数据")
        print("=" * 50)
        
        all_data = []
        
        # 读取现有数据
        existing_datasets = [
            'eccc_manitoba_snow_processed.csv',
            'hydat_streamflow_processed.csv'
        ]
        
        for dataset in existing_datasets:
            filepath = os.path.join(self.processed_dir, dataset)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    df['data_source'] = dataset.replace('_processed.csv', '').upper()
                    all_data.append(df)
                    print(f"✅ 加载现有数据: {dataset} ({len(df)} 条记录)")
                except Exception as e:
                    print(f"❌ 加载现有数据失败: {dataset}: {e}")
        
        # 读取扩展数据
        extended_sources = ['nasa_snow', 'era5_extended']
        
        for source in extended_sources:
            source_dir = os.path.join(self.processed_dir, source)
            if os.path.exists(source_dir):
                csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
                for csv_file in csv_files:
                    filepath = os.path.join(source_dir, csv_file)
                    try:
                        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                        df['data_source'] = source.upper()
                        all_data.append(df)
                        print(f"✅ 加载扩展数据: {csv_file} ({len(df)} 条记录)")
                    except Exception as e:
                        print(f"❌ 加载扩展数据失败: {csv_file}: {e}")
        
        if not all_data:
            print("❌ 没有找到任何数据")
            return pd.DataFrame()
        
        # 合并数据
        print(f"\n🔄 开始合并 {len(all_data)} 个数据集...")
        
        # 标准化列名
        standardized_data = []
        for df in all_data:
            # 重命名列以保持一致性
            column_mapping = {
                'snow_depth_mm': 'snow_depth_mm',
                'snow_fall_mm': 'snow_fall_mm',
                'snow_water_equivalent_mm': 'snow_water_equivalent_mm',
                'snow_depth': 'snow_depth_mm',
                'snowfall': 'snow_fall_mm',
                'snow_depth_water_equivalent': 'snow_water_equivalent_mm'
            }
            
            df_renamed = df.rename(columns=column_mapping)
            
            # 确保必要的列存在
            required_columns = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm']
            for col in required_columns:
                if col not in df_renamed.columns:
                    df_renamed[col] = np.nan
            
            standardized_data.append(df_renamed)
        
        # 合并所有数据
        merged_data = pd.concat(standardized_data, ignore_index=True)
        
        # 去重和排序
        if 'date' in merged_data.columns:
            merged_data = merged_data.drop_duplicates(subset=['date']).sort_values('date')
        
        print(f"✅ 数据合并完成: {len(merged_data)} 条记录")
        
        # 保存合并后的数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"extended_comprehensive_dataset_{timestamp}.csv"
        output_path = os.path.join(self.extended_dir, output_file)
        
        merged_data.to_csv(output_path, index=False)
        print(f"✅ 扩展数据集已保存: {output_path}")
        print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return merged_data
    
    def generate_expansion_report(self, analysis: Dict[str, Any], 
                                availability: Dict[str, Any]) -> Dict[str, Any]:
        """生成数据扩展报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'target_region': self.target_region,
            'current_data_analysis': analysis,
            'data_source_availability': availability,
            'expansion_plan': self._generate_expansion_plan(analysis, availability),
            'recommendations': []
        }
        
        # 生成建议
        if analysis['data_volume']['total_records'] < 50000:
            report['recommendations'].append("立即开始下载高优先级数据源")
        
        if not any(source['status'] == 'ready' for source in availability['sources'].values()):
            report['recommendations'].append("优先解决依赖和配置问题")
        
        report['recommendations'].extend(analysis['recommendations'])
        
        return report
    
    def _generate_expansion_plan(self, analysis: Dict[str, Any], 
                               availability: Dict[str, Any]) -> Dict[str, Any]:
        """生成数据扩展计划"""
        plan = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_goals': [],
            'priority_order': []
        }
        
        # 立即行动
        ready_sources = [key for key, source in availability['sources'].items() 
                        if source['status'] == 'ready']
        
        if ready_sources:
            plan['immediate_actions'].append(f"开始下载就绪的数据源: {', '.join(ready_sources)}")
        
        # 短期目标
        if analysis['data_volume']['total_records'] < 100000:
            plan['short_term_goals'].append("将数据量扩展到100,000条记录以上")
        
        # 长期目标
        plan['long_term_goals'].append("建立多源、长期、高质量的训练数据集")
        plan['long_term_goals'].append("实现数据的自动更新和维护")
        
        # 优先级顺序
        high_priority = [key for key, source in availability['sources'].items() 
                        if source['priority'] == 'high' and source['status'] == 'ready']
        medium_priority = [key for key, source in availability['sources'].items() 
                          if source['priority'] == 'medium' and source['status'] == 'ready']
        
        plan['priority_order'] = high_priority + medium_priority
        
        return plan

def main():
    """主函数"""
    print("🚀 数据源扩展器启动")
    print("=" * 50)
    
    expander = DataSourceExpander()
    
    # 分析当前数据状况
    analysis = expander.analyze_current_data()
    
    # 检查数据源可用性
    availability = expander.check_data_source_availability()
    
    # 生成扩展报告
    report = expander.generate_expansion_report(analysis, availability)
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(expander.extended_dir, f"expansion_report_{timestamp}.json")
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📊 扩展报告已保存: {report_path}")
    
    # 显示扩展计划
    print(f"\n📋 数据扩展计划:")
    print(f"   立即行动: {report['expansion_plan']['immediate_actions']}")
    print(f"   短期目标: {report['expansion_plan']['short_term_goals']}")
    print(f"   长期目标: {report['expansion_plan']['long_term_goals']}")
    print(f"   优先级顺序: {report['expansion_plan']['priority_order']}")
    
    # 询问是否开始下载
    if report['expansion_plan']['immediate_actions']:
        print(f"\n❓ 是否开始下载数据源？")
        print(f"   就绪的数据源: {[key for key, source in availability['sources'].items() if source['status'] == 'ready']}")
        
        # 这里可以添加用户交互逻辑
        # 暂时自动开始下载高优先级数据源
        
        print(f"\n🚀 自动开始下载高优先级数据源...")
        
        for source_key in report['expansion_plan']['priority_order'][:2]:  # 限制为前2个
            print(f"\n🎯 下载数据源: {source_key}")
            success = expander.download_data_source(source_key)
            if success:
                print(f"✅ {source_key} 下载完成")
            else:
                print(f"❌ {source_key} 下载失败")
        
        # 合并扩展数据
        print(f"\n🔄 开始合并扩展数据...")
        merged_data = expander.merge_all_extended_data()
        
        if not merged_data.empty:
            print(f"🎉 数据扩展完成！")
            print(f"   最终数据集大小: {len(merged_data):,} 条记录")
        else:
            print(f"❌ 数据扩展失败")
    
    else:
        print(f"\n⚠️ 没有就绪的数据源，请先解决依赖和配置问题")

if __name__ == "__main__":
    main()

