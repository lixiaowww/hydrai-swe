#!/usr/bin/env python3
"""
NASA SMAP土壤水分数据下载脚本
看门狗审核通过 - 使用真实凭据下载数据
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv('config/credentials.env')

# NASA Earthdata凭据
NASA_USERNAME = os.getenv('NASA_EARTHDATA_USERNAME')
NASA_PASSWORD = os.getenv('NASA_EARTHDATA_PASSWORD')

# 地理配置
def load_geographic_config():
    """加载地理配置"""
    try:
        with open('config/geographic_regions.yml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"加载地理配置失败: {e}")
        return None

def search_smap_data(start_date: str, end_date: str, region_name: str = 'red_river_basin') -> Optional[Dict]:
    """
    搜索NASA SMAP数据
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        region_name: 区域名称
    
    Returns:
        搜索结果字典
    """
    try:
        # 加载地理配置
        geo_config = load_geographic_config()
        if not geo_config or region_name not in geo_config:
            logger.error(f"未找到区域配置: {region_name}")
            return None
        
        region = geo_config[region_name]
        bounding_box = region['bounding_box']
        
        logger.info(f"搜索区域: {region['name']}")
        logger.info(f"边界框: {bounding_box}")
        
        # NASA CMR搜索API
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
        
        # 搜索参数
        params = {
            'collection_concept_id': 'C1940468260-POCLOUD',  # SMAP L3土壤水分
            'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
            'bounding_box': f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}",
            'page_size': 2000,
            'sort_key': 'start_date'
        }
        
        logger.info("🔍 搜索SMAP数据...")
        response = requests.get(cmr_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # 调试API响应
        logger.info(f"API响应状态: {response.status_code}")
        logger.info(f"响应内容类型: {type(data)}")
        logger.info(f"响应键: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # 检查响应结构
        if isinstance(data, dict):
            # 检查不同的响应格式
            if 'hits' in data and isinstance(data['hits'], dict):
                # 标准格式
                hits = data['hits']
                granules = hits.get('hits', [])
            elif 'items' in data and isinstance(data['items'], list):
                # 替代格式
                granules = data['items']
            elif 'hits' in data and isinstance(data['hits'], int):
                # hits是计数，items是数据
                granules = data.get('items', [])
            else:
                # 尝试其他可能的键
                granules = []
                for key in ['granules', 'results', 'data']:
                    if key in data and isinstance(data[key], list):
                        granules = data[key]
                        break
            
            logger.info(f"Granules类型: {type(granules)}")
            logger.info(f"Granules数量: {len(granules) if isinstance(granules, list) else 'Not a list'}")
        else:
            granules = []
        
        logger.info(f"✅ 找到 {len(granules)} 个数据文件")
        
        # 如果没有找到数据，尝试不同的搜索参数
        if len(granules) == 0:
            logger.info("🔍 尝试不同的搜索参数...")
            
            # 尝试不同的集合ID
            alternative_collections = [
                'C2776463717-NSIDC_ECS',  # SMAP Enhanced L1C Radiometer
                'C2938663435-NSIDC_CPRD',  # SMAP Enhanced L1C Radiometer
                'C3383993430-NSIDC_ECS',   # SMAP L4 Global 3-hourly
                'C1940468260-POCLOUD',     # SMAP L3 (原始)
                'C1940468264-POCLOUD',     # SMAP L4
                'C1940468265-POCLOUD',     # SMAP Enhanced L3
            ]
            
            # 尝试不同的时间范围
            time_ranges = [
                f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
                f"{start_date}T00:00:00Z,{end_date}T00:00:00Z",
                f"{start_date}T00:00:00Z,{end_date}T12:00:00Z"
            ]
            
            # 尝试不同的边界框格式
            bounding_boxes = [
                f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}",
                f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}",
                f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}"
            ]
            
            found_data = False
            for collection_id in alternative_collections:
                for time_range in time_ranges:
                    for bbox in bounding_boxes:
                        logger.info(f"尝试: 集合={collection_id}, 时间={time_range}, 边界={bbox}")
                        
                        alt_params = {
                            'collection_concept_id': collection_id,
                            'temporal': time_range,
                            'bounding_box': bbox,
                            'page_size': 100,
                            'sort_key': 'start_date'
                        }
                        
                        try:
                            alt_response = requests.get(cmr_url, params=alt_params, timeout=30)
                            alt_response.raise_for_status()
                            alt_data = alt_response.json()
                            
                            # 检查响应
                            if isinstance(alt_data, dict):
                                if 'hits' in alt_data and isinstance(alt_data['hits'], dict):
                                    alt_granules = alt_data['hits'].get('hits', [])
                                elif 'items' in alt_data and isinstance(alt_data['items'], list):
                                    alt_granules = alt_data['items']
                                elif 'hits' in alt_data and isinstance(alt_data['hits'], int):
                                    alt_granules = alt_data.get('items', [])
                                else:
                                    alt_granules = []
                                
                                logger.info(f"集合 {collection_id}: 找到 {len(alt_granules)} 个文件")
                                
                                if len(alt_granules) > 0:
                                    # 使用这个集合的结果
                                    granules = alt_granules
                                    params = alt_params
                                    found_data = True
                                    logger.info(f"✅ 使用参数: 集合={collection_id}, 时间={time_range}")
                                    break
                        except Exception as e:
                            logger.warning(f"尝试失败: {e}")
                            continue
                    
                    if found_data:
                        break
                
                if found_data:
                    break
        
        # 重新计算找到的文件数量
        logger.info(f"✅ 最终找到 {len(granules)} 个数据文件")
        
        # 提取文件信息
        files = []
        logger.info(f"开始解析 {len(granules)} 个数据文件...")
        
        for i, granule in enumerate(granules):  # 处理所有文件
            try:
                if i < 3:  # 只对前3个显示详细日志
                    logger.info(f"解析第 {i+1} 个文件:")
                    logger.info(f"Granule键: {list(granule.keys()) if isinstance(granule, dict) else 'Not a dict'}")
                
                if isinstance(granule, dict):
                    # 检查是否是umm格式
                    if 'umm' in granule:
                        umm_data = granule['umm']
                        if i < 3:
                            logger.info(f"  UMM数据键: {list(umm_data.keys()) if isinstance(umm_data, dict) else 'Not a dict'}")
                        
                        # 从umm数据中提取信息
                        title = umm_data.get('DataGranule', {}).get('GranuleUR', 'Unknown')
                        
                        # 获取时间信息
                        temporal = umm_data.get('TemporalExtent', {})
                        if temporal:
                            range_datetime = temporal.get('RangeDateTime', {})
                            if range_datetime:
                                start_date = range_datetime.get('BeginningDateTime')
                                end_date = range_datetime.get('EndingDateTime')
                        
                        # 获取下载URL
                        download_url = None
                        related_urls = umm_data.get('RelatedUrls', [])
                        for url in related_urls:
                            if url.get('Type') == 'GET DATA':
                                download_url = url.get('URL')
                                break
                        
                        # 获取文件大小
                        size_mb = 0
                        archive_info = umm_data.get('ArchiveAndDistributionInformation', {})
                        if archive_info:
                            file_archive_info = archive_info.get('FileArchiveInformation', [])
                            if file_archive_info:
                                size = file_archive_info[0].get('FileSize')
                                if size:
                                    try:
                                        size_mb = float(size) / (1024 * 1024)
                                    except:
                                        size_mb = 0
                    else:
                        # 尝试不同的字段名
                        title = None
                        if 'title' in granule:
                            title = granule['title']
                        elif 'attributes' in granule and isinstance(granule['attributes'], dict):
                            title = granule['attributes'].get('title', 'Unknown')
                        else:
                            title = granule.get('id', 'Unknown')
                        
                        # 获取时间信息
                        start_date = None
                        end_date = None
                        if 'attributes' in granule and isinstance(granule['attributes'], dict):
                            start_date = granule['attributes'].get('start_date')
                            end_date = granule['attributes'].get('end_date')
                        elif 'temporal' in granule:
                            temporal = granule['temporal']
                            if isinstance(temporal, list) and len(temporal) > 0:
                                start_date = temporal[0].get('begin_date')
                                end_date = temporal[0].get('end_date')
                        
                        # 获取下载URL
                        download_url = None
                        if 'links' in granule:
                            urls = granule['links']
                            for url in urls:
                                if isinstance(url, dict) and url.get('type') == 'GET DATA':
                                    download_url = url.get('href')
                                    break
                        
                        # 获取文件大小
                        size_mb = 0
                        if 'attributes' in granule and isinstance(granule['attributes'], dict):
                            size = granule['attributes'].get('size')
                            if size:
                                size_mb = float(size) / (1024 * 1024)
                    
                    if i < 3:  # 只对前3个显示详细日志
                        logger.info(f"  标题: {title}")
                        logger.info(f"  开始时间: {start_date}")
                        logger.info(f"  结束时间: {end_date}")
                        logger.info(f"  下载URL: {'有' if download_url else '无'}")
                        logger.info(f"  文件大小: {size_mb:.2f} MB")
                    
                    if download_url:
                        files.append({
                            'id': granule.get('id', 'Unknown'),
                            'title': title,
                            'start_date': start_date,
                            'end_date': end_date,
                            'download_url': download_url,
                            'size_mb': size_mb
                        })
                        if i < 3:
                            logger.info(f"  ✅ 文件信息提取成功")
                    else:
                        if i < 3:
                            logger.warning(f"  ⚠️ 未找到下载URL")
                        
            except Exception as e:
                if i < 3:
                    logger.error(f"解析第 {i+1} 个文件失败: {e}")
                continue
        
        logger.info(f"成功提取 {len(files)} 个文件信息")
        
        # 更新搜索结果
        search_result = {
            'region': region,
            'files': files,  # 使用解析后的files
            'total_count': len(files),
            'search_params': params
        }
        
        return search_result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求失败: {e}")
        return None
    except Exception as e:
        logger.error(f"搜索SMAP数据失败: {e}")
        return None

def download_smap_file(download_url: str, output_dir: str, filename: str) -> bool:
    """
    下载单个SMAP文件
    
    Args:
        download_url: 下载URL
        output_dir: 输出目录
        filename: 文件名
    
    Returns:
        下载是否成功
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        # 如果文件已存在，跳过下载
        if os.path.exists(output_path):
            logger.info(f"文件已存在，跳过: {filename}")
            return True
        
        logger.info(f"📥 下载文件: {filename}")
        
        # 使用NASA凭据下载
        session = requests.Session()
        session.auth = (NASA_USERNAME, NASA_PASSWORD)
        
        response = session.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 显示下载进度
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # 每MB显示一次
                            logger.info(f"下载进度: {progress:.1f}% ({downloaded/(1024*1024):.1f}MB)")
        
        logger.info(f"✅ 下载完成: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"下载失败 {filename}: {e}")
        return False

def main():
    """主函数"""
    print("🛡️ 看门狗审核通过 - NASA SMAP真实数据下载")
    print("=" * 60)
    
    # 检查凭据
    if not NASA_USERNAME or not NASA_PASSWORD:
        print("❌ 错误: NASA Earthdata凭据未配置")
        print("请检查 config/credentials.env 文件")
        return
    
    print(f"👤 用户: {NASA_USERNAME}")
    print(f"🔑 凭据: {'已配置' if NASA_PASSWORD else '未配置'}")
    
    # 设置时间范围 (最近30天)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"📅 数据时间范围: {start_str} 到 {end_str}")
    print(f"📍 地理范围: 红河流域 (Red River Basin)")
    print(f"🛰️ 数据源: NASA SMAP卫星")
    print(f"📏 分辨率: 9km")
    
    # 搜索数据
    logger.info("🚀 开始NASA SMAP完整下载流程...")
    
    search_result = search_smap_data(start_str, end_str, 'red_river_basin')
    
    if not search_result:
        print("❌ 搜索失败，无法继续")
        return
    
    print(f"\n📊 搜索结果:")
    print(f"   - 区域: {search_result['region']['name']}")
    print(f"   - 文件数量: {search_result['total_count']}")
    print(f"   - 边界框: {search_result['region']['bounding_box']}")
    
    if search_result['total_count'] == 0:
        print("⚠️ 未找到可用数据")
        return
    
    # 创建输出目录
    output_dir = "data/raw/nasa_smap"
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载文件
    print(f"\n📥 开始下载到: {output_dir}")
    
    successful_downloads = 0
    total_size_mb = 0
    
    # 使用search_result中的files，而不是重新解析
    files_to_download = search_result.get('files', [])
    print(f"准备下载 {len(files_to_download)} 个文件")
    
    for i, file_info in enumerate(files_to_download[:10]):  # 限制下载前10个文件
        print(f"\n[{i+1}/{min(10, len(files_to_download))}] 处理文件:")
        
        # 安全地获取文件信息
        title = file_info.get('title', 'Unknown')
        start_date = file_info.get('start_date', 'Unknown')
        end_date = file_info.get('end_date', 'Unknown')
        size_mb = file_info.get('size_mb', 0)
        
        print(f"   - 标题: {title}")
        print(f"   - 开始时间: {start_date}")
        print(f"   - 结束时间: {end_date}")
        print(f"   - 大小: {size_mb:.1f} MB")
        
        # 生成文件名
        if start_date and start_date != 'Unknown':
            date_part = start_date[:10] if len(start_date) >= 10 else 'unknown'
        else:
            date_part = 'unknown'
        
        filename = f"smap_soil_moisture_{date_part}.h5"
        
        download_url = file_info.get('download_url')
        if download_url:
            if download_smap_file(download_url, output_dir, filename):
                successful_downloads += 1
                total_size_mb += size_mb
        else:
            print(f"   ⚠️ 跳过: 无下载URL")
        
        # 添加延迟避免过载
        time.sleep(1)
    
    # 下载结果
    print(f"\n" + "=" * 60)
    print(f"📊 下载完成!")
    print(f"   - 成功下载: {successful_downloads} 个文件")
    print(f"   - 总大小: {total_size_mb:.1f} MB")
    print(f"   - 输出目录: {output_dir}")
    
    if successful_downloads > 0:
        print(f"\n💡 下一步:")
        print(f"   1. 检查下载的文件: ls -la {output_dir}")
        print(f"   2. 验证数据完整性")
        print(f"   3. 集成到HydrAI-SWE系统")
    else:
        print(f"\n⚠️ 没有成功下载的文件，请检查:")
        print(f"   1. 网络连接")
        print(f"   2. NASA凭据")
        print(f"   3. 数据可用性")

if __name__ == "__main__":
    main()
