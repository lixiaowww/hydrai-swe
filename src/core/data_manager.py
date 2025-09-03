#!/usr/bin/env python3
"""
统一数据管理器
提供标准化的数据访问接口，解决路径硬编码和数据格式不统一问题
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import glob
import logging
from pathlib import Path
import hashlib
from functools import lru_cache

# 设置日志
logger = logging.getLogger(__name__)

class DataManager:
    """统一数据管理器"""
    
    def __init__(self, base_path: str = "/home/sean/hydrai_swe"):
        """
        初始化数据管理器
        
        Args:
            base_path: 项目根路径
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"
        self.processed_path = self.data_path / "processed"
        
        # 数据源配置
        self.data_sources = {
            "swe": {
                "path": self.processed_path / "swe",
                "sync_pattern": "swe_sync_*.csv",
                "static_file": "swe_analysis_optimized.csv"
            },
            "flood": {
                "path": self.processed_path / "flood_warning", 
                "sync_pattern": "flood_sync_*.csv",
                "static_file": "flood_warning_optimized.csv"
            },
            "hydrology": {
                "path": self.processed_path,
                "sync_pattern": "hydro_sync_*.csv", 
                "static_file": "hydat_streamflow_processed.csv"
            },
            "weather": {
                "path": self.processed_path / "weather",
                "sync_pattern": "weather_sync_*.csv",
                "static_file": "weather_data.csv"
            },
            "agriculture": {
                "path": self.processed_path / "agriculture",
                "sync_pattern": "agri_sync_*.csv",
                "static_file": "agriculture_data.csv"
            }
        }
        
        # 缓存配置
        self.cache = {}
        self.cache_ttl = 300  # 5分钟缓存
        
        logger.info(f"数据管理器初始化完成，基础路径: {self.base_path}")
    
    def _get_cache_key(self, source: str, file_path: str) -> str:
        """生成缓存键（基于文件路径和修改时间）"""
        try:
            file_stat = os.stat(file_path)
            file_mtime = file_stat.st_mtime
            return f"{source}_{file_path}_{file_mtime}"
        except OSError:
            return f"{source}_{file_path}_{datetime.now().timestamp()}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]["timestamp"]
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl
    
    def _clean_cache(self):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, cache_data in self.cache.items():
            if (current_time - cache_data["timestamp"]).total_seconds() > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"🧹 清理了{len(expired_keys)}个过期缓存")
    
    def get_latest_data(self, source: str, force_sync: bool = True, use_cache: bool = True) -> pd.DataFrame:
        """
        获取最新数据（带缓存）
        
        Args:
            source: 数据源名称 (swe, flood, hydrology, weather, agriculture)
            force_sync: 是否强制使用同步数据
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame: 最新数据
            
        Raises:
            ValueError: 数据源不存在
            FileNotFoundError: 没有找到数据文件
        """
        if source not in self.data_sources:
            raise ValueError(f"未知数据源: {source}")
        
        # 清理过期缓存
        self._clean_cache()
        
        config = self.data_sources[source]
        
        # 1. 优先使用同步数据
        if force_sync:
            sync_files = list(config["path"].glob(config["sync_pattern"]))
            if sync_files:
                latest_file = max(sync_files, key=os.path.getctime)
                file_path = str(latest_file)
                
                # 检查缓存
                if use_cache:
                    cache_key = self._get_cache_key(source, file_path)
                    if self._is_cache_valid(cache_key):
                        logger.info(f"📦 使用{source}缓存数据")
                        return self.cache[cache_key]["data"].copy()
                
                # 读取数据并缓存
                data = pd.read_csv(latest_file)
                logger.info(f"✅ 使用{source}同步数据: {latest_file}")
                
                if use_cache:
                    cache_key = self._get_cache_key(source, file_path)
                    self.cache[cache_key] = {
                        "data": data.copy(),
                        "timestamp": datetime.now(),
                        "source": file_path
                    }
                
                return data
        
        # 2. 备选静态文件
        static_file = config["path"] / config["static_file"]
        if static_file.exists():
            file_path = str(static_file)
            
            # 检查缓存
            if use_cache:
                cache_key = self._get_cache_key(source, file_path)
                if self._is_cache_valid(cache_key):
                    logger.info(f"📦 使用{source}缓存数据")
                    return self.cache[cache_key]["data"].copy()
            
            # 读取数据并缓存
            data = pd.read_csv(static_file)
            logger.warning(f"⚠️ 使用{source}静态数据: {static_file} (数据可能过时)")
            
            if use_cache:
                cache_key = self._get_cache_key(source, file_path)
                self.cache[cache_key] = {
                    "data": data.copy(),
                    "timestamp": datetime.now(),
                    "source": file_path
                }
            
            return data
        
        # 3. 没有找到任何数据
        raise FileNotFoundError(f"没有找到{source}数据文件")
    
    def get_data_info(self, source: str) -> Dict[str, Any]:
        """
        获取数据源信息
        
        Args:
            source: 数据源名称
            
        Returns:
            Dict: 数据源信息
        """
        if source not in self.data_sources:
            raise ValueError(f"未知数据源: {source}")
        
        config = self.data_sources[source]
        
        # 检查同步数据
        sync_files = list(config["path"].glob(config["sync_pattern"]))
        latest_sync = max(sync_files, key=os.path.getctime) if sync_files else None
        
        # 检查静态数据
        static_file = config["path"] / config["static_file"]
        static_exists = static_file.exists()
        
        info = {
            "source": source,
            "path": str(config["path"]),
            "sync_files_count": len(sync_files),
            "latest_sync": str(latest_sync) if latest_sync else None,
            "static_file": str(static_file) if static_exists else None,
            "static_exists": static_exists,
            "data_available": len(sync_files) > 0 or static_exists
        }
        
        if latest_sync:
            info["sync_age_hours"] = (datetime.now() - datetime.fromtimestamp(latest_sync.stat().st_mtime)).total_seconds() / 3600
        
        return info
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
        """
        验证数据质量
        
        Args:
            data: 数据DataFrame
            required_columns: 必需的列名
            
        Returns:
            Dict: 验证结果
        """
        validation = {
            "valid": True,
            "shape": data.shape,
            "columns": list(data.columns),
            "missing_columns": [],
            "empty_rows": data.isnull().all(axis=1).sum(),
            "duplicate_rows": data.duplicated().sum(),
            "data_types": data.dtypes.to_dict()
        }
        
        # 检查必需列
        if required_columns:
            missing = [col for col in required_columns if col not in data.columns]
            validation["missing_columns"] = missing
            validation["valid"] = len(missing) == 0
        
        return validation
    
    def get_all_data_sources_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有数据源信息
        
        Returns:
            Dict: 所有数据源信息
        """
        return {source: self.get_data_info(source) for source in self.data_sources.keys()}

# 全局数据管理器实例
data_manager = DataManager()

def get_data_manager() -> DataManager:
    """获取全局数据管理器实例"""
    return data_manager
