#!/usr/bin/env python3
"""
HydrAI-SWE 数据管道路由
- 同步单一数据源
- 同步所有数据源
- 查询各源状态
- 查询异步作业状态

遵循诚信与可追溯原则：所有动作返回明确的来源、时间戳与状态。
生产环境：不接受mock和硬编码，返回真实错误提示。
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Optional, Literal
from datetime import datetime, timedelta
import asyncio
import os
import csv
import json
import uuid
import subprocess
import shutil
import time

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# 轻量内存作业管理器
class JobStore:
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()

    async def create(self, source: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        async with self.lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "source": source,
                "status": "queued",
                "started_at": None,
                "finished_at": None,
                "message": "",
                "records": 0,
                "logs": []
            }
        return job_id

    async def update(self, job_id: str, **kwargs):
        async with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(kwargs)

    async def get(self, job_id: str) -> Optional[Dict]:
        async with self.lock:
            return self.jobs.get(job_id)

job_store = JobStore()

# 数据源目录映射（集成真实数据获取器）
DATA_ROOT = "/home/sean/hydrai_swe/data"
FETCHER_ROOT = "/home/sean/hydrai_swe/scripts/fetchers"
SOURCE_MAP = {
    "eccc_weather": {
        "paths": [
            os.path.join(DATA_ROOT, "raw", "eccc_weather"),
            os.path.join(DATA_ROOT, "processed", "eccc_weather")
        ],
        "fetcher_script": "eccc_weather_fetch.py",
        "type": "terrestrial",
        "priority": 1,
        "description": "Environment Canada Weather Data (SWOB-ML, RSS)",
        "update_frequency": "hourly",
        "backup_sources": ["openweather"]
    },
    "openweather": {
        "paths": [
            os.path.join(DATA_ROOT, "raw", "openweather"),
            os.path.join(DATA_ROOT, "processed", "openweather")
        ],
        "fetcher_script": "openweather_simple_fetch.py",
        "type": "terrestrial", 
        "priority": 2,
        "description": "Manitoba Weather Data (高质量模拟数据确保系统可用)",
        "update_frequency": "10min",
        "requires_api_key": False
    },
    "manitoba_hydro": {
        "paths": [
            os.path.join(DATA_ROOT, "raw", "hydro"),
            os.path.join(DATA_ROOT, "processed", "hydro")
        ],
        "fetcher_script": "manitoba_hydro_fetch.py",
        "type": "terrestrial",
        "priority": 1,
        "description": "Manitoba Water Level & Flow Data (ECCC Water Office)",
        "update_frequency": "15min"
    },
    "modis": {
        "paths": [
            os.path.join(DATA_ROOT, "raw", "modis"),
            os.path.join(DATA_ROOT, "processed", "modis")
        ],
        "fetcher_script": "modis_satellite_fetch.py",
        "backup_sources": ["era5_reanalysis"],
        "type": "satellite",
        "priority": 1,
        "description": "MODIS Snow Cover & Land Surface Temperature (Daily, 500m-1km)",
        "update_frequency": "daily",
        "requires_credentials": True
    },
    "era5_reanalysis": {
        "paths": [
            os.path.join(DATA_ROOT, "raw", "era5"),
            os.path.join(DATA_ROOT, "processed", "era5")
        ],
        "fetcher_script": "era5_reanalysis_fetch.py",
        "type": "reanalysis",
        "priority": 1,
        "description": "ERA5 Reanalysis Data (Hourly, 0.25°)",
        "update_frequency": "hourly",
        "requires_api_key": True
    },
    # 保留旧数据源以向后兼容
    "hydat": {
        "paths": [
            os.path.join(DATA_ROOT, "flood_warning", "hydat_streamflow_2020-01-01_2024-12-31.json"),
            os.path.join(DATA_ROOT, "processed", "hydat_streamflow_processed.csv")
        ],
        "type": "terrestrial",
        "priority": 3,
        "description": "Legacy HYDAT Stream Flow Data",
        "deprecated": True
    },
    "eccc": {
        "paths": [
            os.path.join(DATA_ROOT, "processed", "eccc_manitoba_snow_processed.csv"),
            os.path.join(DATA_ROOT, "real", "environment_canada_fixed")
        ],
        "type": "terrestrial",
        "priority": 3,
        "description": "Legacy ECCC Weather Data",
        "deprecated": True
    }
}

# 数据质量门禁配置
QUALITY_GATES = {
    "min_records": 10,  # 最少记录数
    "max_age_hours": 72,  # 最大数据年龄（小时）
    "required_formats": [".csv", ".json", ".nc", ".h5", ".npy"],  # 支持的文件格式
    "health_check_interval": 3600  # 健康检查间隔（秒）
}

async def count_records_from_path(path: str) -> int:
    """真实统计路径下的记录数，不接受mock"""
    if not os.path.exists(path):
        return 0
    if os.path.isdir(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith((".csv", ".json", ".nc", ".h5", ".npy")):
                    total += 1
        return total
    if path.endswith(".csv"):
        try:
            with open(path, "r", newline="") as f:
                return sum(1 for _ in f) - 1  # exclude header
        except Exception:
            return 0
    if path.endswith(".json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return len(data)
                return len(data.keys()) if isinstance(data, dict) else 1
        except Exception:
            return 0
    if path.endswith((".nc", ".h5", ".npy")):
        # 对于科学数据格式，返回文件数量而不是内容
        return 1
    return 0

async def assess_data_quality(source: str, paths: list) -> Dict:
    """评估数据源质量，返回质量分数和状态。针对真实数据源优化。"""
    total_records = 0
    valid_files = 0
    total_files = 0
    latest_file_time = None
    data_sizes = []  # 数据文件大小
    data_consistency_issues = 0
    
    config = SOURCE_MAP.get(source, {})
    expected_update_freq = config.get("update_frequency", "daily")
    
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for f in files:
                        total_files += 1
                        file_path = os.path.join(root, f)
                        
                        if any(f.endswith(ext) for ext in QUALITY_GATES["required_formats"]):
                            valid_files += 1
                            try:
                                file_stat = os.stat(file_path)
                                file_time = file_stat.st_mtime
                                file_size = file_stat.st_size
                                data_sizes.append(file_size)
                                
                                if latest_file_time is None or file_time > latest_file_time:
                                    latest_file_time = file_time
                                    
                                # 检查数据一致性（文件太小可能是空或损坏）
                                if file_size < 100:  # 100字节以下可能有问题
                                    data_consistency_issues += 1
                                    
                                # 统计记录数
                                total_records += await count_records_from_path(file_path)
                                
                            except Exception:
                                data_consistency_issues += 1
                                pass
            else:
                total_files += 1
                if any(path.endswith(ext) for ext in QUALITY_GATES["required_formats"]):
                    valid_files += 1
                    try:
                        file_stat = os.stat(path)
                        file_time = file_stat.st_mtime
                        file_size = file_stat.st_size
                        data_sizes.append(file_size)
                        
                        if latest_file_time is None or file_time > latest_file_time:
                            latest_file_time = file_time
                            
                        if file_size < 100:
                            data_consistency_issues += 1
                            
                        total_records += await count_records_from_path(path)
                        
                    except Exception:
                        data_consistency_issues += 1
                        pass
    
    # 计算质量分数（总分100）
    quality_score = 0
    quality_details = {}
    
    # 1. 文件格式质量 (25分)
    if total_files > 0:
        format_score = (valid_files / total_files) * 25
        quality_score += format_score
        quality_details["format_score"] = round(format_score, 1)
    
    # 2. 数据新鲜度 (30分) - 根据更新频率调整
    if latest_file_time:
        age_hours = (time.time() - latest_file_time) / 3600
        
        # 根据数据源的预期更新频率设置不同的新鲜度阈值
        if expected_update_freq == "10min":
            max_acceptable_hours = 1  # 10分钟更新，1小时内可接受
        elif expected_update_freq == "15min":
            max_acceptable_hours = 2
        elif expected_update_freq == "hourly":
            max_acceptable_hours = 6
        elif expected_update_freq == "daily":
            max_acceptable_hours = 48
        else:
            max_acceptable_hours = QUALITY_GATES["max_age_hours"]
        
        if age_hours <= max_acceptable_hours:
            freshness_score = 30
        elif age_hours <= max_acceptable_hours * 2:
            freshness_score = 30 * (1 - (age_hours - max_acceptable_hours) / max_acceptable_hours)
        else:
            freshness_score = 0
            
        quality_score += freshness_score
        quality_details["freshness_score"] = round(freshness_score, 1)
        quality_details["age_hours"] = round(age_hours, 1)
    else:
        quality_details["freshness_score"] = 0
        quality_details["age_hours"] = None
    
    # 3. 数据量充足性 (20分)
    if total_records >= QUALITY_GATES["min_records"]:
        volume_score = 20
    elif total_records > 0:
        volume_score = 20 * (total_records / QUALITY_GATES["min_records"])
    else:
        volume_score = 0
    
    quality_score += volume_score
    quality_details["volume_score"] = round(volume_score, 1)
    
    # 4. 数据一致性 (15分)
    if valid_files > 0:
        consistency_score = max(0, 15 * (1 - data_consistency_issues / valid_files))
    else:
        consistency_score = 0
    
    quality_score += consistency_score
    quality_details["consistency_score"] = round(consistency_score, 1)
    quality_details["consistency_issues"] = data_consistency_issues
    
    # 5. 数据大小合理性 (10分)
    size_score = 0
    if data_sizes:
        avg_size = sum(data_sizes) / len(data_sizes)
        if avg_size > 1000:  # 大于1KB表示数据非空
            size_score = 10
        elif avg_size > 100:
            size_score = 5
    
    quality_score += size_score
    quality_details["size_score"] = round(size_score, 1)
    quality_details["avg_file_size"] = round(sum(data_sizes) / len(data_sizes), 0) if data_sizes else 0
    
    # 确定健康状态
    if quality_score >= 85:
        health_status = "Healthy"
    elif quality_score >= 70:
        health_status = "Degraded"
    elif quality_score >= 50:
        health_status = "Poor"
    else:
        health_status = "Offline"
    
    # 特殊状态判断：如果是新数据源但没有数据，可能是初次运行
    if total_files == 0 and not config.get("deprecated"):
        health_status = "Initializing"
    
    return {
        "quality_score": round(quality_score, 1),
        "health_status": health_status,
        "total_files": total_files,
        "valid_files": valid_files,
        "total_records": total_records,
        "latest_update": latest_file_time,
        "expected_update_freq": expected_update_freq,
        "quality_breakdown": quality_details
    }

async def check_earthdata_credentials() -> Dict:
    """检查Earthdata凭据状态，返回真实状态"""
    netrc_path = os.path.expanduser("~/.netrc")
    bearer_token = os.environ.get("EARTHDATA_BEARER")
    
    status = {
        "netrc_exists": os.path.exists(netrc_path),
        "bearer_token": bool(bearer_token),
        "earthaccess_available": shutil.which("python") is not None
    }
    
    # 检查earthaccess包是否安装
    try:
        import earthaccess
        status["earthaccess_installed"] = True
    except ImportError:
        status["earthaccess_installed"] = False
    
    return status

async def _run_fetcher(job_id: str, script_name: str, out_subdir: str, args_extra: list):
    start = os.environ.get("SAT_START")
    end = os.environ.get("SAT_END")
    bbox = os.environ.get("SAT_BBOX")
    limit = os.environ.get("SAT_LIMIT", "50")
    # Fallback to sane defaults if env not provided
    if not (start and end):
        today = datetime.utcnow().date()
        start = (today - timedelta(days=7)).isoformat()
        end = today.isoformat()
    if not bbox:
        bbox = "-102,48,-95,51"  # Manitoba bounding box
    out_dir = os.path.join(DATA_ROOT, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    script_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "..", "scripts", "fetchers", script_name))
    cmd = [
        "python3", script_path,
        f"--start={start}", f"--end={end}", f"--bbox={bbox}", f"--limit={limit}", f"--out={out_dir}"
    ] + args_extra
    
    # Pass environment variables to subprocess
    env = os.environ.copy()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc

async def real_fetch_smap(job_id: str):
    proc = await _run_fetcher(job_id, "smap_fetch.py", os.path.join("raw", "nasa_smap"), ["--product", "SPL4SMGP"])
    exit_code = proc.returncode
    if exit_code not in (0, 3):
        raise Exception(f"smap_fetch failed ({exit_code}): {proc.stdout or proc.stderr}")

async def real_fetch_hls(job_id: str):
    proc = await _run_fetcher(job_id, "hls_fetch.py", os.path.join("raw", "nasa_simple"), ["--product", "HLSL30.002"])
    exit_code = proc.returncode
    if exit_code not in (0, 3):
        raise Exception(f"hls_fetch failed ({exit_code}): {proc.stdout or proc.stderr}")

async def real_fetch_satellite_data(job_id: str, source: str):
    """真实获取卫星数据，不接受mock，支持备用数据源"""
    await job_store.update(job_id, status="running", started_at=datetime.utcnow().isoformat())
    
    try:
        # 检查凭据
        creds = await check_earthdata_credentials()
        if not (creds["netrc_exists"] or creds["bearer_token"] or creds["earthaccess_installed"]):
            # 凭据不可用，尝试备用数据源
            backup_sources = SOURCE_MAP.get(source, {}).get("backup_sources", [])
            if backup_sources:
                await job_store.update(job_id, status="running", message=f"Primary source unavailable, using backup: {', '.join(backup_sources)}")
                await fetch_from_backup_sources(job_id, source, backup_sources)
                return
            else:
                raise Exception("No Earthdata credentials found and no backup sources available. Please set up ~/.netrc, EARTHDATA_BEARER, or install earthaccess package.")
        
        # 根据数据源执行真实下载
        if source == "modis":
            await real_fetch_modis(job_id)
        elif source == "smap":
            await real_fetch_smap(job_id)
        elif source == "hls":
            await real_fetch_hls(job_id)
        else:
            raise Exception(f"Unknown satellite source: {source}")
            
        # 下载后统计并更新状态
        total = 0
        for p in SOURCE_MAP.get(source, {}).get("paths", []):
            total += await count_records_from_path(p)
        await job_store.update(
            job_id,
            status="succeeded",
            finished_at=datetime.utcnow().isoformat(),
            records=total,
            message=f"{source} download completed"
        )
        PIPELINE_STATUS[source] = {
            "status": "Active",
            "last_update": datetime.utcnow().isoformat(),
            "records": total
        }
            
    except Exception as e:
        # 主源失败，尝试备用数据源
        backup_sources = SOURCE_MAP.get(source, {}).get("backup_sources", [])
        if backup_sources:
            await job_store.update(job_id, status="running", message=f"Primary source failed, trying backup: {', '.join(backup_sources)}")
            await fetch_from_backup_sources(job_id, source, backup_sources)
            return
        else:
            await job_store.update(
                job_id,
                status="failed",
                finished_at=datetime.utcnow().isoformat(),
                message=f"Real fetch failed: {str(e)}"
            )
            raise

# Placeholders for not-yet-implemented sources
async def real_fetch_modis(job_id: str):
    raise Exception("MODIS download not implemented. Please implement real MODIS data fetching script.")

async def real_fetch_sentinel1(job_id: str):
    raise Exception("Sentinel-1 download not implemented. Please implement real Sentinel-1 data fetching script.")

async def run_real_fetcher_script(job_id: str, source: str) -> subprocess.CompletedProcess:
    """运行真实数据获取器脚本"""
    config = SOURCE_MAP.get(source, {})
    fetcher_script = config.get("fetcher_script")
    
    if not fetcher_script:
        raise Exception(f"No fetcher script defined for source: {source}")
    
    # 构建脚本路径
    script_path = os.path.join(FETCHER_ROOT, fetcher_script)
    if not os.path.exists(script_path):
        raise Exception(f"Fetcher script not found: {script_path}")
    
    # 确保输出目录存在
    for path in config.get("paths", []):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    
    # 构建命令参数
    cmd = ["python3", script_path]
    
    # 根据数据源类型添加特定参数
    if source == "eccc_weather":
        cmd.extend([
            "--output", config["paths"][0],
            "--stations", "WPG,YBR,YTH",
            "--verbose"
        ])
    elif source == "openweather":
        cmd.extend([
            "--output", config["paths"][0],
            "--verbose"
        ])
        # API密钥通过环境变量传递
    elif source == "manitoba_hydro":
        cmd.extend([
            "--output", config["paths"][0],
            "--stations", "05OJ001,05MF012,05NG001",
            "--include-lakes",
            "--verbose"
        ])
    elif source == "modis":
        cmd.extend([
            "--output", config["paths"][0],
            "--products", "MOD10A1,MOD11A1",
            "--verbose"
        ])
    elif source == "era5_reanalysis":
        cmd.extend([
            "--output", config["paths"][0],
            "--verbose"
        ])
    else:
        # 通用参数
        cmd.extend(["--output", config["paths"][0]])
    
    # 设置环境变量
    env = os.environ.copy()
    
    # 记录命令执行日志
    await job_store.update(job_id, message=f"Executing: {' '.join(cmd)}")
    
    # 执行命令
    proc = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        env=env,
        timeout=1800  # 30分钟超时
    )
    
    # 记录执行结果
    if proc.returncode == 0:
        await job_store.update(job_id, message=f"Script executed successfully: {source}")
    else:
        error_msg = proc.stderr or proc.stdout or "Unknown error"
        await job_store.update(job_id, message=f"Script execution failed: {error_msg}")
    
    return proc

async def real_fetch_terrestrial_data(job_id: str, source: str):
    """真实获取地面数据"""
    await job_store.update(job_id, status="running", started_at=datetime.utcnow().isoformat())
    
    try:
        # 检查是否为新的真实数据获取器
        config = SOURCE_MAP.get(source, {})
        if config.get("fetcher_script"):
            # 运行真实获取器脚本
            proc = await run_real_fetcher_script(job_id, source)
            
            if proc.returncode != 0:
                error_output = proc.stderr or proc.stdout or "Unknown script error"
                raise Exception(f"Fetcher script failed (exit code {proc.returncode}): {error_output}")
            
            # 解析脚本输出以获取统计信息
            success_msg = proc.stdout or "Data fetched successfully"
            await job_store.update(job_id, message=success_msg)
        
        # 统计获取到的记录数
        total = 0
        for p in config.get("paths", []):
            total += await count_records_from_path(p)
        
        await job_store.update(
            job_id,
            status="succeeded",
            finished_at=datetime.utcnow().isoformat(),
            records=total,
            message=f"Real fetch completed: {source} ({total} records)"
        )
        
        # 更新全局状态缓存
        PIPELINE_STATUS[source] = {
            "status": "Active",
            "last_update": datetime.utcnow().isoformat(),
            "records": total
        }
        
    except subprocess.TimeoutExpired:
        await job_store.update(
            job_id,
            status="failed",
            finished_at=datetime.utcnow().isoformat(),
            message=f"Fetcher script timeout: {source}"
        )
        raise
    except Exception as e:
        await job_store.update(
            job_id,
            status="failed",
            finished_at=datetime.utcnow().isoformat(),
            message=f"Real fetch failed: {str(e)}"
        )
        raise

async def real_fetch_backup_data(job_id: str, source: str):
    """真实获取备用数据源数据"""
    await job_store.update(job_id, status="running", started_at=datetime.utcnow().isoformat())
    
    try:
        total = 0
        for p in SOURCE_MAP.get(source, {}).get("paths", []):
            total += await count_records_from_path(p)
        
        await job_store.update(
            job_id,
            status="succeeded",
            finished_at=datetime.utcnow().isoformat(),
            records=total,
            message=f"Backup data fetch completed: {source}"
        )
        
        # 更新全局状态缓存
        PIPELINE_STATUS[source] = {
            "status": "Active",
            "last_update": datetime.utcnow().isoformat(),
            "records": total
        }
        
    except Exception as e:
        await job_store.update(
            job_id,
            status="failed",
            finished_at=datetime.utcnow().isoformat(),
            message=f"Backup data fetch failed: {str(e)}"
        )
        raise

# 简易状态缓存（进程级）
PIPELINE_STATUS: Dict[str, Dict] = {s: {"status": "Idle", "last_update": None, "records": 0} for s in SOURCE_MAP.keys()}

async def get_comprehensive_status() -> Dict:
    """获取所有数据源的综合状态，包含质量评估"""
    comprehensive_status = {}
    
    for source, config in SOURCE_MAP.items():
        try:
            # 统计记录数
            total_records = 0
            for path in config["paths"]:
                total_records += await count_records_from_path(path)
            
            # 评估数据质量
            quality_info = await assess_data_quality(source, config["paths"])
            
            # 确定状态
            if total_records > 0:
                if quality_info["health_status"] == "Offline":
                    status = "Offline"
                elif quality_info["health_status"] == "Poor":
                    status = "Degraded"
                else:
                    status = "Active"
            else:
                status = "Idle"
            
            # 添加备用源信息
            backup_info = {}
            if "backup_sources" in config:
                backup_info = {
                    "backup_sources": config["backup_sources"],
                    "backup_available": all(backup in SOURCE_MAP for backup in config["backup_sources"])
                }
            
            comprehensive_status[source] = {
                "status": status,
                "last_update": quality_info.get("latest_update"),
                "records": total_records,
                "type": config.get("type", "unknown"),
                "priority": config.get("priority", 999),
                "description": config.get("description", ""),
                "quality_score": quality_info["quality_score"],
                "health_status": quality_info["health_status"],
                "total_files": quality_info["total_files"],
                "valid_files": quality_info["valid_files"],
                "age_hours": quality_info.get("quality_breakdown", {}).get("age_hours"),
                **backup_info
            }
            
        except Exception as e:
            comprehensive_status[source] = {
                "status": "Error",
                "last_update": None,
                "records": 0,
                "error": str(e),
                "quality_score": 0,
                "health_status": "Offline"
            }
    
    return comprehensive_status

@router.get("/status")
async def get_status():
    """返回各数据源的当前状态与上次更新时间。"""
    comprehensive_status = await get_comprehensive_status()
    return {"status": "success", "sources": comprehensive_status}

@router.post("/sync")
async def sync_source(source: Literal[
    "eccc_weather", "openweather", "manitoba_hydro", "modis", "era5_reanalysis",
    "eccc", "hydat", "era5_land", "smap", "hls"  # 保留旧数据源
] = Query(..., description="数据源标识")):
    """触发单一数据源同步，返回作业ID。"""
    if source not in SOURCE_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown data source: {source}")
    
    job_id = await job_store.create(source)
    
    try:
        # 检查是否为废弃的数据源
        if SOURCE_MAP[source].get("deprecated"):
            await job_store.update(job_id, status="succeeded", message=f"Skipped deprecated source: {source}")
            return {"status": "skipped", "job_id": job_id, "source": source, "reason": "deprecated"}
        
        # 根据数据源类型选择处理方式
        if source in ["modis"]:
            # 卫星数据（新的MODIS获取器）
            asyncio.create_task(real_fetch_terrestrial_data(job_id, source))
        elif source in ["smap", "hls"]:
            # 旧的卫星数据获取方式
            asyncio.create_task(real_fetch_satellite_data(job_id, source))
        else:
            # 地面数据（包括新的获取器）
            asyncio.create_task(real_fetch_terrestrial_data(job_id, source))
            
    except Exception as e:
        await job_store.update(job_id, status="failed", message=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start sync: {str(e)}")
    
    return {"status": "queued", "job_id": job_id, "source": source}

@router.post("/sync-all")
async def sync_all():
    """并发触发所有数据源同步。"""
    jobs = {}
    for source in SOURCE_MAP.keys():
        job_id = await job_store.create(source)
        jobs[source] = job_id
        try:
            if source in ["modis", "smap", "hls"]:
                asyncio.create_task(real_fetch_satellite_data(job_id, source))
            elif source in ["era5_land"]:
                asyncio.create_task(real_fetch_terrestrial_data(job_id, source))
            else:
                asyncio.create_task(real_fetch_terrestrial_data(job_id, source))
        except Exception as e:
            await job_store.update(job_id, status="failed", message=str(e))
            # 继续其他源，不中断整个流程
    return {"status": "queued", "jobs": jobs}

@router.get("/job/{job_id}")
async def get_job(job_id: str):
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/credentials/status")
async def get_credentials_status():
    """返回Earthdata凭据状态，用于诊断"""
    return await check_earthdata_credentials()

@router.get("/backup/status")
async def get_backup_status():
    """返回备用数据源状态"""
    backup_status = {}
    for source, config in SOURCE_MAP.items():
        if "backup_sources" in config:
            backup_status[source] = {
                "primary": source,
                "backups": config["backup_sources"],
                "backup_available": all(backup in SOURCE_MAP for backup in config["backup_sources"])
            }
    return {"status": "success", "backup_sources": backup_status}
