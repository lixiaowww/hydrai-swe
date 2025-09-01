#!/usr/bin/env python3
"""
HydrAI-SWE 数据管道测试脚本
- 测试所有数据获取器
- 验证数据质量评估
- 端到端测试数据流程
- 检查前端界面显示

使用方法:
    python3 test_data_pipeline.py --sources all
    python3 test_data_pipeline.py --sources eccc_weather,openweather
"""

import os
import sys
import argparse
import json
import time
import requests
from datetime import datetime
import subprocess
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp

# API服务器URL
API_BASE_URL = "http://localhost:8000/api/v1"
FETCHER_SCRIPTS_DIR = "/home/sean/hydrai_swe/scripts/fetchers"
DATA_ROOT = "/home/sean/hydrai_swe/data"

# 颜色常量
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

# 数据源列表
DATA_SOURCES = [
    "eccc_weather",
    "openweather",
    "manitoba_hydro",
    "modis",
    "era5_reanalysis"
]

def print_header(message: str):
    """打印带有格式的标题"""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}== {message}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 80}{RESET}\n")

def print_status(message: str, status: str):
    """打印状态信息"""
    if status.lower() == "pass":
        status_str = f"{GREEN}✓ PASS{RESET}"
    elif status.lower() == "fail":
        status_str = f"{RED}✗ FAIL{RESET}"
    elif status.lower() == "warning":
        status_str = f"{YELLOW}⚠ WARNING{RESET}"
    else:
        status_str = status
    
    print(f"{message}: {status_str}")

async def test_api_connectivity() -> bool:
    """测试API服务器连接"""
    print_header("测试API服务器连接")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/pipeline/status") as response:
                if response.status == 200:
                    print_status("API服务器连接", "PASS")
                    return True
                else:
                    print_status(f"API服务器连接 (状态码: {response.status})", "FAIL")
                    return False
    except Exception as e:
        print_status(f"API服务器连接 (错误: {str(e)})", "FAIL")
        return False

async def test_directory_structure() -> bool:
    """测试目录结构"""
    print_header("测试数据目录结构")
    all_good = True
    
    # 检查根目录
    if not os.path.exists(DATA_ROOT):
        print_status("数据根目录存在", "FAIL")
        print(f"  创建目录: {DATA_ROOT}")
        os.makedirs(DATA_ROOT, exist_ok=True)
        all_good = False
    else:
        print_status("数据根目录存在", "PASS")
    
    # 检查raw和processed目录
    for dir_name in ["raw", "processed"]:
        dir_path = os.path.join(DATA_ROOT, dir_name)
        if not os.path.exists(dir_path):
            print_status(f"{dir_name}目录存在", "FAIL")
            print(f"  创建目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            all_good = False
        else:
            print_status(f"{dir_name}目录存在", "PASS")
    
    # 检查各个数据源目录
    for source in DATA_SOURCES:
        raw_dir = os.path.join(DATA_ROOT, "raw", source)
        processed_dir = os.path.join(DATA_ROOT, "processed", source)
        
        for dir_path in [raw_dir, processed_dir]:
            if not os.path.exists(dir_path):
                print_status(f"{dir_path}目录存在", "FAIL")
                print(f"  创建目录: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)
                all_good = False
            else:
                print_status(f"{dir_path}目录存在", "PASS")
    
    return all_good

async def test_fetcher_scripts() -> bool:
    """测试获取器脚本"""
    print_header("测试数据获取器脚本")
    all_good = True
    
    # 检查获取器脚本目录
    if not os.path.exists(FETCHER_SCRIPTS_DIR):
        print_status("获取器脚本目录存在", "FAIL")
        print(f"  目录不存在: {FETCHER_SCRIPTS_DIR}")
        return False
    else:
        print_status("获取器脚本目录存在", "PASS")
    
    # 检查各个获取器脚本
    script_files = {
        "eccc_weather": "eccc_weather_fetch.py",
        "openweather": "openweather_fetch.py",
        "manitoba_hydro": "manitoba_hydro_fetch.py",
        "modis": "modis_satellite_fetch.py",
        "era5_reanalysis": "era5_reanalysis_fetch.py"
    }
    
    for source, script_file in script_files.items():
        script_path = os.path.join(FETCHER_SCRIPTS_DIR, script_file)
        if not os.path.exists(script_path):
            print_status(f"{source}获取器脚本存在", "FAIL")
            print(f"  脚本不存在: {script_path}")
            all_good = False
        else:
            # 检查脚本是否可执行
            if not os.access(script_path, os.X_OK):
                print_status(f"{source}获取器脚本可执行", "WARNING")
                print(f"  添加执行权限: chmod +x {script_path}")
                try:
                    os.chmod(script_path, 0o755)
                except:
                    pass
            
            # 检查脚本帮助信息
            try:
                result = subprocess.run(
                    ["python3", script_path, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print_status(f"{source}获取器脚本可运行", "PASS")
                else:
                    print_status(f"{source}获取器脚本可运行", "WARNING")
                    print(f"  脚本帮助返回非零状态: {result.returncode}")
                    print(f"  错误信息: {result.stderr.strip()}")
            except Exception as e:
                print_status(f"{source}获取器脚本可运行", "FAIL")
                print(f"  运行错误: {str(e)}")
                all_good = False
    
    return all_good

async def test_api_status():
    """测试API状态端点"""
    print_header("测试API状态端点")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/pipeline/status") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "sources" in data:
                        print_status("API状态返回数据源信息", "PASS")
                        
                        # 打印数据源信息
                        print("\n数据源状态:")
                        for source, info in data["sources"].items():
                            status = info.get("status", "Unknown")
                            health = info.get("health_status", "Unknown")
                            score = info.get("quality_score", 0)
                            last_update = info.get("last_update", None)
                            
                            if last_update:
                                try:
                                    last_time = datetime.fromtimestamp(last_update)
                                    time_str = last_time.strftime("%Y-%m-%d %H:%M:%S")
                                except:
                                    time_str = "Unknown"
                            else:
                                time_str = "Never"
                            
                            status_color = GREEN if status == "Active" else (YELLOW if status == "Degraded" else RED)
                            health_color = GREEN if health == "Healthy" else (YELLOW if health in ["Degraded", "Initializing"] else RED)
                            
                            print(f"  {source}: {status_color}{status}{RESET}, 健康: {health_color}{health}{RESET}, 得分: {score}, 最后更新: {time_str}")
                    else:
                        print_status("API状态返回数据源信息", "FAIL")
                else:
                    print_status(f"API状态请求 (状态码: {response.status})", "FAIL")
                    print(await response.text())
    except Exception as e:
        print_status(f"API状态请求 (错误: {str(e)})", "FAIL")

async def test_sync_source(source: str) -> Dict:
    """测试同步单个数据源"""
    print(f"\n测试同步数据源: {BLUE}{source}{RESET}")
    
    result = {
        "source": source,
        "success": False,
        "job_id": None,
        "details": {}
    }
    
    try:
        # 发起同步请求
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{API_BASE_URL}/pipeline/sync?source={source}") as response:
                if response.status == 200:
                    data = await response.json()
                    job_id = data.get("job_id")
                    result["job_id"] = job_id
                    
                    if job_id:
                        print_status(f"同步请求已提交 (Job ID: {job_id})", "PASS")
                        
                        # 等待作业完成
                        max_wait_time = 300  # 最长等待5分钟
                        wait_time = 0
                        poll_interval = 5  # 每5秒轮询一次
                        
                        while wait_time < max_wait_time:
                            print(f"  等待作业完成... ({wait_time}秒)", end="\r")
                            await asyncio.sleep(poll_interval)
                            wait_time += poll_interval
                            
                            # 检查作业状态
                            async with session.get(f"{API_BASE_URL}/pipeline/job/{job_id}") as job_response:
                                if job_response.status == 200:
                                    job_data = await job_response.json()
                                    status = job_data.get("status")
                                    
                                    if status in ["succeeded", "failed"]:
                                        if status == "succeeded":
                                            print_status(f"\n{source}同步作业完成", "PASS")
                                            result["success"] = True
                                        else:
                                            print_status(f"\n{source}同步作业完成", "FAIL")
                                            print(f"  错误信息: {job_data.get('message', 'Unknown error')}")
                                        
                                        result["details"] = job_data
                                        break
                                else:
                                    print_status(f"\n{source}作业状态请求 (状态码: {job_response.status})", "FAIL")
                                    break
                        
                        if wait_time >= max_wait_time:
                            print_status(f"\n{source}同步作业超时", "FAIL")
                    else:
                        print_status("同步请求返回作业ID", "FAIL")
                else:
                    print_status(f"同步请求 (状态码: {response.status})", "FAIL")
                    print(await response.text())
    except Exception as e:
        print_status(f"同步请求 (错误: {str(e)})", "FAIL")
    
    return result

async def verify_data_files(source: str):
    """验证数据文件"""
    print(f"\n验证{source}数据文件:")
    
    raw_dir = os.path.join(DATA_ROOT, "raw", source)
    
    if not os.path.exists(raw_dir):
        print_status(f"{source}原始数据目录存在", "FAIL")
        return
    
    files = os.listdir(raw_dir)
    valid_extensions = [".json", ".csv", ".nc", ".h5"]
    valid_files = [f for f in files if any(f.endswith(ext) for ext in valid_extensions)]
    
    if valid_files:
        print_status(f"找到{len(valid_files)}个有效数据文件", "PASS")
        
        # 检查文件修改时间
        now = time.time()
        recent_files = [f for f in valid_files if (now - os.path.getmtime(os.path.join(raw_dir, f))) < 600]  # 10分钟内
        
        if recent_files:
            print_status(f"找到{len(recent_files)}个最近更新的文件", "PASS")
            
            # 显示最新文件
            print("\n最新数据文件:")
            for f in sorted(recent_files, key=lambda x: os.path.getmtime(os.path.join(raw_dir, x)), reverse=True)[:3]:
                file_path = os.path.join(raw_dir, f)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  {f} ({file_time}, {file_size:.1f} KB)")
        else:
            print_status("找到最近更新的文件", "FAIL")
    else:
        print_status("找到有效数据文件", "FAIL")

async def main():
    parser = argparse.ArgumentParser(description="测试HydrAI-SWE数据管道")
    parser.add_argument("--sources", default="all",
                      help="要测试的数据源，逗号分隔，或'all'表示全部")
    args = parser.parse_args()
    
    # 确定要测试的数据源
    if args.sources.lower() == "all":
        sources_to_test = DATA_SOURCES
    else:
        sources_to_test = [s.strip() for s in args.sources.split(",")]
        # 验证数据源有效性
        invalid_sources = [s for s in sources_to_test if s not in DATA_SOURCES]
        if invalid_sources:
            print(f"{RED}错误: 无效的数据源: {', '.join(invalid_sources)}{RESET}")
            print(f"有效的数据源: {', '.join(DATA_SOURCES)}")
            return 1
    
    print_header(f"HydrAI-SWE数据管道测试 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"测试数据源: {', '.join(sources_to_test)}")
    
    # 测试基础设施
    api_ok = await test_api_connectivity()
    if not api_ok:
        print(f"\n{RED}错误: API服务器不可用，请确保API服务器正在运行{RESET}")
        return 1
    
    dir_ok = await test_directory_structure()
    scripts_ok = await test_fetcher_scripts()
    
    if not (dir_ok and scripts_ok):
        print(f"\n{YELLOW}警告: 基础设施检查发现问题，但将继续测试{RESET}")
    
    # 测试API状态
    await test_api_status()
    
    # 测试数据源同步
    sync_results = []
    for source in sources_to_test:
        result = await test_sync_source(source)
        sync_results.append(result)
        
        if result["success"]:
            await verify_data_files(source)
    
    # 最终测试API状态
    print_header("同步后的API状态")
    await test_api_status()
    
    # 打印测试结果摘要
    print_header("测试结果摘要")
    success_count = sum(1 for r in sync_results if r["success"])
    print(f"成功同步: {success_count}/{len(sync_results)}")
    
    for result in sync_results:
        status = "PASS" if result["success"] else "FAIL"
        print_status(f"数据源 {result['source']}", status)
    
    if success_count == len(sync_results):
        print(f"\n{GREEN}所有数据源同步成功！{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}部分数据源同步失败，请检查详细日志{RESET}")
        return 1

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    exit_code = loop.run_until_complete(main())
    sys.exit(exit_code)
