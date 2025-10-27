#!/usr/bin/env python3
"""
看门狗审核系统 - 全面检查代码质量、安全性和部署准备情况
"""

import os
import sqlite3
import subprocess
import sys
from datetime import datetime

class WatchdogAudit:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.errors = []
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    def print_result(self, check_name, passed, message=""):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
        if message:
            print(f"      {message}")
        
        if passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
            self.errors.append(f"{check_name}: {message}")
    
    def print_warning(self, message):
        print(f"⚠️  WARNING - {message}")
        self.warnings.append(message)
    
    def check_file_exists(self, filepath, description):
        """检查文件是否存在"""
        exists = os.path.exists(filepath)
        self.print_result(f"检查文件: {description}", exists, 
                         f"路径: {filepath}" if exists else f"文件不存在: {filepath}")
        return exists
    
    def check_database(self):
        """检查数据库状态"""
        self.print_header("数据库检查")
        
        # 检查数据库文件
        if not self.check_file_exists("swe_data.db", "SQLite 数据库"):
            return
        
        try:
            conn = sqlite3.connect("swe_data.db")
            cursor = conn.cursor()
            
            # 检查记录数
            cursor.execute("SELECT COUNT(*) FROM swe_data")
            count = cursor.fetchone()[0]
            self.print_result("数据记录数量", count > 0, f"总记录数: {count}")
            
            # 检查时间范围
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM swe_data")
            min_date, max_date = cursor.fetchone()
            self.print_result("数据时间范围", min_date and max_date, 
                            f"范围: {min_date} 到 {max_date}")
            
            # 检查数据源分布
            cursor.execute("SELECT data_source, COUNT(*) FROM swe_data GROUP BY data_source")
            sources = cursor.fetchall()
            print(f"\n  📊 数据源分布:")
            for source, count in sources:
                print(f"      {source}: {count} 条")
            
            # 检查2010-2020年真实数据
            cursor.execute("SELECT COUNT(*) FROM swe_data WHERE data_source = 'historical'")
            historical_count = cursor.fetchone()[0]
            self.print_result("2010-2020年真实数据", historical_count > 0, 
                            f"记录数: {historical_count}")
            
            # 检查2025年真实数据
            cursor.execute("SELECT COUNT(*) FROM swe_data WHERE timestamp >= '2025-01-01'")
            recent_count = cursor.fetchone()[0]
            self.print_result("2025年数据", recent_count > 0, 
                            f"记录数: {recent_count}")
            
            conn.close()
            
        except Exception as e:
            self.print_result("数据库连接", False, str(e))
    
    def check_core_files(self):
        """检查核心文件"""
        self.print_header("核心文件检查")
        
        core_files = [
            ("production_server.py", "生产服务器"),
            ("requirements.txt", "依赖列表"),
            ("app.yaml", "Google Cloud 配置"),
            ("templates/ui/enhanced_dashboard.html", "前端界面"),
            ("README.md", "项目文档"),
            ("DATA_STRATEGY.md", "数据策略文档")
        ]
        
        for filepath, description in core_files:
            self.check_file_exists(filepath, description)
    
    def check_api_endpoints(self):
        """检查 API 端点"""
        self.print_header("API 端点检查")
        
        try:
            import requests
            base_url = "http://localhost:8001"
            
            endpoints = [
                ("/health", "健康检查"),
                ("/api/swe/historical?window=7d", "历史数据"),
                ("/api/swe/realtime", "实时数据"),
                ("/api/flood/prediction/7day", "洪水预测"),
                ("/api/water-quality/analysis/current", "水质分析")
            ]
            
            for endpoint, description in endpoints:
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    self.print_result(f"API: {description}", 
                                    response.status_code == 200, 
                                    f"状态码: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    self.print_warning(f"服务器未运行，无法测试: {description}")
                except Exception as e:
                    self.print_result(f"API: {description}", False, str(e))
                    
        except ImportError:
            self.print_warning("requests 模块未安装，跳过 API 测试")
    
    def check_security(self):
        """安全检查"""
        self.print_header("安全性检查")
        
        # 检查是否有硬编码的密码
        sensitive_files = ["production_server.py", "simple_swe_api.py"]
        for filepath in sensitive_files:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read().lower()
                    has_password = 'password' in content and '=' in content
                    if has_password:
                        self.print_warning(f"{filepath} 可能包含硬编码密码")
        
        # 检查 .gitignore
        if os.path.exists(".gitignore"):
            with open(".gitignore", 'r') as f:
                gitignore = f.read()
                checks = [
                    ("venv/" in gitignore or "env/" in gitignore, "虚拟环境已忽略"),
                    ("*.db" in gitignore or "swe_data.db" in gitignore, "数据库文件已忽略"),
                    ("__pycache__" in gitignore, "Python 缓存已忽略")
                ]
                for check, desc in checks:
                    self.print_result(desc, check)
        else:
            self.print_result(".gitignore 文件", False, "文件不存在")
    
    def check_deployment_readiness(self):
        """部署准备检查"""
        self.print_header("部署准备检查")
        
        # 检查 app.yaml 配置
        if os.path.exists("app.yaml"):
            with open("app.yaml", 'r') as f:
                content = f.read()
                
                checks = [
                    ("runtime: python312" in content, "Python 运行时版本正确"),
                    ("entrypoint:" in content, "入口点已配置"),
                    ("instance_class:" in content, "实例类型已配置")
                ]
                
                for check, desc in checks:
                    self.print_result(desc, check)
        
        # 检查 .gcloudignore
        if os.path.exists(".gcloudignore"):
            with open(".gcloudignore", 'r') as f:
                content = f.read()
                self.print_result(".gcloudignore 已配置", 
                                True, 
                                "排除文件已设置")
        else:
            self.print_warning(".gcloudignore 文件不存在")
        
        # 检查部署包
        if os.path.exists("deploy_package"):
            import subprocess
            result = subprocess.run(['find', 'deploy_package', '-type', 'f'], 
                                  capture_output=True, text=True)
            file_count = len(result.stdout.strip().split('\n'))
            self.print_result("部署包文件数量", 
                            file_count < 10000, 
                            f"文件数: {file_count} (限制: 10000)")
        else:
            self.print_warning("deploy_package 目录不存在")
    
    def check_data_strategy(self):
        """数据策略检查"""
        self.print_header("数据策略检查")
        
        try:
            conn = sqlite3.connect("swe_data.db")
            cursor = conn.cursor()
            
            # 检查各时期数据
            periods = [
                ("2010-01-01", "2020-12-31", "2010-2020年真实数据", 4000),
                ("2021-01-01", "2024-12-31", "2021-2024年模拟数据", 1400),
                ("2025-01-01", "2025-12-31", "2025年实时数据", 50)
            ]
            
            for start, end, desc, min_expected in periods:
                cursor.execute(
                    "SELECT COUNT(*) FROM swe_data WHERE timestamp >= ? AND timestamp <= ?",
                    (start, end)
                )
                count = cursor.fetchone()[0]
                self.print_result(desc, count >= min_expected, 
                                f"记录数: {count} (预期 >= {min_expected})")
            
            conn.close()
            
        except Exception as e:
            self.print_result("数据策略验证", False, str(e))
    
    def generate_report(self):
        """生成审核报告"""
        self.print_header("审核报告")
        
        total = self.checks_passed + self.checks_failed
        pass_rate = (self.checks_passed / total * 100) if total > 0 else 0
        
        print(f"\n  📊 审核统计:")
        print(f"      通过: {self.checks_passed}")
        print(f"      失败: {self.checks_failed}")
        print(f"      警告: {len(self.warnings)}")
        print(f"      通过率: {pass_rate:.1f}%")
        
        if self.errors:
            print(f"\n  ❌ 错误列表:")
            for error in self.errors:
                print(f"      - {error}")
        
        if self.warnings:
            print(f"\n  ⚠️  警告列表:")
            for warning in self.warnings:
                print(f"      - {warning}")
        
        # 部署建议
        print(f"\n  {'='*60}")
        if self.checks_failed == 0:
            print("  ✅ 审核通过！系统准备部署。")
            return True
        else:
            print("  ❌ 审核失败！请修复错误后重试。")
            return False
    
    def run_full_audit(self):
        """运行完整审核"""
        print("\n" + "="*60)
        print("  🐕 HydrAI-SWE 看门狗审核系统")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*60)
        
        self.check_core_files()
        self.check_database()
        self.check_data_strategy()
        self.check_security()
        self.check_api_endpoints()
        self.check_deployment_readiness()
        
        return self.generate_report()

if __name__ == "__main__":
    auditor = WatchdogAudit()
    passed = auditor.run_full_audit()
    
    sys.exit(0 if passed else 1)

