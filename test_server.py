#!/usr/bin/env python3
"""
简单的测试启动脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🌊 HydrAI-SWE 测试启动")
    
    # 检查当前目录
    if not Path("src/api/main.py").exists():
        print("❌ 请在项目根目录运行此脚本")
        print("当前目录:", Path.cwd())
        return
    
    # 设置环境变量
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    try:
        # 简单启动服务器
        cmd = [
            "python3", "-m", "uvicorn", 
            "src.api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ]
        
        print("📡 启动服务器...")
        print(f"命令: {' '.join(cmd)}")
        print()
        print("访问地址:")
        print("   • End User Interface: http://localhost:8000/ui (English)")
        print("   • Model Training:     http://localhost:8000/model")
        print("   • Chinese Interface:  http://localhost:8000/ui/enhanced")
        print("   • Legacy UI:          http://localhost:8000/ui/legacy")
        print("   • API Docs:           http://localhost:8000/docs")
        print()
        print("按 Ctrl+C 停止服务器")
        
        # 启动服务器
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n✅ 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
