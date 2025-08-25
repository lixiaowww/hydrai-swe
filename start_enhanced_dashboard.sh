#!/bin/bash

# HydrAI-SWE Enhanced Dashboard 启动脚本
# 这个脚本会启动 enhanced_en dashboard - SWE 的真正首页

echo "🌊 启动 HydrAI-SWE Enhanced Dashboard..."
echo "📍 主页地址: http://localhost:8000/ui/enhanced_en"
echo "🔧 API文档: http://localhost:8000/docs"
echo ""

# 进入项目目录
cd /home/sean/hydrai_swe

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source venv/bin/activate

# 设置 PYTHONPATH
export PYTHONPATH=/home/sean/hydrai_swe/src/api:/home/sean/hydrai_swe

# 检查虚拟环境是否激活
if [[ "$VIRTUAL_ENV" != *"hydrai_swe/venv"* ]]; then
    echo "⚠️  虚拟环境未正确激活，正在重新激活..."
    source venv/bin/activate
fi

# 启动服务器
echo "🚀 启动服务器..."
echo "请在浏览器中访问: http://localhost:8000/ui/enhanced_en"
echo "按 Ctrl+C 停止服务器"
echo ""

# 使用虚拟环境中的 uvicorn
./venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
