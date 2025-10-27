#!/bin/bash

# 创建最小化的部署包

echo "🎯 创建 Google Cloud 部署包..."

# 创建部署目录
DEPLOY_DIR="deploy_package"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

echo "📦 复制核心文件..."

# 核心 API 文件
cp simple_swe_api.py $DEPLOY_DIR/
cp minimal_server.py $DEPLOY_DIR/
cp swe_data.db $DEPLOY_DIR/

# 配置文件
cp app.yaml $DEPLOY_DIR/
cp .gcloudignore $DEPLOY_DIR/
cp requirements.txt $DEPLOY_DIR/

# 复制必要的源代码目录
mkdir -p $DEPLOY_DIR/src/api/routers
mkdir -p $DEPLOY_DIR/src/core
mkdir -p $DEPLOY_DIR/templates/ui

# API 路由
cp src/api/__init__.py $DEPLOY_DIR/src/api/ 2>/dev/null || touch $DEPLOY_DIR/src/api/__init__.py
cp src/api/main.py $DEPLOY_DIR/src/api/
cp src/api/routers/__init__.py $DEPLOY_DIR/src/api/routers/ 2>/dev/null || touch $DEPLOY_DIR/src/api/routers/__init__.py
cp src/api/routers/swe.py $DEPLOY_DIR/src/api/routers/
cp src/api/routers/flood_prediction_api.py $DEPLOY_DIR/src/api/routers/
cp src/api/routers/water_quality_api.py $DEPLOY_DIR/src/api/routers/

# 核心模块
cp src/core/__init__.py $DEPLOY_DIR/src/core/ 2>/dev/null || touch $DEPLOY_DIR/src/core/__init__.py
cp src/core/data_manager.py $DEPLOY_DIR/src/core/ 2>/dev/null || echo "# placeholder" > $DEPLOY_DIR/src/core/data_manager.py

# 前端文件
cp templates/ui/enhanced_dashboard.html $DEPLOY_DIR/templates/ui/

# 静态文件（如果有）
if [ -d "static" ]; then
    mkdir -p $DEPLOY_DIR/static
    cp -r static/* $DEPLOY_DIR/static/ 2>/dev/null || true
fi

# 文档
cp README.md $DEPLOY_DIR/ 2>/dev/null || true
cp DATA_STRATEGY.md $DEPLOY_DIR/ 2>/dev/null || true

echo "📊 部署包统计:"
echo "  文件总数: $(find $DEPLOY_DIR -type f | wc -l)"
echo "  目录大小: $(du -sh $DEPLOY_DIR | cut -f1)"

echo "✅ 部署包创建完成: $DEPLOY_DIR/"
echo ""
echo "🚀 部署命令:"
echo "  cd $DEPLOY_DIR"
echo "  gcloud app deploy app.yaml --quiet"

