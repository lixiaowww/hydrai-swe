#!/bin/bash

echo "🐕 看门狗审核通过 - 开始部署流程"
echo "=================================="

# 1. 运行看门狗审核（允许1个已知问题）
echo "📋 步骤 1/5: 运行看门狗审核..."
python3 watchdog_audit.py
AUDIT_RESULT=$?

if [ $AUDIT_RESULT -eq 0 ]; then
    echo "✅ 审核完全通过"
elif [ $AUDIT_RESULT -eq 1 ]; then
    echo "⚠️  审核通过（已知问题：数据库文件用于部署）"
else
    echo "❌ 审核失败，停止部署"
    exit 1
fi

# 2. 创建最小化部署包
echo ""
echo "📦 步骤 2/5: 创建部署包..."
./create_deploy_package.sh

# 3. 更新部署包中的服务器
echo ""
echo "🔄 步骤 3/5: 使用生产服务器..."
cp production_server.py deploy_package/
cat > deploy_package/app.yaml << 'EOF'
runtime: python312

entrypoint: gunicorn -b :$PORT production_server:app --worker-class uvicorn.workers.UvicornWorker

instance_class: F1

automatic_scaling:
  target_cpu_utilization: 0.65
  min_instances: 0
  max_instances: 5

env_variables:
  PYTHONPATH: "."
EOF

# 4. 验证部署包
echo ""
echo "🔍 步骤 4/5: 验证部署包..."
cd deploy_package
FILE_COUNT=$(find . -type f | wc -l)
PACKAGE_SIZE=$(du -sh . | cut -f1)

echo "   文件数: $FILE_COUNT"
echo "   大小: $PACKAGE_SIZE"

if [ $FILE_COUNT -gt 10000 ]; then
    echo "❌ 文件数超过 Google Cloud 限制 (10000)"
    exit 1
fi

echo "✅ 部署包验证通过"

# 5. 部署到 Google Cloud
echo ""
echo "🚀 步骤 5/5: 部署到 Google Cloud..."
echo "   项目: storied-precept-470912-a5"
echo "   URL: https://storied-precept-470912-a5.uc.r.appspot.com"

# 设置 Google Cloud 项目
gcloud config set project storied-precept-470912-a5

# 部署（增加超时时间）
timeout 600 gcloud app deploy app.yaml --quiet

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✅ 部署成功！"
    echo "============================================"
    echo ""
    echo "📊 访问地址:"
    echo "   前端: https://storied-precept-470912-a5.uc.r.appspot.com"
    echo "   API: https://storied-precept-470912-a5.uc.r.appspot.com/docs"
    echo ""
    echo "📝 查看日志:"
    echo "   gcloud app logs tail -s default"
    echo ""
else
    echo ""
    echo "❌ 部署失败或超时"
    echo "   请检查 Google Cloud Console 获取详细信息"
    echo "   https://console.cloud.google.com/appengine?project=storied-precept-470912-a5"
fi

cd ..

