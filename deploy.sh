#!/bin/bash

# HydrAI-SWE 生产环境部署脚本
# 使用方法: ./deploy.sh [environment]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查部署依赖..."
    
    # 检查kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl 未安装，请先安装 Kubernetes CLI"
        exit 1
    fi
    
    # 检查docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    # 检查helm
    if ! command -v helm &> /dev/null; then
        log_warning "Helm 未安装，将跳过 Helm 相关部署"
        HELM_AVAILABLE=false
    else
        HELM_AVAILABLE=true
    fi
    
    log_success "依赖检查完成"
}

# 构建Docker镜像
build_docker_image() {
    log_info "构建 HydrAI-SWE Docker 镜像..."
    
    # 检查Dockerfile是否存在
    if [ ! -f "Dockerfile" ]; then
        log_error "Dockerfile 不存在"
        exit 1
    fi
    
    # 构建镜像
    docker build -t hydrai-swe:latest .
    
    if [ $? -eq 0 ]; then
        log_success "Docker 镜像构建成功"
    else
        log_error "Docker 镜像构建失败"
        exit 1
    fi
}

# 创建命名空间
create_namespaces() {
    log_info "创建 Kubernetes 命名空间..."
    
    # 创建 hydrai 命名空间
    kubectl create namespace hydrai --dry-run=client -o yaml | kubectl apply -f -
    
    # 创建 monitoring 命名空间
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "命名空间创建完成"
}

# 部署数据库服务
deploy_database() {
    log_info "部署数据库服务..."
    
    # 部署 PostgreSQL
    kubectl apply -f k8s/database-services.yaml
    
    # 等待数据库就绪
    log_info "等待 PostgreSQL 就绪..."
    kubectl wait --for=condition=ready pod -l app=postgresql -n hydrai --timeout=300s
    
    log_info "等待 Redis 就绪..."
    kubectl wait --for=condition=ready pod -l app=redis -n hydrai --timeout=300s
    
    log_success "数据库服务部署完成"
}

# 部署监控系统
deploy_monitoring() {
    log_info "部署监控系统..."
    
    # 部署 Prometheus, Grafana, Alertmanager
    kubectl apply -f k8s/monitoring.yaml
    
    # 等待监控服务就绪
    log_info "等待监控服务就绪..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=300s
    kubectl wait --for=condition=condition=ready pod -l app=grafana -n monitoring --timeout=300s
    kubectl wait --for=condition=ready pod -l app=alertmanager -n monitoring --timeout=300s
    
    log_success "监控系统部署完成"
}

# 部署主应用
deploy_main_application() {
    log_info "部署 HydrAI-SWE 主应用..."
    
    # 部署主应用
    kubectl apply -f k8s/hydrai-swe-deployment.yaml
    
    # 等待应用就绪
    log_info "等待应用就绪..."
    kubectl wait --for=condition=ready pod -l app=hydrai-swe -n hydrai --timeout=600s
    
    log_success "主应用部署完成"
}

# 配置SSL证书
setup_ssl() {
    log_info "配置 SSL 证书..."
    
    # 检查 cert-manager 是否安装
    if kubectl get namespace cert-manager &> /dev/null; then
        log_info "cert-manager 已安装，配置证书..."
        
        # 创建 ClusterIssuer
        cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@hydrai-swe.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
        
        log_success "SSL 证书配置完成"
    else
        log_warning "cert-manager 未安装，跳过 SSL 配置"
        log_info "请手动安装 cert-manager 或配置证书"
    fi
}

# 验证部署
verify_deployment() {
    log_info "验证部署状态..."
    
    echo ""
    echo "=== 部署状态 ==="
    
    # 检查命名空间
    echo "命名空间:"
    kubectl get namespaces | grep -E "(hydrai|monitoring)"
    
    echo ""
    echo "=== HydrAI 命名空间 ==="
    kubectl get all -n hydrai
    
    echo ""
    echo "=== Monitoring 命名空间 ==="
    kubectl get all -n monitoring
    
    echo ""
    echo "=== 服务端点 ==="
    kubectl get svc -n hydrai
    kubectl get svc -n monitoring
    
    echo ""
    echo "=== 持久化存储 ==="
    kubectl get pvc -n hydrai
    kubectl get pvc -n monitoring
    
    log_success "部署验证完成"
}

# 显示访问信息
show_access_info() {
    log_info "显示访问信息..."
    
    echo ""
    echo "=== 访问信息 ==="
    echo "HydrAI-SWE API: http://localhost:8000 (端口转发)"
    echo "Grafana Dashboard: http://localhost:3000 (端口转发)"
    echo "Prometheus: http://localhost:9090 (端口转发)"
    
    echo ""
    echo "=== 端口转发命令 ==="
    echo "HydrAI-SWE API: kubectl port-forward -n hydrai svc/hydrai-swe-service 8000:80"
    echo "Grafana: kubectl port-forward -n monitoring svc/grafana-service 3000:3000"
    echo "Prometheus: kubectl port-forward -n monitoring svc/prometheus-service 9090:9090"
    
    echo ""
    echo "=== 默认凭据 ==="
    echo "Grafana: admin / admin123"
    echo "PostgreSQL: hydrai_user / hydrai_password123"
    echo "Redis: 无用户名 / hydrai_redis_123"
}

# 清理函数
cleanup() {
    log_warning "清理部署..."
    
    # 删除应用
    kubectl delete -f k8s/hydrai-swe-deployment.yaml --ignore-not-found=true
    kubectl delete -f k8s/database-services.yaml --ignore-not-found=true
    kubectl delete -f k8s/monitoring.yaml --ignore-not-found=true
    
    # 删除命名空间
    kubectl delete namespace hydrai --ignore-not-found=true
    kubectl delete namespace monitoring --ignore-not-found=true
    
    log_success "清理完成"
}

# 主函数
main() {
    local environment=${1:-production}
    
    echo "🚀 HydrAI-SWE 生产环境部署脚本"
    echo "=================================="
    echo "环境: $environment"
    echo ""
    
    # 检查依赖
    check_dependencies
    
    # 构建镜像
    build_docker_image
    
    # 创建命名空间
    create_namespaces
    
    # 部署数据库
    deploy_database
    
    # 部署监控
    deploy_monitoring
    
    # 部署主应用
    deploy_main_application
    
    # 配置SSL
    setup_ssl
    
    # 验证部署
    verify_deployment
    
    # 显示访问信息
    show_access_info
    
    log_success "🎉 HydrAI-SWE 部署完成!"
    echo ""
    echo "下一步:"
    echo "1. 配置域名和DNS"
    echo "2. 设置监控告警"
    echo "3. 配置备份策略"
    echo "4. 运行系统测试"
}

# 脚本入口
case "${1:-}" in
    "cleanup")
        cleanup
        ;;
    "verify")
        verify_deployment
        ;;
    "info")
        show_access_info
        ;;
    "help"|"-h"|"--help")
        echo "使用方法: $0 [command]"
        echo ""
        echo "命令:"
        echo "  (无参数)    执行完整部署"
        echo "  cleanup     清理部署"
        echo "  verify      验证部署状态"
        echo "  info        显示访问信息"
        echo "  help        显示帮助信息"
        ;;
    *)
        main "$@"
        ;;
esac
