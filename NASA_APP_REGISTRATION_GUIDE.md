# 🔐 NASA SMAP数据应用注册指南

## 📋 问题描述
当前HydrAI-SWE系统无法访问NASA SMAP土壤湿度数据，因为应用尚未在NASA Earthdata注册。

## 👤 用户信息
- 用户名: lixiaowww
- 应用名称: HydrAI-SWE
- 应用类型: web

## 🌐 应用注册步骤

### 1. 访问NASA Earthdata开发者中心
打开浏览器访问: https://urs.earthdata.nasa.gov/oauth/applications

### 2. 登录账户
使用您的凭据登录NASA Earthdata账户

### 3. 创建新应用
- 点击 "New Application" 或 "新建应用" 按钮
- 选择应用类型: web

### 4. 填写应用信息
- **应用名称**: HydrAI-SWE
- **应用类型**: web
- **描述**: 智能水资源管理系统 - 土壤湿度预测和洪水预警
- **重定向URI**: https://n5eil01u.ecs.nsidc.org/OPS/redirect
- **主页URL**: https://github.com/your-org/hydrai-swe
- **支持URL**: https://github.com/your-org/hydrai-swe/issues

### 5. 配置权限范围
选择以下权限:
- **read**: 读取数据文件
- **search**: 搜索可用数据

### 6. 提交注册
- 检查所有信息无误后点击提交
- 等待NASA审核（通常几分钟到几小时）

### 7. 获取凭据
注册成功后，您将获得:
- **Client ID**: 用于OAuth2认证
- **Client Secret**: 用于安全验证

### 8. 更新配置文件
将获得的Client ID和Secret更新到系统配置中

## 🔧 技术细节

### 重定向URI说明
重定向URI `https://n5eil01u.ecs.nsidc.org/OPS/redirect` 是NASA SMAP数据服务的标准端点，用于OAuth2认证流程。

### 权限范围说明
- **read**: 允许应用读取SMAP数据文件
- **search**: 允许应用搜索可用的SMAP数据

### 应用类型说明
应用类型 `web` 表示这是一个Web应用，需要服务器端OAuth2流程。

## 📞 技术支持
如果遇到问题，请联系NASA Earthdata支持团队或查看官方文档。

## 🔄 注册后步骤
1. 更新系统配置文件中的Client ID和Secret
2. 重新运行NASA认证测试
3. 开始SMAP数据下载
4. 集成到HydrAI-SWE系统

---
**生成时间**: 2025-08-21 11:32:41
**状态**: 待注册
**下一步**: 完成应用注册
