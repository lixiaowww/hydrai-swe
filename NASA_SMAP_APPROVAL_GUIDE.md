# 🔐 NASA SMAP数据访问授权指南

## 📋 问题描述
当前HydrAI-SWE系统无法访问NASA SMAP土壤湿度数据，需要完成应用授权流程。

## 👤 用户信息
- 用户名: lixiaowww
- 应用名称: HydrAI-SWE
- 客户端ID: _JLuwMHxb2xX6NwYTb4dRA

## 🌐 授权步骤

### 1. 访问NASA Earthdata用户中心
打开浏览器访问: https://urs.earthdata.nasa.gov/profile

### 2. 登录账户
使用您的凭据登录NASA Earthdata账户

### 3. 导航到应用管理
- 点击左侧菜单中的 "Applications" 或 "应用"
- 或者直接访问: https://urs.earthdata.nasa.gov/oauth/applications

### 4. 查找HydrAI-SWE应用
在应用列表中找到名为 "HydrAI-SWE" 的应用

### 5. 授权应用
- 点击应用名称进入详情页
- 点击 "Approve" 或 "批准" 按钮
- 选择必要的权限范围（至少需要read和search权限）
- 确认授权

### 6. 验证授权
授权完成后，重新运行数据下载脚本验证是否成功

## 🔧 权限说明
- **read**: 读取数据文件
- **search**: 搜索可用数据
- **redirect_uri**: https://n5eil01u.ecs.nsidc.org/OPS/redirect

## 📞 技术支持
如果遇到问题，请联系NASA Earthdata支持团队或查看官方文档。

---
**生成时间**: 2025-08-21 11:09:32
**状态**: 待授权
