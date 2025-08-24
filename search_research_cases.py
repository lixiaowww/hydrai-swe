#!/usr/bin/env python3
"""
搜索GitHub和Kaggle上的SWE气候变化研究案例
分析"气候变化影响SWE"和"融雪洪水风险"相关研究
"""

import requests
import json
import time
from datetime import datetime
import pandas as pd

def search_github_repos(query, max_results=50):
    """搜索GitHub仓库"""
    print(f"🔍 搜索GitHub: {query}")
    
    # GitHub API搜索仓库
    url = "https://api.github.com/search/repositories"
    params = {
        'q': query,
        'sort': 'stars',
        'order': 'desc',
        'per_page': min(max_results, 100)
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('items', [])
        else:
            print(f"❌ GitHub API错误: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return []

def search_kaggle_datasets(query, max_results=20):
    """搜索Kaggle数据集"""
    print(f"🔍 搜索Kaggle数据集: {query}")
    
    # 注意：Kaggle API需要认证，这里提供搜索建议
    print("📊 Kaggle数据集搜索建议:")
    print(f"  关键词: {query}")
    print("  访问: https://www.kaggle.com/datasets")
    print("  搜索相关数据集")
    
    # 返回一些已知的相关数据集
    known_datasets = [
        {
            'title': 'Snow Water Equivalent (SWE) Data',
            'description': 'Historical SWE measurements for climate analysis',
            'url': 'https://www.kaggle.com/datasets/example/swe-data'
        }
    ]
    
    return known_datasets

def analyze_climate_change_swe():
    """分析气候变化影响SWE的研究案例"""
    print("\n🌍 分析气候变化影响SWE研究案例")
    print("=" * 60)
    
    # 搜索关键词
    queries = [
        'climate change snow water equivalent SWE',
        'SWE trend analysis Mann-Kendall',
        'snow water equivalent climate impact',
        'SWE time series analysis 30 years',
        'snow cover climate change detection'
    ]
    
    all_repos = []
    for query in queries:
        repos = search_github_repos(query, max_results=20)
        all_repos.extend(repos)
        time.sleep(1)  # 避免API限制
    
    # 去重和排序
    unique_repos = {}
    for repo in all_repos:
        if repo['id'] not in unique_repos:
            unique_repos[repo['id']] = repo
    
    # 按星标排序
    sorted_repos = sorted(unique_repos.values(), key=lambda x: x['stargazers_count'], reverse=True)
    
    print(f"\n📊 找到 {len(sorted_repos)} 个相关仓库")
    
    # 分析前10个仓库
    top_repos = sorted_repos[:10]
    for i, repo in enumerate(top_repos, 1):
        print(f"\n{i}. {repo['full_name']}")
        print(f"   描述: {repo['description'] or '无描述'}")
        print(f"   星标: {repo['stargazers_count']}")
        print(f"   语言: {repo['language'] or '未知'}")
        print(f"   更新: {repo['updated_at'][:10]}")
        print(f"   URL: {repo['html_url']}")
    
    return sorted_repos

def analyze_snowmelt_flood_risk():
    """分析融雪洪水风险研究案例"""
    print("\n🌊 分析融雪洪水风险研究案例")
    print("=" * 60)
    
    # 搜索关键词
    queries = [
        'snowmelt flood risk assessment',
        'SWE runoff prediction flood',
        'snow melt flood modeling',
        'basin scale snowmelt analysis',
        'daily SWE temperature radiation flood'
    ]
    
    all_repos = []
    for query in queries:
        repos = search_github_repos(query, max_results=20)
        all_repos.extend(repos)
        time.sleep(1)
    
    # 去重和排序
    unique_repos = {}
    for repo in all_repos:
        if repo['id'] not in unique_repos:
            unique_repos[repo['id']] = repo
    
    sorted_repos = sorted(unique_repos.values(), key=lambda x: x['stargazers_count'], reverse=True)
    
    print(f"\n📊 找到 {len(sorted_repos)} 个相关仓库")
    
    # 分析前10个仓库
    top_repos = sorted_repos[:10]
    for i, repo in enumerate(top_repos, 1):
        print(f"\n{i}. {repo['full_name']}")
        print(f"   描述: {repo['description'] or '无描述'}")
        print(f"   星标: {repo['stargazers_count']}")
        print(f"   语言: {repo['language'] or '未知'}")
        print(f"   更新: {repo['updated_at'][:10]}")
        print(f"   URL: {repo['html_url']}")
    
    return sorted_repos

def generate_research_summary():
    """生成研究案例总结"""
    print("\n📋 研究案例总结和建议")
    print("=" * 60)
    
    print("\n🌍 气候变化影响SWE研究要点:")
    print("1. 时间序列要求: 至少30年连续数据")
    print("2. 数据同质化: 处理观测方法变化、站点迁移等")
    print("3. 基准期: 1991-2020年作为气候基准期")
    print("4. 异常计算: 相对于基准期的偏差")
    print("5. 趋势检验: Mann-Kendall非参数检验")
    print("6. 斜率估计: Theil-Sen稳健斜率")
    
    print("\n🌊 融雪洪水风险研究要点:")
    print("1. 空间尺度: 流域/子流域为主")
    print("2. 时间聚焦: 积雪-融雪期")
    print("3. 驱动因子: 日尺度SWE、气温、辐射")
    print("4. 模型设置: 热身期 + 分期校验")
    print("5. 验证方法: 交叉验证、独立测试")
    
    print("\n💡 技术实现建议:")
    print("1. 使用pymannkendall库进行趋势检验")
    print("2. 使用scipy.stats进行Theil-Sen斜率估计")
    print("3. 实现数据同质化检测算法")
    print("4. 建立30年基准期计算框架")
    print("5. 开发流域尺度SWE-径流模型")

def main():
    """主函数"""
    print("🚀 开始搜索SWE气候变化研究案例")
    print("=" * 60)
    print(f"搜索时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 搜索气候变化影响SWE案例
    climate_repos = analyze_climate_change_swe()
    
    # 搜索融雪洪水风险案例
    flood_repos = analyze_snowmelt_flood_risk()
    
    # 生成总结
    generate_research_summary()
    
    # 保存结果
    results = {
        'climate_change_swe': climate_repos,
        'snowmelt_flood_risk': flood_repos,
        'search_time': datetime.now().isoformat()
    }
    
    with open('research_cases_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 搜索结果已保存到: research_cases_results.json")
    print("\n🎉 搜索完成!")

if __name__ == "__main__":
    main()
