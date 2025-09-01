#!/usr/bin/env python3
"""
页面功能验证脚本
验证修复后的无监督学习页面是否能正常工作
"""

import requests
import re
import json
from urllib.parse import urljoin

def check_page_accessibility():
    """检查页面是否可访问"""
    print("🌐 检查页面访问性...")
    
    try:
        response = requests.get('http://localhost:8080/complete_test_page.html', timeout=10)
        if response.status_code == 200:
            print("✅ 页面访问正常 (HTTP 200)")
            return True, response.text
        else:
            print(f"❌ 页面访问失败 (HTTP {response.status_code})")
            return False, None
    except requests.exceptions.RequestException as e:
        print(f"❌ 页面访问异常: {e}")
        return False, None

def check_javascript_syntax(html_content):
    """检查JavaScript语法"""
    print("\n🔧 检查JavaScript语法...")
    
    # 提取所有JavaScript代码块
    script_patterns = [
        r'<script[^>]*>(.*?)</script>',
    ]
    
    total_scripts = 0
    for pattern in script_patterns:
        scripts = re.findall(pattern, html_content, re.DOTALL)
        total_scripts += len(scripts)
    
    print(f"✅ 找到 {total_scripts} 个脚本块")
    
    # 检查大括号匹配
    bracket_count = 0
    paren_count = 0
    square_count = 0
    
    for pattern in script_patterns:
        scripts = re.findall(pattern, html_content, re.DOTALL)
        for script in scripts:
            # 移除注释和字符串
            clean_script = re.sub(r'//.*$', '', script, flags=re.MULTILINE)
            clean_script = re.sub(r'/\*.*?\*/', '', clean_script, flags=re.DOTALL)
            clean_script = re.sub(r'"[^"]*"', '""', clean_script)
            clean_script = re.sub(r"'[^']*'", "''", clean_script)
            clean_script = re.sub(r'`[^`]*`', '``', clean_script)
            
            bracket_count += clean_script.count('{') - clean_script.count('}')
            paren_count += clean_script.count('(') - clean_script.count(')')
            square_count += clean_script.count('[') - clean_script.count(']')
    
    syntax_issues = []
    if bracket_count != 0:
        syntax_issues.append(f"大括号不匹配: {bracket_count}")
    if paren_count != 0:
        syntax_issues.append(f"圆括号不匹配: {paren_count}")
    if square_count != 0:
        syntax_issues.append(f"方括号不匹配: {square_count}")
    
    if not syntax_issues:
        print("✅ JavaScript语法检查通过")
        return True
    else:
        print("❌ JavaScript语法问题:")
        for issue in syntax_issues:
            print(f"  - {issue}")
        return False

def check_required_elements(html_content):
    """检查必要的HTML元素"""
    print("\n📋 检查必要元素...")
    
    required_elements = [
        ('Chart.js库', r'chart\.js'),
        ('Bootstrap库', r'bootstrap'),
        ('Canvas元素', r'<canvas[^>]*id="[^"]*Chart"'),
        ('动态内容容器', r'id="[^"]*-dynamic-content"'),
        ('标签导航', r'data-bs-toggle="tab"'),
        ('初始化函数', r'function\s+initialize\w*Chart'),
        ('解释生成函数', r'function\s+generate\w*Interpretation'),
    ]
    
    missing_elements = []
    for name, pattern in required_elements:
        if re.search(pattern, html_content, re.IGNORECASE):
            print(f"✅ {name}: 找到")
        else:
            print(f"❌ {name}: 未找到")
            missing_elements.append(name)
    
    return len(missing_elements) == 0

def check_chart_initialization(html_content):
    """检查图表初始化逻辑"""
    print("\n📊 检查图表初始化逻辑...")
    
    # 检查是否有图表初始化函数
    chart_functions = [
        'initializeDecompositionChart',
        'initializeAnomalyChart', 
        'initializeClusteringChart',
        'initializeStatisticalChart',
        'initializeFactorsChart'
    ]
    
    found_functions = []
    for func in chart_functions:
        if f'function {func}' in html_content:
            found_functions.append(func)
            print(f"✅ {func}: 找到")
        else:
            print(f"❌ {func}: 未找到")
    
    # 检查是否有事件监听器
    event_listeners = [
        'DOMContentLoaded',
        'load',
        'shown.bs.tab'
    ]
    
    for event in event_listeners:
        if event in html_content:
            print(f"✅ {event} 事件监听器: 找到")
        else:
            print(f"❌ {event} 事件监听器: 未找到")
    
    return len(found_functions) >= 4

def main():
    """主验证函数"""
    print("=== 无监督学习模块验证报告 ===\n")
    
    # 检查页面访问性
    accessible, html_content = check_page_accessibility()
    if not accessible:
        print("\n❌ 页面无法访问，验证终止")
        return False
    
    # 检查JavaScript语法
    syntax_ok = check_javascript_syntax(html_content)
    
    # 检查必要元素
    elements_ok = check_required_elements(html_content)
    
    # 检查图表初始化
    charts_ok = check_chart_initialization(html_content)
    
    # 生成总结报告
    print("\n" + "="*50)
    print("📋 验证总结:")
    print("="*50)
    
    results = [
        ("页面访问性", accessible),
        ("JavaScript语法", syntax_ok),
        ("必要元素", elements_ok),
        ("图表初始化", charts_ok)
    ]
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:<15}: {status}")
        if not result:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 所有验证项目都通过！无监督学习模块修复成功！")
        print("\n📝 使用说明:")
        print("1. 在浏览器中访问: http://localhost:8080/complete_test_page.html")
        print("2. 检查所有标签页的图表是否正常显示")
        print("3. 验证动态解释内容是否正确生成")
        print("4. 测试标签页切换功能")
        return True
    else:
        print("⚠️  部分验证项目未通过，请检查相关问题")
        return False

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
