#!/usr/bin/env python3
"""
JavaScript语法检查工具 - 专门用于检查大括号匹配问题
"""

import re
import sys

def extract_javascript_from_html(html_content):
    """从HTML中提取JavaScript代码"""
    # 查找script标签中的内容
    pattern = r'<script[^>]*>(.*?)</script>'
    scripts = re.findall(pattern, html_content, re.DOTALL)
    return '\n'.join(scripts)

def check_bracket_matching(js_content):
    """检查大括号匹配"""
    stack = []
    issues = []
    lines = js_content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        # 移除注释和字符串中的括号
        clean_line = re.sub(r'//.*$', '', line)  # 移除单行注释
        clean_line = re.sub(r'"[^"]*"', '""', clean_line)  # 移除字符串内容
        clean_line = re.sub(r"'[^']*'", "''", clean_line)  # 移除字符串内容
        clean_line = re.sub(r'`[^`]*`', '``', clean_line)  # 移除模板字符串
        
        for char_pos, char in enumerate(clean_line):
            if char in '({[':
                stack.append({
                    'char': char,
                    'line': line_num,
                    'pos': char_pos,
                    'context': line.strip()
                })
            elif char in ')}]':
                if not stack:
                    issues.append(f"行 {line_num}: 多余的 '{char}' - {line.strip()}")
                    continue
                    
                last = stack.pop()
                expected_closing = {'(': ')', '{': '}', '[': ']'}
                
                if expected_closing[last['char']] != char:
                    issues.append(f"行 {line_num}: 括号不匹配 - 期望 '{expected_closing[last['char']]}', 实际 '{char}' - {line.strip()}")
    
    # 检查未闭合的括号
    for unclosed in stack:
        issues.append(f"行 {unclosed['line']}: 未闭合的 '{unclosed['char']}' - {unclosed['context']}")
    
    return issues

def analyze_function_structure(js_content):
    """分析函数结构"""
    issues = []
    lines = js_content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        # 检查函数定义
        if 'function' in line and '{' not in line:
            # 函数定义行没有开括号，检查下一行
            if line_num < len(lines):
                next_line = lines[line_num]
                if '{' not in next_line:
                    issues.append(f"行 {line_num}: 函数定义可能缺少开括号 - {line.strip()}")
        
        # 检查控制结构
        control_keywords = ['if', 'else', 'for', 'while', 'try', 'catch']
        for keyword in control_keywords:
            pattern = rf'\b{keyword}\s*\([^)]*\)\s*$'
            if re.search(pattern, line.strip()):
                if line_num < len(lines) and '{' not in lines[line_num]:
                    issues.append(f"行 {line_num}: {keyword} 语句可能缺少开括号 - {line.strip()}")
    
    return issues

def main():
    filename = '/home/sean/hydrai_swe/complete_test_page.html'
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        js_content = extract_javascript_from_html(html_content)
        
        print("=== JavaScript语法检查结果 ===\n")
        
        # 检查大括号匹配
        bracket_issues = check_bracket_matching(js_content)
        if bracket_issues:
            print("大括号匹配问题:")
            for issue in bracket_issues[:10]:  # 只显示前10个问题
                print(f"  ❌ {issue}")
            if len(bracket_issues) > 10:
                print(f"  ... 还有 {len(bracket_issues) - 10} 个问题")
        else:
            print("✅ 大括号匹配正常")
        
        # 检查函数结构
        function_issues = analyze_function_structure(js_content)
        if function_issues:
            print("\n函数结构问题:")
            for issue in function_issues[:10]:
                print(f"  ❌ {issue}")
            if len(function_issues) > 10:
                print(f"  ... 还有 {len(function_issues) - 10} 个问题")
        else:
            print("\n✅ 函数结构正常")
            
        return len(bracket_issues) + len(function_issues)
        
    except FileNotFoundError:
        print(f"❌ 文件未找到: {filename}")
        return 1
    except Exception as e:
        print(f"❌ 分析出错: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
