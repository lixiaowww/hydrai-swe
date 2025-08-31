#!/usr/bin/env python3
"""
测试无监督模块解读功能
验证新增的interpret_insights方法是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from src.models.exploration.insight_discovery import InsightDiscoveryModule

def test_interpretation_functionality():
    """测试解读功能"""
    print("🧪 开始测试无监督模块解读功能...")
    
    try:
        # 1. 创建探索模块实例
        print("🔧 步骤1: 创建探索模块实例...")
        explorer = InsightDiscoveryModule()
        
        # 2. 创建模拟数据
        print("🔧 步骤2: 创建模拟数据...")
        np.random.seed(42)
        n_samples = 100
        
        # 创建包含各种特征的模拟数据
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
            'Year': [2024] * n_samples,
            'Month': [(i % 12) + 1 for i in range(n_samples)],
            'Day': [(i % 28) + 1 for i in range(n_samples)],
            'Temp (°C)': np.random.normal(10, 15, n_samples),
            'Total Precip (mm)': np.random.exponential(5, n_samples),
            'Snow Depth (cm)': np.random.exponential(10, n_samples),
            'Soil Moisture': np.random.uniform(0.1, 0.9, n_samples),
            'Wind Speed (km/h)': np.random.exponential(10, n_samples),
            'Humidity (%)': np.random.uniform(30, 90, n_samples),
            'Pressure (kPa)': np.random.normal(101.3, 2, n_samples),
            'estimated_soil_moisture': np.random.uniform(0.2, 0.8, n_samples)
        })
        
        # 添加一些异常值
        data.loc[10:15, 'Temp (°C)'] = np.random.normal(50, 5, 6)  # 异常高温
        data.loc[20:25, 'Total Precip (mm)'] = np.random.exponential(50, 6)  # 异常降水
        
        # 添加一些缺失值
        data.loc[30:35, 'Snow Depth (cm)'] = np.nan
        data.loc[40:45, 'Soil Moisture'] = np.nan
        
        print(f"✅ 模拟数据创建完成: {data.shape}")
        
        # 3. 运行模式发现
        print("🔧 步骤3: 运行模式发现...")
        insights = explorer.discover_patterns(data, 'estimated_soil_moisture')
        
        if 'status' in insights and insights['status'] == 'error':
            print(f"❌ 模式发现失败: {insights['error']}")
            return False
        
        print("✅ 模式发现完成")
        
        # 4. 测试解读功能
        print("🔧 步骤4: 测试解读功能...")
        interpretation = explorer.interpret_insights(insights)
        
        if 'status' in interpretation and interpretation['status'] == 'error':
            print(f"❌ 解读失败: {interpretation['error']}")
            return False
        
        print("✅ 解读功能测试完成")
        
        # 5. 验证解读结果结构
        print("🔧 步骤5: 验证解读结果结构...")
        required_sections = [
            'executive_summary',
            'business_insights', 
            'risk_assessment',
            'data_quality_insights',
            'actionable_recommendations'
        ]
        
        for section in required_sections:
            if section not in interpretation:
                print(f"❌ 缺少必要部分: {section}")
                return False
            print(f"✅ 验证部分: {section}")
        
        # 6. 验证执行摘要
        print("🔧 步骤6: 验证执行摘要...")
        executive_summary = interpretation['executive_summary']
        if 'total_discoveries' not in executive_summary:
            print("❌ 执行摘要缺少total_discoveries")
            return False
        if 'key_message' not in executive_summary:
            print("❌ 执行摘要缺少key_message")
            return False
        
        print(f"✅ 执行摘要验证完成: {executive_summary['total_discoveries']} 个发现")
        print(f"🔍 关键信息: {executive_summary['key_message']}")
        
        # 7. 验证业务洞察
        print("🔧 步骤7: 验证业务洞察...")
        business_insights = interpretation['business_insights']
        
        if 'anomaly_analysis' in business_insights:
            anomaly_analysis = business_insights['anomaly_analysis']
            if 'anomaly_rate_interpretation' in anomaly_analysis:
                print(f"✅ 异常检测解读: {anomaly_analysis['anomaly_rate_interpretation']}")
        
        if 'clustering_analysis' in business_insights:
            clustering_analysis = business_insights['clustering_analysis']
            if 'cluster_interpretation' in clustering_analysis:
                print(f"✅ 聚类分析解读: {clustering_analysis['cluster_interpretation']}")
        
        if 'dimension_analysis' in business_insights:
            dimension_analysis = business_insights['dimension_analysis']
            if 'dimension_interpretation' in dimension_analysis:
                print(f"✅ 降维分析解读: {dimension_analysis['dimension_interpretation']}")
        
        # 8. 验证风险评估
        print("🔧 步骤8: 验证风险评估...")
        risk_assessment = interpretation['risk_assessment']
        if 'overall_risk_assessment' in risk_assessment:
            print(f"✅ 整体风险评估: {risk_assessment['overall_risk_assessment']}")
        
        # 9. 验证可操作建议
        print("🔧 步骤9: 验证可操作建议...")
        actionable_recommendations = interpretation['actionable_recommendations']
        
        if 'immediate_actions' in actionable_recommendations:
            immediate_actions = actionable_recommendations['immediate_actions']
            print(f"✅ 立即行动 ({len(immediate_actions)} 项):")
            for i, action in enumerate(immediate_actions[:3], 1):  # 显示前3项
                print(f"   {i}. {action}")
        
        if 'short_term_actions' in actionable_recommendations:
            short_term_actions = actionable_recommendations['short_term_actions']
            print(f"✅ 短期行动 ({len(short_term_actions)} 项):")
            for i, action in enumerate(short_term_actions[:3], 1):  # 显示前3项
                print(f"   {i}. {action}")
        
        # 10. 保存解读结果
        print("🔧 步骤10: 保存解读结果...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        interpretation_file = f"test_interpretation_results_{timestamp}.json"
        
        with open(interpretation_file, 'w', encoding='utf-8') as f:
            json.dump(interpretation, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 解读结果已保存: {interpretation_file}")
        
        # 11. 生成测试报告
        print("🔧 步骤11: 生成测试报告...")
        report = {
            'test_name': '无监督模块解读功能测试',
            'test_timestamp': datetime.now().isoformat(),
            'test_status': 'PASSED',
            'test_summary': {
                'total_sections': len(required_sections),
                'verified_sections': len(required_sections),
                'executive_summary_discoveries': executive_summary.get('total_discoveries', 0),
                'risk_level': risk_assessment.get('overall_risk_assessment', 'unknown'),
                'immediate_actions_count': len(actionable_recommendations.get('immediate_actions', [])),
                'short_term_actions_count': len(actionable_recommendations.get('short_term_actions', []))
            },
            'test_details': {
                'data_shape': data.shape,
                'insights_keys': list(insights.keys()),
                'interpretation_keys': list(interpretation.keys()),
                'required_sections_verified': required_sections
            }
        }
        
        report_file = f"test_interpretation_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 测试报告已保存: {report_file}")
        
        # 12. 测试完成
        print("\n🎉 无监督模块解读功能测试完成！")
        print("=" * 60)
        print("📊 测试结果摘要:")
        print(f"   ✅ 模式发现: 成功")
        print(f"   ✅ 解读功能: 成功")
        print(f"   ✅ 结果验证: 通过")
        print(f"   ✅ 发现数量: {executive_summary.get('total_discoveries', 0)}")
        print(f"   ✅ 风险等级: {risk_assessment.get('overall_risk_assessment', 'unknown')}")
        print(f"   ✅ 立即行动: {len(actionable_recommendations.get('immediate_actions', []))} 项")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 无监督模块解读功能测试")
    print("=" * 60)
    
    success = test_interpretation_functionality()
    
    if success:
        print("\n✅ 所有测试通过！解读功能正常工作。")
    else:
        print("\n❌ 测试失败！请检查代码实现。")
    
    return success

if __name__ == "__main__":
    main()
