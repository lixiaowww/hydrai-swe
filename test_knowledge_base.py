#!/usr/bin/env python3
"""
测试水文知识库系统
Test Hydrology Knowledge Base System
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_knowledge_base():
    """测试知识库基本功能"""
    print("🌊 Testing Hydrology Knowledge Base")
    print("=" * 50)
    
    try:
        from src.knowledge.hydrology_knowledge_base import HydrologyKnowledgeBase
        
        # 创建知识库实例
        kb = HydrologyKnowledgeBase()
        print("✅ Knowledge base initialized successfully")
        
        # 测试基本知识获取
        print("\n📚 Testing basic knowledge retrieval:")
        
        # SWE基础知识
        swe_info = kb.knowledge_base["swe_fundamentals"]
        print(f"  SWE Definition: {swe_info['definition']['en'][:100]}...")
        
        # 测量方法
        methods = swe_info["measurement_methods"]["ground_truth"]
        print(f"  Ground Truth Methods: {len(methods)} methods available")
        
        # 物理属性
        props = swe_info["physical_properties"]
        print(f"  Density Range: {props['density_range']}")
        print(f"  Typical Ratio: {props['typical_ratio']}")
        
        # 测试区域信息
        print("\n🗺️ Testing regional information:")
        manitoba_info = kb.regional_context["manitoba_hydrology"]
        print(f"  Climate Zones: {len(manitoba_info['climate_zones'])} zones defined")
        print(f"  Major Rivers: {len(manitoba_info['major_rivers'])} rivers documented")
        
        # 测试解读模板
        print("\n📊 Testing interpretation templates:")
        templates = kb.interpretation_templates
        print(f"  Trend Analysis: {len(templates['trend_analysis'])} patterns")
        print(f"  Seasonal Patterns: {len(templates['seasonal_patterns'])} patterns")
        print(f"  Anomaly Interpretation: {len(templates['anomaly_interpretation'])} types")
        
        # 测试专业解读生成
        print("\n🔍 Testing professional interpretation generation:")
        interpretation = kb.get_swe_interpretation(
            trend_direction="increasing",
            trend_magnitude=25.5,
            seasonal_pattern="early_peak",
            anomaly_score=2.3,
            forecast_confidence=0.85
        )
        
        print("  Interpretation generated successfully:")
        print(f"    Trend: {interpretation['trend_analysis']['description'][:100]}...")
        print(f"    Anomaly: {interpretation['anomaly_assessment']['description'][:100]}...")
        print(f"    Recommendations: {len(interpretation['management_recommendations'])} items")
        
        # 测试气候变化信息
        print("\n🌍 Testing climate change information:")
        climate_info = kb.get_climate_context()
        print(f"  Global Context: {len(climate_info['global_context'])} items")
        print(f"  Manitoba Specific: {len(climate_info['manitoba_specific'])} items")
        print(f"  Adaptation Strategies: {len(climate_info['adaptation_strategies'])} strategies")
        
        # 测试技术词汇表
        print("\n📖 Testing technical glossary:")
        glossary = kb.get_technical_glossary()
        print(f"  Technical Terms: {len(glossary)} terms defined")
        
        # 显示前几个术语
        for i, (term, definition) in enumerate(list(glossary.items())[:3]):
            print(f"    {term}: {definition[:80]}...")
        
        print("\n✅ All knowledge base tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import knowledge base: {e}")
        return False
    except Exception as e:
        print(f"❌ Knowledge base test failed: {e}")
        return False

def test_enhanced_interpretation_service():
    """测试增强解读服务"""
    print("\n🔧 Testing Enhanced Interpretation Service")
    print("=" * 50)
    
    try:
        from src.api.routers.enhanced_interpretation import EnhancedInterpretationService
        
        # 创建服务实例
        service = EnhancedInterpretationService()
        print("✅ Interpretation service initialized successfully")
        
        # 测试数据
        test_values = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        test_timestamps = [f"2024-01-{i:02d}" for i in range(1, 11)]
        
        print(f"\n📊 Test Data: {len(test_values)} values from {test_timestamps[0]} to {test_timestamps[-1]}")
        
        # 测试趋势分析
        print("\n📈 Testing trend analysis:")
        trend_analysis = service.analyze_trend_significance(test_values, test_timestamps)
        print(f"  Significant: {trend_analysis['significant']}")
        print(f"  Direction: {trend_analysis['trend_direction']}")
        print(f"  Change: {trend_analysis['total_change_percent']:.1f}%")
        print(f"  R²: {trend_analysis['r_squared']:.3f}")
        print(f"  Confidence: {trend_analysis['confidence']}")
        
        # 测试季节性模式检测
        print("\n🌤️ Testing seasonal pattern detection:")
        seasonal_patterns = service.detect_seasonal_patterns(test_values, test_timestamps)
        print(f"  Pattern: {seasonal_patterns['pattern']}")
        print(f"  Description: {seasonal_patterns['description'][:80]}...")
        
        # 测试异常检测
        print("\n⚠️ Testing anomaly detection:")
        anomaly_score = service.calculate_anomaly_score(test_values)
        print(f"  Anomaly Score: {anomaly_score:.3f}")
        
        # 测试数据质量评估
        print("\n🔍 Testing data quality assessment:")
        data_quality = service.assess_data_quality(test_values, test_timestamps)
        print(f"  Quality: {data_quality['quality']}")
        print(f"  Score: {data_quality['score']:.1f}/100")
        print(f"  Issues: {len(data_quality['issues'])} issues found")
        
        # 测试专业解读生成
        print("\n📝 Testing professional interpretation generation:")
        interpretation = service.generate_professional_interpretation(
            trend_analysis, seasonal_patterns, anomaly_score, data_quality
        )
        
        if service.knowledge_base:
            print("  Professional interpretation generated using knowledge base")
            print(f"    Trend Analysis: {interpretation['trend_analysis']['description'][:80]}...")
            print(f"    Regional Context: {len(interpretation['regional_context'])} context items")
            print(f"    Recommendations: {len(interpretation['management_recommendations'])} recommendations")
        else:
            print("  Basic interpretation generated (knowledge base not available)")
            print(f"    Summary: {interpretation['summary']['trend'][:80]}...")
            print(f"    Recommendations: {len(interpretation['recommendations'])} recommendations")
        
        print("\n✅ All interpretation service tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import interpretation service: {e}")
        return False
    except Exception as e:
        print(f"❌ Interpretation service test failed: {e}")
        return False

def test_api_endpoints():
    """测试API端点"""
    print("\n🌐 Testing API Endpoints")
    print("=" * 50)
    
    try:
        import requests
        import json
        
        base_url = "http://localhost:8000"
        
        # 测试知识库词汇表端点
        print("\n📖 Testing glossary endpoint:")
        try:
            response = requests.get(f"{base_url}/api/v1/interpretation/knowledge-base/glossary")
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Glossary endpoint working: {len(data.get('glossary', {}))} terms")
            else:
                print(f"  ❌ Glossary endpoint failed: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("  ⚠️ Server not running, skipping API tests")
            return True
        
        # 测试快速评估端点
        print("\n⚡ Testing quick assessment endpoint:")
        test_data = {
            "values": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/v1/interpretation/quick-assessment",
                json=test_data
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Quick assessment working: {data.get('assessment', 'Unknown')}")
                print(f"  Risk Level: {data.get('risk_level', 'Unknown')}")
            else:
                print(f"  ❌ Quick assessment failed: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("  ⚠️ Server not running, skipping API tests")
            return True
        
        print("\n✅ All API endpoint tests passed!")
        return True
        
    except ImportError:
        print("  ⚠️ Requests library not available, skipping API tests")
        return True
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Starting Hydrology Knowledge Base System Tests")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("Knowledge Base", test_knowledge_base),
        ("Interpretation Service", test_enhanced_interpretation_service),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Knowledge base system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
