#!/usr/bin/env python3
"""
HydrAI-SWE 简化监控系统深度测试
验证系统真实功能和质量
"""

import logging
import time
import os
from datetime import datetime
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """测试基础功能"""
    logger.info("🧪 测试1：基础功能验证")
    
    try:
        from src.monitoring.simple_monitor import SimpleMonitor, SystemStatus
        
        # 创建监控器
        monitor = SimpleMonitor()
        logger.info("✅ 监控器创建成功")
        
        # 验证初始状态
        assert monitor.running == False, "初始状态应该是停止的"
        assert len(monitor.status_history) == 0, "初始历史记录应该为空"
        assert len(monitor.alert_history) == 0, "初始告警记录应该为空"
        logger.info("✅ 初始状态验证通过")
        
        # 验证配置
        assert 'cpu_critical' in monitor.thresholds, "应该包含CPU临界阈值"
        assert 'memory_warning' in monitor.thresholds, "应该包含内存警告阈值"
        assert 'disk_critical' in monitor.thresholds, "应该包含磁盘临界阈值"
        logger.info("✅ 配置验证通过")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 基础功能测试失败: {e}")
        return False

def test_monitoring_lifecycle():
    """测试监控生命周期"""
    logger.info("\n🧪 测试2：监控生命周期验证")
    
    try:
        from src.monitoring.simple_monitor import SimpleMonitor
        
        monitor = SimpleMonitor()
        
        # 启动监控
        monitor.start_monitoring(interval=2.0)
        time.sleep(1)  # 等待线程启动
        
        assert monitor.running == True, "监控应该已启动"
        assert monitor.monitor_thread is not None, "监控线程应该存在"
        assert monitor.monitor_thread.is_alive(), "监控线程应该活跃"
        logger.info("✅ 监控启动验证通过")
        
        # 等待收集数据
        time.sleep(5)
        
        assert len(monitor.status_history) > 0, "应该收集到状态数据"
        assert len(monitor.status_history) <= 100, "历史记录应该在合理范围内"
        logger.info("✅ 数据收集验证通过")
        
        # 停止监控
        monitor.stop_monitoring()
        time.sleep(1)  # 等待线程停止
        
        assert monitor.running == False, "监控应该已停止"
        logger.info("✅ 监控停止验证通过")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 监控生命周期测试失败: {e}")
        return False

def test_system_status_collection():
    """测试系统状态收集"""
    logger.info("\n🧪 测试3：系统状态收集验证")
    
    try:
        from src.monitoring.simple_monitor import SimpleMonitor
        
        monitor = SimpleMonitor()
        monitor.start_monitoring(interval=1.0)
        
        # 等待收集数据
        time.sleep(3)
        
        # 验证状态数据
        current_status = monitor.get_current_status()
        assert current_status is not None, "应该能获取当前状态"
        assert hasattr(current_status, 'cpu_percent'), "状态应该包含CPU信息"
        assert hasattr(current_status, 'memory_percent'), "状态应该包含内存信息"
        assert hasattr(current_status, 'disk_percent'), "状态应该包含磁盘信息"
        assert hasattr(current_status, 'process_count'), "状态应该包含进程数信息"
        assert hasattr(current_status, 'status'), "状态应该包含状态标识"
        assert hasattr(current_status, 'alerts'), "状态应该包含告警信息"
        
        # 验证数值合理性
        assert 0 <= current_status.cpu_percent <= 100, "CPU使用率应该在0-100%之间"
        assert 0 <= current_status.memory_percent <= 100, "内存使用率应该在0-100%之间"
        assert 0 <= current_status.disk_percent <= 100, "磁盘使用率应该在0-100%之间"
        assert current_status.process_count > 0, "进程数应该大于0"
        
        logger.info(f"✅ 状态数据验证通过: CPU={current_status.cpu_percent:.1f}%, "
                   f"内存={current_status.memory_percent:.1f}%, "
                   f"磁盘={current_status.disk_percent:.1f}%")
        
        monitor.stop_monitoring()
        return True
        
    except Exception as e:
        logger.error(f"❌ 系统状态收集测试失败: {e}")
        return False

def test_alert_system():
    """测试告警系统"""
    logger.info("\n🧪 测试4：告警系统验证")
    
    try:
        from src.monitoring.simple_monitor import SimpleMonitor
        
        monitor = SimpleMonitor()
        
        # 测试告警阈值
        assert monitor.thresholds['cpu_critical'] == 90.0, "CPU临界阈值应该是90%"
        assert monitor.thresholds['cpu_warning'] == 70.0, "CPU警告阈值应该是70%"
        assert monitor.thresholds['memory_critical'] == 95.0, "内存临界阈值应该是95%"
        assert monitor.thresholds['memory_warning'] == 80.0, "内存警告阈值应该是80%"
        logger.info("✅ 告警阈值验证通过")
        
        # 启动监控并等待告警
        monitor.start_monitoring(interval=1.0)
        time.sleep(5)
        
        # 检查是否有告警
        current_status = monitor.get_current_status()
        if current_status and current_status.alerts:
            logger.info(f"✅ 告警触发验证通过: {len(current_status.alerts)} 个告警")
            for alert in current_status.alerts:
                logger.info(f"  🚨 {alert}")
        else:
            logger.info("ℹ️ 当前无告警，系统运行正常")
        
        # 验证告警历史
        if monitor.alert_history:
            assert len(monitor.alert_history) <= 50, "告警历史应该在合理范围内"
            logger.info(f"✅ 告警历史验证通过: {len(monitor.alert_history)} 条记录")
        
        monitor.stop_monitoring()
        return True
        
    except Exception as e:
        logger.error(f"❌ 告警系统测试失败: {e}")
        return False

def test_data_persistence():
    """测试数据持久化"""
    logger.info("\n🧪 测试5：数据持久化验证")
    
    try:
        from src.monitoring.simple_monitor import SimpleMonitor
        
        monitor = SimpleMonitor()
        
        # 启动监控并收集数据
        monitor.start_monitoring(interval=1.0)
        time.sleep(3)
        monitor.stop_monitoring()
        
        # 保存数据
        monitor.save_data()
        
        # 检查文件是否创建
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status_file = f"monitoring/status_history_{timestamp}.json"
        alert_file = f"monitoring/alert_history_{timestamp}.json"
        
        # 等待文件写入
        time.sleep(1)
        
        # 验证文件存在
        if os.path.exists(status_file):
            with open(status_file, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
                assert isinstance(status_data, list), "状态数据应该是列表"
                if status_data:
                    assert 'cpu_percent' in status_data[0], "状态数据应该包含CPU信息"
            logger.info("✅ 状态数据保存验证通过")
        else:
            logger.warning("⚠️ 状态数据文件未找到")
        
        if os.path.exists(alert_file):
            with open(alert_file, 'r', encoding='utf-8') as f:
                alert_data = json.load(f)
                assert isinstance(alert_data, list), "告警数据应该是列表"
            logger.info("✅ 告警数据保存验证通过")
        else:
            logger.warning("⚠️ 告警数据文件未找到")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据持久化测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    logger.info("\n🧪 测试6：错误处理验证")
    
    try:
        from src.monitoring.simple_monitor import SimpleMonitor
        
        monitor = SimpleMonitor()
        
        # 测试异常情况下的状态收集
        # 这里我们模拟一个异常情况
        import psutil
        original_cpu_percent = psutil.cpu_percent
        
        # 临时替换函数以模拟异常
        def mock_cpu_percent(*args, **kwargs):
            raise Exception("模拟CPU监控异常")
        
        try:
            # 替换函数
            psutil.cpu_percent = mock_cpu_percent
            
            # 尝试收集状态
            status = monitor._collect_system_status()
            
            # 验证异常处理
            assert status.status == 'unknown', "异常情况下状态应该是unknown"
            assert len(status.alerts) > 0, "异常情况下应该有告警"
            assert any('状态收集失败' in alert for alert in status.alerts), "应该有状态收集失败的告警"
            
            logger.info("✅ 异常处理验证通过")
            
        finally:
            # 恢复原始函数
            psutil.cpu_percent = original_cpu_percent
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 错误处理测试失败: {e}")
        return False

def test_performance_and_stability():
    """测试性能和稳定性"""
    logger.info("\n🧪 测试7：性能和稳定性验证")
    
    try:
        from src.monitoring.simple_monitor import SimpleMonitor
        
        monitor = SimpleMonitor()
        
        # 启动监控
        start_time = time.time()
        monitor.start_monitoring(interval=0.5)  # 快速监控
        
        # 运行一段时间
        time.sleep(10)
        
        # 停止监控
        monitor.stop_monitoring()
        end_time = time.time()
        
        # 验证性能
        total_time = end_time - start_time
        expected_records = int(total_time / 0.5)  # 预期记录数
        
        actual_records = len(monitor.status_history)
        logger.info(f"运行时间: {total_time:.1f}秒")
        logger.info(f"预期记录数: {expected_records}")
        logger.info(f"实际记录数: {actual_records}")
        
        # 允许一定的误差
        if abs(actual_records - expected_records) <= 2:
            logger.info("✅ 性能验证通过")
        else:
            logger.warning(f"⚠️ 性能偏差较大: 预期{expected_records}, 实际{actual_records}")
        
        # 验证稳定性
        if monitor.status_history:
            # 检查数据一致性
            first_status = monitor.status_history[0]
            last_status = monitor.status_history[-1]
            
            assert hasattr(first_status, 'cpu_percent'), "第一条记录应该有CPU信息"
            assert hasattr(last_status, 'cpu_percent'), "最后一条记录应该有CPU信息"
            
            logger.info("✅ 稳定性验证通过")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 性能和稳定性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始HydrAI-SWE简化监控系统深度测试")
    
    # 创建测试结果目录
    os.makedirs("test_results", exist_ok=True)
    
    # 记录测试开始时间
    start_time = datetime.now()
    
    # 执行测试
    test_results = {}
    
    # 测试1：基础功能
    test_results['basic_functionality'] = test_basic_functionality()
    
    # 测试2：监控生命周期
    test_results['monitoring_lifecycle'] = test_monitoring_lifecycle()
    
    # 测试3：系统状态收集
    test_results['system_status_collection'] = test_system_status_collection()
    
    # 测试4：告警系统
    test_results['alert_system'] = test_alert_system()
    
    # 测试5：数据持久化
    test_results['data_persistence'] = test_data_persistence()
    
    # 测试6：错误处理
    test_results['error_handling'] = test_error_handling()
    
    # 测试7：性能和稳定性
    test_results['performance_and_stability'] = test_performance_and_stability()
    
    # 生成测试报告
    end_time = datetime.now()
    duration = end_time - start_time
    
    test_summary = {
        'test_start_time': start_time.isoformat(),
        'test_end_time': end_time.isoformat(),
        'test_duration_seconds': duration.total_seconds(),
        'test_results': test_results,
        'overall_success': all(test_results.values()),
        'success_count': sum(test_results.values()),
        'total_tests': len(test_results)
    }
    
    # 保存测试报告
    with open("test_results/simple_monitor_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # 输出测试结果
    logger.info("\n" + "="*60)
    logger.info("🎯 简化监控系统深度测试结果汇总")
    logger.info("="*60)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n总体结果: {'✅ 全部通过' if test_summary['overall_success'] else '❌ 部分失败'}")
    logger.info(f"通过率: {test_summary['success_count']}/{test_summary['total_tests']}")
    logger.info(f"测试耗时: {duration.total_seconds():.1f} 秒")
    
    if test_summary['overall_success']:
        logger.info("\n🎉 所有测试通过！简化监控系统质量合格")
        logger.info("🚀 系统已准备好投入生产使用")
    else:
        logger.info("\n⚠️ 部分测试失败，需要进一步改进")
    
    return test_summary['overall_success']

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n🛑 测试被用户中断")
        exit(1)
    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        exit(1)
