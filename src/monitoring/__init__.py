#!/usr/bin/env python3
"""
HydrAI-SWE 监控系统模块
简化版监控系统，合并重复功能，提高可维护性
"""

from .simple_monitor import SimpleMonitor, SystemStatus

__all__ = ['SimpleMonitor', 'SystemStatus']

def get_monitoring_system():
    """获取监控系统实例"""
    return SimpleMonitor()

def start_monitoring(interval: float = 10.0):
    """启动监控系统"""
    monitor = SimpleMonitor()
    monitor.start_monitoring(interval)
    return monitor

def get_system_status():
    """获取系统状态"""
    monitor = SimpleMonitor()
    return monitor.get_status_summary()
