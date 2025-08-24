#!/usr/bin/env python3
"""
HydrAI-SWE 简化监控系统
合并重复功能，减少代码量，提高可维护性
"""

import psutil
import time
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """系统状态数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    process_count: int
    status: str  # healthy, warning, critical
    alerts: list

class SimpleMonitor:
    """简化监控系统 - 合并所有功能"""
    
    def __init__(self):
        self.running = False
        self.monitor_thread = None
        self.status_history = []
        self.alert_history = []
        
        # 创建目录
        os.makedirs("monitoring", exist_ok=True)
        
        # 告警阈值
        self.thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 70.0,
            'memory_critical': 95.0,
            'memory_warning': 80.0,
            'disk_critical': 95.0,
            'disk_warning': 85.0
        }
        
        logger.info("🚀 简化监控系统已初始化")
    
    def start_monitoring(self, interval: float = 10.0):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.monitor_thread.start()
        logger.info("✅ 监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 监控已停止")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.running:
            try:
                status = self._collect_system_status()
                self.status_history.append(status)
                
                # 检查告警
                if status.alerts:
                    self._handle_alerts(status)
                
                # 保持历史记录在合理范围内
                if len(self.status_history) > 100:
                    self.status_history = self.status_history[-100:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"❌ 监控错误: {e}")
                time.sleep(interval)
    
    def _collect_system_status(self) -> SystemStatus:
        """收集系统状态"""
        try:
            # 基础指标
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            process_count = len(psutil.pids())
            
            # 状态评估
            status = 'healthy'
            alerts = []
            
            # CPU检查
            if cpu_percent >= self.thresholds['cpu_critical']:
                status = 'critical'
                alerts.append(f"CPU使用率过高: {cpu_percent:.1f}%")
            elif cpu_percent >= self.thresholds['cpu_warning']:
                status = 'warning'
                alerts.append(f"CPU使用率较高: {cpu_percent:.1f}%")
            
            # 内存检查
            if memory.percent >= self.thresholds['memory_critical']:
                status = 'critical'
                alerts.append(f"内存使用率过高: {memory.percent:.1f}%")
            elif memory.percent >= self.thresholds['memory_warning']:
                status = 'warning'
                alerts.append(f"内存使用率较高: {memory.percent:.1f}%")
            
            # 磁盘检查
            if disk.percent >= self.thresholds['disk_critical']:
                status = 'critical'
                alerts.append(f"磁盘使用率过高: {disk.percent:.1f}%")
            elif disk.percent >= self.thresholds['disk_warning']:
                status = 'warning'
                alerts.append(f"磁盘使用率较高: {disk.percent:.1f}%")
            
            return SystemStatus(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                process_count=process_count,
                status=status,
                alerts=alerts
            )
            
        except Exception as e:
            logger.error(f"❌ 收集系统状态失败: {e}")
            return SystemStatus(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                process_count=0,
                status='unknown',
                alerts=[f"状态收集失败: {e}"]
            )
    
    def _handle_alerts(self, status: SystemStatus):
        """处理告警"""
        for alert in status.alerts:
            alert_record = {
                'timestamp': status.timestamp.isoformat(),
                'message': alert,
                'level': status.status,
                'cpu_percent': status.cpu_percent,
                'memory_percent': status.memory_percent,
                'disk_percent': status.disk_percent
            }
            
            self.alert_history.append(alert_record)
            logger.warning(f"🚨 {alert}")
            
            # 保持告警历史在合理范围内
            if len(self.alert_history) > 50:
                self.alert_history = self.alert_history[-50:]
    
    def get_current_status(self) -> Optional[SystemStatus]:
        """获取当前状态"""
        if self.status_history:
            return self.status_history[-1]
        return None
    
    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        if not self.status_history:
            return {'status': 'no_data'}
        
        current = self.status_history[-1]
        recent = self.status_history[-10:] if len(self.status_history) >= 10 else self.status_history
        
        # 计算趋势
        if len(recent) >= 2:
            cpu_trend = 'stable'
            if recent[-1].cpu_percent > recent[0].cpu_percent + 10:
                cpu_trend = 'increasing'
            elif recent[-1].cpu_percent < recent[0].cpu_percent - 10:
                cpu_trend = 'decreasing'
        else:
            cpu_trend = 'insufficient_data'
        
        return {
            'timestamp': current.timestamp.isoformat(),
            'current_status': current.status,
            'cpu_percent': current.cpu_percent,
            'memory_percent': current.memory_percent,
            'disk_percent': current.disk_percent,
            'process_count': current.process_count,
            'cpu_trend': cpu_trend,
            'alerts': current.alerts,
            'total_records': len(self.status_history),
            'total_alerts': len(self.alert_history)
        }
    
    def save_data(self):
        """保存监控数据"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存状态历史
            status_file = f"monitoring/status_history_{timestamp}.json"
            status_data = []
            for status in self.status_history:
                status_data.append({
                    'timestamp': status.timestamp.isoformat(),
                    'cpu_percent': status.cpu_percent,
                    'memory_percent': status.memory_percent,
                    'disk_percent': status.disk_percent,
                    'process_count': status.process_count,
                    'status': status.status,
                    'alerts': status.alerts
                })
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存告警历史
            alert_file = f"monitoring/alert_history_{timestamp}.json"
            with open(alert_file, 'w', encoding='utf-8') as f:
                json.dump(self.alert_history, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ 监控数据已保存: {status_file}, {alert_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存监控数据失败: {e}")

def main():
    """主函数：演示简化监控系统"""
    logger.info("🚀 启动简化监控系统演示")
    
    monitor = SimpleMonitor()
    
    try:
        # 启动监控
        monitor.start_monitoring(interval=5.0)
        
        # 运行演示
        logger.info("监控系统运行中，按Ctrl+C停止...")
        
        for i in range(6):  # 运行30秒
            time.sleep(5)
            
            # 获取状态摘要
            summary = monitor.get_status_summary()
            logger.info(f"状态检查 {i+1}: {summary['current_status']}")
            logger.info(f"CPU: {summary['cpu_percent']:.1f}%, 内存: {summary['memory_percent']:.1f}%, 磁盘: {summary['disk_percent']:.1f}%")
            
            if summary['alerts']:
                for alert in summary['alerts']:
                    logger.info(f"  🚨 {alert}")
        
        # 保存数据
        monitor.save_data()
        
        logger.info("✅ 简化监控系统演示完成")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 监控演示被用户中断")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
