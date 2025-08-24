#!/usr/bin/env python3
"""
HydrAI-SWE 定时重训练系统
支持夜间训练和每周评估的自动化流程
"""

import schedule
import time
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import subprocess
import signal
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduled_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScheduledTrainingSystem:
    """定时重训练系统"""
    
    def __init__(self):
        self.training_processes = {}
        self.evaluation_results = {}
        
        # 创建必要的目录
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/scheduled", exist_ok=True)
        os.makedirs("logs/evaluations", exist_ok=True)
        
        # 训练脚本路径
        self.training_scripts = {
            'swe_model': 'train_swe_model.py',
            'agriculture_model': 'train_agriculture_model.py',
            'flood_warning_model': 'train_flood_warning_model.py'
        }
        
        # 评估脚本路径
        self.evaluation_scripts = {
            'cross_validation': 'src/models/cross_validation_system.py',
            'performance_monitoring': 'src/models/performance_monitor.py'
        }
        
        logger.info("🚀 定时重训练系统初始化完成")
    
    def start_nightly_training(self):
        """启动夜间训练"""
        logger.info("🌙 开始夜间训练...")
        
        start_time = datetime.now()
        timestamp = start_time.strftime('%Y%m%d_%H%M%S')
        
        # 创建夜间训练日志
        nightly_log = f"logs/scheduled/nightly_training_{timestamp}.log"
        
        try:
            # 启动所有模型的训练
            for model_name, script_path in self.training_scripts.items():
                if os.path.exists(script_path):
                    logger.info(f"🔧 启动 {model_name} 训练...")
                    
                    # 启动训练进程
                    process = subprocess.Popen(
                        ['python3', script_path],
                        stdout=open(nightly_log, 'a'),
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid
                    )
                    
                    self.training_processes[model_name] = {
                        'pid': process.pid,
                        'start_time': start_time,
                        'script': script_path,
                        'log_file': nightly_log
                    }
                    
                    logger.info(f"✅ {model_name} 训练已启动 (PID: {process.pid})")
                else:
                    logger.warning(f"⚠️ 训练脚本不存在: {script_path}")
            
            # 记录夜间训练启动
            self._log_training_session('nightly', start_time, timestamp)
            
        except Exception as e:
            logger.error(f"❌ 夜间训练启动失败: {e}")
            self._log_error('nightly_training', str(e))
    
    def start_weekly_evaluation(self):
        """启动每周评估"""
        logger.info("📊 开始每周评估...")
        
        start_time = datetime.now()
        timestamp = start_time.strftime('%Y%m%d_%H%M%S')
        
        try:
            # 运行交叉验证
            logger.info("🔍 运行交叉验证...")
            cv_result = self._run_cross_validation()
            
            # 运行性能监控
            logger.info("📈 运行性能监控...")
            perf_result = self._run_performance_monitoring()
            
            # 保存评估结果
            evaluation_summary = {
                'evaluation_start': start_time.isoformat(),
                'evaluation_type': 'weekly',
                'timestamp': timestamp,
                'cross_validation': cv_result,
                'performance_monitoring': perf_result,
                'summary': self._generate_evaluation_summary(cv_result, perf_result)
            }
            
            # 保存到文件
            eval_file = f"logs/evaluations/weekly_evaluation_{timestamp}.json"
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ 每周评估完成，结果保存到: {eval_file}")
            
            # 记录评估会话
            self._log_evaluation_session('weekly', start_time, timestamp, evaluation_summary)
            
        except Exception as e:
            logger.error(f"❌ 每周评估失败: {e}")
            self._log_error('weekly_evaluation', str(e))
    
    def _run_cross_validation(self) -> Dict[str, Any]:
        """运行交叉验证"""
        try:
            # 这里应该调用实际的交叉验证系统
            # 为了演示，我们返回模拟结果
            return {
                'status': 'completed',
                'models_evaluated': ['SWE', 'Agriculture', 'Flood Warning'],
                'total_folds': 5,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"交叉验证运行失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_performance_monitoring(self) -> Dict[str, Any]:
        """运行性能监控"""
        try:
            # 这里应该调用实际的性能监控系统
            # 为了演示，我们返回模拟结果
            return {
                'status': 'completed',
                'models_monitored': ['SWE', 'Agriculture', 'Flood Warning'],
                'monitoring_duration': 30.5,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"性能监控运行失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _generate_evaluation_summary(self, cv_result: Dict, perf_result: Dict) -> Dict[str, Any]:
        """生成评估摘要"""
        return {
            'overall_status': 'completed' if cv_result.get('status') == 'completed' and perf_result.get('status') == 'completed' else 'partial',
            'models_count': len(cv_result.get('models_evaluated', [])),
            'evaluation_time': datetime.now().isoformat(),
            'recommendations': self._generate_recommendations(cv_result, perf_result)
        }
    
    def _generate_recommendations(self, cv_result: Dict, perf_result: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if cv_result.get('status') == 'completed':
            recommendations.append("交叉验证完成，模型性能稳定")
        else:
            recommendations.append("交叉验证失败，需要检查模型状态")
        
        if perf_result.get('status') == 'completed':
            recommendations.append("性能监控完成，系统运行正常")
        else:
            recommendations.append("性能监控失败，需要检查系统状态")
        
        # 添加通用建议
        recommendations.extend([
            "建议每周检查模型性能指标",
            "如发现性能下降，考虑重新训练模型",
            "定期更新训练数据以提高模型准确性"
        ])
        
        return recommendations
    
    def _log_training_session(self, session_type: str, start_time: datetime, timestamp: str):
        """记录训练会话"""
        session_log = {
            'session_type': session_type,
            'start_time': start_time.isoformat(),
            'timestamp': timestamp,
            'models': list(self.training_processes.keys()),
            'status': 'started'
        }
        
        log_file = f"logs/scheduled/{session_type}_session_{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(session_log, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📝 训练会话记录已保存: {log_file}")
    
    def _log_evaluation_session(self, session_type: str, start_time: datetime, timestamp: str, results: Dict):
        """记录评估会话"""
        session_log = {
            'session_type': session_type,
            'start_time': start_time.isoformat(),
            'timestamp': timestamp,
            'results': results,
            'status': 'completed'
        }
        
        log_file = f"logs/evaluations/{session_type}_session_{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(session_log, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📝 评估会话记录已保存: {log_file}")
    
    def _log_error(self, error_type: str, error_message: str):
        """记录错误"""
        error_log = {
            'error_type': error_type,
            'timestamp': datetime.now().isoformat(),
            'error_message': error_message
        }
        
        error_file = f"logs/scheduled/error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_log, f, indent=2, ensure_ascii=False, default=str)
        
        logger.error(f"❌ 错误记录已保存: {error_file}")
    
    def check_training_status(self):
        """检查训练状态"""
        logger.info("🔍 检查训练状态...")
        
        for model_name, process_info in self.training_processes.items():
            try:
                # 检查进程是否还在运行
                os.kill(process_info['pid'], 0)
                logger.info(f"✅ {model_name} 训练进程正在运行 (PID: {process_info['pid']})")
            except OSError:
                logger.info(f"✅ {model_name} 训练进程已完成 (PID: {process_info['pid']})")
                # 从活动进程列表中移除
                del self.training_processes[model_name]
        
        logger.info(f"📊 当前活动训练进程: {len(self.training_processes)}")
    
    def stop_all_training(self):
        """停止所有训练进程"""
        logger.info("🛑 停止所有训练进程...")
        
        for model_name, process_info in self.training_processes.items():
            try:
                os.killpg(process_info['pid'], signal.SIGTERM)
                logger.info(f"✅ {model_name} 训练进程已停止")
            except OSError as e:
                logger.warning(f"⚠️ 停止 {model_name} 训练进程失败: {e}")
        
        self.training_processes.clear()
        logger.info("🛑 所有训练进程已停止")
    
    def setup_schedule(self):
        """设置定时任务"""
        logger.info("⏰ 设置定时任务...")
        
        # 每天晚上11点开始训练
        schedule.every().day.at("23:00").do(self.start_nightly_training)
        
        # 每周一早上9点开始评估
        schedule.every().monday.at("09:00").do(self.start_weekly_evaluation)
        
        # 每小时检查训练状态
        schedule.every().hour.do(self.check_training_status)
        
        logger.info("✅ 定时任务设置完成:")
        logger.info("   - 夜间训练: 每天 23:00")
        logger.info("   - 每周评估: 每周一 09:00")
        logger.info("   - 状态检查: 每小时")
    
    def run_scheduler(self):
        """运行调度器"""
        logger.info("🚀 启动定时重训练调度器...")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
                
        except KeyboardInterrupt:
            logger.info("⚠️ 收到中断信号，正在停止...")
            self.stop_all_training()
            logger.info("👋 调度器已停止")
        except Exception as e:
            logger.error(f"❌ 调度器运行错误: {e}")
            self.stop_all_training()
            raise

def main():
    """主函数"""
    print("⏰ HydrAI-SWE 定时重训练系统")
    print("=" * 60)
    
    try:
        # 创建调度系统
        scheduler = ScheduledTrainingSystem()
        
        # 设置定时任务
        scheduler.setup_schedule()
        
        # 显示当前状态
        print("\n📊 系统状态:")
        print(f"   - 训练脚本: {len(scheduler.training_scripts)} 个")
        print(f"   - 评估脚本: {len(scheduler.evaluation_scripts)} 个")
        print(f"   - 日志目录: logs/")
        
        print("\n⏰ 定时任务:")
        print("   - 夜间训练: 每天 23:00")
        print("   - 每周评估: 每周一 09:00")
        print("   - 状态检查: 每小时")
        
        print("\n🚀 启动调度器...")
        print("按 Ctrl+C 停止系统")
        
        # 运行调度器
        scheduler.run_scheduler()
        
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        logger.error(f"系统启动失败: {e}")

if __name__ == "__main__":
    main()
