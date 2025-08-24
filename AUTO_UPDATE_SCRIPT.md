# Auto-Update Script Documentation

**Purpose**: Automatically update daily reports with current project status  
**Reports**: Development Progress Report and Model Training Report  
**Language**: Chinese (中文)  
**Update Frequency**: Daily at 18:00 UTC  

## 🎯 Overview

The auto-update script automatically refreshes two key daily reports with current project metrics, development status, and model performance data.

## 📊 Reports to Update

### 1. PROJECT_DEVELOPMENT_PROGRESS_REPORT.md
**Content**: Development progress, milestone status, current priorities  
**Key Sections**:
- Core function development status
- Application function development status  
- System architecture development status
- Development milestones and progress
- Next development priorities
- Key metrics monitoring
- Risk assessment

### 2. PROJECT_MODEL_TRAINING_REPORT.md
**Content**: Model training status, performance metrics, training plans  
**Key Sections**:
- Current model status
- Training performance metrics
- Model performance trends
- Model update and maintenance
- Next training plans
- Training resource usage
- Success indicators and benchmarks

## 🔧 Implementation Options

### Option 1: Python Script with Git Integration

```python
#!/usr/bin/env python3
# auto_update_reports.py

import os
import datetime
import subprocess
from pathlib import Path

class DailyReportUpdater:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.reports = [
            "PROJECT_DEVELOPMENT_PROGRESS_REPORT.md",
            "PROJECT_MODEL_TRAINING_REPORT.md"
        ]
        
    def update_timestamp(self, file_path):
        """Update the timestamp in the report file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update report date
        current_date = datetime.datetime.now().strftime("%Y年%m月%d日")
        content = content.replace(
            "**报告日期**: 2025年8月17日",
            f"**报告日期**: {current_date}"
        )
        
        # Update next update time
        next_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y年%m月%d日")
        content = content.replace(
            "**下次更新**: 2025年8月18日 18:00 UTC",
            f"**下次更新**: {next_date} 18:00 UTC"
        )
        
        # Update generation time
        current_time = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M UTC")
        content = content.replace(
            "**报告生成时间**: 2025年8月17日 18:00 UTC",
            f"**报告生成时间**: {current_time}"
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def update_development_metrics(self):
        """Update development progress metrics"""
        # This would integrate with your project management system
        # For now, we'll just update timestamps
        pass
    
    def update_model_metrics(self):
        """Update model training metrics"""
        # This would integrate with your ML training pipeline
        # For now, we'll just update timestamps
        pass
    
    def commit_changes(self):
        """Commit updated reports to git"""
        try:
            subprocess.run(['git', 'add'] + self.reports, check=True)
            subprocess.run([
                'git', 'commit', 
                '-m', f'Daily report update - {datetime.datetime.now().strftime("%Y-%m-%d")}'
            ], check=True)
            print("✅ Reports committed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Git commit failed: {e}")
    
    def run_daily_update(self):
        """Run the complete daily update process"""
        print(f"🔄 Starting daily report update at {datetime.datetime.now()}")
        
        for report in self.reports:
            report_path = self.project_root / report
            if report_path.exists():
                print(f"📝 Updating {report}")
                self.update_timestamp(report_path)
            else:
                print(f"⚠️  Report not found: {report}")
        
        # Update metrics (placeholder for future integration)
        self.update_development_metrics()
        self.update_model_metrics()
        
        # Commit changes
        self.commit_changes()
        
        print("✅ Daily report update completed")

if __name__ == "__main__":
    updater = DailyReportUpdater()
    updater.run_daily_update()
```

### Option 2: Shell Script with Cron

```bash
#!/bin/bash
# daily_report_update.sh

# Set project directory
PROJECT_DIR="/home/sean/hydrai_swe"
cd "$PROJECT_DIR"

# Get current date
CURRENT_DATE=$(date '+%Y年%m月%d日')
NEXT_DATE=$(date -d '+1 day' '+%Y年%m月%d日')
CURRENT_TIME=$(date '+%Y年%m月%d日 %H:%M UTC')

# Update Development Progress Report
echo "🔄 Updating Development Progress Report..."
sed -i "s/**报告日期**: .*/**报告日期**: $CURRENT_DATE/" PROJECT_DEVELOPMENT_PROGRESS_REPORT.md
sed -i "s/**下次更新**: .*/**下次更新**: $NEXT_DATE 18:00 UTC/" PROJECT_DEVELOPMENT_PROGRESS_REPORT.md
sed -i "s/**报告生成时间**: .*/**报告生成时间**: $CURRENT_TIME/" PROJECT_DEVELOPMENT_PROGRESS_REPORT.md

# Update Model Training Report
echo "🔄 Updating Model Training Report..."
sed -i "s/**报告日期**: .*/**报告日期**: $CURRENT_DATE/" PROJECT_MODEL_TRAINING_REPORT.md
sed -i "s/**下次更新**: .*/**下次更新**: $NEXT_DATE 18:00 UTC/" PROJECT_MODEL_TRAINING_REPORT.md
sed -i "s/**报告生成时间**: .*/**报告生成时间**: $CURRENT_TIME/" PROJECT_MODEL_TRAINING_REPORT.md

# Git operations
git add PROJECT_DEVELOPMENT_PROGRESS_REPORT.md PROJECT_MODEL_TRAINING_REPORT.md
git commit -m "Daily report update - $(date '+%Y-%m-%d')"

echo "✅ Daily reports updated successfully"
```

## ⏰ Scheduling

### Cron Job Setup

```bash
# Edit crontab
crontab -e

# Add daily update at 18:00 UTC (adjust for your timezone)
0 18 * * * /home/sean/hydrai_swe/daily_report_update.sh >> /home/sean/hydrai_swe/logs/auto_update.log 2>&1
```

### Systemd Timer (Alternative)

```ini
# /etc/systemd/system/daily-report-update.timer
[Unit]
Description=Daily Report Update Timer
Requires=daily-report-update.service

[Timer]
OnCalendar=*-*-* 18:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```ini
# /etc/systemd/system/daily-report-update.service
[Unit]
Description=Daily Report Update Service
Type=oneshot
User=sean
WorkingDirectory=/home/sean/hydrai_swe

[Service]
ExecStart=/home/sean/hydrai_swe/daily_report_update.sh
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## 🔄 Integration Points

### Development Metrics
- **Git commits**: Count and analyze recent commits
- **Issue status**: Track open/closed issues
- **Milestone progress**: Update completion percentages
- **Build status**: Check CI/CD pipeline status

### Model Metrics
- **Training logs**: Parse training output files
- **Performance metrics**: Extract from model evaluation results
- **Resource usage**: Monitor CPU, memory, GPU usage
- **Model versions**: Track model file updates

### Data Sources
```python
# Example integration with project data
def get_development_metrics():
    """Get current development metrics"""
    metrics = {
        'total_commits': get_git_commit_count(),
        'open_issues': get_github_issue_count(),
        'milestone_progress': get_milestone_status(),
        'build_status': get_ci_status()
    }
    return metrics

def get_model_metrics():
    """Get current model training metrics"""
    metrics = {
        'latest_training': get_latest_training_time(),
        'model_performance': get_model_performance(),
        'resource_usage': get_system_resources(),
        'data_quality': get_data_quality_score()
    }
    return metrics
```

## 📝 Report Template Updates

### Development Progress Report
```markdown
## 🔄 当前开发状态

### 正在进行
- **真实数据验证**: 获取2020+真实HYDAT日流量数据
- **模型性能优化**: 基于真实数据的模型调优
- **API扩展**: 风险评估和地理区域管理端点
- **文档完善**: 技术文档和用户指南更新
- **农业模块规划**: 完成技术调研和开发路线图制定

### 下一步计划
- **土壤湿度集成**: ERA5-Land/SMAP数据接入
- **预报模式**: ECCC/ECMWF预报数据支持
- **事件阈值**: 与曼省标准对齐的分类指标
- **性能监控**: Prometheus + Grafana监控系统
- **农业模块开发**: 基于调研结果开始农业AI功能开发
```

### Model Training Report
```markdown
## 📊 当前模型状态

### 1. SWE积雪水当量预测模型 ⭐⭐⭐⭐⭐
**状态**: ✅ 生产就绪  
**模型类型**: NeuralHydrology (LSTM/TCN/GRU)  
**最后训练**: 2025年8月15日  
**训练数据**: 2020-2024年完整数据集  

#### 当前状态
- **模型精度**: ✅ 达到预期目标
- **训练状态**: ✅ 收敛稳定
- **过拟合**: ✅ 无过拟合现象
- **泛化能力**: ✅ 验证集和测试集性能一致

#### 下一步优化
- [ ] 获取2020+真实HYDAT数据重新训练
- [ ] 超参数自动调优 (Optuna)
- [ ] 集成学习 (Ensemble)
- [ ] 不确定性量化
```

## 🚀 Future Enhancements

### Automated Metrics Collection
- **GitHub API integration**: Real-time commit and issue tracking
- **MLflow integration**: Automatic model performance tracking
- **Prometheus integration**: System metrics collection
- **Database integration**: Real-time data quality metrics

### Smart Content Updates
- **AI-powered summaries**: Generate progress summaries
- **Trend analysis**: Identify performance trends
- **Recommendation engine**: Suggest next actions
- **Natural language generation**: Create human-readable updates

### Multi-language Support
- **English reports**: For international stakeholders
- **French reports**: For Canadian government users
- **Auto-translation**: Using translation APIs

## 📞 Support and Maintenance

### Troubleshooting
- **Script failures**: Check logs and permissions
- **Git issues**: Verify credentials and repository access
- **File permissions**: Ensure write access to report files
- **Cron issues**: Check system time and cron service

### Maintenance
- **Weekly**: Review and optimize update scripts
- **Monthly**: Update report templates and metrics
- **Quarterly**: Review and improve automation
- **Annually**: Major template and process updates

---

**Maintained by**: Sean Li  
**Last Updated**: 2025-08-17  
**Next Review**: 2025-09-01
