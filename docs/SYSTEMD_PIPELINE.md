# HydrAI-SWE 数据管道 systemd 调度指南

## 目标
- 每日/每小时定时触发数据同步
- 无交互、可观测、可重启

## 前置
- API 运行在本机 8000 端口
- curl 可用

## 示例：每日 02:30 触发全量同步

创建服务文件：`/etc/systemd/system/hydrai-pipeline-sync.service`
```ini
[Unit]
Description=HydrAI Pipeline Sync
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/curl -s -X POST http://localhost:8000/api/v1/pipeline/sync-all | /usr/bin/jq .
User=sean
Group=sean

[Install]
WantedBy=multi-user.target
```

创建定时器：`/etc/systemd/system/hydrai-pipeline-sync.timer`
```ini
[Unit]
Description=Run HydrAI Pipeline Sync daily at 02:30

[Timer]
OnCalendar=*-*-* 02:30:00
Persistent=true

[Install]
WantedBy=timers.target
```

启用并启动：
```bash
sudo systemctl daemon-reload
sudo systemctl enable hydrai-pipeline-sync.service
sudo systemctl enable --now hydrai-pipeline-sync.timer
sudo systemctl status hydrai-pipeline-sync.timer | cat
```

## 每小时执行 ECCC 同步（示例）

服务：`/etc/systemd/system/hydrai-pipeline-eccc.service`
```ini
[Unit]
Description=HydrAI ECCC hourly sync
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/curl -s -X POST 'http://localhost:8000/api/v1/pipeline/sync?source=eccc' | /usr/bin/jq .
User=sean
Group=sean
```

定时器：`/etc/systemd/system/hydrai-pipeline-eccc.timer`
```ini
[Unit]
Description=Run HydrAI ECCC sync hourly

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

启用：
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hydrai-pipeline-eccc.timer
```
