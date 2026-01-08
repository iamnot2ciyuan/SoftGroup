#!/bin/bash
# 快速检查训练状态脚本

echo "=== 训练状态检查 $(date) ==="
echo ""

# 检查进程
TRAIN_PID=$(ps aux | grep "train.py.*configs" | grep -v grep | grep -v "DataLoader" | awk '{print $2}' | head -1)
if [ -n "$TRAIN_PID" ]; then
    echo "✓ 训练进程运行中 (PID: $TRAIN_PID)"
    echo "  进程状态:"
    ps aux | grep "$TRAIN_PID" | grep -v grep | awk '{print "    CPU: " $3 "%, MEM: " $4 "%, VSZ: " $5}'
else
    echo "✗ 训练进程未运行"
fi

echo ""
echo "监控进程:"
MONITOR_PID=$(ps aux | grep "monitor.*\.sh" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$MONITOR_PID" ]; then
    echo "✓ 监控进程运行中 (PID: $MONITOR_PID)"
else
    echo "✗ 监控进程未运行"
fi

echo ""
echo "日志文件状态:"
if [ -f "training_nohup.log" ]; then
    LOG_SIZE=$(stat -c%s training_nohup.log 2>/dev/null || stat -f%z training_nohup.log 2>/dev/null)
    LOG_TIME=$(stat -c%y training_nohup.log 2>/dev/null | cut -d. -f1)
    echo "  training_nohup.log: $(numfmt --to=iec-i --suffix=B $LOG_SIZE 2>/dev/null || echo "${LOG_SIZE}B") (更新于: $LOG_TIME)"
    echo "  最后一行:"
    tail -1 training_nohup.log | sed 's/^/    /'
fi

echo ""
echo "实时查看命令:"
echo "  tail -f training_nohup.log      # 训练日志"
echo "  tail -f training_monitor.log    # 监控日志"
