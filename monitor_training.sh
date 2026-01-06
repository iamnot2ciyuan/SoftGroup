#!/bin/bash
# 训练监控脚本 - 实时检测训练中断原因

LOG_FILE="training_nohup.log"
MONITOR_LOG="training_monitor.log"
PID_FILE="training.pid"

echo "=== 训练监控脚本启动 ===" | tee -a "$MONITOR_LOG"
echo "监控开始时间: $(date)" | tee -a "$MONITOR_LOG"

# 获取训练进程PID
get_train_pid() {
    ps aux | grep "train.py" | grep -v grep | awk '{print $2}' | head -1
}

# 检查进程是否还在运行
check_process() {
    PID=$(get_train_pid)
    if [ -z "$PID" ]; then
        return 1  # 进程不存在
    else
        return 0  # 进程存在
    fi
}

# 监控循环
while true; do
    if ! check_process; then
        echo "" | tee -a "$MONITOR_LOG"
        echo "===== 训练进程已停止 =====" | tee -a "$MONITOR_LOG"
        echo "停止时间: $(date)" | tee -a "$MONITOR_LOG"
        
        # 检查日志最后的错误
        echo "检查最后50行日志中的错误:" | tee -a "$MONITOR_LOG"
        tail -50 "$LOG_FILE" | grep -i "error\|exception\|traceback\|killed\|oom\|segfault" | tee -a "$MONITOR_LOG"
        
        # 检查系统资源
        echo "" | tee -a "$MONITOR_LOG"
        echo "系统资源状态:" | tee -a "$MONITOR_LOG"
        free -h | tee -a "$MONITOR_LOG"
        echo "" | tee -a "$MONITOR_LOG"
        nvidia-smi | tee -a "$MONITOR_LOG"
        
        # 检查日志文件最后的内容
        echo "" | tee -a "$MONITOR_LOG"
        echo "日志文件最后20行:" | tee -a "$MONITOR_LOG"
        tail -20 "$LOG_FILE" | tee -a "$MONITOR_LOG"
        
        break
    fi
    
    # 每30秒检查一次
    sleep 30
done

echo "监控脚本结束时间: $(date)" | tee -a "$MONITOR_LOG"
