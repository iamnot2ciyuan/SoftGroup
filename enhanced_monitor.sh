#!/bin/bash
# 增强版训练监控脚本

TRAIN_LOG="training_nohup.log"
MONITOR_LOG="training_monitor.log"
CHECK_INTERVAL=30  # 每30秒检查一次

echo "=== 训练监控启动 $(date) ===" | tee -a "$MONITOR_LOG"

# 获取主训练进程PID（排除worker进程）
get_main_pid() {
    ps aux | grep "train.py.*configs" | grep -v grep | grep -v "DataLoader" | awk '{print $2}' | head -1
}

last_log_size=0
last_log_time=$(date +%s)

while true; do
    PID=$(get_main_pid)
    
    if [ -z "$PID" ]; then
        echo "" | tee -a "$MONITOR_LOG"
        echo "===== 训练进程停止检测 $(date) =====" | tee -a "$MONITOR_LOG"
        echo "主进程PID: 不存在" | tee -a "$MONITOR_LOG"
        
        # 检查日志是否还在增长
        current_log_size=$(stat -f%z "$TRAIN_LOG" 2>/dev/null || stat -c%s "$TRAIN_LOG" 2>/dev/null || echo "0")
        echo "日志文件大小: $current_log_size" | tee -a "$MONITOR_LOG"
        
        # 检查最后的错误
        echo "" | tee -a "$MONITOR_LOG"
        echo "=== 错误信息检查 ===" | tee -a "$MONITOR_LOG"
        tail -100 "$TRAIN_LOG" | grep -i -A 10 "error\|exception\|traceback\|killed\|segfault\|abort\|fatal" | head -30 | tee -a "$MONITOR_LOG"
        
        # 检查系统资源
        echo "" | tee -a "$MONITOR_LOG"
        echo "=== 系统资源状态 ===" | tee -a "$MONITOR_LOG"
        free -h | tee -a "$MONITOR_LOG"
        echo "" | tee -a "$MONITOR_LOG"
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv | tee -a "$MONITOR_LOG"
        
        # 日志最后20行
        echo "" | tee -a "$MONITOR_LOG"
        echo "=== 日志最后20行 ===" | tee -a "$MONITOR_LOG"
        tail -20 "$TRAIN_LOG" | tee -a "$MONITOR_LOG"
        
        break
    fi
    
    # 检查日志是否还在更新
    current_log_size=$(stat -f%z "$TRAIN_LOG" 2>/dev/null || stat -c%s "$TRAIN_LOG" 2>/dev/null || echo "0")
    current_time=$(date +%s)
    
    if [ "$current_log_size" -eq "$last_log_size" ] && [ $(($current_time - $last_log_time)) -gt 300 ]; then
        echo "警告: 日志文件已300秒未更新 $(date)" | tee -a "$MONITOR_LOG"
        echo "进程PID: $PID, 日志大小: $current_log_size" | tee -a "$MONITOR_LOG"
    fi
    
    last_log_size=$current_log_size
    last_log_time=$current_time
    
    sleep $CHECK_INTERVAL
done

echo "监控结束 $(date)" | tee -a "$MONITOR_LOG"
