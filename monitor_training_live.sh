#!/bin/bash
# 实时监控训练日志脚本

LOG_FILE="work_dirs/softgroup++_forinstance/20260106_003506.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "日志文件不存在: $LOG_FILE"
    echo "正在查找最新的日志文件..."
    LOG_FILE=$(find work_dirs -name "*.log" -type f -mtime -1 | head -1)
    if [ -z "$LOG_FILE" ]; then
        echo "未找到训练日志文件"
        exit 1
    fi
    echo "使用日志文件: $LOG_FILE"
fi

echo "=========================================="
echo "  训练日志实时监控"
echo "  日志文件: $LOG_FILE"
echo "  按 Ctrl+C 退出"
echo "=========================================="
echo ""

# 实时跟踪日志
tail -f "$LOG_FILE"

