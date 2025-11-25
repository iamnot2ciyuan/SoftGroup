#!/bin/bash
# SoftGroup++ 训练进度监控脚本

LOG_FILE="/tmp/softgroup_train.log"
WORK_DIR="work_dirs/softgroup++_forinstance"

echo "============================================================"
echo "SoftGroup++ 训练进度监控"
echo "============================================================"
echo "监控时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 检查训练进程
echo "1. 训练进程状态:"
if pgrep -f "train.py" > /dev/null; then
    echo "   ✓ 训练进程正在运行"
    ps aux | grep "train.py" | grep -v grep | head -1 | awk '{print "   PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "   ✗ 训练进程未运行"
fi
echo ""

# 2. 训练进度
echo "2. 训练进度:"
if [ -f "$LOG_FILE" ]; then
    # 提取最新epoch
    LATEST_EPOCH=$(grep "Epoch \[" "$LOG_FILE" | tail -1 | grep -oP 'Epoch \[\K\d+' | head -1)
    TOTAL_EPOCH=$(grep "Epoch \[" "$LOG_FILE" | tail -1 | grep -oP 'Epoch \[\d+/\K\d+' | head -1)
    if [ ! -z "$LATEST_EPOCH" ] && [ ! -z "$TOTAL_EPOCH" ]; then
        PROGRESS=$(echo "scale=1; $LATEST_EPOCH * 100 / $TOTAL_EPOCH" | bc)
        echo "   当前Epoch: $LATEST_EPOCH / $TOTAL_EPOCH ($PROGRESS%)"
    fi
    
    # 提取最新loss
    LATEST_LOSS=$(grep "loss:" "$LOG_FILE" | tail -1 | grep -oP 'loss: \K[\d.]+')
    if [ ! -z "$LATEST_LOSS" ]; then
        echo "   最新Loss: $LATEST_LOSS"
    fi
    
    # 提取最新mIoU
    LATEST_MIOU=$(grep "mIoU:" "$LOG_FILE" | tail -1 | grep -oP 'mIoU: \K[\d.]+')
    if [ ! -z "$LATEST_MIOU" ]; then
        echo "   最新mIoU: $LATEST_MIOU%"
    fi
    
    # 提取最新准确率
    LATEST_ACC=$(grep "Acc:" "$LOG_FILE" | tail -1 | grep -oP 'Acc: \K[\d.]+')
    if [ ! -z "$LATEST_ACC" ]; then
        echo "   最新准确率: $LATEST_ACC%"
    fi
    
    # 提取ETA
    LATEST_ETA=$(grep "eta:" "$LOG_FILE" | tail -1 | grep -oP 'eta: \K[\d:]+')
    if [ ! -z "$LATEST_ETA" ]; then
        echo "   预计剩余时间: $LATEST_ETA"
    fi
else
    echo "   ✗ 日志文件不存在"
fi
echo ""

# 3. 检查点文件
echo "3. 检查点文件:"
if [ -d "$WORK_DIR" ]; then
    CHECKPOINTS=$(ls -1 "$WORK_DIR"/*.pth 2>/dev/null | wc -l)
    if [ "$CHECKPOINTS" -gt 0 ]; then
        echo "   ✓ 已保存 $CHECKPOINTS 个检查点"
        LATEST_CKPT=$(ls -t "$WORK_DIR"/epoch_*.pth 2>/dev/null | head -1)
        if [ ! -z "$LATEST_CKPT" ]; then
            CKPT_NAME=$(basename "$LATEST_CKPT")
            CKPT_SIZE=$(du -h "$LATEST_CKPT" | cut -f1)
            echo "   最新检查点: $CKPT_NAME ($CKPT_SIZE)"
        fi
    else
        echo "   ✗ 尚未生成检查点文件"
    fi
else
    echo "   ✗ 工作目录不存在"
fi
echo ""

# 4. GPU状态
echo "4. GPU状态:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | \
    awk -F', ' '{printf "   使用率: %s\n   显存: %s / %s\n   温度: %s\n", $1, $2, $3, $4}'
else
    echo "   无法查询GPU状态"
fi
echo ""

# 5. 最近训练日志
echo "5. 最近训练日志 (最后5条):"
if [ -f "$LOG_FILE" ]; then
    grep -E "(Epoch|loss|mIoU|Validation|INFO)" "$LOG_FILE" | tail -5 | while read line; do
        echo "   ${line:0:100}"
    done
fi

echo ""
echo "============================================================"

