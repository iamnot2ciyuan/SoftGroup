#!/bin/bash
# SoftGroup 环境激活脚本

# 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate newsoftgrouppp

# 设置 PYTHONPATH
export PYTHONPATH=/root/autodl-tmp/SoftGroup:$PYTHONPATH

echo "✓ SoftGroup 环境已激活"
echo "  - Conda 环境: newsoftgrouppp"
echo "  - PYTHONPATH: $PYTHONPATH"
echo ""
echo "您现在可以运行:"
echo "  - 训练: python tools/train.py configs/softgroup/softgroup_scannet.yaml"
echo "  - 测试: python tools/test.py configs/softgroup/softgroup_scannet.yaml CHECKPOINT"
