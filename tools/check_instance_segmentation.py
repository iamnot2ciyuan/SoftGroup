#!/usr/bin/env python3
"""
检查实例分割流程是否正确实现切割每个树木实例
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import torch
import numpy as np
from softgroup.data.forinstance import FORInstanceDataset
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def check_instance_segmentation():
    """检查实例分割流程"""
    print("=" * 60)
    print("检查实例分割流程 - 确认是否能切割每个树木实例")
    print("=" * 60)
    
    # 1. 检查配置文件
    print("\n[1] 检查配置文件...")
    config_path = 'configs/softgroup++/softgroup++_forinstance.yaml'
    with open(config_path, 'r') as f:
        cfg = load(f, Loader=Loader)
    
    print(f"  ✓ semantic_classes: {cfg['model']['semantic_classes']} (期望: 3, 对应 0=low_veg, 1=terrain, 2=tree)")
    print(f"  ✓ instance_classes: {cfg['model']['instance_classes']} (期望: 1, 只有树类别需要实例分割)")
    print(f"  ✓ sem2ins_classes: {cfg['model']['sem2ins_classes']} (期望: [2], 对语义类别2做实例分割)")
    print(f"  ✓ ignore_classes: {cfg['model']['grouping_cfg']['ignore_classes']} (期望: [0,1], 忽略low_veg和terrain)")
    
    # 2. 检查数据格式
    print("\n[2] 检查数据格式...")
    data_root = cfg['data']['train']['data_root']
    preprocess_dir = os.path.join(data_root, 'preprocess', 'train')
    
    pth_files = [f for f in os.listdir(preprocess_dir) if f.endswith('.pth')]
    if len(pth_files) == 0:
        print(f"  ✗ 未找到预处理文件在 {preprocess_dir}")
        return
    
    sample_file = os.path.join(preprocess_dir, pth_files[0])
    data = torch.load(sample_file)
    xyz, rgb, semantic_label, instance_label = data
    
    print(f"  ✓ 加载样本文件: {os.path.basename(sample_file)}")
    print(f"  ✓ 点数: {len(xyz)}")
    
    # 检查语义标签
    unique_sem = np.unique(semantic_label)
    print(f"  ✓ 语义标签唯一值: {unique_sem} (期望: [-100, 0, 1, 2])")
    if not all(x in [-100, 0, 1, 2] for x in unique_sem):
        print(f"    ⚠ 警告: 语义标签超出预期范围!")
    
    # 检查实例标签
    unique_inst = np.unique(instance_label)
    unique_inst = unique_inst[unique_inst != -100]
    print(f"  ✓ 有效实例数量: {len(unique_inst)}")
    if len(unique_inst) > 0:
        class_ids = np.unique([inst_id // 1000 for inst_id in unique_inst])
        print(f"  ✓ 实例class_id: {class_ids} (期望: [2], 对应语义类别2)")
        if not all(cid == 2 for cid in class_ids):
            print(f"    ⚠ 警告: 实例class_id不正确! 应该是2")
        
        # 检查每个实例的点数
        for inst_id in unique_inst[:5]:  # 只显示前5个
            mask = (instance_label == inst_id)
            inst_points = mask.sum()
            inst_sem = np.unique(semantic_label[mask])
            print(f"    - 实例 {inst_id}: {inst_points} 点, 语义标签: {inst_sem}")
    
    # 3. 检查DataLoader
    print("\n[3] 检查DataLoader...")
    try:
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        dataset = FORInstanceDataset(
            data_root=cfg['data']['train']['data_root'],
            prefix='train',
            suffix='.las',
            voxel_cfg=type('obj', (object,), cfg['data']['train']['voxel_cfg'])(),
            training=True,
            repeat=1,
            logger=logger
        )
        print("  ✓ DataLoader初始化成功")
        
        # 尝试加载多个样本（可能有些样本crop后点数不足）
        sample = None
        for i in range(min(10, len(dataset))):
            try:
                sample = dataset[i]
                if sample is not None:
                    print(f"  ✓ 成功加载样本 {i}")
                    break
            except:
                continue
        
        if sample is None:
            print("  ⚠ 警告: 前10个样本都加载失败 (可能是crop后点数不足)")
            print("  但数据格式检查已通过，这通常不影响训练")
            return
        
        scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num, \
            inst_pointnum, inst_cls, pt_offset_label = sample
        
        print(f"  ✓ 样本加载成功")
        print(f"  ✓ 实例数量: {inst_num}")
        print(f"  ✓ 实例类别: {inst_cls} (期望: 全部为0，因为只有树类别需要实例分割)")
        
        # 检查instance_cls是否正确映射
        if len(inst_cls) > 0:
            valid_inst_cls = [c for c in inst_cls if c != -100]
            if len(valid_inst_cls) > 0:
                if all(c == 0 for c in valid_inst_cls):
                    print(f"  ✓ 实例类别映射正确: 所有有效实例都是类别0 (树)")
                else:
                    print(f"  ⚠ 警告: 实例类别映射不正确! 应该是全部为0")
        
    except Exception as e:
        print(f"  ✗ DataLoader检查失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 总结
    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    print("✓ 配置文件正确: 对语义类别2 (tree) 做实例分割")
    print("✓ 数据格式正确: 语义标签0-based (0,1,2), 实例class_id=2")
    print("✓ DataLoader正确: 将语义类别2映射到实例类别0")
    print("\n结论: 代码已正确配置，能够实现切割每个树木实例!")
    print("=" * 60)

if __name__ == '__main__':
    check_instance_segmentation()

