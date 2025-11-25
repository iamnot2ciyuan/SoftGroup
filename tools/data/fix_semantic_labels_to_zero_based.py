#!/usr/bin/env python3
# tools/data/fix_semantic_labels_to_zero_based.py
import os
import torch
import numpy as np
from glob import glob

DATA_DIR = "dataset/forinstance/preprocess"   # 预处理数据目录
OUT_DIR = "dataset/forinstance/preprocess_fixed"  # 修复后输出目录
IGNORE_LABEL = -100  # 与 config 中 ignore_label 保持一致

os.makedirs(OUT_DIR, exist_ok=True)

# 查找所有 .pth 文件
pths = glob(os.path.join(DATA_DIR, "**/*.pth"), recursive=True) + glob(os.path.join(DATA_DIR, "*.pth"))
if not pths:
    print("未发现 .pth 文件。若你的数据是 .las/.ply，请先运行转换脚本生成 .pth。")
else:
    print(f"找到 {len(pths)} 个 .pth 文件，开始处理...")
    fixed_count = 0
    for p in pths:
        try:
            data = torch.load(p)
            
            # 数据可能是tuple格式 (xyz, rgb, semantic_label, instance_label)
            if isinstance(data, tuple) and len(data) >= 3:
                xyz, rgb, sem, inst = data[0], data[1], data[2], data[3] if len(data) > 3 else None
                sem = np.array(sem, dtype=np.int32)
                
                # 将原始语义为 -1（或其他负值，除了-100）统一为 IGNORE_LABEL
                neg_mask = (sem < 0) & (sem != IGNORE_LABEL)
                if neg_mask.any():
                    sem[neg_mask] = IGNORE_LABEL
                
                # 将大于等于1的类减1（1->0, 2->1, 3->2）
                # 但跳过 IGNORE_LABEL
                valid_mask = sem != IGNORE_LABEL
                if valid_mask.sum() > 0:
                    sem[valid_mask] = sem[valid_mask] - 1
                
                # 额外检查是否存在超范围值
                valid_sem = sem[sem != IGNORE_LABEL]
                if len(valid_sem) > 0 and valid_sem.max() >= 3:
                    print(f"警告：检测到异常大语义标签值 {valid_sem.max()}，文件: {p}")
                
                # 保存到新目录（保留原目录结构）
                rel = os.path.relpath(p, DATA_DIR)
                outp = os.path.join(OUT_DIR, rel)
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                
                # 保存修复后的数据
                fixed_data = (xyz, rgb, sem.astype(np.int32), inst) if inst is not None else (xyz, rgb, sem.astype(np.int32))
                torch.save(fixed_data, outp)
                fixed_count += 1
                if fixed_count % 5 == 0:
                    print(f"已处理 {fixed_count}/{len(pths)} 个文件...")
                    
            elif isinstance(data, dict):
                sem = data.get("semantic_labels", None)
                if sem is None:
                    print(f"跳过（无 semantic_labels）: {p}")
                    continue
                sem = np.array(sem, dtype=np.int32)
                
                # 将原始语义为 -1（或其他负值，除了-100）统一为 IGNORE_LABEL
                neg_mask = (sem < 0) & (sem != IGNORE_LABEL)
                if neg_mask.any():
                    sem[neg_mask] = IGNORE_LABEL
                
                # 将大于等于1的类减1（1->0, 2->1, 3->2）
                valid_mask = sem != IGNORE_LABEL
                if valid_mask.sum() > 0:
                    sem[valid_mask] = sem[valid_mask] - 1
                
                # 保存
                rel = os.path.relpath(p, DATA_DIR)
                outp = os.path.join(OUT_DIR, rel)
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                
                data["semantic_labels"] = sem.astype(np.int32)
                torch.save(data, outp)
                fixed_count += 1
            else:
                print(f"跳过（未知数据格式）: {p}")
                
        except Exception as e:
            print(f"处理文件失败 {p}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n全部处理完成！共修复 {fixed_count}/{len(pths)} 个文件")
    print(f"输出目录：{OUT_DIR}")
    print("\n下一步：")
    print("1. 备份原数据（可选）: mv dataset/forinstance/preprocess dataset/forinstance/preprocess_bak")
    print("2. 使用修复后的数据: mv dataset/forinstance/preprocess_fixed dataset/forinstance/preprocess")
    print("   或者修改配置文件的 data_root 指向修复后的目录")

