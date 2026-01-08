#!/usr/bin/env python3
# tools/data/fix_semantic_labels_to_zero_based.py
import os
import torch
import numpy as np
from glob import glob

DATA_DIR = "dataset/forinstance/preprocess_tiled"   # 预处理数据目录（直接覆盖原文件）
IGNORE_LABEL = -100  # 与 config 中 ignore_label 保持一致

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
                
                # 检查是否需要转换：如果标签最大值 > 2，说明是1-based，需要减1
                # 如果最大值 <= 2 且最小值 >= 0，说明已经是0-based，不需要转换
                valid_mask = sem != IGNORE_LABEL
                if valid_mask.sum() > 0:
                    valid_sem = sem[valid_mask]
                    max_sem = valid_sem.max()
                    min_sem = valid_sem.min()
                    # 如果最大值 > 2 或最小值 > 0，说明可能是1-based，需要减1
                    if max_sem > 2 or (min_sem > 0 and max_sem > 2):
                        sem[valid_mask] = sem[valid_mask] - 1
                    # 如果已经是0-based（最大值 <= 2 且最小值 >= 0），不需要转换
                    # 但需要确保所有负值（除了-100）都被设置为-100
                
                # 额外检查是否存在超范围值
                valid_sem = sem[sem != IGNORE_LABEL]
                if len(valid_sem) > 0 and valid_sem.max() >= 3:
                    print(f"警告：检测到异常大语义标签值 {valid_sem.max()}，文件: {p}")
                
                # 直接覆盖原文件
                fixed_data = (xyz, rgb, sem.astype(np.int32), inst) if inst is not None else (xyz, rgb, sem.astype(np.int32))
                torch.save(fixed_data, p)
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
                
                # 检查是否需要转换：如果标签最大值 > 2，说明是1-based，需要减1
                # 如果最大值 <= 2 且最小值 >= 0，说明已经是0-based，不需要转换
                valid_mask = sem != IGNORE_LABEL
                if valid_mask.sum() > 0:
                    valid_sem = sem[valid_mask]
                    max_sem = valid_sem.max()
                    min_sem = valid_sem.min()
                    # 如果最大值 > 2 或最小值 > 0，说明可能是1-based，需要减1
                    if max_sem > 2 or (min_sem > 0 and max_sem > 2):
                        sem[valid_mask] = sem[valid_mask] - 1
                    # 如果已经是0-based（最大值 <= 2 且最小值 >= 0），不需要转换
                    # 但需要确保所有负值（除了-100）都被设置为-100
                
                # 直接覆盖原文件
                data["semantic_labels"] = sem.astype(np.int32)
                torch.save(data, p)
                fixed_count += 1
            else:
                print(f"跳过（未知数据格式）: {p}")
                
        except Exception as e:
            print(f"处理文件失败 {p}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n全部处理完成！共修复 {fixed_count}/{len(pths)} 个文件")
    print(f"文件已直接覆盖到：{DATA_DIR}")

