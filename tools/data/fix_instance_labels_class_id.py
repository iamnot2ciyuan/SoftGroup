#!/usr/bin/env python3
# 修复实例标签的class_id，使其与语义标签匹配
import os
import torch
import numpy as np
from glob import glob

DATA_DIR = "dataset/forinstance/preprocess"
OUT_DIR = "dataset/forinstance/preprocess_fixed_inst"
IGNORE_LABEL = -100

os.makedirs(OUT_DIR, exist_ok=True)

# 查找所有 .pth 文件
pths = glob(os.path.join(DATA_DIR, "**/*.pth"), recursive=True)
if not pths:
    print("未发现 .pth 文件")
else:
    print(f"找到 {len(pths)} 个 .pth 文件，开始修复实例标签的class_id...")
    fixed_count = 0
    
    for p in pths:
        try:
            data = torch.load(p)
            
            if isinstance(data, tuple) and len(data) >= 4:
                xyz, rgb, sem, inst = data[0], data[1], data[2], data[3]
                sem = np.array(sem, dtype=np.int32)
                inst = np.array(inst, dtype=np.int32)
                
                # 修复：将 class_id=3 改为 class_id=2（因为语义标签中树是类别2）
                valid_inst_mask = inst != IGNORE_LABEL
                if valid_inst_mask.sum() > 0:
                    # 提取class_id和instance_id
                    old_class_ids = inst[valid_inst_mask] // 1000
                    old_inst_ids = inst[valid_inst_mask] % 1000
                    
                    # 将class_id从3改为2
                    new_inst = inst.copy()
                    new_inst[valid_inst_mask] = 2 * 1000 + old_inst_ids
                    
                    print(f"{os.path.basename(p)}: class_id {np.unique(old_class_ids)} -> 2, instances: {len(np.unique(old_inst_ids))}")
                else:
                    new_inst = inst
                
                # 保存到新目录
                rel = os.path.relpath(p, DATA_DIR)
                outp = os.path.join(OUT_DIR, rel)
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                
                fixed_data = (xyz, rgb, sem, new_inst.astype(np.int32))
                torch.save(fixed_data, outp)
                fixed_count += 1
                
        except Exception as e:
            print(f"处理文件失败 {p}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n全部处理完成！共修复 {fixed_count}/{len(pths)} 个文件")
    print(f"输出目录：{OUT_DIR}")

