#!/usr/bin/env python3
# 修复实例标签的class_id，使其与语义标签匹配
import os
import torch
import numpy as np
from glob import glob

DATA_DIR = "dataset/forinstance/preprocess_tiled"  # 预处理数据目录（直接覆盖原文件）
IGNORE_LABEL = -100

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
                
                # 修复：将 class_id 改为 2（因为语义标签中树是类别2）
                valid_inst_mask = inst != IGNORE_LABEL
                if valid_inst_mask.sum() > 0:
                    # 提取class_id和instance_id
                    old_class_ids = inst[valid_inst_mask] // 1000
                    old_inst_ids = inst[valid_inst_mask] % 1000
                    
                    # 检查：如果实例标签不是 class_id * 1000 + instance_id 格式（即 class_id 为 0 或很大的值）
                    # 说明是直接的 tree_id，需要先编码
                    unique_class_ids = np.unique(old_class_ids)
                    if len(unique_class_ids) > 0 and (unique_class_ids[0] == 0 or unique_class_ids.max() > 100):
                        # 如果 class_id 是 0 或者很大（说明可能是直接的 tree_id），则编码为 2 * 1000 + tree_id
                        new_inst = inst.copy()
                        # 将所有有效的实例标签（包括可能的直接 tree_id）编码为 2 * 1000 + instance_id
                        new_inst[valid_inst_mask] = 2 * 1000 + old_inst_ids
                        if fixed_count % 50 == 0:
                            print(f"{os.path.basename(p)}: 直接 tree_id 编码为 class_id=2, instances: {len(np.unique(old_inst_ids))}")
                    else:
                        # 如果已经是正确的格式，只需修复 class_id（将非 2 的改为 2）
                        new_inst = inst.copy()
                        needs_fix = (old_class_ids != 2) & (old_class_ids > 0)
                        if needs_fix.any():
                            fix_mask = valid_inst_mask.copy()
                            fix_mask[valid_inst_mask] = needs_fix
                            new_inst[fix_mask] = 2 * 1000 + old_inst_ids[needs_fix]
                            if fixed_count % 50 == 0:
                                print(f"{os.path.basename(p)}: class_id {np.unique(old_class_ids[needs_fix])} -> 2, instances: {len(np.unique(old_inst_ids))}")
                        else:
                            new_inst = inst  # 已经是正确的格式，不需要修改
                else:
                    new_inst = inst
                
                # 直接覆盖原文件
                fixed_data = (xyz, rgb, sem, new_inst.astype(np.int32))
                torch.save(fixed_data, p)
                fixed_count += 1
                
        except Exception as e:
            print(f"处理文件失败 {p}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n全部处理完成！共修复 {fixed_count}/{len(pths)} 个文件")
    print(f"文件已直接覆盖到：{DATA_DIR}")

