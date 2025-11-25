import os
import os.path as osp
import glob
import numpy as np
import torch
import laspy
from tqdm import tqdm

def convert_las_to_pth(data_root, save_root, split='train'):
    """
    读取 FOR-instance 的 .las 文件，处理坐标和标签，保存为 .pth 文件
    根据 data_split_metadata.csv 来划分数据
    """
    import pandas as pd
    
    # 读取数据分割信息
    split_file = osp.join(data_root, 'data_split_metadata.csv')
    if not osp.exists(split_file):
        print(f"[错误] 找不到数据分割文件: {split_file}")
        return
    
    df = pd.read_csv(split_file)
    
    # 根据split筛选文件
    if split == 'train':
        # dev数据的80%作为训练集
        dev_files = df[df['split'] == 'dev']
        split_files = dev_files.iloc[len(dev_files)//5:]
    elif split == 'val':
        # dev数据的20%作为验证集
        dev_files = df[df['split'] == 'dev']
        split_files = dev_files.iloc[:len(dev_files)//5]
    elif split == 'test':
        # test数据作为测试集
        split_files = df[df['split'] == 'test']
    else:
        print(f"[错误] 未知的split: {split}")
        return
    
    # 构建完整路径
    las_files = []
    for _, row in split_files.iterrows():
        las_path = osp.join(data_root, row['path'])
        if osp.exists(las_path):
            las_files.append(las_path)
    
    if len(las_files) == 0:
        print(f"[警告] 没有找到 {split} 集的 .las 文件，跳过")
        return
    
    # 创建输出目录
    output_folder = osp.join(save_root, split)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"正在处理 {split} 集: 共 {len(las_files)} 个文件...")
    print(f"输出目录: {output_folder}")

    for las_path in tqdm(las_files):
        try:
            # --- A. 读取 LAS ---
            las = laspy.read(las_path)
            
            # --- B. 提取并归一化坐标 (关键步骤!) ---
            xyz = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
            
            # [核心修复] 立即将坐标移动到原点 (0,0,0)
            # 这确保了后续训练时的 Offset 计算是基于小数值的，解决了 MAE 过大的问题
            xyz -= xyz.min(0)

            # --- C. 提取颜色 ---
            if hasattr(las, 'red'):
                rgb = np.vstack((las.red, las.green, las.blue)).transpose()
                # 处理 16位颜色 (0-65535) -> 8位 (0-255)
                if rgb.max() > 255:
                    rgb = (rgb / 65535.0 * 255.0).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
            else:
                rgb = np.ones_like(xyz).astype(np.uint8) * 128

            # --- D. 处理语义标签 (Semantic) ---
            # 初始化为 -100 (Ignore)
            semantic_label = np.full(xyz.shape[0], -100, dtype=np.int32)
            
            if hasattr(las, 'classification'):
                raw_cls = np.array(las.classification, dtype=np.int32)
                
                # 映射规则 (FOR-instance -> SoftGroup)
                # 原始: 0:Unclassified, 1:Low-vegetation, 2:Terrain, 3:Out-points, 4:Stem, 5:Live-branches, 6:Woody-branches
                # SoftGroup 0-based: 0=low_vegetation, 1=terrain, 2=tree
                mask_low_veg = (raw_cls == 1)
                mask_terrain = (raw_cls == 2)
                mask_tree = (raw_cls == 4) | (raw_cls == 5) | (raw_cls == 6)
                
                semantic_label[mask_low_veg] = 0  # Low-vegetation -> 0 (0-based)
                semantic_label[mask_terrain] = 1  # Terrain -> 1 (0-based)
                semantic_label[mask_tree] = 2     # Tree -> 2 (0-based)

            # --- E. 处理实例标签 (Instance) ---
            # 初始化为 -100 (Ignore)
            instance_label = np.full(xyz.shape[0], -100, dtype=np.int32)
            
            if hasattr(las, 'treeID'):
                tree_id = np.array(las.treeID, dtype=np.int32)
                
                # 找到所有有效的树 ID (排除 0)
                unique_ids = np.unique(tree_id)
                unique_ids = unique_ids[unique_ids > 0]
                
                for i, uid in enumerate(unique_ids):
                    mask = (tree_id == uid)
                    
                    # SoftGroup 要求格式: class_id * 1000 + instance_count
                    # 树的语义类别是 2 (0-based)，所以 class_id = 2
                    # 每个树实例的ID: 2 * 1000 + instance_index
                    new_inst_id = 2 * 1000 + i
                    
                    instance_label[mask] = new_inst_id
                    
                    # 强制修正语义: 有 TreeID 的点必须是 Tree (语义类别 2, 0-based)
                    semantic_label[mask] = 2

            # --- F. 保存为 PTH ---
            file_name = osp.basename(las_path).replace('.las', '.pth')
            save_path = osp.join(output_folder, file_name)
            
            # 保存这4个 Tensor
            torch.save((xyz, rgb, semantic_label, instance_label), save_path)

        except Exception as e:
            print(f"处理文件失败 {las_path}: {e}")

if __name__ == '__main__':
    # === 配置区域 ===
    # 原始 LAS 数据的根目录
    DATA_ROOT = 'dataset/forinstance' 
    
    # 转换后的输出目录 (建议新建一个 preprocess 文件夹)
    SAVE_ROOT = 'dataset/forinstance/preprocess' 
    
    # 执行转换
    convert_las_to_pth(DATA_ROOT, SAVE_ROOT, split='train')
    convert_las_to_pth(DATA_ROOT, SAVE_ROOT, split='val')
    # convert_las_to_pth(DATA_ROOT, SAVE_ROOT, split='test')