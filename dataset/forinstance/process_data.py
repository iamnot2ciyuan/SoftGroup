import os
import os.path as osp
import numpy as np
import torch
import laspy
from tqdm import tqdm
import glob
import random

# === 配置区域 ===
# 目标文件夹列表 (确保这些文件夹都在 DATA_ROOT 下)
TARGET_FOLDERS = ['CULS', 'NIBIO', 'RMIT', 'SCION', 'TUWIEN']

CHUNK_SIZE = [35.0, 35.0]  # 切块大小 35m
STRIDE = [10.0, 10.0]      # 步长 10m (重叠率)
MIN_POINTS = 1000          # 忽略稀疏块

def get_all_las_files(data_root):
    """
    强制扫描指定文件夹下的所有 .las 文件
    """
    all_files = []
    print(f"[扫描中] 正在遍历以下文件夹: {TARGET_FOLDERS}")
    
    for folder in TARGET_FOLDERS:
        # 构建搜索路径: data_root/FOLDER/*.las
        search_path = osp.join(data_root, folder, "*.las")
        found_files = glob.glob(search_path)
        
        # 兼容性检查：如果找不到，尝试递归搜索或检查大小写
        if len(found_files) == 0:
             search_path_recursive = osp.join(data_root, folder, "**", "*.las")
             found_files = glob.glob(search_path_recursive, recursive=True)
        
        print(f"  > {folder}: 找到 {len(found_files)} 个文件")
        all_files.extend(found_files)
        
    return all_files

def process_and_save(las_files, output_folder, split_name):
    """
    处理文件列表并保存
    """
    os.makedirs(output_folder, exist_ok=True)
    chunk_idx_global = 0
    
    for las_path in tqdm(las_files, desc=f"Processing {split_name}"):
        try:
            # --- A. 读取 LAS ---
            las = laspy.read(las_path)
            
            # 获取原始坐标
            xyz_global = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
            
            # B. 提取颜色 (如果缺失则填充灰色)
            if hasattr(las, 'red'):
                rgb = np.vstack((las.red, las.green, las.blue)).transpose()
                # 修复 16位颜色
                if rgb.max() > 255:
                    rgb = (rgb / 65535.0 * 255.0).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
            else:
                rgb = np.ones_like(xyz_global).astype(np.uint8) * 128

            # C. 语义标签映射
            # 原始分类：0=Unclassified, 1=Low-veg, 2=Terrain, 3=Out-points, 4=Stem, 5=Live-branches, 6=Woody-branches
            # 目标：0=Low_Veg, 1=Terrain, 2=Tree, -100=ignore
            semantic_label = np.full(xyz_global.shape[0], -100, dtype=np.int32)
            if hasattr(las, 'classification'):
                raw_cls = np.array(las.classification, dtype=np.int32)
                semantic_label[raw_cls == 1] = 0 # Low Veg
                semantic_label[raw_cls == 2] = 1 # Terrain
                # 树的各部分（Stem, Live-branches, Woody-branches）都映射到Tree类别
                semantic_label[(raw_cls == 4) | (raw_cls == 5) | (raw_cls == 6)] = 2 # Tree
                # Out-points (3) 保持为 -100 (忽略)

            # D. 实例标签处理
            # 实例标签格式：class_id * 1000 + instance_id
            # 树的实例类别ID是2，所以格式为：2 * 1000 + tree_id
            instance_label = np.full(xyz_global.shape[0], -100, dtype=np.int32)
            if hasattr(las, 'treeID'):
                tree_id = np.array(las.treeID, dtype=np.int32)
                valid_tree_mask = tree_id > 0
                
                # 强制将有 TreeID 的点设为 Tree (Class 2)
                semantic_label[valid_tree_mask] = 2 
                # 实例标签格式：2 * 1000 + tree_id
                instance_label[valid_tree_mask] = 2 * 1000 + tree_id[valid_tree_mask]

            # --- E. 滑动窗口切块 (Sliding Window) ---
            x_min, y_min = xyz_global[:, 0].min(), xyz_global[:, 1].min()
            x_max, y_max = xyz_global[:, 0].max(), xyz_global[:, 1].max()
            
            x_steps = np.arange(x_min, x_max, STRIDE[0])
            y_steps = np.arange(y_min, y_max, STRIDE[1])
            
            # 使用文件名（不带扩展名）作为前缀，方便追溯
            file_basename = osp.basename(las_path).rsplit('.', 1)[0]
            
            # 获取文件夹名（如 CULS），避免不同文件夹下同名文件冲突
            folder_name = osp.basename(osp.dirname(las_path)) 

            for x_start in x_steps:
                for y_start in y_steps:
                    x_end = x_start + CHUNK_SIZE[0]
                    y_end = y_start + CHUNK_SIZE[1]
                    
                    # 裁剪
                    mask = (xyz_global[:, 0] >= x_start) & (xyz_global[:, 0] < x_end) & \
                           (xyz_global[:, 1] >= y_start) & (xyz_global[:, 1] < y_end)
                    
                    if np.sum(mask) < MIN_POINTS:
                        continue
                    
                    # 复制数据
                    xyz_chunk = xyz_global[mask].copy()
                    rgb_chunk = rgb[mask].copy()
                    sem_chunk = semantic_label[mask].copy()
                    inst_chunk = instance_label[mask].copy()
                    
                    # [关键步骤] 局部坐标归一化
                    chunk_center_x = x_start + CHUNK_SIZE[0] / 2.0
                    chunk_center_y = y_start + CHUNK_SIZE[1] / 2.0
                    
                    xyz_chunk[:, 0] -= chunk_center_x
                    xyz_chunk[:, 1] -= chunk_center_y
                    # Z轴归零 (落地)
                    xyz_chunk[:, 2] -= xyz_chunk[:, 2].min()
                    
                    # 保存
                    # 命名格式: 文件夹_文件名_序号.pth
                    save_name = f"{folder_name}_{file_basename}_chunk_{chunk_idx_global:06d}.pth"
                    save_path = osp.join(output_folder, save_name)
                    
                    torch.save((xyz_chunk, rgb_chunk, sem_chunk, inst_chunk), save_path)
                    chunk_idx_global += 1

        except Exception as e:
            print(f"[错误] 文件 {las_path} 处理失败: {e}")

def main():
    # === 路径配置 ===
    DATA_ROOT = 'dataset/forinstance' 
    SAVE_ROOT = 'dataset/forinstance/preprocess_tiled'
    
    # 1. 获取所有文件
    all_las_files = get_all_las_files(DATA_ROOT)
    total_files = len(all_las_files)
    
    if total_files == 0:
        print("[错误] 未找到任何 .las 文件！请检查 DATA_ROOT 和文件夹结构。")
        return

    print(f"共发现 {total_files} 个文件。开始随机划分数据集...")

    # 2. 随机打乱并划分 (80% Train, 20% Val)
    random.seed(42) # 固定种子，保证复现
    random.shuffle(all_las_files)
    
    num_train = int(total_files * 0.8)
    train_files = all_las_files[:num_train]
    val_files = all_las_files[num_train:]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    # 3. 执行处理
    if len(train_files) > 0:
        process_and_save(train_files, osp.join(SAVE_ROOT, 'train'), 'train')
    
    if len(val_files) > 0:
        process_and_save(val_files, osp.join(SAVE_ROOT, 'val'), 'val')

    print("所有处理完成！")

if __name__ == '__main__':
    main()