import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset

class ForestDataset(Dataset):
    def __init__(self, data_root, prefix, suffix, training=True, 
                 voxel_size=0.1, inner_square_size=15.0):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.training = training
        self.voxel_size = voxel_size
        
        # 核心区域大小 (米)，建议设为切块大小的一半左右
        # 只有在中心区域的点才计算 Loss，避免边缘截断误差
        self.inner_square_size = inner_square_size 

        self.filenames = self.get_filenames()
        print(f"Loaded {len(self.filenames)} samples from {data_root}")

    def get_filenames(self):
        filenames = [f for f in os.listdir(self.data_root) if f.endswith(self.suffix)]
        return filenames

    def get_stem_guided_offset(self, xyz, instance_label):
        """
        【核心需求】以树干质心为偏移中心
        """
        pt_offset_label = np.zeros_like(xyz)
        mask_valid = np.zeros_like(instance_label, dtype=bool)
        
        unique_ids = np.unique(instance_label)
        for uid in unique_ids:
            if uid <= 0: continue # 忽略背景和无效点
            
            # 1. 找到这棵树的所有点
            idxs = np.where(instance_label == uid)[0]
            tree_points = xyz[idxs]
            
            # 2. 提取树基 (Bottom Part)
            # 逻辑：取 Z 值最小的 0.5m 范围
            # 抗噪：使用 np.partition 找到第 3 小的 Z 值作为基准，防止噪点干扰
            if len(tree_points) > 10:
                k = min(3, len(tree_points)-1)
                min_z = np.partition(tree_points[:, 2], k)[k]
            else:
                min_z = tree_points[:, 2].min()
                
            # 截取底部 0.5m
            base_mask = tree_points[:, 2] <= (min_z + 0.5)
            base_points = tree_points[base_mask]
            
            if len(base_points) > 0:
                # 3. 计算树干质心 (Stem Centroid)
                stem_center = np.mean(base_points, axis=0)
                
                # 4. 生成 Offset：当前点 -> 指向树干质心
                pt_offset_label[idxs] = stem_center - tree_points
                mask_valid[idxs] = True
            else:
                # 如果找不到树基（比如树被切断只剩树冠），则不计算这棵树的 Offset Loss
                mask_valid[idxs] = False
                
        return pt_offset_label, mask_valid

    def get_inner_mask(self, xyz):
        """
        生成核心区域 Mask，只对切块中心的点计算 Loss
        """
        # 计算无穷范数 (点到中心的 XY 最大距离)
        inf_norm = np.linalg.norm(xyz[:, :2], ord=np.inf, axis=1)
        
        # 只有在核心区域内的点，mask 为 True
        mask_inner = inf_norm <= (self.inner_square_size / 2)
        return mask_inner

    def __getitem__(self, index):
        # 1. 加载数据
        filepath = osp.join(self.data_root, self.filenames[index])
        # 读取 preprocess_data.py 生成的 .pth
        xyz, rgb, semantic_label, instance_label = torch.load(filepath)
        
        # 2. 数据增强 (仅训练时)
        if self.training:
            # 随机旋转 (Z轴)
            theta = np.random.uniform(0, 2 * np.pi)
            rot_mat = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            xyz = xyz @ rot_mat.T
            # 随机抖动
            xyz += np.random.normal(0, 0.01, xyz.shape)

        # 3. 计算树干引导偏移 (Stem Centroid Offset)
        pt_offset_label, mask_valid_offset = self.get_stem_guided_offset(xyz, instance_label)
        
        # 4. 计算核心区域 Mask
        mask_inner = self.get_inner_mask(xyz)
        
        # 5. 组合 Loss Mask
        # 只有同时满足：[在核心区] 且 [偏移有效]，才计算 Offset Loss
        loss_mask = mask_inner & mask_valid_offset
        
        # 6. 体素化 (Voxelization)
        locs = xyz / self.voxel_size
        
        return {
            'locs': locs.astype(np.float32), 
            'feats': rgb.astype(np.float32) / 255.0 - 0.5, # 归一化颜色
            'semantic_labels': semantic_label.astype(np.int64),
            'instance_labels': instance_label.astype(np.int64),
            'instance_pointnum': np.array([len(xyz)]), 
            'instance_cls': np.array([2]), # 假设树是 Class 2
            'pt_offset_label': pt_offset_label.astype(np.float32),
            'loss_mask': loss_mask.astype(np.bool_) # 传给 Loss 使用
        }

    def collate_fn(self, batch):
        # 请保留您 SoftGroup 原有的 collate_fn，但要确保它能处理 loss_mask
        # 这里给出一个标准示例
        locs, feats, semantic_labels, instance_labels = [], [], [], []
        pt_offset_labels, loss_masks = [], []
        instance_pointnum, instance_cls = [], []
        batch_offsets = [0]

        for i, data in enumerate(batch):
            # 添加 Batch 索引到 locs (N, 4) -> (batch_idx, x, y, z)
            item_locs = data['locs']
            item_locs = torch.cat([torch.LongTensor(item_locs.shape[0], 1).fill_(i), torch.from_numpy(item_locs)], 1)
            locs.append(item_locs)
            
            feats.append(torch.from_numpy(data['feats']))
            semantic_labels.append(torch.from_numpy(data['semantic_labels']))
            instance_labels.append(torch.from_numpy(data['instance_labels']))
            pt_offset_labels.append(torch.from_numpy(data['pt_offset_label']))
            loss_masks.append(torch.from_numpy(data['loss_mask']))
            
            instance_pointnum.append(torch.from_numpy(data['instance_pointnum']))
            instance_cls.append(torch.from_numpy(data['instance_cls']))
            batch_offsets.append(batch_offsets[-1] + item_locs.shape[0])

        return {
            'locs': torch.cat(locs, 0),
            'feats': torch.cat(feats, 0),
            'semantic_labels': torch.cat(semantic_labels, 0),
            'instance_labels': torch.cat(instance_labels, 0),
            'instance_pointnum': torch.cat(instance_pointnum, 0),
            'instance_cls': torch.cat(instance_cls, 0),
            'pt_offset_label': torch.cat(pt_offset_labels, 0),
            'loss_mask': torch.cat(loss_masks, 0),
            'batch_offsets': torch.tensor(batch_offsets, dtype=torch.int)
        }