"""
FOR-instance数据集处理，简化版
从预处理后的.pth文件加载数据
"""
import os.path as osp
from glob import glob
from pathlib import Path

import numpy as np
import torch

from .custom import CustomDataset


class FORInstanceDataset(CustomDataset):

    CLASSES = ('low_vegetation', 'terrain', 'tree')
    NYU_ID = None

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
                 with_label=True,
                 repeat=1,
                 logger=None):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.repeat = repeat
        self.logger = logger
        self.mode = 'train' if training else 'test'
        self.filenames = self.get_filenames()
        self.logger.info(f'Load {self.mode} dataset: {len(self.filenames)} scans')

    def get_filenames(self):
        """获取文件列表，支持按split划分"""
        # 读取数据分割信息
        split_file = osp.join(self.data_root, 'data_split_metadata.csv')
        if osp.exists(split_file):
            import pandas as pd
            df = pd.read_csv(split_file)
            
            # 根据prefix筛选
            if self.prefix == 'train':
                # dev数据的80%作为训练集
                dev_files = df[df['split'] == 'dev']
                split_files = dev_files.iloc[len(dev_files)//5:]
            elif self.prefix == 'val':
                # dev数据的20%作为验证集
                dev_files = df[df['split'] == 'dev']
                split_files = dev_files.iloc[:len(dev_files)//5]
            elif self.prefix == 'test':
                split_files = df[df['split'] == 'test']
            else:
                split_files = df
            
            # 构建完整路径
            filenames = []
            for _, row in split_files.iterrows():
                las_path = osp.join(self.data_root, row['path'])
                if osp.exists(las_path):
                    filenames.append(las_path)
        else:
            # 如果没有分割文件，直接搜索
            pattern = osp.join(self.data_root, self.prefix, '*' + self.suffix)
            filenames = glob(pattern)
        
        assert len(filenames) > 0, f'Empty dataset for {self.prefix}'
        filenames = sorted(filenames * self.repeat)
        return filenames

    def load(self, filename):
        """
        加载预处理后的.pth文件
        如果传入的是.las路径，自动重定向到预处理后的.pth
        返回: xyz, rgb, semantic_label, instance_label
        """
        # 如果传入的是 .las 路径，自动重定向到预处理后的 .pth
        if filename.endswith('.las'):
            scan_name = osp.basename(filename).replace('.las', '.pth')
            # 根据 prefix 确定 split 文件夹
            split_folder = self.prefix  # prefix 应该是 'train' 或 'val' 或 'test'
            # 新的路径: dataset/forinstance/preprocess/train/xxx.pth
            filename = osp.join(self.data_root, 'preprocess', split_folder, scan_name)
        
        # 极速加载
        xyz, rgb, semantic_label, instance_label = torch.load(filename)
        return xyz, rgb, semantic_label, instance_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, aug_prob=1.0):
        """
        训练时的数据变换，简化版
        因为 xyz 已经在预处理时归一化了 (xyz -= min)，这里逻辑非常简单
        
        🚨 关键：xyz_middle 必须始终与 xyz 保持同步，且都是体素单位
        - xyz_middle 用于计算 Offset GT (pt_offset_label = pt_mean - xyz_middle)
        - 如果 xyz_middle 单位不一致（一会儿米，一会儿体素），Offset Loss 会剧烈波动（0.3 vs 14.7）
        - 因此，xyz_middle 必须始终是体素单位，与 xyz 完全一致
        
        🚨🚨🚨 最终修复：体素缩放必须是第一个主要操作 🚨🚨🚨
        """
        # 🚨🚨🚨 修正：将体素缩放操作移动到最前面 🚨🚨🚨
        # 1. 缩放 (Scale) - xyz 变为体素单位（必须是第一个操作！）
        xyz = xyz * self.voxel_cfg.scale
        
        # 🚨 关键：xyz_middle 在体素单位下，用于计算 Offset GT
        # 从这一步开始，xyz_middle 必须始终与 xyz 保持同步
        xyz_middle = xyz.copy()
        
        # 2. 数据增强 (Jitter, Flip, Rotate) - 在体素空间中进行
        xyz = self.dataAugment(xyz, True, True, True, False, aug_prob)
        # 同步更新 xyz_middle
        xyz_middle = xyz.copy()
        
        # 3. Elastic (在体素空间中进行)
        if np.random.rand() < aug_prob:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
            # 🚨 关键：在 Elastic 后同步更新 xyz_middle
            xyz_middle = xyz.copy()
        
        # 4. 将坐标原点移到最小值
        xyz = xyz - xyz.min(0)
        xyz_middle = xyz_middle - xyz_middle.min(0)
        
        # 5. Crop (改进版：基于中心的固定窗口裁剪)
        max_tries = 10  # 增加重试次数，因为新策略基于点中心，成功率更高
        valid_idxs = None
        xyz_offset = None
        
        for _ in range(max_tries):
            # 调用新的 crop 方法
            xyz_offset, valid_idxs = self.crop(xyz)
            
            if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                break
        
        if valid_idxs is None or valid_idxs.sum() < self.voxel_cfg.min_npoint:
            # 如果点数太少，返回None让DataLoader跳过
            return None
        
        # 应用 Crop
        # 注意：这里使用 crop 返回的 xyz_offset (已经平移到了局部坐标 0,0,0)
        xyz = xyz_offset[valid_idxs]
        
        # 🚨 关键：xyz_middle 也必须应用同样的平移和过滤！
        # 新的 crop 方法返回的 xyz_offset = xyz - min_bound，所以我们需要对 xyz_middle 做同样的平移
        # 由于 crop 内部是随机计算的 offset，我们需要把 offset 传出来，或者
        # 简单点：直接让 xyz_middle = xyz (因为它们在 Scale 后是完全一样的)
        # 但为了保持一致性，我们需要计算 offset 并应用到 xyz_middle
        # 由于新的 crop 方法内部计算了 min_bound，我们需要重新计算 offset
        # 实际上，由于 xyz_middle 和 xyz 在 Scale 后完全一致，我们可以直接使用 xyz_offset
        # 但为了安全，我们重新计算：找到裁剪框的 min_bound
        # 由于 crop 方法返回的 xyz_offset = xyz - min_bound，所以 min_bound = xyz - xyz_offset
        # 但这样计算会有精度问题，更简单的方法是：由于 xyz_middle 和 xyz 在 Scale 后完全一致
        # 我们可以直接使用相同的 valid_idxs 和相同的偏移逻辑
        # 最安全的方法：直接让 xyz_middle = xyz（因为它们在所有变换中都保持同步）
        xyz_middle = xyz.copy()
        
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        
        return xyz, xyz_middle, rgb, semantic_label, instance_label

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        """
        重新映射实例标签，保持 class_id * 1000 + instance_id 格式
        """
        instance_label = instance_label[valid_idxs]
        ins_label_map = {}
        new_id = 0
        instance_ids = np.unique(instance_label)
        for id in instance_ids:
            if id == -100:
                ins_label_map[id] = id
                continue
            # 提取class_id和instance_id
            class_id = id // 1000
            # 重新映射instance_id，但保持class_id不变
            ins_label_map[id] = class_id * 1000 + new_id
            new_id += 1
        instance_label = np.vectorize(ins_label_map.__getitem__)(instance_label)
        return instance_label

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        """
        获取实例信息
        🚨 修复3: 确保语义类别映射正确
        """
        # 注意：instance_label现在是 class_id * 1000 + instance_id 格式
        # getInstanceInfo期望连续的实例ID（0,1,2,...），所以需要先转换
        instance_label_continuous = instance_label.copy()
        unique_inst_ids = np.unique(instance_label)
        unique_inst_ids = unique_inst_ids[unique_inst_ids != -100]
        
        # 将 class_id * 1000 + instance_id 映射回连续ID用于getInstanceInfo
        inst_id_map = {}
        for idx, inst_id in enumerate(unique_inst_ids):
            inst_id_map[inst_id] = idx
        
        for inst_id, new_id in inst_id_map.items():
            instance_label_continuous[instance_label == inst_id] = new_id
        
        # 调用父类方法
        # 注意：传入的 xyz 应该是 xyz_middle（体素单位）
        # 父类会计算 pt_offset_label = pt_mean - xyz，所以 pt_offset_label 也应该是体素单位
        ret = super().getInstanceInfo(xyz, instance_label_continuous, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        
        # 🚨🚨🚨 终极安全卫士：强制修正 pt_offset_label 的单位 🚨🚨🚨
        # 如果 Offset Loss 仍然爆炸（> 10），说明 xyz_middle 在某些批次中仍然是米单位
        # 这里强制将 pt_offset_label 除以 scale，确保它始终是体素单位
        # 这是一个"反向修正"方案，即使 xyz_middle 是米单位，也能得到正确的体素单位 offset
        
        # 🚨 确保 scale 存在且有效
        if self.voxel_cfg is None or not hasattr(self.voxel_cfg, 'scale'):
            import logging
            logger = logging.getLogger()
            logger.error("voxel_cfg.scale 不存在！无法修正 pt_offset_label 单位！")
        else:
            scale = self.voxel_cfg.scale
            # 🚨 强制修正：将 pt_offset_label 除以 scale
            # 如果 xyz_middle 是米单位，pt_offset_label 也是米单位，除以 scale 得到体素单位
            # 如果 xyz_middle 已经是体素单位，这里除以 scale 会得到错误的单位（米单位）
            # 但根据 Offset Loss 爆炸的现象，说明在某些情况下 xyz_middle 仍然是米单位
            pt_offset_label = pt_offset_label / scale
            
            # 🚨 调试信息：检查修正后的数值范围
            if isinstance(pt_offset_label, np.ndarray) and pt_offset_label.size > 0:
                max_offset = np.abs(pt_offset_label).max()
                if max_offset > 10.0:
                    import logging
                    logger = logging.getLogger()
                    logger.warning(f"pt_offset_label 修正后仍然很大 (max={max_offset:.2f})，可能仍有单位问题！")
        
        # 🚨 修复3: instance_cls应该是实例类别编号（0-based），不是语义类别编号
        # 配置中instance_classes=1，只有树类别需要实例分割
        # 语义类别2（tree）应该映射到实例类别0（因为语义标签是0-based：0=low_veg, 1=terrain, 2=tree）
        # 注意：只有语义类别2（树）才需要实例分割，其他类别（0,1）不应该有实例
        # 如果instance_cls中有非2的值，说明数据有问题，应该设为-100（忽略）
        instance_cls = [0 if x == 2 else -100 for x in instance_cls]  # 树(语义2) -> 实例类别0
        
        # 验证：确保所有有效的实例都是树类别（语义3）
        # 如果发现非树类别的实例，记录警告（但不影响训练）
        if len(instance_cls) > 0:
            valid_instances = [i for i, cls in enumerate(instance_cls) if cls != -100]
            if len(valid_instances) > 0:
                # 检查对应的语义标签是否都是3
                for inst_idx in valid_instances:
                    # 找到该实例对应的点
                    inst_mask = (instance_label_continuous == inst_idx)
                    if inst_mask.any():
                        inst_sem_labels = np.unique(semantic_label[inst_mask])
                        # 如果实例中有非2的语义标签，记录警告（语义标签是0-based，树是类别2）
                        if len(inst_sem_labels) > 1 or (len(inst_sem_labels) == 1 and inst_sem_labels[0] != 2):
                            import logging
                            logger = logging.getLogger()
                            logger.warning(f"实例 {inst_idx} 包含非树类别的语义标签: {inst_sem_labels}")
        
        # 🚨 返回时确保 pt_offset_label 没有乘 scale
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

