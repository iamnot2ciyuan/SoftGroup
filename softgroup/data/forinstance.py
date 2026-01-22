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
        🚨 [核心修改] 实现0.5米树干质心约束
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
        
        # 🚨🚨🚨 [核心修改] 实现0.5米树干质心约束 🚨🚨🚨
        # 注意：xyz 是 xyz_middle（体素单位），需要转换为米单位来计算0.5米约束
        scale = self.voxel_cfg.scale if self.voxel_cfg else 10.0
        xyz_meters = xyz / scale  # 转换为米单位
        
        pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
        instance_pointnum = []
        instance_cls = []
        instance_num = max(int(instance_label_continuous.max()) + 1, 0)
        
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label_continuous == i_)
            if inst_idx_i[0].size == 0:
                continue
                
            xyz_i_meters = xyz_meters[inst_idx_i]  # 米单位
            
            # 🚨 提取底部0.5米的点（树干部分）
            # 1. 找到最小Z值（抗噪：使用第3小的Z值作为基准）
            if len(xyz_i_meters) > 10:
                k = min(3, len(xyz_i_meters) - 1)
                min_z = np.partition(xyz_i_meters[:, 2], k)[k]
            else:
                min_z = xyz_i_meters[:, 2].min()
            
            # 2. 截取底部0.5米范围的点
            base_mask = xyz_i_meters[:, 2] <= (min_z + 0.5)
            base_points = xyz_i_meters[base_mask]
            
            if len(base_points) > 0:
                # 3. 计算树干质心（米单位）
                stem_center_meters = np.mean(base_points, axis=0)
                # 转换为体素单位
                stem_center = stem_center_meters * scale
                pt_mean[inst_idx_i] = stem_center
            else:
                # 如果找不到树基（比如树被切断只剩树冠），使用整棵树的质心作为fallback
                pt_mean[inst_idx_i] = xyz_i_meters.mean(0) * scale
            
            instance_pointnum.append(inst_idx_i[0].size)
            cls_idx = inst_idx_i[0][0]
            instance_cls.append(semantic_label[cls_idx])
        
        # 计算 offset label（体素单位）
        # pt_mean 和 xyz 都是体素单位，所以 pt_offset_label 也是体素单位
        pt_offset_label = pt_mean - xyz
        
        # 🚨 修复3: instance_cls应该是实例类别编号（0-based），不是语义类别编号
        # 配置中instance_classes=3，但实际只有树需要实例分割
        # 语义类别2（tree）应该映射到实例类别2（因为语义标签是0-based：0=low_veg, 1=terrain, 2=tree）
        # instance_cls 已经是从 semantic_label 获取的，所以树(语义2) -> 实例类别2
        # 但根据实际需求，只有树需要实例分割，所以将非树类别设为-100
        # 注意：如果配置中instance_classes=3，则保持原样；如果instance_classes=1，则映射为0
        # 这里根据配置保持原样，因为instance_classes=3
        
        # 验证：确保所有有效的实例都是树类别（语义2）
        # 如果发现非树类别的实例，记录警告（但不影响训练）
        if len(instance_cls) > 0:
            valid_instances = [i for i, cls in enumerate(instance_cls) if cls != -100]
            if len(valid_instances) > 0:
                # 检查对应的语义标签是否都是2（树）
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
        
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

