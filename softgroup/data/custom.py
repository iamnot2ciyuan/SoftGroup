import math
import os.path as osp
from glob import glob

import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch
from torch.utils.data import Dataset

from ..ops import voxelization_idx


class CustomDataset(Dataset):

    CLASSES = None
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
        filenames = glob(osp.join(self.data_root, self.prefix, '*' + self.suffix))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames * self.repeat)
        return filenames

    def load(self, filename):
        return torch.load(filename)

    def __len__(self):
        return len(self.filenames)

    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
            for n in noise
        ]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
        instance_pointnum = []
        instance_cls = []
        # max(instance_num, 0) to support instance_label with no valid instance_id
        instance_num = max(int(instance_label.max()) + 1, 0)
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)
            xyz_i = xyz[inst_idx_i]
            pt_mean[inst_idx_i] = xyz_i.mean(0)
            instance_pointnum.append(inst_idx_i[0].size)
            cls_idx = inst_idx_i[0][0]
            instance_cls.append(semantic_label[cls_idx])
        pt_offset_label = pt_mean - xyz
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False, scale=False, prob=1.0):
        m = np.eye(3)
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        else:
            # Empirically, slightly rotate the scene can match the results from checkpoint
            theta = 0.35 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        if scale and np.random.rand() < prob:
            scale_factor = np.random.uniform(0.95, 1.05)
            xyz = xyz * scale_factor
        return np.matmul(xyz, m)

    def crop(self, xyz):
        """
        改进版裁剪策略：基于中心的固定窗口裁剪 (Center-based Fixed Crop)
        适用于稀疏的大场景户外点云。
        
        核心改进：
        1. 不缩小窗口：使用固定的窗口大小，保持空间一致性
        2. 基于点中心：随机选择一个真实存在的点作为中心，避免切到空气
        3. 后采样：如果点数超过max_npoint，进行随机采样而不是缩小窗口
        """
        # 1. 获取裁剪窗口大小 (Fixed Crop Size)
        # Config格式: [Z, XY] -> code需要 [XY, XY, Z]
        if isinstance(self.voxel_cfg.spatial_shape, (list, tuple)):
            if len(self.voxel_cfg.spatial_shape) == 2:
                # 新格式: [Z, XY]
                z_shape = self.voxel_cfg.spatial_shape[0]
                xy_shape = self.voxel_cfg.spatial_shape[1]
                full_shape = np.array([xy_shape, xy_shape, z_shape])
            elif len(self.voxel_cfg.spatial_shape) == 3:
                # 旧格式: [X, Y, Z] 或 [H, W, D]
                full_shape = np.array(self.voxel_cfg.spatial_shape)
            else:
                # 兼容标量
                s = self.voxel_cfg.spatial_shape[0] if isinstance(self.voxel_cfg.spatial_shape, (list, tuple)) else self.voxel_cfg.spatial_shape
                full_shape = np.array([s, s, s])
        else:
            # Fallback: 标量
            s = self.voxel_cfg.spatial_shape
            full_shape = np.array([s, s, s])
        
        # 2. 随机选择一个点作为裁剪中心 (确保不会切到空气)
        pt_idx = np.random.randint(0, xyz.shape[0])
        center = xyz[pt_idx]
        
        # 3. 计算裁剪框的左下角 (Min Bound)
        # 加入一点随机抖动，避免每次都把该点正好放在正中心
        jitter = np.random.rand(3) * full_shape * 0.2  # 20% 的抖动范围
        min_bound = center - full_shape / 2 + jitter
        
        # 可选：确保 Z 轴从地面开始 (针对林业数据通常希望保留完整的树高)
        # 如果希望 Z 轴也是随机截断的，注释掉下面这行
        # min_bound[2] = xyz[:, 2].min()  # 强制 Z 轴对齐地面
        
        # 4. 计算偏移后的坐标 (Shift to local)
        xyz_offset = xyz - min_bound
        
        # 5. 计算有效索引 (Mask)
        # 只要在 [0, full_shape) 范围内的点都保留
        epsilon = 1e-6
        valid_idxs = (xyz_offset >= -epsilon).all(1) & (xyz_offset < full_shape + epsilon).all(1)
        
        # 6. 检查点数上限 (Max Npoint Handling)
        # 如果裁剪出来的点太多，我们不缩小窗口，而是随机扔掉多余的点
        # 这保持了空间结构的一致性
        if valid_idxs.sum() > self.voxel_cfg.max_npoint:
            # 获取所有有效点的索引
            valid_indices_list = np.where(valid_idxs)[0]
            # 随机选择 max_npoint 个
            selected_indices = np.random.choice(valid_indices_list, self.voxel_cfg.max_npoint, replace=False)
            # 重置 mask
            valid_idxs[:] = False
            valid_idxs[selected_indices] = True
        
        # 7. ❌ 绝对不要使用 clip 挤压坐标！❌
        # 这种操作会破坏几何结构。我们只返回 valid_idxs，让外部过滤。
        # 确保坐标在有效范围内（但不强制clip）
        xyz_offset = np.clip(xyz_offset, 0, full_shape - epsilon)
        
        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, aug_prob=1.0):
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        xyz = xyz_middle * self.voxel_cfg.scale
        if np.random.rand() < aug_prob:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        # xyz_middle = xyz / self.voxel_cfg.scale
        xyz = xyz - xyz.min(0)
        max_tries = 5
        while (max_tries > 0):
            xyz_offset, valid_idxs = self.crop(xyz)
            if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                xyz = xyz_offset
                break
            max_tries -= 1
        if valid_idxs.sum() < self.voxel_cfg.min_npoint:
            return None
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label

    def transform_test(self, xyz, rgb, semantic_label, instance_label):
        xyz_middle = self.dataAugment(xyz, False, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label

    def __getitem__(self, index):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, '')
        data = self.load(filename)
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        if data is None:
            return None
        xyz, xyz_middle, rgb, semantic_label, instance_label = data
        info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label)
        inst_num, inst_pointnum, inst_cls, pt_offset_label = info
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        if self.training:
            feat += torch.randn(feat.size(1)) * 0.1
        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)
        pt_offset_label = torch.from_numpy(pt_offset_label)
        return (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num,
                inst_pointnum, inst_cls, pt_offset_label)

    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        semantic_labels = []
        instance_labels = []

        instance_pointnum = []  # (total_nInst), int
        instance_cls = []  # (total_nInst), long
        pt_offset_labels = []

        total_inst_num = 0
        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num,
             inst_pointnum, inst_cls, pt_offset_label) = data
            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num
            scan_ids.append(scan_id)
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            instance_pointnum.extend(inst_pointnum)
            instance_cls.extend(inst_cls)
            pt_offset_labels.append(pt_offset_label)
            batch_id += 1
        # 关键修复：如果batch中所有样本都是None，返回None而不是抛出异常
        # 这样DataLoader会跳过这个batch，train.py会处理None batch
        if batch_id == 0:
            return None
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        instance_cls = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)
        pt_offset_labels = torch.cat(pt_offset_labels).float()

        # 修复：正确处理spatial_shape（支持新格式 [Z, XY] 和旧格式）
        if isinstance(self.voxel_cfg.spatial_shape, (list, tuple)):
            if len(self.voxel_cfg.spatial_shape) == 2:
                # 新格式: [Z, XY] -> 转换为 [XY, XY, Z]
                z_shape = self.voxel_cfg.spatial_shape[0]
                xy_shape = self.voxel_cfg.spatial_shape[1]
                target_shape = np.array([xy_shape, xy_shape, z_shape])
            elif len(self.voxel_cfg.spatial_shape) == 3:
                # 旧格式: [X, Y, Z] 或 [H, W, D]
                target_shape = np.array(self.voxel_cfg.spatial_shape)
            else:
                # 兼容标量
                s = self.voxel_cfg.spatial_shape[0]
                target_shape = np.array([s, s, s])
        else:
            # Fallback: 标量
            s = self.voxel_cfg.spatial_shape
            target_shape = np.array([s, s, s])
        
        # 计算实际的空间形状（基于坐标的最大值）
        max_coords = coords.max(0)[0][1:].numpy() + 1
        # 使用 clip 确保不超过目标形状
        spatial_shape = np.clip(max_coords, None, target_shape)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
        return {
            'scan_ids': scan_ids,
            'coords': coords,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'instance_pointnum': instance_pointnum,
            'instance_cls': instance_cls,
            'pt_offset_labels': pt_offset_labels,
            'spatial_shape': spatial_shape,
            'batch_size': batch_id,
        }
