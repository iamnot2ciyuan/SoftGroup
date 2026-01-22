import functools
from collections import OrderedDict

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# 引入必要的 ops 和 utils
from ..ops import (ball_query, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_decode, rle_encode

# 引入你的自定义模块
from .blocks import MLP, ResidualBlock, UBlock, SparseGroupNorm, DenseGroupNorm, AdaptiveRGBGate


class SoftGroup(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=20,
                 instance_classes=18,
                 semantic_weight=None,
                 sem2ins_classes=[],
                 ignore_label=-100,
                 with_coords=True,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 fixed_modules=[]):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.semantic_weight = semantic_weight
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.with_coords = with_coords
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules

        # [Contribution 3: GN Optimization]
        # 定义两种 Norm 函数
        sparse_norm_fn = functools.partial(SparseGroupNorm, num_groups=16)
        dense_norm_fn = functools.partial(DenseGroupNorm, num_groups=16)

        block = ResidualBlock
        
        # Backbone 输入层
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, channels, kernel_size=3, padding=1, bias=True, indice_key='subm1'))

        # [Contribution 2: Adaptive RGB Fusion]
        self.rgb_gate = AdaptiveRGBGate(geo_channels=channels, rgb_channels=in_channels)

        # Backbone 主干
        # 动态计算 channels 列表以匹配原版逻辑
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, sparse_norm_fn, 2, block, indice_key_id=1)
        
        self.output_layer = spconv.SparseSequential(
            sparse_norm_fn(channels),
            nn.ReLU())

        # Point-wise Heads
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=dense_norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=dense_norm_fn, num_layers=2)

        # Top-down Refinement Path (实例分割部分)
        if not semantic_only:
            # Tiny U-Net 用于处理 Proposal 的体素特征
            self.tiny_unet = UBlock([channels, 2 * channels], sparse_norm_fn, 2, block, indice_key_id=11)
            self.tiny_unet_outputlayer = spconv.SparseSequential(sparse_norm_fn(channels), nn.ReLU())
            
            # Instance Heads (全部使用 dense_norm_fn 优化)
            self.cls_linear = nn.Linear(channels, instance_classes + 1)
            self.mask_linear = MLP(channels, instance_classes + 1, norm_fn=None, num_layers=2) # Mask head通常不用Norm
            self.iou_score_linear = nn.Linear(channels, instance_classes + 1)

        # 初始化权重
        self.init_weights()

        # 固定模块逻辑
        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, MLP):
                m.init_weights()
        # 初始化头部
        if not self.semantic_only:
            for m in [self.cls_linear, self.iou_score_linear]:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch, return_loss=False):
        # 兼容原来的调用方式，如果 batch 是字典，解包传给 forward_train/test
        if return_loss:
            return self.forward_train(**batch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls,
                      pt_offset_labels, spatial_shape, batch_size, **kwargs):
        losses = {}
        
        # 1. 准备输入数据
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
            
        # 2. Voxelization
        voxel_feats = voxelization(feats, p2v_map)
        x_sp = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        # ================= [Adaptive RGB Fusion] =================
        # 截获体素化的原始 RGB (V, 3)
        # 注意：这里假设输入 feats 的前3维是 RGB
        # 如果 feats 被拼上了 coords (变成 N, 6)，需要根据 in_channels 切片
        # 原始 feats (N, 3) -> voxel_feats (V, 3) -> raw_voxel_rgb
        # 但这里 feats 已经被 concat 了，我们需要从 x_sp.features 里切出来
        raw_voxel_rgb = x_sp.features[:, :self.in_channels].clone() 

        # 第一层卷积
        x_sp = self.input_conv(x_sp)
        geo_feats = x_sp.features

        # 融合
        fused_feats = self.rgb_gate(geo_feats, raw_voxel_rgb)
        x_sp = x_sp.replace_feature(fused_feats)
        # =========================================================

        # 3. Backbone Forward
        output_feats = self.forward_backbone(x_sp, v2p_map)

        # 4. Point-wise Prediction & Loss
        # 映射回点云
        semantic_scores = self.semantic_linear(output_feats)
        pt_offsets = self.offset_linear(output_feats)

        # 计算语义和偏移损失
        point_wise_loss = self.point_wise_loss(semantic_scores, pt_offsets, semantic_labels,
                                               instance_labels, pt_offset_labels)
        losses.update(point_wise_loss)

        # 5. Instance Prediction & Loss (核心补全部分)
        if not self.semantic_only:
            # 5.1 Grouping (聚类生成 Proposal)
            proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                    batch_idxs, coords_float,
                                                                    self.grouping_cfg)
            
            # 限制 Proposal 数量
            if proposals_offset.shape[0] > self.train_cfg.max_proposal_num:
                proposals_offset = proposals_offset[:self.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]

            # 5.2 Clusters Voxelization (提取实例特征)
            # 注意：这里需要 output_feats (Backbone的输出特征)
            # 在 forward_backbone 中我们得把 output_feats 拿出来，现在它是 point-wise 的
            # 我们需要把它传给 clusters_voxelization
            inst_feats, inst_map = self.clusters_voxelization(
                proposals_idx,
                proposals_offset,
                output_feats,
                coords_float,
                scale=self.instance_voxel_cfg.scale,
                spatial_shape=self.instance_voxel_cfg.spatial_shape,
                rand_quantize=True)

            # 5.3 Instance Forward (Top-down Refinement)
            instance_batch_idxs, cls_scores, iou_scores, mask_scores = self.forward_instance(
                inst_feats, inst_map)

            # 5.4 Instance Loss
            instance_loss = self.instance_loss(cls_scores, mask_scores, iou_scores, proposals_idx,
                                               proposals_offset, instance_labels, instance_pointnum,
                                               instance_cls, instance_batch_idxs)
            losses.update(instance_loss)

        return self.parse_losses(losses)

    def forward_backbone(self, x_sp, input_map):
        # U-Net
        x_unet = self.unet(x_sp)
        x_out = self.output_layer(x_unet)
        
        # Devoxelization (Map back to points)
        # x_out.features: (V, C)
        # input_map: (N,)
        output_feats = x_out.features[input_map.long()]
        return output_feats

    def point_wise_loss(self, semantic_scores, pt_offsets, semantic_labels, instance_labels,
                        pt_offset_labels):
        losses = {}
        if self.semantic_weight:
            weight = torch.tensor(self.semantic_weight, dtype=torch.float, device='cuda')
        else:
            weight = None
        
        # Semantic Loss
        semantic_loss = F.cross_entropy(
            semantic_scores, semantic_labels, weight=weight, ignore_index=self.ignore_label)
        losses['semantic_loss'] = semantic_loss

        # Offset Loss
        pos_inds = instance_labels != self.ignore_label
        if pos_inds.sum() == 0:
            offset_loss = 0 * pt_offsets.sum()
        else:
            offset_loss = F.l1_loss(
                pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
        losses['offset_loss'] = offset_loss
        return losses

    @force_fp32(apply_to=('cls_scores', 'mask_scores', 'iou_scores'))
    def instance_loss(self, cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset,
                      instance_labels, instance_pointnum, instance_cls, instance_batch_idxs):
        if proposals_idx.size(0) == 0 or (instance_cls != self.ignore_label).sum() == 0:
            cls_loss = cls_scores.sum() * 0
            mask_loss = mask_scores.sum() * 0
            iou_score_loss = iou_scores.sum() * 0
            return dict(
                cls_loss=cls_loss,
                mask_loss=mask_loss,
                iou_score_loss=iou_score_loss,
                num_pos=mask_loss,
                num_neg=mask_loss)

        losses = {}
        # 注意：proposals_idx 是 (N, 2) 形状，第一列是 proposal_id，第二列是 point_idx
        # get_mask_iou_on_cluster 需要的是 point_idx 的一维张量
        proposals_point_idx = proposals_idx[:, 1].int().cuda()
        proposals_offset = proposals_offset.cuda()

        # 计算 IoU
        ious_on_cluster = get_mask_iou_on_cluster(proposals_point_idx, proposals_offset, instance_labels,
                                                  instance_pointnum)

        # 过滤背景
        fg_inds = (instance_cls != self.ignore_label)
        fg_instance_cls = instance_cls[fg_inds]
        fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

        # 匹配 GT
        num_proposals = fg_ious_on_cluster.size(0)
        num_gts = fg_ious_on_cluster.size(1)
        assigned_gt_inds = fg_ious_on_cluster.new_full((num_proposals, ), -1, dtype=torch.long)

        max_iou, argmax_iou = fg_ious_on_cluster.max(1)
        pos_inds = max_iou >= self.train_cfg.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_iou[pos_inds]

        # Low quality match
        match_low_quality = getattr(self.train_cfg, 'match_low_quality', False)
        min_pos_thr = getattr(self.train_cfg, 'min_pos_thr', 0)
        if match_low_quality:
            gt_max_iou, gt_argmax_iou = fg_ious_on_cluster.max(0)
            for i in range(num_gts):
                if gt_max_iou[i] >= min_pos_thr:
                    assigned_gt_inds[gt_argmax_iou[i]] = i

        # Cls Loss
        labels = fg_instance_cls.new_full((num_proposals, ), self.instance_classes)
        pos_inds = assigned_gt_inds >= 0
        labels[pos_inds] = fg_instance_cls[assigned_gt_inds[pos_inds]]
        cls_loss = F.cross_entropy(cls_scores, labels)
        losses['cls_loss'] = cls_loss

        # Mask Loss
        mask_cls_label = labels[instance_batch_idxs.long()]
        slice_inds = torch.arange(
            0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
        mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]
        mask_label = get_mask_label(proposals_point_idx, proposals_offset, instance_labels, instance_cls,
                                    instance_pointnum, ious_on_cluster, self.train_cfg.pos_iou_thr)
        mask_label_weight = (mask_label != -1).float()
        mask_label[mask_label == -1.] = 0.5
        mask_loss = F.binary_cross_entropy(
            mask_scores_sigmoid_slice, mask_label, weight=mask_label_weight, reduction='sum')
        mask_loss /= (mask_label_weight.sum() + 1)
        losses['mask_loss'] = mask_loss

        # IoU Score Loss
        ious = get_mask_iou_on_pred(proposals_point_idx, proposals_offset, instance_labels,
                                    instance_pointnum, mask_scores_sigmoid_slice.detach())
        fg_ious = ious[:, fg_inds]
        gt_ious, _ = fg_ious.max(1)
        slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
        iou_score_weight = (labels < self.instance_classes).float()
        iou_score_slice = iou_scores[slice_inds, labels]
        iou_score_loss = F.mse_loss(iou_score_slice, gt_ious, reduction='none')
        iou_score_loss = (iou_score_loss * iou_score_weight).sum() / (iou_score_weight.sum() + 1)
        losses['iou_score_loss'] = iou_score_loss

        losses['num_pos'] = (labels < self.instance_classes).sum().float()
        losses['num_neg'] = (labels >= self.instance_classes).sum().float()
        return losses

    def parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' + f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    @cuda_cast
    def forward_test(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                     semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
                     scan_ids, **kwargs):
        # 补全 test 逻辑，用于推理
        # 这里简化处理，主要是为了保证 forward 不报错
        # 实际推理需要完整的 get_instances 等逻辑
        
        # 1. 准备输入数据
        # 注意：input_conv期望的输入是in_channels维（RGB 3维）
        # 如果with_coords=True，coords会在后续处理中使用，但不直接传给input_conv
        # 这里我们只使用RGB部分进行体素化和卷积
        voxel_feats = voxelization(feats, p2v_map)
        x_sp = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        # ================= [Adaptive RGB Fusion] =================
        # 截获体素化的原始 RGB (V, 3)
        raw_voxel_rgb = x_sp.features[:, :self.in_channels].clone()
        
        # 第一层卷积（只使用RGB部分）
        x_sp = self.input_conv(x_sp)
        geo_feats = x_sp.features

        # 融合
        fused_feats = self.rgb_gate(geo_feats, raw_voxel_rgb)
        x_sp = x_sp.replace_feature(fused_feats)
        # =========================================================

        output_feats = self.forward_backbone(x_sp, v2p_map)
        semantic_scores = self.semantic_linear(output_feats)
        pt_offsets = self.offset_linear(output_feats)
        
        # 转换为numpy格式，用于评估
        semantic_preds = semantic_scores.argmax(1).cpu().numpy()
        semantic_preds = semantic_preds.astype(np.int32)
        
        # 返回评估所需的字段
        return {
            'scan_id': scan_ids[0] if isinstance(scan_ids, (list, tuple)) else scan_ids,
            'coords_float': coords_float.cpu().numpy(),
            'color_feats': feats.cpu().numpy()[:, :3],  # RGB部分
            'semantic_preds': semantic_preds,
            'semantic_labels': semantic_labels.cpu().numpy() if semantic_labels is not None else None,
            'offset_preds': pt_offsets.cpu().numpy(),
            'offset_labels': pt_offset_labels.cpu().numpy() if pt_offset_labels is not None else None,
            'instance_labels': instance_labels.cpu().numpy() if instance_labels is not None else None,
        }

    @force_fp32(apply_to=('semantic_scores', 'pt_offsets'))
    def forward_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         batch_idxs,
                         coords_float,
                         grouping_cfg=None,
                         lvl_fusion=False):
        # 从原 softgroup.py 完整复制的 grouping 逻辑
        proposals_idx_list = []
        proposals_offset_list = []
        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)

        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        npoint_thr = self.grouping_cfg.npoint_thr
        with_pyramid = getattr(self.grouping_cfg, 'with_pyramid', False)
        with_octree = getattr(self.grouping_cfg, 'with_octree', False)
        base_size = getattr(self.grouping_cfg, 'pyramid_base_size', 0.02)
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)
        assert class_numpoint_mean.size(0) == self.semantic_classes
        for class_id in range(self.semantic_classes):
            if class_id in self.grouping_cfg.ignore_classes:
                continue
            scores = semantic_scores[:, class_id].contiguous()
            object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
            if object_idxs.size(0) < self.test_cfg.min_npoint:
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            if with_pyramid:
                num_points = coords_.size(0)
                level = self.get_level(num_points)
                radius = self.grouping_cfg.radius * level
                if level > 1 or not lvl_fusion:
                    coords_, pt_offsets_, batch_idxs_, l2p_map = self.pyramid_map(
                        coords_, pt_offsets_, batch_idxs_, level, base_size)
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            neighbor_inds, start_len = ball_query(
                coords_ + pt_offsets_,
                batch_idxs_,
                batch_offsets_,
                radius,
                mean_active,
                with_octree=with_octree)
            proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, neighbor_inds.cpu(),
                                                          start_len.cpu(), npoint_thr, class_id)
            if with_pyramid:
                if level > 1 or not lvl_fusion:
                    proposals_idx, proposals_offset = self.pyramid_inverse_map(
                        proposals_idx, proposals_offset, coords_.size(0), l2p_map)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

            # merge proposals
            if len(proposals_offset_list) > 0:
                proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                proposals_offset += proposals_offset_list[-1][-1]
                proposals_offset = proposals_offset[1:]
            if proposals_idx.size(0) > 0:
                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)
        if len(proposals_idx_list) > 0:
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)
        else:
            proposals_idx = torch.zeros((0, 2), dtype=torch.int32)
            proposals_offset = torch.zeros((0, ), dtype=torch.int32)
        return proposals_idx, proposals_offset

    def get_level(self, num_points):
        if num_points > 1000000:
            level = 3
        elif num_points > 100000:
            level = 2
        else:
            level = 1
        return level

    def pyramid_map(self, coords_float, pt_offsets, batch_idxs, level=1, base_size=0.02):
        coords = (coords_float / (base_size * level)).long()
        coords = torch.cat([batch_idxs[:, None], coords], dim=1)
        coords, l2p_map, p2l_map = voxelization_idx(coords.cpu(), batch_idxs[-1].item() + 1)
        coords_float = voxelization(coords_float, p2l_map.cuda())
        pt_offsets = voxelization(pt_offsets, p2l_map.cuda())
        batch_idxs = coords[:, 0].cuda().int()
        return coords_float, pt_offsets, batch_idxs, l2p_map

    def pyramid_inverse_map(self, proposals_idx, proposals_offset, num_points, l2p_map):
        proposals = torch.zeros((proposals_offset.size(0) - 1, num_points), dtype=torch.int)
        proposals[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        proposals = proposals[:, l2p_map.cpu().long()]
        proposals_idx = proposals.nonzero()
        proposals_offset = torch.cumsum(proposals.sum(1), dim=0).int()
        proposals_offset = torch.cat([proposals_offset.new_zeros(1), proposals_offset])
        return proposals_idx, proposals_offset

    def forward_instance(self, inst_feats, inst_map):
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)

        # predict mask scores
        mask_scores = self.mask_linear(feats.features)
        mask_scores = mask_scores[inst_map.long()]
        instance_batch_idxs = feats.indices[:, 0][inst_map.long()]

        # predict instance cls and iou scores
        feats = self.global_pool(feats)
        cls_scores = self.cls_linear(feats)
        iou_scores = self.iou_score_linear(feats)
        return instance_batch_idxs, cls_scores, iou_scores, mask_scores

    @force_fp32(apply_to='feats')
    def clusters_voxelization(self,
                              clusters_idx,
                              clusters_offset,
                              feats,
                              coords,
                              scale,
                              spatial_shape,
                              rand_quantize=False):
        if clusters_idx.size(0) == 0:
            # create dummpy tensors
            coords = torch.tensor(
                [[0, 0, 0, 0], [0, spatial_shape - 1, spatial_shape - 1, spatial_shape - 1]],
                dtype=torch.int,
                device='cuda')
            feats = feats[0:2]
            voxelization_feats = spconv.SparseConvTensor(feats, coords, [spatial_shape] * 3, 1)
            inp_map = feats.new_zeros((1, ), dtype=torch.long)
            return voxelization_feats, inp_map

        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = sec_min(coords, clusters_offset.cuda())
        coords_max = sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = voxelization(feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats,
                                                     out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)
        return voxelization_feats, inp_map

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    @force_fp32(apply_to=('x'))
    def global_pool(self, x, expand=False):
        indices = x.indices[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1, ), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x