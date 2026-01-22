from collections import OrderedDict

import spconv.pytorch as spconv
import torch
import torch.nn as nn
from spconv.pytorch.modules import SparseModule


# [Contribution 3: GN Optimization - Sparse Version]
class SparseGroupNorm(nn.Module):
    """
    专门为 SparseConvTensor 设计的 GroupNorm。
    参数顺序经过封装，兼容 BatchNorm 的调用方式：(num_channels, num_groups)
    """
    def __init__(self, num_channels, num_groups=16):
        super().__init__()
        # 自动调整组数，防止通道数过少报错
        if num_channels < num_groups:
            num_groups = max(1, num_channels)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, x):
        if isinstance(x, spconv.SparseConvTensor):
            features = x.features
            out_features = self.gn(features)
            x = x.replace_feature(out_features)
            return x
        else:
            # 防御性编程：万一传入的是普通 Tensor
            return self.gn(x)


# [Contribution 3: GN Optimization - Dense Version]
class DenseGroupNorm(nn.Module):
    """
    专门为 MLP/Linear 层设计的 GroupNorm 包装器。
    【关键修复】：nn.GroupNorm 的第一个参数是 num_groups，而 BatchNorm 是 num_features。
    这个类将参数顺序翻转，使其可以像 BatchNorm 一样被 MLP 类调用：DenseGroupNorm(channels)
    """
    def __init__(self, num_channels, num_groups=16):
        super().__init__()
        if num_channels < num_groups:
            num_groups = max(1, num_channels)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, x):
        return self.gn(x)


# [Contribution 2: Adaptive RGB Fusion]
class AdaptiveRGBGate(nn.Module):
    def __init__(self, geo_channels, rgb_channels, hidden_dim=16):
        super().__init__()
        # RGB 编码器
        self.rgb_encoder = nn.Sequential(
            nn.Linear(rgb_channels, geo_channels),
            nn.ReLU(inplace=True),
            nn.Linear(geo_channels, geo_channels)
        )
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(geo_channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, geo_channels),
            nn.Sigmoid()
        )

    def forward(self, geo_feats, raw_rgb):
        # 安全检查：确保设备和类型一致
        if raw_rgb.device != geo_feats.device:
            raw_rgb = raw_rgb.to(geo_feats.device)
        if raw_rgb.dtype != geo_feats.dtype:
            raw_rgb = raw_rgb.to(geo_feats.dtype)

        rgb_feats = self.rgb_encoder(raw_rgb)
        
        # 拼接 -> 计算权重
        cat_feats = torch.cat([geo_feats, rgb_feats], dim=1)
        gate = self.gate_net(cat_feats)

        # 融合：Geo + Gate * RGB
        return geo_feats + (gate * rgb_feats)


class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                # 这里 norm_fn(in_channels) 调用必须要求 norm_fn 的第一个参数是 channels
                # 所以必须使用 DenseGroupNorm 而不是原始的 nn.GroupNorm
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


class Custom1x1Subm3d(spconv.SparseConv3d):
    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape,
                                             input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn=None, indice_key=None):
        super().__init__()
        if norm_fn is None:
            norm_fn = nn.BatchNorm1d

        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = norm_fn(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    indice_key=indice_key), norm_fn(out_channels))

    def forward(self, input):
        identity = input
        out = self.conv1(input)
        out = self.bn1(out)
        out = out.replace_feature(self.relu(out.features))
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(input)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        return out


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):
        super().__init__()
        self.nPlanes = nPlanes
        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape,
                                           output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        return output