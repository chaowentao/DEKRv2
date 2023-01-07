# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torchvision.ops as ops

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=False,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AdaptBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 outplanes,
                 stride=1,
                 downsample=None,
                 dilation=1,
                 deformable_groups=1):
        super(AdaptBlock, self).__init__()
        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.deformable_groups = deformable_groups
        self.transform_matrix_conv = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.translation_conv = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)
        self.adapt_conv = ops.DeformConv2d(inplanes, outplanes, kernel_size=3, stride=stride, \
            padding=dilation, dilation=dilation, bias=False, groups=deformable_groups)
        self.bn = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        N, _, H, W = x.shape
        transform_matrix = self.transform_matrix_conv(x)
        transform_matrix = transform_matrix.permute(0, 2, 3, 1).reshape(
            (N * H * W, 2, 2))
        offset = torch.matmul(transform_matrix, self.regular_matrix)
        offset = offset - self.regular_matrix  #  仿射变换矩阵乘以基本矩阵，然后减去基本矩阵得到位置偏移
        offset = offset.transpose(1, 2).reshape(
            (N, H, W, 18)).permute(0, 3, 1, 2)

        translation = self.translation_conv(x)
        offset[:, 0::2, :, :] += translation[:, 0:1, :, :]
        offset[:, 1::2, :, :] += translation[:, 1:2, :, :]
        out = self.adapt_conv(x, offset)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AdaptBlockv2(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 outplanes,
                 stride=1,
                 downsample=None,
                 dilation=1,
                 deformable_groups=1):
        super(AdaptBlockv2, self).__init__()
        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.deformable_groups = deformable_groups
        self.transform_matrix_conv = nn.Conv2d(inplanes,
                                               4 * deformable_groups,
                                               3,
                                               1,
                                               1,
                                               bias=True,
                                               groups=deformable_groups)
        self.translation_conv = nn.Conv2d(inplanes,
                                          2 * deformable_groups,
                                          3,
                                          1,
                                          1,
                                          bias=True,
                                          groups=deformable_groups)
        self.adapt_conv = ops.DeformConv2d(inplanes, outplanes, kernel_size=3, stride=stride, \
            padding=dilation, dilation=dilation, bias=False, groups=deformable_groups)
        self.bn = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        N, _, H, W = x.shape
        transform_matrix = self.transform_matrix_conv(x)
        transform_matrix = transform_matrix.permute(0, 2, 3, 1).reshape(
            (N * H * W * self.deformable_groups, 2, 2))
        offset = torch.matmul(transform_matrix, self.regular_matrix)
        offset = offset - self.regular_matrix  #  仿射变换矩阵乘以基本矩阵，然后减去基本矩阵得到位置偏移
        offset = offset.transpose(1, 2).reshape(
            (N, H, W, 18 * self.deformable_groups)).permute(0, 3, 1, 2)

        translation = self.translation_conv(x)
        if self.deformable_groups != 1:
            offset[:, 0::2, :, :] += translation[:, 0:1, :, :]
            offset[:, 1::2, :, :] += translation[:, 1:2, :, :]
        else:
            offset = offset.reshape(N, 18, self.deformable_groups, H, W)
            translation = translation.reshape(N, 2, self.deformable_groups, H,
                                              W)
            offset[:, 0::2, :, :, :] += translation[:, 0:1, :, :, :]
            offset[:, 1::2, :, :, :] += translation[:, 1:2, :, :, :]
            offset = offset.reshape(N, 18 * self.deformable_groups, H, W)

        out = self.adapt_conv(x, offset)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out