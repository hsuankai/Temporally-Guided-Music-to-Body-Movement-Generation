#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 00:11:44 2019

@author: hsuankai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

#        nn.init.uniform_(self.linear.weight, -0.1, 0.1) # uniform initialization
        nn.init.normal_(self.linear.weight, 0.0, 0.02) # normal initialization
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x)

class Conv1d(nn.Module):
    """
    Convolution Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

#        nn.init.uniform_(self.conv.weight, -0.1, 0.1) # uniform initialization
        nn.init.normal_(self.conv.weight, 0.0, 0.02) # normal initialization
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv2d(nn.Module):
    """
    Convolution Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

#        nn.init.uniform_(self.conv.weight, -0.1, 0.1) # uniform initialization
        nn.init.normal_(self.conv.weight, 0.0, 0.02) # normal initialization
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv(nn.Module):
    """
    (Convolution => BN => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=True):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.residual = residual
        self.double_conv = nn.Sequential(
            Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.bypass = nn.Sequential(
            Conv1d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm1d(out_channels)
                )
    def forward(self, x):
        if self.residual:
            return F.dropout(F.relu(self.double_conv(x) + self.bypass(x)), 0.1)
        else:
            return F.dropout(F.relu(self.double_conv(x)), 0.1)


class Down(nn.Module):
    """
    Downscaling with avgpool then double conv
    """
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool1d(2),
            DoubleConv(in_channels, out_channels, residual=residual),
        )

    def forward(self, x):
        return self.avgpool_conv(x)

class Up(nn.Module):
    """
    Upscaling by linear interpotation then double conv
    """
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, residual=residual)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size(2), mode='linear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x
