import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True):
        """
        Args:
            in_dim: dimension of input
            out_dim: dimension of output
            bias: boolean. if True, bias is included.
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
    Convolution 1d Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True):
        """
        Args:
            in_channels: dimension of input
            out_channels: dimension of output
            kernel_size: size of kernel
            stride: size of stride
            padding: size of padding
            dilation: dilation rate
            bias: boolean. if True, bias is included.
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
        return self.conv(x)

class DoubleConv(nn.Module):
    """
    Convolution block which is used in U-net
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=True):
        super(DoubleConv, self).__init__()
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
        
        if self.residual: 
            self.bypass = nn.Sequential(
                Conv1d(in_channels, out_channels, 1, 1, 0),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        if self.residual:
            return F.relu(self.double_conv(x) + self.bypass(x))
        else:
            return F.relu(self.double_conv(x))


class Down(nn.Module):
    """
    Downscaling with avgpool then double conv
    """
    def __init__(self, in_channels, out_channels, residual=False):
        super(Down, self).__init__()      
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
        super(Up, self).__init__()       
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, residual=residual)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size(2), mode='linear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x
