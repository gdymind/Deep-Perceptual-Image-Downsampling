import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# convolution that ensures out feature map shape == in feature map shape
# note that kernel size must be odd
def ConvHalfPad(in_channels, out_channels, kernel_size = 3, stride = 1, bias = True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride = 1,
        padding = (kernel_size // 2), bias = bias)

def ConvFusion(in_channels, out_channels, bias = True):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 1,
            padding = 0, stride = 1, bias = bias)

# conv + bn + activate(ReLU)
class CBA_Block(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1,
        bias = True, bn = True, act = nn.ReLU(True)):

        mlist = [ConvHalfPad(in_channels, out_channels, kernel_size,
            stride = stride,  bias = bias)]

        if bn:
            mlist.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            mlist.append(act)

        super(CBA_Block, self).__init__(*mlist)

# the block for up-scaling
class SubConvBlock(nn.Sequential):
    def __init__(self, scale, act = None):
        super(SubConvlock, self).__init__()

        mlist = [nn.PixelShuffle(scale)]
        if act is not None:
            mlist.append(act)

        super(SubConvlock, self).__init__(*mlist)

class ResBlock(nn.Module):
    def __init__(self, feature_size, kernel_size,
        bias = True, bn = False, res_scale = 1):

        super(ResBlock, self).__init__()

        mlist = []
        mlist.append(CBA_Block(feature_size, feature_size, kernel_size, bias = bias, bn = bn))
        mlist.append(CBA_Block(feature_size, feature_size, kernel_size, bias = bias, bn = bn, act = None))

        self.body = nn.Sequential(*mlist)
        self.res_scale = res_scale

    def forward(self, x):
        y = self.body(x).mul(self.res_scale)
        y += x

        return y

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size = 3, stride = 1,
                bias = True, bn = False, act = nn.ReLU(True)): # out_channels aka growth_rate
        super(DenseBlock, self).__init__()
        self.conv = CBA_Block(in_channels, growth_rate, kernel_size, stride, bias, act = act)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)# the first dimension is the batch index, so we should cat the second dimension
        return out

class ResDenseBlock(nn.Module)
    def __init__(self, in_channels, growth_rate, n_dense_layers): # out_channels aka growth_rate
        sup(ResDenseBlock, self).__init__()

        cur_channels = in_channels

        mDenlist = []
        for i in range(n_dense_layers):
            mDenlist.append(DenseBlock(in_channels, growth_rate))
            cur_channels += growth_rate

        self.DenseLayers = nn.Sequential(*mDenlist)
        self.ConvFusion = ConvFusion(cur_channels, in_channels)

    def forward(self, x):
        out  = self.DenseLayers(x)
        out = self.Conv1x1(out)
        out += x
        return out