import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = -1 * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

# class MeanShift(nn.Module):
#     def __init__(self, device, data_mean, sign = -1, channels = 3):
#         super(MeanShift, self).__init__()
#         self.sign = sign
#         self.mean = torch.Tensor(data_mean).view(1, channels, 1, 1).contiguous().to(device)
#         self.requires_grad = False
#     def forward(self, x):
#         x += self.sign * self.mean
#         return x


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
        bias = True, bn = False, act = nn.ReLU(True)):

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

class ResDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_dense_layers): # out_channels aka growth_rate
        super(ResDenseBlock, self).__init__()

        cur_channels = in_channels

        mDenlist = []
        for i in range(n_dense_layers):
            mDenlist.append(DenseBlock(cur_channels, growth_rate))
            cur_channels += growth_rate

        self.DenseLayers = nn.Sequential(*mDenlist)
        self.ConvFusion = ConvFusion(cur_channels, in_channels)

    def forward(self, x):
        out  = self.DenseLayers(x)
        out = self.ConvFusion(out)
        out += x
        return out

class CatToLastBlock(nn.Module):
    def __init__(self, mlist):
        super(CatToLastBlock, self).__init__()
        self.mlist = mlist

    def forward(self, x):
        for i, block in enumerate(self.mlist):
            x = block(x)
            if i == 0:
                out = x
            else:
                out = torch.cat((x, out), 1)
        return out


class DownPoolBlock(nn.AvgPool2d):
    def __init__(self, scale):
        super(DownPoolBlock, self).__init__(kernel_size = scale)

class DownConvBlock(nn.Conv2d):
    def __init__(self, in_channels, scale):
        super(DownConvBlock, self).__init__(in_channels, 3, kernel_size = scale, stride = scale, padding = 0, bias = True)