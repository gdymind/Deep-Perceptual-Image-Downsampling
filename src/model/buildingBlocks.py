import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# convolution that ensures out feature map size == in feature map size
# note that kernel size must be odd
def conv_retain_size(in_channels, out_channels, kernel_size, bias = True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, 
        padding = (kernel_size // 2), bias = bias)


# conv + bn + activate(ReLU )
class CBABlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1,
        bias = True, bn = True, act = nn.ReLU(True)):

        mlist = [conv_retain_size(in_channels, out_channels, kernel_size, stride = 1,
            bias = bias)]

        if bn:
            mlist.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            mlist.append(act)

        super(CBABlock, self).__init__(*mlist)

class ResBlock(nn.Module):
    def __init__(self, feature_size, kernel_size,
        bias = True, bn = False, res_scale = 1):

        super(ResBlock, self).__init__()

        mlist = []
        mlist.append(CBABlock(feature_size, feature_size, kernel_size, bias = bias, bn = bn))
        mlist.append(CBABlock(feature_size, feature_size, kernel_size, bias = bias, bn = bn, act = None))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        y = self.body(x).mul(self.res_scale)
        y += x

        return y

"""
class ResDenBlock(nn.Module)
    def __init__(self, C):
        pass
"""