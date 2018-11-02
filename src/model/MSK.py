import torch
import torch.nn as nn
import torch.nn.functional as F

from model.buildingBlocks import *
from utility import *

def make_model(args):
    return MSK(args)

class MyDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size = 3, stride = 1,
                bias = True, bn = False, act = nn.ReLU(True)): # out_channels aka growth_rate
        super(MyDenseBlock, self).__init__()
        self.conv = CBA_Block(in_channels, growth_rate, kernel_size, stride, bias, act = act)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)# the first dimension is the batch index, so we should cat the second dimension
        return out


class MyResDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_dense_layers): # out_channels aka growth_rate
        super(MyResDenseBlock, self).__init__()

        self.ShrinkX = ConvFusion(2*in_channels, in_channels)

        cur_channels = 2*in_channels
        mDenlist = []
        for i in range(n_dense_layers):
            mDenlist.append(MyDenseBlock(cur_channels, growth_rate))
            cur_channels += growth_rate

        self.DenseLayers = nn.Sequential(*mDenlist)
        self.ConvFusion = ConvFusion(cur_channels, in_channels)
     
    def forward(self, x):
        out  = self.DenseLayers(x)
        out = self.ConvFusion(out)
        out += self.ShrinkX(x)
        return out

class FirstToAllBlock(nn.Module):
    def __init__(self, mlist):
        super(FirstToAllBlock, self).__init__()
        self.mlist = mlist
    def forward(self, x):
        x0 = x
        for i, block in enumerate(self.mlist):
            x = torch.cat((x0, x), 1)
            # print('x.size() = ', x.size())
            x = block(x)
        return x

# Deep Perceptual Image Downsampling Net
class  MSK(nn.Module):
    def __init__(self, args):
        super(MSK, self).__init__()

        self.scales = args.scales
        self.cur_scale = self.scales[0]# current scale
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        n_channels = args.n_channels
        n_shallow_feature = args.n_shallow_feature
        n_feature = args.n_feature

        n_ResDenseBlock = args.n_ResDenseBlock
        n_dense_layer = args.n_dense_layer
        growth_rate = args.growth_rate

        mean = imgGlobalMean
        # std = (1.0, 1.0, 1.0)

        self.DownSize = nn.Conv2d(n_channels, n_feature, kernel_size = 2, stride = 2)

        # shallow feature extraction (SFE)
        mlist = []
        mlist.append(ConvHalfPad(n_feature, n_feature))
        mlist.append(ConvHalfPad(n_feature, n_feature))
        self.SFE = nn.Sequential(*mlist)

        # ResDenseBlocks
        mlist = []
        # mlist.append(ResDenseBlock(n_feature, growth_rate, n_dense_layer))
        for i in range(n_ResDenseBlock):
            mlist.append(MyResDenseBlock(n_feature, growth_rate, n_dense_layer).to(self.device))
        self.RDBs = FirstToAllBlock(mlist)

        # global feature fusion (GFF)
        mlist = []
        # mlist.append(ConvFusion(n_feature, n_feature))
        mlist.append(ConvHalfPad(n_feature, n_feature))
        self.GFF = nn.Sequential(*mlist)

        mlist = []
        mlist.append(ConvHalfPad(n_feature, 4 * n_feature))
        mlist.append(nn.PixelShuffle(2))
        self.UpSize = nn.Sequential(*mlist)

        self.Acc = ConvHalfPad(n_feature, 3)

        # down scaling
        mlist = []
        # mlist.append(DownConvBlock(n_feature, self.scales))
        mlist.append(DownConvBlock(6, self.cur_scale))
        self.Down = nn.Sequential(*mlist)

    def forward(self, x):
        x0 = x
        x = self.DownSize(x)
        # print('Shape input:', x.size())
        x = self.SFE(x)
        sfe = x
        # print('Shape SFE:', x.size())
        x = self.RDBs(x)
        # print('Shape ResDense:', x.size())
        x = self.GFF(x)
        # print('Shape GFF:', x.size())
        x = self.UpSize(x)
        x = self.Acc(x)
        # print('Acc:', x.size())
        # print('y:', y.size())
        x = torch.cat((x, x0), 1)
        # print('cat:', x.size())
        x = self.Down(x)
        # print('Shape Down:', x.size())
        return x