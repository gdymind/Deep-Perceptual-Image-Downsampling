import torch
import torch.nn as nn
import torch.nn.functional as F

from model.buildingBlocks import *
from utility import *

def make_model(args):
    return DPIDSK(args)

# Deep Perceptual Image Downsampling Net
class  DPIDSK(nn.Module):
    def __init__(self, args):
        super(DPIDSK, self).__init__()

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
            mlist.append(ResDenseBlock(n_feature, growth_rate, n_dense_layer).to(self.device))
        self.ResDenseBlocks = CatToLastBlock(mlist)

        # global feature fusion (GFF)
        mlist = []
        mlist.append(ConvFusion(n_feature * n_ResDenseBlock, n_feature))
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
        y = x
        x = self.DownSize(x)
        # print('Shape input:', x.size())
        x = self.SFE(x)
        # print('Shape SFE:', x.size())
        x = self.ResDenseBlocks(x)
        # print('Shape ResDense:', x.size())
        x = self.GFF(x)
        # print('Shape GFF:', x.size())
        x = self.UpSize(x)
        x = self.Acc(x)
        # print('Acc:', x.size())
        # print('y:', y.size())
        x = torch.cat((x, y), 1)
        # print('cat:', x.size())
        x = self.Down(x)
        # print('Shape Down:', x.size())
        return x