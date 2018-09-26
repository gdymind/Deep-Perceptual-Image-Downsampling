import torch
import torch.nn as nn
import torch.nn.functional as F

from model.buildingBlocks import *

def make_model(args):
    return DPID(args)

# Deep Perceptual Image Downsampling Net
class  DPID(nn.Module):
    def __init__(self, args):
        super(DPID, self).__init__()

        self.scales = args.scales

        n_channels = args.n_channels
        n_shallow_feature = args.n_shallow_feature
        n_feature = args.n_feature

        n_ResDenseBlock = args.n_ResDenseBlock
        n_dense_layer = args.n_dense_layer
        growth_rate = args.growth_rate        

        # shallow feature extraction (SFE)
        mlist = []
        mlist.append(ConvHalfPad(n_channels, n_shallow_feature))
        mlist.append(ConvHalfPad(n_shallow_feature, n_shallow_feature))
        self.SFE = nn.Sequential(*mlist)

        # ResDenseBlocks
        mlist = []
        for i in range(n_ResDenseBlock):
            mlist.append(ResDenseBlock(n_shallow_feature, growth_rate, n_dense_layer))
        self.ResDenseBlocks = nn.Sequential(*mlist)

        # global feature fusion (GFF)
        mlist = []
        mlist.append(ConvFusion(n_feature * n_ResDenseBlock, n_feature))
        mlist.append(ConvHalfPad(n_feature, n_feature))
        self.GFF = nn.Sequential(*mlist)

        # down scaling
        mlist = []
        mlist.append(DownConvBlock(n_feature, self.scales))
        self.Down = nn.Sequential(*mlist)

    def forward(self, x):
        out = self.SFE(x)
        out = self.ResDenseBlocks(out)
        out = self.GFF(out)
        out = self.Down(out)

        return out