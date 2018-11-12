import torch
import torch.nn as nn
import torch.nn.functional as F

from model.buildingBlocks import *
from utility import *
from model.unet import *

def make_model(args):
    return DUNET(args)

# Deep Perceptual Image Downsampling Net
class  DUNET(nn.Module):
    def __init__(self, args):
        super(DUNET, self).__init__()

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

        # shallow feature extraction (SFE)
        mlist = []
        mlist.append(CBA_Block(n_channels, n_feature))
        self.SFE = nn.Sequential(*mlist)

        # unet
        self.Unet = UNet(n_feature, n_feature)
        self.Acc = CBA_Block(n_feature, 3)

        # down scaling
        mlist = []
        # mlist.append(DownConvBlock(n_feature, self.scales))
        mlist.append(DownConvBlock(6, self.cur_scale))
        self.Down = nn.Sequential(*mlist)

    def forward(self, x):
        y = x
        x = self.SFE(x)
        x = self.Unet(x)
        x = self.Acc(x)
        x = torch.cat((x, y), 1)
        x = self.Down(x)
        return x