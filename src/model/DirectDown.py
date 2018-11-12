import torch
import torch.nn as nn
import torch.nn.functional as F

from model.buildingBlocks import *
from utility import *
from model.unet import *

def make_model(args):
    return DirectDown(args)

# Deep Perceptual Image Downsampling Net
class  DirectDown(nn.Module):
    def __init__(self, args):
        super(DirectDown, self).__init__()

        self.scales = args.scales
        self.cur_scale = self.scales[0]# current scale
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        n_channels = args.n_channels
        self.n_feature = args.n_feature

        # shallow feature extraction (SFE)
        self.FE1 = CBA_Block(n_channels, self.n_feature)
        self.FE2 = CBA_Block(self.n_feature, self.n_feature * 3)

        # down scaling
        mlist = []
        # mlist.append(DownConvBlock(n_feature, self.scales))
        mlist.append(DownConvBlock(self.n_feature * 3, self.cur_scale))
        self.Down = nn.Sequential(*mlist)

    def forward(self, x):
        y = x
        x = self.FE1(x)
        x = self.FE2(x)
        y = y.repeat(1, self.n_feature, 1, 1)
        x = x + y
        x = self.Down(x)
        return x