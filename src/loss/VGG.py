import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utility import *
from model.buildingBlocks import *

class VGG(nn.Module):
    def __init__(self, conv_index = '54', data_range = 1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained = True).features
        mlist = [m for m in vgg_features]

        if conv_index == '22':
            self.vgg = nn.Sequential(*mlist[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*mlist[:35])

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        self.SubMean = MeanShift(mean, std)

        self.vgg.requires_grad = False

    def forward(self, img_ori, img_up):
        vgg_img_up = self.SubMean(img_up)
        vgg_img_up = self.vgg(img_up)

        with torch.no_grad():
            vgg_img_ori = self.SubMean(img_ori.detach())
            vgg_img_ori = self.vgg(vgg_img_ori)

        loss = F.mse_loss(vgg_img_up, vgg_img_ori)

        return loss

