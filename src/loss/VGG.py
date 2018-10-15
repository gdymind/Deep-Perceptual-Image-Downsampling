import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utility import *

class VGG(nn.Module):
    def __init__(self, conv_index = '54', data_range = 1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained = True).features
        mlist = [m for m in vgg_features]

        if conv_index == '22':
            self.vgg = nn.Sequential(*mlist[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*mlist[:35])

        # mean = (0.485, 0.456, 0.406)
        # vgg_std = (0.229 * data_range, 0.224 * data_range, 0.225 * data_range)

        # self.SubMean = MeanShift(mean, std, True, data_range)

        self.vgg.requires_grad = False

    def forward(self, img_down, img):
        vgg_img_down = self.vgg(img_down)
        with torch.no_grad():
            vgg_img = self.vgg(img.detach())

        loss = F.mse_loss(vgg_img_down, vgg_img)

        return loss

