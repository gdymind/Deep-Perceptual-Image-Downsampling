import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from math import exp


class SSIM(nn.Module):
    def __init__(self, window_size = 11, n_channel = 3, size_average = True, device = "cuda"):
        super(SSIM, self).__init__()
        
        self.window_size = window_size
        self.size_average = size_average
        self.window = self._create_weights(window_size, n_channel)
        self.window.to(device)

    def forward(self, img1, img2):
        n_channel = img1.size()[1]

        return _calc_ssim(img1, img2, self.window, self.window_size, n_channel, size_average = self.size_average)
        

    def _gaussian(self, window_size, sigma):
        mid = window_size // 2
        y = np.array([-1.0 * (x - mid) * (x - mid)
                        for x in range(window_size)])
        y /= 2 * sigma * sigma
        y = np.exp(y)
        y /= np.sum(y)

        return torch.Tensor(y)

    def _create_weights(self, window_size, n_channel):
        window = self._gaussian(window_size, 1.5).unsqueeze(1) # create n by 1 matrix
        window = window.mm(window.t()) # create n by n matrix
        window = window.unsqueeze(0).unsqueeze(0).expand(n_channel, 1, window_size, window_size) # add 2 dimensions corresponding to channels and feature maps
        window /= window.sum()

        return window

    def _calc_ssim(img1, img2, window, window_size, n_channel, C1 = 0, C2 = 0, size_average = True):
        # mu is luminance, which is estimated as the mean intensity
        mu1 = F.conv2d(img1, window, groups = n_channel)
        mu2 = F.conv2d(img2, window, groups = n_channel)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12 = mu * mu2

        # sigma is contrast, which is estimated as the standard deviation
        # padding == 0?
        sigma1_sq = F.conv2d(img1 * img1, window, groups = n_channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, groups = n_channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, groups = n_channel) - mu12

        # C1 == C2 == 0

        ssim_map = (2 * mu12 + C1) * (2 * sigma12 + C2)
        ssim_map /= (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)