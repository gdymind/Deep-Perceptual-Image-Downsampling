import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from math import exp


class SSIM(nn.Module):
    def __init__(self, args, window_size = 11, n_channel = 3, size_average = True):
        super(SSIM, self).__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.window = self._create_weights(window_size, n_channel)
        self.scales = args.scales
        self.cur_scale = self.scales[0]

    def forward(self, imgs1, imgs2):
        # print('img size:', img1.size())
        n_channel = imgs1.size()[1]
        # print('n_channel =', n_channel)

        return self._calc_ssim(imgs1, imgs2, self.window, self.window_size, n_channel, size_average = self.size_average)

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

        return window.to(self.device)

    def upscale_imgs(self, imgs, scale):
        #batch, channel, height, width
        n, c, h, w = imgs.size()
        # print('a', imgs.size())
        imgs = imgs.view(n, c, h * w, 1).contiguous()
        # print('b', imgs.size())
        imgs = imgs.repeat(1, 1, 1, scale * scale).contiguous()
        # print('c', imgs.size())
        imgs = imgs.view(n, c, h, w, scale, scale).contiguous()
        # print('d', imgs.size())
        imgs = imgs.permute(0, 1, 2, 4, 3, 5).contiguous()
        # print('e', imgs.size())
        imgs = imgs.view(n, c, h * scale, w * scale).contiguous()
        # print('f', imgs.size())

        return imgs


    def _calc_ssim(self, imgs1, imgs2, window, window_size, n_channel, C1 = 0, C2 = 0, size_average = True):
        # upsample imgs2
        # transfom = transforms.Compose([
        #     transforms.ToPILImage()#,
        #     #transforms.Resize([imgs1.size()[2], imgs1.size()[3]]),
        #     #transforms.ToTensor()
        #     ])
        # imgs2 = [transfom(img) for img in imgs2]
        # print('upsample starts')
        sz = imgs2.size()

        # imgs3 = imgs1.clone()
        # for i in range(sz[0]):
        #     for j in range(sz[1]):
        #         for k in range(sz[2]):
        #             for l in range(sz[3]):
        #                  for m1 in range(self.cur_scale):
        #                     for m2 in range(self.cur_scale):
        #                         imgs3[i][j][k + m1][l + m2] = imgs2[i][j][k][l]
        # imgs2 = imgs3.to(torch.device('cuda'))
        # imgs2 = self.upscale_imgs(imgs2, self.cur_scale)
        # print('imgs2 size:', imgs2.size())

        # mu is luminance, which is estimated as the mean intensity

        # for img1, img2 in zip(img_batch1, img_batch2):
        mu1 = F.conv2d(imgs1, window, groups = n_channel)
        mu2 = F.conv2d(imgs2, window, groups = n_channel)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12 = mu1 * mu2

        # sigma is contrast, which is estimated as the standard deviation
        # padding == 0?
        sigma1_sq = F.conv2d(imgs1 * imgs1, window, groups = n_channel) - mu1_sq
        sigma2_sq = F.conv2d(imgs2 * imgs2, window, groups = n_channel) - mu2_sq
        sigma12 = F.conv2d(imgs1 * imgs2, window, groups = n_channel) - mu12

        # C1 == C2 == 0

        ssim_map = (2 * mu12 + C1) * (2 * sigma12 + C2)
        ssim_map /= (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        if size_average:
            ssim_map = ssim_map.mean()
            return ssim_map
        else:
            return ssim_map.mean(1).mean(1).mean(1)