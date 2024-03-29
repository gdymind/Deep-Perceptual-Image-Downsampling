import torch
import torch.nn.functional as F
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1) # n 1
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) # 1 1 n n
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous() # c 1 n n
    return window

def create_window_uniform(window_size, channel):
    n = window_size * window_size
    c = 1.0 / n
    window = torch.ones(channel, 1, window_size, window_size) * c
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel) # b c n n
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel) # b c n n

    mu1_sq = mu1.pow(2) # b c n n
    mu2_sq = mu2.pow(2) # b c n n
    mu1_mu2 = mu1*mu2 # b c n n

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, args, window_size = 8, n_channel = 3, size_average = True):
        super(SSIM, self).__init__()
        self.scales = args.scales
        self.cur_scale = self.scales[0]# current scale
        self.window_size = self.cur_scale * 2
        self.size_average = size_average
        self.channel = n_channel
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        # self.window = create_window(window_size, self.channel).to(self.device)
        self.window = create_window_uniform(window_size, self.channel).to(self.device)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)