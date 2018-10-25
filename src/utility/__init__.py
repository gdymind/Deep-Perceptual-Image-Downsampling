import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.timer import *

imgGlobalRange = 255.0
imgGlobalMean = (0.4488 , 0.4371, 0.4040)
# imgGlobalStd = (imgGlobalRange, imgGlobalRange, imgGlobalRange)
# imgGlobalStd = (10.0, 10.0, 10.0)

globalTimer = Timer()

# add mean or substract mean, then divided by std
class MeanShift(nn.Conv2d):
    def __init__(self, mean, std, forward = True, data_range = 256):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(std)
        mean = torch.Tensor(mean)

        if forward:
            self.weight.data = torch.eye(3).view(3, 3, 1, 1).contiguous() # use a identical kernel
            self.weight.data.div_(std.view(3, 1, 1, 1).contiguous())
            self.bias.data = data_range * mean
            self.bias.data.div_(std)
        else:
            self.weight.data = torch.eye(3).view(3, 3, 1, 1).contiguous() # use a identical kernel
            self.weight.data.mul_(std.view(3, 1, 1, 1).contiguous())
            self.bias.data = -1 * data_range * mean
            self.bias.data.mul_(std)

        self.requires_grad = False