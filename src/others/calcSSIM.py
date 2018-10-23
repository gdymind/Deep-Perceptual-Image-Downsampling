import numpy as np
import imageio
from skimage.measure import compare_ssim as ssim

# def upscale_imgs(self, img, scale):
#     #batch, channel, height, width
#     n, c, h, w = imgs.size()
#     imgs = imgs.view(n, c, h * w, 1).contiguous()
#     imgs = imgs.repeat(1, 1, 1, scale * scale).contiguous()
#     imgs = imgs.view(n, c, h, w, scale, scale).contiguous()
#     imgs = imgs.permute(0, 1, 2, 4, 3, 5).contiguous()
#     imgs = imgs.view(n, c, h * scale, w * scale).contiguous()

name1 = input("image name:")
img1 = imageio.imread(name1)

name2 = input("image name:")
img2 = imageio.imread(name2)

res = ssim(img1, img2, win_size = 11, multichannel = True, gaussian_weights = True)
print("ssim = {}".format(res))