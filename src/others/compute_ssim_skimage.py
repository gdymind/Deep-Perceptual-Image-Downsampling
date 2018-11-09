import numpy as np
import imageio
from skimage.measure import compare_ssim as ssim
import torch

def upscale_imgs(img, scale):
    # channel, height, width
    c, h, w = img.size()
    img = img.view(c, h * w, 1).contiguous()
    img = img.repeat(1, 1, scale * scale).contiguous()
    img = img.view(c, h, w, scale, scale).contiguous()
    img = img.permute(0, 1, 3, 2, 4).contiguous()
    img = img.view(c, h * scale, w * scale).contiguous()

    return img

# read img1
name1 = input("original image name:")
img1 = imageio.imread(name1)

# read img2
name2 = input("downscaled image name:")
img2 = imageio.imread(name2)

# upscale img2
scale = input("scale:")
scale = int(scale)
img2 = np.ascontiguousarray(np.transpose(img2, (2, 0, 1)))
img2 = upscale_imgs(torch.Tensor(img2), scale).numpy().astype(int)
img2 = np.ascontiguousarray(np.transpose(img2, (1, 2, 0)))

res = ssim(img1, img2, win_size = 9, multichannel = True, gaussian_weights = False)
print("ssim = {}".format(res))