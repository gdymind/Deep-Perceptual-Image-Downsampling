import os
import glob

import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

root = "/home/gdymind/Vis/Dog/"

filenames = sorted(glob.glob(os.path.join(root, '*')))

print(len(filenames), "files")

imgs = []

for f in filenames:
    img = io.imread(f)
    img = color.rgb2lab(img)
    imgs.append(img)
    print(img.shape)

img0 = imgs[0]
img1 = imgs[7]

for i in range(3):
    x = img0[:, :, i]
    y = img1[:, :, i]

    h = np.histogram(x, density = True)[0]
    plt.hist(h, histtype = "step", color = "red")

    h = np.histogram(y, density = True)[0]
    plt.hist(h, histtype = "step", color = "blue")
    plt.show()

"""
for i in range(3):
    for img in imgs:
        x = img[:, :, i]
        x = x.flatten()
        #print(x)
        h = np.histogram(x, density = True)[0]
        #print(h[0])
        plt.hist(h, histtype = "step")
        #plt.show()
    plt.show()

"""

