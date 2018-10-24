import os
import glob
import random
import sys

import pickle
import numpy as np
import imageio
from skimage.viewer import ImageViewer

import torch
import torch.utils.data as data

from utility import *

class BaseDataset(data.Dataset):
    def __init__(self, args, name = "DIV2K", train = True):
        self.name = name

        self.train = train
        self.scales = args.scales# all the possible scales
        self.cur_scale = self.scales[0]# current scale
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.patch_size = args.patch_size

        # set img files' start/end index
        fileIdx = [r.split('-') for r in args.data_range.split('/')]
        if train:
            fileIdx = fileIdx[0]
        else:
            fileIdx = fileIdx[-1]
        fileIdx = [int(x) for x in fileIdx]

        # set path
        path_root = os.path.join(args.dir_data, self.name)
        path_bin = os.path.join(path_root, 'bin')
        os.makedirs(path_bin, exist_ok = True)
        if train:
            path_bin = os.path.join(path_bin, self.name + "_train_bin.pt")
        else:
            path_bin = os.path.join(path_bin, self.name + "_test_bin.pt")

        # set images
        self.filenames = sorted(glob.glob(os.path.join(path_root, '*.png')))
        self.filenames = self.filenames[fileIdx[0] - 1: fileIdx[1]]
        self.length = len(self.filenames)
        # self._load_bin(self.filenames, path_bin, args.reset)


    def __len__(self):
        # return len(self.images)
        return self.length


    def __getitem__(self, idx):
        # for trainset, return the image
        # for testset, return the image and the filename
        if self.train:
            return torch.from_numpy(self.get_patch(idx)).float().to(self.device)
        else:
            return [torch.from_numpy(self.get_patch(idx)).float().to(self.device), self.filenames[idx].split('/')[-1].split('.')[0]]

    def _load_bin(self, names, path_bin, reset):
        #bin_number = len(glob.glob(os.path.join(self.path_root, '*.pt')))
        #make_bin = (bin_number == len(names))
        make_bin = not os.path.isfile(path_bin)
        make_bin = make_bin or reset
        if make_bin:
            print("Generating binary file:\t" + path_bin.split('/')[-1])
            print("Collecting images...")
            self.images = [imageio.imread(iname).astype(float) for iname in names] # iname means 'image name'
            self.filenames = [iname.split('/')[-1].split('.')[0] for iname in names]
            # swap dimensions(channel, height, weight)
            # imgs = [[np.ascontiguousarray(np.transpose(img, (2, 0, 1))), iname] for img, iname in imgs]
            # pre-process
            # for i, data in enumerate(imgs):
            #     img, iname = data
            #     for j, img_channel in enumerate(img):
            #         img[j] = (img_channel - imgGlobalMean[j]) / imgGlobalStd[j]
            #     imgs[i] = [img, iname]
            print("Found",len(self.images), "images")
            with open(path_bin, "wb") as f:
                pickle.dump(self.images, f)
                f.close()
            print("Finished generating binary files")
        else:
            print("Loading binary file:\t" + path_bin.split('/')[-1])
            with open(path_bin, "rb") as f:
                self.images = pickle.load(f)
                print("Found",len(self.images), "images")
                print("Finished loading binary files")
                f.close()
                return self.images

    def set_scale(self, scale):
        self.cur_scale = scale

    def data_augument(self, img):
        flipx = random.random() <= 0.5
        flipy = random.random() <= 0.5
        transpose = random.random() <= 0.5

        if flipx: # reverse the first dimension using slicing
            img = img[:, ::-1, :] 
        if flipy: # reverse the second dimension using slicing
            img = img[:, :, ::-1]
        if transpose:
            img = img.transpose(0, 2, 1) # swap x-y aixes

        return np.ascontiguousarray(img)

    def get_patch(self, idx):
        # img = np.copy(self.images[idx])
        img = imageio.imread(self.filenames[idx])
        img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        for i, img_channel in enumerate(img):
            img[i] = (img_channel - imgGlobalMean[i]) / imgGlobalStd[i]

        if self.train:
            size_i, size_j = img.shape[1:3]
            p = self.patch_size
            i, j = 0, 0
            if p < size_i:
                i = random.randrange(0, size_i - p + 1)
            if p < size_j:
                j = random.randrange(0, size_j - p + 1)
            img = img[:, i: i + p, j: j + p]
            img = self.data_augument(img)

        return img
        # if self.train:
        #     size_i, size_j = img.shape[1:3]
        #     p = self.patch_size
        #     # print(p, size_i, size_j)

        #     i = random.randrange(0, size_i - p + 1)
        #     j = random.randrange(0, size_j - p + 1)
        #     img = img[:, i: i + p, j: j + p]

        #     return self.data_augument(img)
        # else:
        #     return img
        #     img = img[:, :, :]

        #     return self.data_augument(img)