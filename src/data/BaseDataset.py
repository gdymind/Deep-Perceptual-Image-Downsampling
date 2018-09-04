import os
import glob

import pickle
import numpy as np
import imageio
from skimage.viewer import ImageViewer

import torch
import torch.utils.data as data

class BaseDataset(data.Dataset):
    def __init__(self, args, name = "DIV2K", train = True):
        self.args = args
        self.name = name

        self.train = train
        self.scales = args.scales# all the possible scales
        self.cur_scale = self.scales[0]# current scale

        # set img files' start/end index
        self.fileIdx = [r.split('-') for r in args.data_range.split('/')]
        if train:
            self.fileIdx = self.fileIdx[0]
        else:
            self.fileIdx = self.fileIdx[-1]
        self.fileIdx = [int(x) for x in self.fileIdx]

        # set path                
        self.path_root = os.path.join(args.dir_data, self.name)
        self.path_bin = os.path.join(self.path_root, 'bin')
        self.path_binfile = os.path.join(self.path_bin, self.name + "_bin.pt")
        os.makedirs(self.path_bin, exist_ok = True)

        # set images
        filenames = sorted(glob.glob(os.path.join(self.path_root, '*.png')))
        filenames = filenames[self.fileIdx[0] - 1: self.fileIdx[1]]
        self.images = self._load_bin(filenames)

        #print(self.__len__())
        #item = self.__getitem__(150)
        #viewer = ImageViewer(item)
        #viewer.show()

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        return self.images[idx]


    def _load_bin(self, names):
        #bin_number = len(glob.glob(os.path.join(self.path_root, '*.pt')))
        #make_bin = (bin_number == len(names))
        make_bin = not os.path.isfile(self.path_binfile)
        if make_bin:
            print("Generating binary files...")
            imgs = [imageio.imread(i) for i in names]
            print("Found",len(imgs), "images")
            with open(self.path_binfile, "wb") as f: pickle.dump(imgs, f)
            print("Finished generating binary files")
            return imgs
        else:
            print("Loading binary files...")
            with open(self.path_binfile, "rb") as f:
                print(self.path_binfile)
                imgs = pickle.load(f)
                print("Found",len(imgs), "images")
                print("Finished loading binary files")
                return imgs

    def set_scale(self, scale):
        self.cur_scale = scale