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
        filenames = sorted(glob.glob(os.path.join(path_root, '*.png')))
        filenames = filenames[fileIdx[0] - 1: fileIdx[1]]
        self.images = self._load_bin(filenames, path_bin)

        #print(self.__len__())
        #item = self.__getitem__(150)
        #viewer = ImageViewer(item)
        #viewer.show()

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        return self.images[idx]


    def _load_bin(self, names, path_bin):
        #bin_number = len(glob.glob(os.path.join(self.path_root, '*.pt')))
        #make_bin = (bin_number == len(names))
        make_bin = not os.path.isfile(path_bin)
        if make_bin:
            print("Generating binary file:\t" + path_bin.split('/')[-1])
            imgs = [imageio.imread(i) for i in names]
            print("Found",len(imgs), "images")
            with open(path_bin, "wb") as f:
                pickle.dump(imgs, f)
                f.close()
            print("Finished generating binary files")
            return imgs
        else:
            print("Loading binary file:\t" + path_bin.split('/')[-1])
            with open(path_bin, "rb") as f:
                # print(path_bin)
                imgs = pickle.load(f)
                print("Found",len(imgs), "images")
                print("Finished loading binary files")
                f.close()
                return imgs

    def set_scale(self, scale):
        self.cur_scale = scale