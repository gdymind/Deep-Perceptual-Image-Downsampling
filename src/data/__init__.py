from importlib import import_module

from torch.utils.data.dataloader import default_collate
from data.BaseDataset import BaseDataset
from data.BaseDataloader import *

#import data.BaseDataset as ds

class  Data:
    def __init__(self, args):
        # tarin dataset & dataloader
        self.loader_train = None
        trainset = None

        if not args.test_only:
            trainset = BaseDataset(args, train = True)
            self.loader_train = BaseDataloader(args, trainset,
                batch_size = args.batch_size, pin_memory = not args.cpu)

        testset = BaseDataset(args, train = False)
        self.loader_test = BaseDataloader(args, testset,
            batch_size = 1, shuffle = False, pin_memory = not args.cpu)