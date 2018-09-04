from importlib import import_module

from torch.utils.data.dataloader import default_collate

import data.BaseDataset as ds

class  Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            #module_train = import_module('data.' + args.data_train)
            #trainset = getattr(module_train, args.data_train)(args)
            self.dataset = ds.BaseDataset(args) 
