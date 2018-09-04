from importlib import import_module

from torch.utils.data.dataloader import default_collate
import data.BaseDataset as ds
from torch.utils.data.dataloader import DataLoader

class  Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            self.trainset = ds.BaseDataset(args)