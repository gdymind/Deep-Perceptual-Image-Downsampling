from data.BaseDataset import BaseDataset
from torch.utils.data.dataloader import DataLoader

class  Data:
    def __init__(self, args):
        # tarin dataset & dataloader
        self.loader_train = None
        trainset = None

        # load test dataset
        testset = BaseDataset(args, train = False)
        self.loader_test = DataLoader(testset, batch_size = 1,
            shuffle = False, pin_memory = not args.cpu)

        # load train dataset
        if not args.test_only:
            trainset = BaseDataset(args, train = True)
            self.loader_train = DataLoader(trainset, batch_size = args.batch_size,
                shuffle = True, pin_memory = not args.cpu)