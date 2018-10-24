import sys

from data.BaseDataset import BaseDataset
from torch.utils.data.dataloader import DataLoader

class  Data:
    def __init__(self, args):
        train_name, test_name = args.data_name.split('/')

        # tarin dataset & dataloader
        self.loader_train = None
        trainset = None

        # load test dataset
        testset = BaseDataset(args, name = test_name, train = False)
        self.loader_test = DataLoader(testset, batch_size = 1,
            shuffle = False, pin_memory = args.cpu)
        # self.loader_test = MSDataLoader(args, testset, batch_size = 1,
        #     shuffle = False, pin_memory = not args.cpu)

        # load train dataset
        if not args.test_only:
            trainset = BaseDataset(args, name = train_name, train = True)
            self.loader_train = DataLoader(trainset, batch_size = args.batch_size,
                shuffle = True, pin_memory = args.cpu)
            # self.loader_train = MSDataLoader(args, trainset, batch_size = args.batch_size,
            #     shuffle = True, pin_memory = not arg.cpu)

        if args.gen_data_only:
            sys.exit()