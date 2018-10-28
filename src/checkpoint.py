import os
import time
import datetime
import numpy as np

import torch

class Checkpoint():
    def __init__(self, args):
        self.args = args
        self.log = torch.Tensor()

        self.dir = args.dir_log
        self.dir_model = os.path.join(self.dir, "model")
        self.dir_result = os.path.join(self.dir, "results")
        self.dir_log = os.path.join(self.dir, "log.txt")

        if args.reset:
            os.system("rm -rf " + self.dir)
            print("Log files are all cleared")

        os.makedirs(self.dir, exist_ok = True)
        os.makedirs(self.dir_model, exist_ok = True)
        os.makedirs(self.dir_result, exist_ok = True)

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.log_txt = open(self.dir_log, "a")

        # write config.txt
        with open(os.path.join(self.dir, "config.txt"), "a") as f:
            f.write('===============\n')
            f.write(now + '\n')
            for arg in vars(args):
                f.write('{}: \t{}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.log_txt.write('===============\n')
        self.log_txt.write(now + '\n')

    def append_log(self, log):
        self.log = torch.cat([self.log, log])

    def save_log_txt(self, log):
        print(log)
        self.log_txt = open(self.dir_log, "a")
        self.log_txt.write(log + '\n')
        self.log_txt.close()

    def stop(self):
        self.log_file.close()

    def save(self, trainer, epoch, is_best = False):
        trainer.model.save(epoch, is_best)
        trainer.loss.save(epoch, is_best)
        trainer.loss.plot_loss(epoch)