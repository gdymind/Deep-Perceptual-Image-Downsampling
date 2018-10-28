import os
from importlib import import_module
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.modules as modules
from torch.nn import MSELoss


class Loss(modules.loss._Loss):
    def __init__(self, args, checkpoint):
        super(Loss, self).__init__()

        self.ckp = checkpoint
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.batch_size = args.batch_size
        self.dir = os.path.join(args.dir_log, 'loss')
        self.dir_figure = os.path.join(args.dir_log, 'figure')
        os.makedirs(self.dir, exist_ok = True)
        os.makedirs(self.dir_figure, exist_ok = True)

        self.loss = []
        self.loss_module = nn.ModuleList()

        self.load_loss(args, args.resume_version)


    def forward(self, img_down, img, test = False):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](img_down, img)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                if not test:
                    self.log[-1][i] += effective_loss.item()

        loss_sum = sum(losses)
        if (not test) and len(self.loss) > 1:
            self.log[-1][-1] += loss_sum.item()

        self.ckp.save_log_txt(self.display_loss(1 if test else self.batch_size))

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()
            else:
                print('no scheduler for', l)

    def start_log(self): # start a new log line
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches): # finish a new log line
        self.log[-1].div_(n_batches)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def plot_loss(self, epoch):
        axis = np.linspace(1, epoch, epoch) # start, stop, num
        for i, l in enumerate(self.loss): # for each type of loss
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].cpu().numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel(label)
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(self.dir_figure, l['type']))
            plt.close(fig)

    def display_loss(self, batch):
        n_samples = batch
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{} {}] {:.4f}'.format(l['type'], l['weight'], c/n_samples))

        return ''.join(log)

    def load_loss(self, args, version):
        if version != 'X':
            resume_file = os.path.join(self.dir, 'loss_{}.pt'.format(version))
            self.load_state_dict(
                torch.load(resume_file, map_location = self.device))
            # self.log.load_state_dict(
            #     torch.load(os.path.join(self.dir, 'loss_log_{}.pt'.format(version)),
            #         map_location = self.device))
        else:
            for l in args.loss.split('+'):
                loss_weight, loss_type = l.split('*')

                if loss_type.find("SSIM") >= 0:
                    module = import_module('loss.SSIM')
                    loss_function = getattr(module, 'SSIM')(args).to(self.device)
                elif loss_type.find("MSE") >= 0:
                    loss_function = MSELoss()
                elif loss_type.find("VGG") >= 0:
                    module = import_module('loss.VGG')
                    loss_function = getattr(module, 'VGG')().to(self.device)
                else:
                    pass

                self.loss.append({
                    'type': loss_type,
                    'weight': float(loss_weight),
                    'function': loss_function
                    })

            # just for displaying the total loss
            if len(self.loss) > 1:
                self.loss.append({
                    'type': 'Total',
                    'weight': 1.0,
                    'function': None
                    })

            for l in self.loss:
                if l['function'] is not None:
                    print('{:.3f} * {}'.format(l['weight'], l['type']))
                    self.loss_module.append(l['function'])

            self.log = torch.Tensor()
            self.log.to(self.device)
            self.loss_module.to(self.device)

            # if args.precision == 'half': self.loss_module.half()
            # if not args.cpu and args.n_GPUs > 1:
            #     self.loss_module = nn.DataParallel(
            #         self.loss_module, range(args.n_GPUs)
            #     )


        # for l in self.loss_module:
        #     if hasattr(l, 'scheduler'):
        #         for _ in range(len(self.log)): l.scheduler.step()

    def save(self, epoch, is_best = False):
        torch.save(self.state_dict(), os.path.join(self.dir, 'loss_{}.pt'.format(epoch)))
        torch.save(self.log, os.path.join(self.dir, 'loss_log_{}.pt'.format(epoch)))

        torch.save(self.state_dict(), os.path.join(self.dir, 'loss_latest.pt'))
        torch.save(self.log, os.path.join(self.dir, 'loss_log_latest.pt'))

        if is_best:
            torch.save(self.state_dict(), os.path.join(self.dir, 'loss_best.pt'))
            torch.save(self.log, os.path.join(self.dir, 'loss_log_best.pt'))