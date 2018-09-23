import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.modules as modules

"""
class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
"""


class Loss(modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.n_GPU = args.n_GPU
        self.loss = []
        self.loss_module = nn.ModuleList()

        for l in args.loss.split('+'):
            loss_weight, loss_type = l.split('_')

            if loss_type.find("SSIM") >= 0:
                module = import_module('loss.SSIM')
                loss_function = getattr(module, 'SSIM')()

            self.loss.append({
                'type': loss_type,
                'weight': float(loss_weight),
                'function': loss_function
                })

        if len(self.loss) > 1:
            self.loss.append({
                'type': 'Total',
                'weight': 0,
                'function': None
                })

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)

        # if args.precision == 'half': self.loss_module.half()
        # if not args.cpu and args.n_GPUs > 1:
        #     self.loss_module = nn.DataParallel(
        #         self.loss_module, range(args.n_GPUs)
        #     )

        # if args.load != '.': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, img_down, img):
        losses = []
        for i, l in enumerate(self.loss):
            if l['funciton'] is not None:
                loss = l['function'](img_down, img)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # self.log[-1, i] += effective_loss.item()

        loss_sum = sum(losses)
        # if len(self.loss) > 1:
        #     self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()
            else:
                print('no scheduler for', l)

    def start_log(self):
        self.log.cat_(torch.zeros(1, len(self.loss)))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)