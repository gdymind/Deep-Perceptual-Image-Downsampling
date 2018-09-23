import os
from importlib import import_module

import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, args, ckp):
        print('Making model...')

        self.scale = args.scale
        # self.self_ensemble = args.self_ensemble
        # self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        # if args.precision == 'half': self.model.half()

        # if not args.cpu and args.n_GPUs > 1:
        #     self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train = args.pre_train,
            resume = args.resume,
            cpus = args.cpu
            )

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    # not read yet
    def load(self, apath, pre_train = '.', resume = -1, cpu = False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )

    def save(self, apath, epoch, is_best = False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def forward(self, x, scale):
        target = self.get_model()       


        # if self.self_ensemble and not self.training:
        #     if self.chop:
        #         forward_function = self.forward_chop
        #     else:
        #         forward_function = self.model.forward

        #     return self.forward_x8(x, forward_function)
        # elif self.chop and not self.training:
        #     return self.forward_chop(x)
        # else:
        #     return self.model(x)

        return self.model(x)