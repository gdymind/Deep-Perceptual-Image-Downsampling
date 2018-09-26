import os
from importlib import import_module

import torch
import torch.nn as nn

# from model.DPID import DPID

class BaseModel(nn.Module):
    def __init__(self, args, ckp):
        super(BaseModel, self).__init__()
        print('Making model...')

        self.scale = args.scales
        # self.self_ensemble = args.self_ensemble
        # self.chop = args.chop
        # self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs

        if args.dir_model == '@default':
            args.dir_model = 'model'
        self.dir = os.path.join(args.dir_root, args.dir_model)

        # import corresponding model
        module = import_module('model.' + args.model)
        self.model = module.make_model(args).to(self.device)
        # if self.n_GPUs > 1:
        #     self.model = self.model.module

        # if args.precision == 'half': self.model.half()

        # if not args.cpu and args.n_GPUs > 1:
        #     self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        #     self.model = self.model.module


        # now resume from certain checkpoint
        # no resumption is possible
        self.load_model(args.resume_version)

    def load_model(self, version):
        resume_file = ''

        if version == 'X':
            resume_file = 'X'
        elif version[0] == '@':
            resume_file = version[1: ]
        else:
            resume_file = os.path.join(self.dir, 'model_{}.pt'.format(version))

        if resume_file != 'X':
            self.load_state_dict(
                torch.load(resume_file, map_location = self.device))

    def save(self, apath, epoch, is_best = False):
        torch.save(
            self.model.state_dict(), 
            os.path.join(apath, 'model', 'model_latest.pt'))
        if is_best:
            torch.save(
                self.model.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt'))
        if self.save_models:
            torch.save(
                self.model.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch)))

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