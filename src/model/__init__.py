import os
from importlib import import_module

import torch
import torch.nn as nn

# from model.DPID import DPID

class BaseModel(nn.Module):
    def __init__(self, args, ckp):
        super(BaseModel, self).__init__()
        print('Making model...')

        # self.self_ensemble = args.self_ensemble
        # self.chop = args.chop
        # self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.dir = os.path.join(args.dir_log, 'model')
        os.makedirs(self.dir, exist_ok = True)

        # import corresponding model
        module = import_module('model.' + args.model)
        self.model = module.make_model(args).to(self.device)
        # if self.n_GPUs > 1:
        #     self.model = self.model.module

        # if args.precision == 'half': self.model.half()

        # if not args.cpu and args.n_GPUs > 1:
        #     self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        #     self.model = self.model.module
        print('Done with making model')


        # now resume from certain checkpoint
        # no resumption is possible
        self.load_model(args.resume_version)

    def load_model(self, version):
        if version != 'X':
            resume_file = os.path.join(self.dir, 'model_{}.pt'.format(version))
            self.load_state_dict(
                torch.load(resume_file, map_location = self.device))

    def save(self, epoch, is_best = False):
        sd = self.state_dict()

        torch.save(
            sd,
            os.path.join(self.dir, 'model_{}.pt'.format(epoch)))

        torch.save(
            sd,
            os.path.join(self.dir, 'model_latest.pt'))

        if is_best:
            torch.save(
                sd,
                os.path.join(self.dir, 'model_best.pt'))

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def forward(self, x):
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