import os
import math
import numpy as np
import scipy.misc as misc
from tqdm import tqdm
from decimal import Decimal

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from utility.timer import *
from utility import *
import datetime

class Trainer():
    def __init__(self, args, loader, my_model, loss, ckp):
        self.args = args
        self.scales = args.scales
        self.cur_scale = self.scales[0]# current scale
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')

        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = loss
        self.ckp = ckp
        self.dir = os.path.join(args.dir_log, 'optimizer')
        self.dir_log = self.ckp.dir

        os.makedirs(self.dir, exist_ok = True)
        os.makedirs(self.dir_log, exist_ok = True)

        self.load(args.resume_version)
        self.scheduler = self._create_scheduler(self.args, self.optimizer)
        for _ in range(len(self.loss.log)): self.scheduler.step()

        self.error_last = 1e8 # error in the last step

        # to do: load

        """
                if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()
        """


    def _create_optimizer(self, args, my_model):
        trainable = filter(lambda x: x.requires_grad, my_model.parameters())

        if args.optimizer == 'SGD':
            optimizer_function = optim.SGD
            kwargs = {'momentum': args.momentum}
        elif args.optimizer == 'ADAM':
            optimizer_function = optim.Adam
            kwargs = {
                'betas': (args.beta1, args.beta2),
                'eps': args.epsilon,
                'amsgrad': True
            }
        elif args.optimizer == 'RMSprop':
            optimizer_function = optim.RMSprop
            kwargs = {'eps': args.epsilon}

        kwargs['lr'] = args.lr
        kwargs['weight_decay'] = args.weight_decay

        return optimizer_function(trainable, **kwargs)

    def _create_scheduler(self, args, optimizer):
        if args.decay_type == 'step':
            scheduler = lrs.StepLR(optimizer, step_size = args.lr_decay, gamma = args.gamma)
        else:
        # elif args.decay_type.find('step') >= 0:
            milstones = args.decay_type.split('_')
            milstones.pop(0)
            milstones = list(map(lambda x: int(x), milstones))
            scheduler = lrs.MultiStepLR(optimizer, milstones = milstones, gamma = args.gamma)

        return scheduler

    def load(self, version):
        # if version != 'X':
        #     resume_file = os.path.join(self.dir, 'optimizer_{}.pt'.format(version))
        #     self.optimizer = self._create_optimizer(self.args, self.model)
        #     self.optimizer.load_state_dict(torch.load(resume_file, map_location = self.device))
        # else:
        #     self.optimizer = self._create_optimizer(self.args, self.model)
        self.optimizer = self._create_optimizer(self.args, self.model)

    def save(self, version, is_best = False):
        resume_file = os.path.join(self.dir, 'optimizer_{}.pt'.format(version))
        torch.save(self.optimizer.state_dict(), resume_file)
        resume_file = os.path.join(self.dir, 'optimizer_latest.pt'.format(version))
        torch.save(self.optimizer.state_dict(), resume_file)

        if is_best:
            resume_file = os.path.join(self.dir, 'optimizer_best.pt'.format(version))
            torch.save(self.optimizer.state_dict(), resume_file)

    # def convert_tensor_device(self, *tensors):
    #     device = torch.device('cpu' if self.args.cpu else 'cuda')

    #     return [t.to(device) for t in tensors]

    def should_terminate(self):
        if self.args.test_only:
            self.test(True)
            return True
        else:
            return self.scheduler.last_epoch + 1 >= self.args.epochs

    def upscale_imgs(self, imgs, scale):
        #batch, channel, height, width
        n, c, h, w = imgs.size()
        imgs = imgs.view(n, c, h * w, 1).contiguous()
        imgs = imgs.repeat(1, 1, 1, scale * scale).contiguous()
        imgs = imgs.view(n, c, h, w, scale, scale).contiguous()
        imgs = imgs.permute(0, 1, 2, 4, 3, 5).contiguous()
        imgs = imgs.view(n, c, h * scale, w * scale).contiguous()

        return imgs

    def train(self):
        self.model.train(True)

        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        lr = self.scheduler.get_lr()[0]
        self.ckp.save_log_txt('[Epoch {}] Learning rate: {:.2e}'.format(epoch, Decimal(lr)))

        timer = Timer()

        self.loss.start_log() # start a new log line

        for batch, img in enumerate(self.loader_train):
            timer.tic()
            self.ckp.save_log_txt('[Epoch {} Batch {}] lr = {:.2e}'.format(epoch, batch, Decimal(lr)))

            self.optimizer.zero_grad()
            if self.args.model == 'REC':
                img_rec, img_down = self.model(img)
                img_up = self.upscale_imgs(img_down, self.cur_scale)
                loss_rec = self.loss(img, img_rec)
                loss_up = self.loss(img, img_up)
                loss = loss_rec + loss_up
            else:
                img_down = self.model(img)
                img_up = self.upscale_imgs(img_down, self.cur_scale)
                loss = self.loss(img, img_up)
            self.ckp.save_log_txt('[Epoch {} Batch {}] Total loss = {:.2e}'.format(epoch, batch, loss))

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip batch {}. (Loss = {})'.format(batch + 1, loss.item()))

            timer.hold()
            print('[Epoch {} Batch {}] Batch time = {:.1}s'.format(epoch, batch, timer.toc()))
            print('Program Total time = {}'.format(str(datetime.timedelta(seconds = globalTimer.toc()))))

        print('len of loader_train', len(self.loader_train))
        self.loss.end_log(len(self.loader_train))

        print('[Epoch {}] Epoch Time = {:.2e}'.format(epoch, timer.release()))
        print('Program Total time = {}\n'.format(str(datetime.timedelta(seconds = globalTimer.toc()))))

    def test(self, save_results = False):
        def save_result_imgs(filename, img, scale):
            apath = os.path.join(self.dir_log, 'results')
            os.makedirs(apath, exist_ok = True)
            filename = os.path.join(apath, filename + '_{}.png'.format(scale))
            ndarr = img.cpu().numpy()
            # recover img
            # for i, data in enumerate(ndarr):
            #     ndarr[i] = data * imgGlobalStd[i] + imgGlobalMean[i]
            for i in range(ndarr.shape[0]):
                ndarr[i] *= imgGlobalRange
            ndarr = np.transpose(ndarr, (1, 2, 0)).astype(int)
            ndarr = ndarr.clip(0, 255)
            misc.imsave(filename, ndarr)

        self.model.eval() # set test mode
        epoch = self.scheduler.last_epoch + 1

        # self.ckp.write_log('\nEvaluation:')
        # self.ckp.add_log(torch.zeros(1, len(self.scale)))

        timer_test = Timer()
        with torch.no_grad():
            # self.loader_test.set_scale(self.scale)
            tqdm_test = tqdm(self.loader_test) # show progress bar
            for i, data in enumerate(tqdm_test):
                img = data[0]
                filename = data[1][0]

                if self.args.model == 'REC':
                    img_rec, img_down = self.model(img)
                    img_up = self.upscale_imgs(img_down, self.cur_scale)
                    loss_rec = self.loss(img, img_rec, test = True)
                    loss_up = self.loss(img, img_up, test = True)
                    loss = loss_rec + loss_up
                else:
                    img_down = self.model(img)
                    img_up = self.upscale_imgs(img_down, self.cur_scale)
                    loss = self.loss(img, img_up, test = True)
                self.ckp.save_log_txt('[Epoch {} Test] Total loss = {:.2e}'.format(epoch, loss))

                if save_results:
                    save_result_imgs(filename, img_down.squeeze(0), self.cur_scale)

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best = False)

        self.ckp.save_log_txt('Test total time: {:.2f}s\n'.format(timer_test.toc()))
        # if not self.args.test_only:
        #     self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
