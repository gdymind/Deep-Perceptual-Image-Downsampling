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
        self.dir = os.path.join(args.dir_root, 'optimizer')
        self.dir_log = self.ckp.dir


        self.load_optimizer(args.resume_version)
        self.scheduler = self._create_scheduler(self.args, self.optimizer)

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
                'eps': args.epsilon
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

    def load_optimizer(self, version):
        if version != 'X':
            resume_file = os.path.join(self.dir, 'optimizer_{}.pt'.format(version))
            self.optimizer.load_state_dict(
                torch.load(resume_file, map_location = self.device))
        else:
            self.optimizer = self._create_optimizer(self.args, self.model)


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
        # self.loss.start_log()

        timer_data, timer_model = Timer(), Timer()

        # timer_data.tic()

        for batch, img in enumerate(self.loader_train):
            # timer_data.hold()
            # print('batch {} load time: {}'.format(batch, timer_data.toc()))
            print('[batch {}] starts'.format(batch))
            # print('img size:', img.size())
            timer_model.tic()
            self.optimizer.zero_grad()
            img_down = self.model(img)
            img_up = self.upscale_imgs(img_down, self.cur_scale)
            # print('img_down.size() =', img_down.size())
            loss = self.loss(img, img_up)
            print('Batch otal loss =', loss)

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip batch {}. (Loss = {})'.format(batch + 1))
            # timer_model.hold()
            print('[Batch {}] time: {}'.format(batch, timer_model.toc()))

            # save_result_imgs('aa', img_down.squeeze(0), 2)
            # print('img_down mean:', img_down.mean())
            # a = input('input anything:')
            # if (batch + 1) % self.args.print_every == 0:
            #      self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
            #         (batch + 1) * self.args.batch_size,
            #         len(self.loader_train.dataset),
            #         self.loss.display_loss(batch),
            #         timer_model.release(),
            #         timer_data.release()))

            # timer_data.tic()


    def test(self, save_results = False):
        def save_result_imgs(filename, img, scale):
            apath = os.path.join(self.dir_log, 'results')
            os.makedirs(apath, exist_ok = True)
            filename = os.path.join(apath, filename + '_{}.png'.format(scale))
            ndarr = img.cpu().numpy()
            print('ndarr mean Before', ndarr.mean())
            # recover img
            print
            for i, data in enumerate(ndarr):
                # print(data)
                ndarr[i] = data * imgGlobalStd[i] + imgGlobalMean[i]
                # print(ndarr[i])
            ndarr = np.transpose(ndarr, (1, 2, 0)).astype(int)
            print('ndarr mean After', ndarr.mean())
            ndarr = ndarr.clip(0, 255)
            print('ndarr mean clip', ndarr.mean())
            misc.imsave(filename, ndarr)
            a = input('input anything...')

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
                # filename = str(data[1].numpy()[0])
                # if len(filename) < 4:
                #     filename = ('0' * (4 - len(filename))) + filename
                # print(filename)

                img_down = self.model(img).squeeze(0)

                if save_results:
                    save_result_imgs(filename, img_down, self.cur_scale)

                # self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                # best = self.ckp.log.max(0)
                # self.ckp.write_log(
                #     '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                #         self.args.data_test,
                #         scale,
                #         self.ckp.log[-1, idx_scale],
                #         best[0][idx_scale],
                #         best[1][idx_scale] + 1
                #     )
                # )

        self.ckp.save_log_txt('Test total time: {:.2f}s\n'.format(timer_test.toc()))
        # if not self.args.test_only:
        #     self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
