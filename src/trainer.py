import os
import math
import numpy as np
import scipy.misc as misc
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

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


    def _create_optimizer(self, args, model):
        trainable = filter(lambda x: x.requires_grad, model.parameters())

        if args.optimizer == 'SGD':
            opt_algorithm = optim.SGD
            opt_args = {'momentum': args.momentum}
        elif args.optimizer == 'ADAM':
            opt_algorithm = optim.Adam
            opt_args = {
                'betas': (args.beta1, args.beta2),
                'eps': args.epsilon
            }
        elif ars.optimizer == 'RMSprop':
            opt_algorithm = optim.RMSprop
            opt_args = {'eps': args.epsilon}

        opt_args['lr'] = args.lr
        opt_args['weight_decay'] = args.weight_decay

        return opt_algorithm(trainable, **opt_args)

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
            self.test()
            return True
        else:
            return self.scheduler.last_epoch + 1 >= self.args.epochs

    def train(self):
        self.model.train(True)

        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        #lr = self.scheduler.get_lr()[0]
        """to do
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        """


        # timer_data, timer_model = utility.timer(), utility.timer()

        for batch, img in enumerate(self.loader_train):
            # timer_data.hold()
            # timer_model.tic()

            self.optimizer.zero_grad()
            img_down = self.model(img)
            loss = self.loss(img, img_down)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()

            else:
                print('Skip batch {}. (Loss = {})'.format(
                        batch + 1))

            # timer_model.hold()

            # if (batch + 1) % self.args.print_every == 0:
            #      self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
            #         (batch + 1) * self.args.batch_size,
            #         len(self.loader_train.dataset),
            #         self.loss.display_loss(batch),
            #         timer_model.release(),
            #         timer_data.release()))

            # timer_data.tic()


    def test(self):
        def save_result_imgs(self, filename, save_list, scale):
            filename = os.path.join(self.dir_log, 'results', '{}x{}'.format(filename, scale))
            for img in save_list:
                ndarr = img.data.byte().permute(1, 2, 0).cpu().numpy()
                misc.imsave('{}.png'.format(filename), ndarr)

        self.model.eval() # set test mode
        epoch = self.scheduler.last_epoch + 1

        # self.ckp.write_log('\nEvaluation:')
        # self.ckp.add_log(torch.zeros(1, len(self.scale)))

        # timer_test = utility.timer()
        with torch.no_grad():
            # self.loader_test.set_scale(self.scale)
            tqdm_test = tqdm(self.loader_test) # show progress bar
            for i, img in enumerate(tqdm_test):
                filename = 'hahaha'
                """to do
                    get filename
                """

                img_down = self.model(img)
                img_down = img.clamp(0, 255)


                save_list = [img_down, img]

                # if self.args.save_results:
                #         self.ckp.save_results(filename, save_list, scale)

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

        # self.ckp.write_log(
        #     'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        # )
        # if not self.args.test_only:
        #     self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
