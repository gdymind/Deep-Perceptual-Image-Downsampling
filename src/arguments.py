import argparse
import template

parser = argparse.ArgumentParser(description='DPID')
"""
parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in arguments.py')
"""
# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_false',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
                    

# Data specifications
parser.add_argument('--dir_data', type=str, default='/home/gdymind/DPID/Dataset',
                    help='dataset directory')

parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')

parser.add_argument('--data_range', type=str, default='1-850/851-900',
                    help='train/test data range')

parser.add_argument('--scales', type=str, default='2',
                    help='all the possible down scaling scales')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')

parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='DPID',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

args = parser.parse_args()

args.scales = list(map(lambda x: int(x), args.scales.split('+')))

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
