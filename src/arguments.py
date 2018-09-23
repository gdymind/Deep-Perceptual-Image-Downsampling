import argparse
import template

parser = argparse.ArgumentParser(description = 'DPID')
# Hardware specifications
parser.add_argument('--n_threads', type = int, default = 6,
             help = 'number of threads for data loading')
parser.add_argument('--cpu', action = 'store_false',
             help = 'use cpu only')
parser.add_argument('--n_GPU', type = int, default = 1,
             help = 'number of GPU')
parser.add_argument('--seed', type = int, default = 1,
             help = 'random seed')
             

# Data specifications
    # directory
parser.add_argument('--dir_data', type = str, default = '/home/gdymind/DPID/Dataset',
             help = 'dataset directory')
    # dataset
parser.add_argument('--data_train', type = str, default = 'DIV2K',
             help = 'train dataset name')
parser.add_argument('--data_test', type = str, default = 'DIV2K',
             help = 'test dataset name')
parser.add_argument('--data_range', type = str, default = '1-850/851-900',
             help = 'train/test data range')
parser.add_argument('--n_channels', type = int, default = 3, # using Lab color space
             help = 'the number of channels')
    # techiniques
parser.add_argument('--chop', action = 'store_true',
             help = 'enable memory-efficient forward')

# Model specifications
    # model
parser.add_argument('--model', default = 'DPID',
             help = 'model name')
parser.add_argument('--pre_train', type = str, default = '.',
             help = 'pre-trained model directory')
parser.add_argument('--scales', type = str, default = '2',
             help = 'all the possible down scaling scales')
    # block
parser.add_argument('--n_dense_layer', type = int, default = 6,
             help = 'the number of dense layers inside a residual dense block')
parser.add_argument('--n_ResDenseBlock', type = int, default = 3,
             help = 'the number of residual dense blocks')
parser.add_argument('--n_feature', type = int,
             help = 'the number of feature maps SFE and RDBs produce')
parser.add_argument('--growth_rate', type = int, default = 32,
             help = "growrh rate of the residual dense blocks")
    # techniques
parser.add_argument('--shift_mean', default = True,
             help = 'subtract pixel mean from the input')
parser.add_argument('--dilation', action = 'store_true',
             help = 'use dilated convolution')
parser.add_argument('--res_scale', type = float, default = 0.1,
             help = 'residual scaling')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Training specifications
parser.add_argument('--reset', action = 'store_true',
             help = 'reset the training and start from the very beginning')
parser.add_argument('--test_every', type = int, default = 1000,
             help = 'do test per every N batches')
parser.add_argument('--epochs', type = int, default = 300,
             help = 'number of epochs to train')
parser.add_argument('--batch_size', type = int, default = 16,
             help = 'input batch size for training')
parser.add_argument('--split_batch', type = int, default = 1,
             help = 'split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action = 'store_true',
             help = 'use self-ensemble method for test')
parser.add_argument('--test_only', action = 'store_true',
             help = 'set this option to test the model')
parser.add_argument('--gan_k', type = int, default = 1,
             help = 'k value for adversarial loss')
parser.add_argument('--patch_size', type = int, default = 192,
             help = 'output patch size')

# Loss specifications
parser.add_argument('--loss', type = str, default = '1*SSIM',
                    help = """set of losses and their parameters.
                    Use '+' to split different type of losses,
                    and insert '*' between weights and loss_type
                    """)
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--dir_log', type = str, default = '/home/gdymind/DPID/Experiment',
             help = 'log directory')
parser.add_argument('--log_name', type = str, default = 'test',
             help = 'log folder name')
parser.add_argument('--resume', type = int, default = 0,
             help = 'resume from specific checkpoint')
parser.add_argument('--save_models', action = 'store_true',
             help = 'save all intermediate models')
parser.add_argument('--print_every', type = int, default = 100,
             help = 'how many batches to wait before logging training status')
parser.add_argument('--save_results', action = 'store_true',
             help = 'save output results')

args = parser.parse_args()

args.scales = list(map(lambda x: int(x), args.scales.split('+')))

for arg in vars(args):
    if vars(args)[arg] = = 'True':
      vars(args)[arg] = True
    elif vars(args)[arg] = = 'False':
      vars(args)[arg] = False
