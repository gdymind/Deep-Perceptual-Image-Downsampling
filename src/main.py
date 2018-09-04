import torch
from arguments import args
import data
from checkpoint import Checkpoint

# ensure that every time you train, the initial parameters are exactly the same
torch.manual_seed(args.seed)

checkpoint = Checkpoint(args)
loader = data.Data(args)

print("The whole program has exited.")