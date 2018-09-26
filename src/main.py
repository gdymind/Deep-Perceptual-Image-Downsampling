import torch

from arguments import args
from checkpoint import Checkpoint
import data
import model


# ensure that every time you train, the initial parameters are exactly the same
torch.manual_seed(args.seed)

checkpoint = Checkpoint(args)
loader = data.Data(args)
model = model.BaseModel(args, checkpoint)

print("The whole program has exited.")