import torch

from arguments import args
from checkpoint import Checkpoint
import data
import model
import loss
import trainer

# ensure that every time you train, the initial parameters are exactly the same
torch.manual_seed(args.seed)

print('Use {}'.format('CPU' if args.cpu else 'GPU'))

checkpoint = Checkpoint(args)
loader = data.Data(args)
model = model.BaseModel(args, checkpoint)
loss = loss.Loss(args, checkpoint) if not args.test_only else None
trainer = trainer.Trainer(args, loader, model, loss, checkpoint)

while not trainer.should_terminate():
    trainer.train()
    trainer.test()

print("The whole program has exited.")