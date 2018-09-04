import sys
import threading
import queue
import random
import collections

import torch
import torch.multiprocessing as multiprocessing

from torch._C import _set_worker_signal_handlers, _update_worker_pids, \
    _remove_worker_pids, _error_if_any_worker_fails
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _DataLoaderIter

from torch.utils.data.dataloader import ExceptionWrapper
from torch.utils.data.dataloader import _use_shared_memory
from torch.utils.data.dataloader import _worker_manager_loop
from torch.utils.data.dataloader import numpy_type_map
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import pin_memory_batch
from torch.utils.data.dataloader import _SIGCHLD_handler_set
from torch.utils.data.dataloader import _set_SIGCHLD_handler


class  MSDataloder(DataLoader):
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=default_collate, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None):
        pass
