import sys
import threading
import queue
import random
import collections
import numpy as np

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

def _base_worker_loop(dataset, index_queue, data_queue, collate_fn, seed, init_fn, worker_id, scales):
    global _use_shared_memory
    _use_shared_memory = True

    # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
    # module's handlers are executed after Python returns from C low-level
    # handlers, likely when the same fatal signal happened again already.
    # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)

    if init_fn is not None:
        init_fn(worker_id)

    watchdog = ManagerWatchdog()

    while True:
        try:
            r = index_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            if watchdog.is_alive():
                continue
            else:
                break
        if r is None:
            break
        idx, batch_indices = r
        try:
            # set scale
            cur_scale = np.random.choice(scales)
            dataset.set_scale(cur_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            # append scale
            samples.append(cur_scale)
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))
            del samples    

class _BaseDataLoaderIter(_DataLoaderIter):
    def __init__(self, loader):
        self.scales = loader.scales#add scales

        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        base_seed = torch.LongTensor(1).random_().item()

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queues = [multiprocessing.Queue() for _ in range(self.num_workers)]
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_base_worker_loop,
                    args=(self.dataset, self.index_queues[i],
                          self.worker_result_queue, self.collate_fn, base_seed + i,
                          self.worker_init_fn, i, self.scales))#add scales
                for i in range(self.num_workers)]

            if self.pin_memory or self.timeout > 0:
                self.data_queue = queue.Queue()
                if self.pin_memory:
                    maybe_device_id = torch.cuda.current_device()
                else:
                    # do not initialize cuda context if not necessary
                    maybe_device_id = None
                self.worker_manager_thread = threading.Thread(
                    target=_worker_manager_loop,
                    args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
                          maybe_device_id))
                self.worker_manager_thread.daemon = True
                self.worker_manager_thread.start()
            else:
                self.data_queue = self.worker_result_queue

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()


class  BaseDataloader(DataLoader):
    def __init__(
        self, args, dataset, batch_size = 4, shuffle = True,
        sampler = None, batch_sampler = None, num_workers = 0,
        collate_fn = default_collate, pin_memory = True, drop_last = True,
        timeout = 0, worker_init_fn = None):

        super(BaseDataloader, self).__init__(
            dataset, batch_size = batch_size, shuffle = shuffle,
            sampler = sampler, batch_sampler = batch_sampler, num_workers = args.n_threads,
            collate_fn = collate_fn, pin_memory = pin_memory, drop_last = drop_last,
            timeout = timeout, worker_init_fn = worker_init_fn)

        #self.args = args
        self.scales = args.scales

    def __iter__(self):
        return _BaseDataLoaderIter(self)

