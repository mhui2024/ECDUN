import sys
import threading

if sys.version > '3':
    pass
else:
    pass
import random

import torch
import torch.multiprocessing as multiprocessing

from torch._C import _set_worker_signal_handlers, _update_worker_pids
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _DataLoaderIter

from torch.utils.data.dataloader import ExceptionWrapper
from torch.utils.data.dataloader import _pin_memory_loop
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import _set_SIGCHLD_handler

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, init_fn, worker_id):
    global _use_shared_memory
    _use_shared_memory = True
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            idx_scale = 0
            if len(scale) > 1 and dataset.train:
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            samples.append(idx_scale)

        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


class _MSDataLoaderIter(_DataLoaderIter):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queues = [
                multiprocessing.Queue() for _ in range(self.num_workers)
            ]
            self.worker_queue_idx = 0
            # self.worker_result_queue = multiprocessing.SimpleQueue()
            self.worker_result_queue = multiprocessing.Queue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            base_seed = torch.LongTensor(1).random_()[0]
            self.workers = [
                multiprocessing.Process(
                    target=_ms_loop,
                    args=(
                        self.dataset,
                        self.index_queues[i],
                        self.worker_result_queue,
                        self.collate_fn,
                        self.scale,
                        base_seed + i,
                        self.worker_init_fn,
                        i
                    )
                )
                for i in range(self.num_workers)]

            if self.pin_memory or self.timeout > 0:
                self.data_queue = queue.Queue()
                if self.pin_memory:
                    maybe_device_id = torch.cuda.current_device()
                else:
                    # do not initialize cuda context if not necessary
                    maybe_device_id = None
                self.pin_memory_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(self.worker_result_queue, self.data_queue, maybe_device_id, self.done_event))
                self.pin_memory_thread.daemon = True
                self.pin_memory_thread.start()

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


class MSDataLoader(DataLoader):
    # dataset数据集对象，必须传入，表示带加载的数据集
    # batch_size每一个min-batch得大小
    # shuffle如果为True表示每一个epoch之前都会对数据进行洗牌
    # num_worker指定数据加载时候使用的进程数量
    # collate_fn定义一个函数，用于将单个样本数据进行打包组合成mini-batch，默认为default_collate即按照样本数据维度进行张量拼接
    # pin_memory是否加载数据到GPU显存
    # drop_last指示样本数据不能被batch_size整除的时候，是否去除最后一个小于batch_size的mini-batch，默认为false
    # timeout指示每个数据加载器的工作时间限制，单位为秒
    # worker_init_fn用于初始化每一个工作进程，默认为none
    # sampler: 提供一个自定义的样本抽样器，常与 shuffle 参数一起使用。如果设置了sampler参数，则 shuffle 参数会被忽略。
    def __init__(
            self, args, dataset, batch_size=1, shuffle=False,
            sampler=None, batch_sampler=None,
            collate_fn=default_collate, pin_memory=False, drop_last=False,
            timeout=0, worker_init_fn=None):
        super(MSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.n_threads, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)

        self.scale = args.scale

    def __iter__(self):
        return _MSDataLoaderIter(self)
    # 其中MSDataLoaderIter 是 MSDataLoader 对象的一个内部类，它实现了数据集的迭代器。
    # _iter__ 函数的作用是返回一个迭代器对象，用于对数据集进行迭代，即使是在DataLoader
    # 类中也是一个必须被实现的函数。返回一个 _MSDataLoaderIter类的实例，该类包含
    # 了对数据集进行迭代的方法。由于 _MSDataLoaderIter 类已经实现了 __next__ 函数，
    # 因此可以直接通过迭代器对象来获取下一个 mini-batch 数据。
