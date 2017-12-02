# -*- coding: utf-8 -*-
# Global Imports
import abc
import numpy as np
import threading

# Local Imports


class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        index_array = None

        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if self.shuffle:
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * self.batch_size) % N
            if N >= current_index + self.batch_size:
                current_batch_size = self.batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow_ex(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    __metaclass__ = abc.ABCMeta

    ##########################################################################
    # Protected Interface
    ##########################################################################
    @abc.abstractmethod
    def next(self, *args, **kwargs):
        # must override this method
        pass
