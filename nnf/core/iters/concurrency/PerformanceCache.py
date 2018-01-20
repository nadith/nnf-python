# -*- coding: utf-8 -*-
"""
.. module:: PerformanceCache
   :platform: Unix, Windows
   :synopsis: Represent multi-producer consumer, memory cache for high performance.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
from abc import ABCMeta, abstractmethod
from multiprocessing import JoinableQueue

# Local Imports
from nnf.core.iters.concurrency.Producer import *


class PerformanceCache(object):
    """PerformanceCache facilitate mutl-producer consumer cache."""

    __metaclass__ = ABCMeta

    def __init__(self, iterator, count, cache_size=32):

        self.concurrencies = []     # Thread or process
        self.dataset_flags = []
        self.shuffle = iterator.shuffle

        rng_st = 0
        for i in range(count):

            # Divide the workload to processes/threads
            indices = np.arange(rng_st, iterator.N, count)
            rng_st += 1

            if self.shuffle:
                indices = indices[np.random.permutation(indices.size)]

            # Clone iterator (semi deep copy), hand it over to the producer thread
            iter_copy = iterator.clone()

            # Reinitialize the iterator
            iter_copy.reinit(batch_size=1, indices=indices)

            # Construct a producer thread or process and start
            cname = str(iterator.edataset) + "_" + self._prefix() + "_" + str(i)
            concurrency = self._create_concurrency(cname, iter_copy, cache_size)
            concurrency.start()

            # Save the producer thread or process
            self.concurrencies.append(concurrency)

            # Track the dataset flags for each concurrency
            self.dataset_flags.append(False)

    ##########################################################################
    # Public Interface
    ##########################################################################
    # Called by the consumer
    def get_item(self):
        """Get an item from the data queue.

        Notes
        -----
        If last item is flagged as the end of dataset (usually 1 epoch),
        send a signal to the concurrency (thread or process).
        """

        c_idx = 0
        while True:
            # Reset the flag
            continue_flag = False

            item = None
            concurrency = self.concurrencies[c_idx]
            cur_dataset_flag = self.dataset_flags[c_idx]

            # If dataset has been fully traversed and the last record has been read
            # by the consumer in the previous iteration (PERF: MEMORY SAVING)
            if self.shuffle:
                if cur_dataset_flag:
                    print_clog("CONSUMER CONCURRENCY:" + concurrency.name + " SENDING SIGNAL")
                    concurrency.signal_q.put(CSignal.READ_FROM_BEGINING)
                    cur_dataset_flag = self.dataset_flags[c_idx] = False

            else:
                if self.__all_concurrencies_flagged():
                    self.__clear_concurrency_flags()
                    c_idx = 0
                    concurrency = self.concurrencies[c_idx]
                    cur_dataset_flag = self.dataset_flags[c_idx]

            if not cur_dataset_flag:
                try:
                    timeout = 0.5 if self.shuffle else None

                    # Blocking call till data becomes available again
                    item = concurrency.data_q.get(block=True, timeout=timeout)
                    concurrency.data_q.task_done()

                    # COSTLY: qsize() utilizes a mutex lock
                    print_clog("CONSUMER CONCURRENCY:" + concurrency.name + " ITEM:" + str(item[0]) +
                               " QSIZE:" + str(concurrency.data_q.qsize()))

                    # Item structure: (<sample_index>, x, cls, xt, <dataset_flag>)
                    self.dataset_flags[c_idx] = item[-1]
                    item = item[1:-1]

                except queue.Empty:
                    continue_flag = True

            if cur_dataset_flag:
                continue_flag = True

            # Fix the thread index for next iteration
            c_idx = 0 if c_idx + 1 == len(self.concurrencies) else c_idx + 1

            if continue_flag:
                continue

            assert(item is not None)
            yield item

    # Called by the consumer
    def delete(self):
        """Delete the this `PerformanceCache` instance."""
        # Ask producers to stop
        for concurrency in self.concurrencies:
            concurrency.signal_q.put(CSignal.EXIT)

        # Block this thread till all producers kill themselves :D
        for concurrency in self.concurrencies:
            concurrency.join()

    ##########################################################################
    # Protected Interface
    ##########################################################################
    @abstractmethod
    def _prefix(self):
        """Easy to follow name prefix for concurrency. Useful for debugging.

        Returns
        -------
        str
            name prefix.

        Note
        ----
        Extend this method to provide a name prefix.
        """
        pass

    @abstractmethod
    def _create_concurrency(self, name, iterator, cache_size):
        """Create a producer concurrency (thread or process) to be used in combination with the PerformanceCache.

        Parameters
        ----------
        name : str
            The Neural Network Model.

        iterator : :obj:`Iterator'
            The `Iterator` instance which will call get_item(...).

        cache_size : int
            The cache size of the data queue between the producer concurrency (thread or process) and the consumer.

        Returns
        -------
        :obj:`ProducerThread` or :obj:`ProducerProducer`
            Producer concurrency (thread or process).

        Note
        ----
        Extend this method to construct the specific concurrency.
        """
        pass

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __clear_concurrency_flags(self):
        """Send the signal to all threads waiting."""
        for c_idx, concurrency in enumerate(self.concurrencies):
            print_clog("CONSUMER CONCURRENCY:" + concurrency.name + " SENDING SIGNAL")
            concurrency.signal_q.put(CSignal.READ_FROM_BEGINING)
            self.dataset_flags[c_idx] = False

    def __all_concurrencies_flagged(self):
        """Whether all concurrencies (threads or processes) are flagged.

        Notes
        -----
        If a thread or process is flagged, it indicates that it is waiting for a signal
        from the consumer thread.
        """
        # Test whether all array elements along a given axis evaluate to True
        return np.all(self.dataset_flags)


class ThreadPerformanceCache(PerformanceCache):
    """ThreadPerformanceCache facilitate mutl-producer consumer cache with multi-thread support."""

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _create_concurrency(self, name, iterator, cache_size):
        return ProducerThread(name, iterator, queue.Queue(cache_size), queue.Queue(1))

    def _prefix(self):
        return 'thread'


class ProcessPerformanceCache(PerformanceCache):
    """ProcessPerformanceCache facilitate mutl-producer consumer cache with multi-process support."""

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _create_concurrency(self, name, iterator, cache_size):
        return ProducerProcess(name, iterator, JoinableQueue(cache_size), JoinableQueue(1))

    def _prefix(self):
        return 'process'
