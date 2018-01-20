# -*- coding: utf-8 -*-
"""
.. module:: Iterator
   :platform: Unix, Windows
   :synopsis: Represent Iterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import os
import abc
import threading
from warnings import warn as warning

# Local Imports
from nnf.utl.internals import *
from nnf.core.Globals import Globals
from nnf.core.iters.concurrency.PerformanceCache import ProcessPerformanceCache, ThreadPerformanceCache


class Iterator(object):
    """Iterator represents the parent class for core iterators in NNFramework.
   
    Attributes
    ----------
    N : int
        Number of samples. If `indices` available, N will be calculated from indices (=indices.size)
        See also self.indices

    indices : ndarray
        Vector indicating indices to be used in `index_array` that provides the index to read data.
        See also self._flow_index()

    batch_size : int or None
        Number of samples per gradient update. If unspecified, batch_size will default to 32.

    shuffle : bool
        Whether to shuffle the order of the batches at the beginning of each epoch.

    batch_index : int
        Looping index to keep track of the batches during the dataset traversal.
        (=0): At the start.
        (>0): At the end.
        This index will be reset to 0 after each full traversal of the dataset.

    total_batches_seen : int
        Number of batches seen in the lifetime of `self.index_generator`, generator object.
        (=0): At the start or if reinitialized.
        (>0): At all other times.

    index_generator : int
        The core python generator object that yields the sample indices to be read from the dataset.

    edataset : :obj:`Dataset`
        Dataset enumeration key.

    params : :obj:`dict`
        Core iterator parameters.

    performance_cache : :obj:PerformanceCache
        High performance cache to work with multi-threads or processes when reading data.

    performance_cache_gen_next : :obj:`Generator`
        Generator for consuming items when using `performance_cache`.
    """
    
    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, N, batch_size, shuffle, seed, indices=None, params=None):
        """Constructor of the abstract class `Iterator`."""

        # Used by `PerformanceCache`
        self.indices = indices
        self.N = N if indices is None else indices.size
        assert N is not None or self.indices is not None

        # Used by `ProducerProcess`
        self.seed = seed

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(self.N, seed)

        # Following parameters will stay fixed for the lifetime of the object
        # Mandatory field in RTNumpyIterator
        self.edataset = params['_edataset'] if '_edataset' in params else 'TEMP'

        # To support clone functionality
        self.params = params

        # Performance cache for PERF (must call initiate_parallel_operations(...) to initialize
        # Do not instantiate the performance_cache in this constructor since PerformanceCache requires
        # this iterator to be clone(...) for multiple producer threads
        self.performance_cache = None
        self.performance_cache_gen_next = None  # Generator for item consuming

        # Reference count
        self.ref_count = 0
        self.increment_ref_count()

    def increment_ref_count(self):
        self.ref_count += 1

    def decrement_ref_count(self):
        self.ref_count -= 1

    def reset(self):
        """Perform minimal reset to parameters to use this `Iterator` object from the beginning.

        Notes
        -----
        This method is a replica of keras implemented `Iterator` class.
        See also self.reinit()
        """
        # Minimal iterator reset parameters
        self.batch_index = 0

    def reinit(self, N=None, batch_size=None, shuffle=None, seed=None, indices=None):
        """Reinitialize this `Iterator` object with related traversing parameters.

        Notes
        -----
        Unlike `self.reset()` method, reinit() will create a new `self.index_generator` generator object.
        See also self.reset()
        """
        # Used by `PerformanceCache`
        self.indices = indices
        N = N if indices is None else indices.size
        self.N = self.N if N is None else N
        assert self.N is not None or self.indices is not None

        # Used by `ProducerProcess` process based concurrency
        self.seed = seed

        self.batch_size = self.batch_size if batch_size is None else batch_size
        self.shuffle = self.shuffle if shuffle is None else shuffle
        self.reset()
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        # New generator object
        self.index_generator = self._flow_index(self.N, seed)

    def is_dataset_traversed(self):
        """Is dataset fully traversed once.

        Returns
        -------
        bool
            Whether the dataset is fully traversed once or not.

        Notes
        -----
        Usually 1 epoch should cover the whole dataset once.
        """
        return self.batch_index == 0 and self.total_batches_seen > 0

    def release(self):
        """Release internal resources used by the iterator."""

        self.decrement_ref_count()

        if self.ref_count == 0:
            if self.performance_cache is not None:
                self.terminate_parallel_operations()

    @abc.abstractmethod
    def clone(self):
        """Create a copy of this Iterator object."""
        pass

    ##########################################################################
    # Special Interface
    ##########################################################################
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

    def _flow_index(self, N, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        index_array = None

        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N) \
                    if self.indices is None \
                    else self.indices

                if self.shuffle:
                    index_array = np.random.permutation(N) \
                        if self.indices is None \
                        else self.indices[np.random.permutation(N)]

            current_index = (self.batch_index * self.batch_size) % N
            if N >= current_index + self.batch_size:
                current_batch_size = self.batch_size
                self.batch_index += 1

                # BUG: This was a bug from original keras implementation; FIXED locally.
                if N == current_index + self.batch_size:
                    self.batch_index = 0
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1

            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    ##########################################################################
    # Common Routines For Child Iterators
    ##########################################################################
    @staticmethod
    def get_pcache_config(params):
        """Fetch the performance cache configuration for multi-process or multi-thread operations."""
        # Multi-process concurrency
        dict_ppcache = params['process_performance_cache'] \
            if params is not None and  ('process_performance_cache' in params) else None

        # Multi-thread concurrency
        dict_tpcache = params['thread_performance_cache'] \
            if params is not None and  ('thread_performance_cache' in params) else None

        # Initialization
        cache_size = 0
        processes = 0
        threads = 0

        # Set defaults
        # Priority to `process` performance cache
        if dict_ppcache is not None:
            cache_size = dict_ppcache['size'] if ('size' in dict_ppcache) else 32
            processes = dict_ppcache['processes'] if ('processes' in dict_ppcache) else 1

        elif dict_tpcache is not None:
            cache_size = dict_tpcache['size'] if ('size' in dict_tpcache) else 32
            threads = dict_tpcache['threads'] if ('threads' in dict_tpcache) else 1

        return cache_size, processes, threads

    def initiate_parallel_operations(self):
        """Start the data flow of the iterators.

        Notes
        -----
        This will start the data reading threads and initiate the cache (if supported).
        See also :obj:`DirectoryIterator`
        """
        if not Globals.USE_PERFORMANCE_CACHE:
            print("------ INITIATING [" + str(self.edataset).upper() + "] DATA FLOW <WITHOUT PARALLEL POOL>------")
            return

        # Get the performance cache configuration
        cache_size, processes, threads = Iterator.get_pcache_config(self.params)

        if processes > 0:
            # Create a process performance cache
            self.performance_cache = ProcessPerformanceCache(self, count=processes, cache_size=cache_size)
            self.performance_cache_gen_next = self.performance_cache.get_item()

        elif threads > 0:
            # Create a thread performance cache
            self.performance_cache = ThreadPerformanceCache(self, count=threads, cache_size=cache_size)
            self.performance_cache_gen_next = self.performance_cache.get_item()

        else:
            warning("[" + str(self.edataset).upper() + "] Performance cache is not used. Performance of reading data may not be efficient.")

        if processes <= 0 and threads <= 0:
            print("------ INITIATING [" + str(self.edataset).upper() + "] DATA FLOW <WITHOUT PARALLEL POOL>------")
        else:
            print("------ INITIATING [" + str(self.edataset).upper() + "] DATA FLOW <WITH PARALLEL POOL>------")


    def terminate_parallel_operations(self):
        """Stop the data flow of the iterators."""

        # Invocations
        # ------------
        # NNModel._start_train() -> terminate_parallel_operations()
        # self.release() -> terminate_parallel_operations()

        print("------ TERMINATING [" + str(self.edataset).upper() + "] DATA FLOW ------")

        if self.performance_cache is not None:
            self.performance_cache.delete()
            self.performance_cache = None


    def _save_to_disk(self, current_batch_size, current_index, generators, list_batch_data, classes=None):
        """Save the data from the generators with or without class folders."""

        for ig, generator in enumerate(generators):
            batch_data = list_batch_data[ig]

            for i in range(current_batch_size):
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=generator.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=generator.save_im_index,
                                                                  # REMOVE: np.random.randint(10000),
                                                                  format=generator.save_format)
                generator.save_im_index += 1
                if (classes is not None):
                    directory = os.path.join(generator.save_to_dir, str(generator.edataset), str(classes[i]))
                else:
                    directory = os.path.join(generator.save_to_dir, str(generator.edataset))

                if not os.path.exists(directory):
                    os.makedirs(directory)

                keep_max_precision = False if generator.window_iter is None else True
                img = array_to_img(batch_data[i], generator.data_format,
                                   scale=(not keep_max_precision),
                                   keep_max_precision=keep_max_precision)
                img.save(os.path.join(directory, fname))