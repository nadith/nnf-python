# -*- coding: utf-8 -*-
"""
.. module:: WindowIter
   :platform: Unix, Windows
   :synopsis: Represent WindowIter class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import queue
import timeit
import threading
from enum import Enum
from scipy.misc import imresize
from warnings import warn as warning
from multiprocessing import Process
from multiprocessing import JoinableQueue
from concurrent.futures import ThreadPoolExecutor

# Local Imports
from nnf.utl.internals import *
from nnf.core.iters.memory.SHM_RTNumpyIterator import SHM_RTNumpyIterator
from nnf.core.iters.Iterator import Iterator


class WndIterMode(Enum):
    ITER_NO = 0
    ITER_MEM = 1
    ITER_DSK = 2

    def int(self):
        return self.value


class WindowIter(object):
    """WindowIter preprocess data in chunks read by a moving window and provided a distance matrix as the feature."""

    def __init__(self, sizes, strides, mode):
        """Construct a obj:`WindowIter` instance.
            Assumption: len(sizes) == len(strides)

            Parameters
            ----------
            sizes : ndarray | list | None
                Window sizes.

            strides : ndarray | list | None
                x-direction stride to move the window.

            mode : :obj:`WndIterMode`
                Whether the window iterator working on im-memory data or not.

        Notes
        -----
            Real-time data stream is only WndIterMode.ITER_MEM.
            See also: init(), __next__()
        """
        self.sizes = sizes
        self.strides = strides
        self.mode = mode
        self.imdata_pp = None
        self.input_shape = None
        self.data_format = None
        self.fn_read_sample = None
        self.fn_reshape_input = None
        self.icurwnd = None
        self.cur_wnd_offset = None
        self.cls_lbl = None
        self.X = None
        self.y = None
        self.frecords = None
        self.indices = None
        self.loop_data = None

    def init(self, imdata_pp, input_shape, data_format, fn_read_sample, fn_reshape_input=None,
             X=None, y=None, frecords=None, indices=None, loop_data=True):
        """Initialize  obj:`WindowIter` instance.

            Parameters
            ----------
            imdata_pp : :obj:`ImageDataPreProcessor`
                Image data pre-processor.

            input_shape : ndarray | None
                -direction stride to move the window.

            data_format : str | None
                Image data format. ('channel_first' or 'channel_last').

            fn_read_sample : callable |  None
                Function to read a sample from the database.

            fn_reshape_input : callable, optional
                Function to reshape the distance matrix to input_shape provided. (Default = None)
                By default, imresize and tile operations are applied.

            X : ndarray
                Data in 2D matrix. Format: Samples x Features.

            y : ndarray
                Vector indicating the class labels.

            frecords : :obj:`list`
                List of file records. frecord = [fpath or class_path, fpos or filename, cls_lbl]

            indices : ndarray
                Vector indicating indices to be used in iterating the in-memory database.

            loop_data : bool
                Keep looping the database once the window reach the end of the database.
        """
        # Error handling
        if self.mode == WndIterMode.ITER_MEM:
            if X is None and indices is None:
                raise Exception(
                    "WindowIter must be initialized with data or atleast indices (for real-time data generation.")

            # Currently real-time data stream is only supported in in-memory mode
            if X is None and indices is not None:
                warning("WindowIter will be working with real-time data stream.")

        if self.mode == WndIterMode.ITER_DSK:
            if frecords is None:
                raise Exception("WindowIter must be initialized with data (frecords).")

        self.imdata_pp = imdata_pp
        self.input_shape = input_shape  # (H x W x CH) depending on data_format
        self.data_format = data_format
        self.fn_read_sample = fn_read_sample
        self.fn_reshape_input = fn_reshape_input
        self.icurwnd = 0
        self.cur_wnd_offset = 0
        self.cls_lbl = None
        self.X = X
        self.y = y
        self.frecords = frecords
        self.indices = indices
        self.loop_data = loop_data  # Loop the data set over and over again

    def reinit(self, loop_data=True):
        """Reinitialize obj:`WindowIter` instance.

        Parameters
        ----------
        loop_data : bool
            Keep looping the database once the `self.cur_wnd_offset` reach the end of the database.
        """
        self.init(self.imdata_pp, self.input_shape, self.data_format, self.fn_read_sample,
                  self.fn_reshape_input, self.X, self.y, self.frecords, self.indices, loop_data)

    def clone(self):
        """Create a copy of this WindowIter object."""
        obj = WindowIter(self.sizes, self.strides, self.mode)
        obj.init(self.imdata_pp, self.input_shape, self.data_format, self.fn_read_sample, self.fn_reshape_input,
                 self.X, self.y, self.frecords, self.indices, self.loop_data)
        return obj

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow_ex(...):
        return self

    def __next__(self, *args, **kwargs):
        if self.mode == WndIterMode.ITER_MEM:
            return self.__get_data()

        if self.mode == WndIterMode.ITER_DSK:
            return self.__get_data_from_frecords()

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __get_data(self):
        """Get data from a in-memory database."""

        X = self.X
        y = self.y
        nb_sample = self.indices.size
        cur_wnd_offset = self.cur_wnd_offset

        # Handling dataset end
        # If the current window_iter offset is already at the end
        if (cur_wnd_offset >= nb_sample):
            cur_wnd_offset = 0
            self.cur_wnd_offset = cur_wnd_offset
            self.cls_lbl = None

            # Increment the current window_iter index
            self.icurwnd = self.icurwnd + 1
            if (self.icurwnd >= len(self.sizes)):
                if self.loop_data:
                    self.icurwnd = 0
                else:
                    raise StopIteration()

        cur_wnd_size = self.sizes[self.icurwnd]
        cur_wnd_stride = self.strides[self.icurwnd]

        data = None
        cls_lbl = None
        for i in range(0, cur_wnd_size):
            index = self.indices[cur_wnd_offset]
            feature, cls_lbl = self.fn_read_sample(X, y, index, False)

            # Handling class boundary
            # Assumption: Data belong to same class need to be placed in consecutive blocks.
            # Hence the class labels should be in sorted order.
            if (self.cls_lbl is None):
                self.cls_lbl = cls_lbl

            elif (self.cls_lbl != cls_lbl):
                if (i == 0):                # The very first iteration
                    self.cls_lbl = cls_lbl  # Track the class label
                else:
                    cls_lbl = self.cls_lbl
                    break

            # Vectorize the input
            feature = np.reshape(feature, (1, np.prod(feature.shape)))

            # Collect the data into 2D matrix
            data = feature if (data is None) else np.concatenate((feature, data), axis=0)

            # For next iteration
            index = cur_wnd_offset + i

            # For half way through window but not the full window
            if index == nb_sample:
                break

        # Save the current window_iter offset for next iteration
        cur_wnd_offset += cur_wnd_stride
        self.cur_wnd_offset = cur_wnd_offset

        # print('WND_SIZE:{0} STRD_SIZE:{1} WND_OFFSET:{2} CLS_LBL:{3}'.format(cur_wnd_size, cur_wnd_stride, self.cur_wnd_offset, cls_lbl))
        return self.__process_data(data), cls_lbl

    def __get_data_from_frecords(self):
        """Get data from a in-disk database."""

        frecords = self.frecords
        nb_sample = self.indices.size
        cur_wnd_offset = self.cur_wnd_offset

        # Handling dataset end
        # If the current window_iter offset is already at the end
        if (cur_wnd_offset >= len(frecords)):
            cur_wnd_offset = 0
            self.cur_wnd_offset = cur_wnd_offset
            self.cls_lbl = None

            # Increment the current window_iter index
            self.icurwnd = self.icurwnd + 1
            if (self.icurwnd >= len(self.sizes)):
                if self.loop_data:
                    self.icurwnd = 0
                else:
                    raise StopIteration()

        cur_wnd_size = self.sizes[self.icurwnd]
        cur_wnd_stride = self.strides[self.icurwnd]

        data = None
        cls_lbl = None
        for i in range(0, cur_wnd_size):
            index = self.indices[cur_wnd_offset]

            # Process frecord
            frecord = frecords[index]

            # For image stored under class folder names, fpath indicates the file path to the class folder
            # For big data, fpath indicates the path to the data file
            fpath = frecord[0] if isinstance(frecord[0], str) else frecords[frecord[0]][0]

            # For image stored under class folder names, fpos indicates the image file name
            # For big data, fpos indicates the row index to be read
            fpos = frecord[1]
            cls_lbl = frecord[2]

            feature = self.fn_read_sample(fpath, fpos, False)

            # Handling class boundary
            # Assumption: Data belong to same class need to be placed in consecutive blocks.
            # Hence the class labels should be in sorted order.
            if (self.cls_lbl is None):
                self.cls_lbl = cls_lbl

            elif (self.cls_lbl != cls_lbl):
                if (i == 0):  # The very first frecord
                    self.cls_lbl = cls_lbl
                else:
                    cls_lbl = self.cls_lbl
                    break

            # Vectorize the input
            feature = np.reshape(feature, (1, np.prod(feature.shape)))

            # Collect the data into 2D matrix
            data = feature if (data is None) else np.concatenate((feature, data), axis=0)

            # For next iteration
            index = cur_wnd_offset + i

            # For half way through window but not the full window
            if index == nb_sample:
                break

        # Save the current window_iter offset for next iteration
        cur_wnd_offset += cur_wnd_stride
        self.cur_wnd_offset = cur_wnd_offset

        # print('WND_SIZE:{0} STRD_SIZE:{1} WND_OFFSET:{2} CLS_LBL:{3}'.format(cur_wnd_size, cur_wnd_stride, self.cur_wnd_offset, cls_lbl))
        return self.__process_data(data), cls_lbl

    def __process_data(self, data):
        """Calculate the distance matrix/image from data, resize and tile it to the input_shape."""

        if (self.imdata_pp is None):
            return data

        imshape = None
        n_ch = 0
        if self.data_format == 'channels_last':
            imshape = tuple(self.input_shape[0:2])
            n_ch = self.input_shape[2]
        else:
            imshape = tuple(self.input_shape[1:])
            n_ch = self.input_shape[0]

        # Generate the image from the data,
        img = self.imdata_pp.dop_default(data, 2)

        # Resize to imshape
        img = imresize(img, imshape, 'bicubic', mode='F') \
            if (self.fn_reshape_input is None) else \
            self.fn_reshape_input(img, self.input_shape, self.data_format)

        x = img_to_array(img, data_format=self.data_format)

        # Replicate to other channels
        if self.data_format == 'channels_last':
            x = np.tile(x, (1, 1, n_ch))
        else:
            x = np.expand_dims(x, 0)
            x = np.tile(x, (n_ch, 1, 1))

        x = self.imdata_pp.standardize_window(x)
        return x


class RTWindowState:
    """RTWindowState is a helper used internally to iterate the real-time data to fetch window counts."""
    def __init__(self, edataset, rt_stream):

        self.rt_stream = rt_stream
        self.nb_sample, self.n_per_class, self.dc_boundaries = SHM_RTNumpyIterator.process_rt_stream(edataset, rt_stream)

    def fn_read_sample(self, X, y, index, reshape):
        return SHM_RTNumpyIterator.generate_sample(index, self.rt_stream, self.dc_boundaries, self.n_per_class)


class Window(object):
    """Window consists of utility function for `WindowIter`."""

    @staticmethod
    def shm_rtstream_get_total_wnd_counts(window_iter, iter_params, edataset, rt_stream):
        """Dry runs the `WindowIter`, fetch the counts for real-time data stream.

         Notes
         -----
         This method follows the same logic written in `NNF` when calculating the total window counts.
         """
        rtwndstate = RTWindowState(edataset, rt_stream)
        count = 0

        start = timeit.default_timer()

        from nnf.core.NNFramework import NNFramework  # to avoid circular imports
        iter_param_map = NNFramework.get_param_map(iter_params)
        ds_iter_param = iter_param_map[edataset]

        # Get the performance cache configuration
        cache_size, processes, threads = Iterator.get_pcache_config(ds_iter_param)
        if processes > 0:
            concurrency_count = processes
        elif threads > 0:
            concurrency_count = threads
        else:  # Default option
            concurrency_count = 1

        # Create a queue to collect the counts (used only in parallel operation)
        q = None
        if processes > 0:
            q = JoinableQueue()
        elif threads > 0:
            q = queue.Queue()

        rng_st = 0
        for i in range(concurrency_count):
            indices = np.arange(rng_st, rtwndstate.nb_sample, concurrency_count)
            rng_st += 1

            if processes > 0:
                witer_copy = WindowIter(window_iter.sizes, window_iter.strides, window_iter.mode)
                p = Process(target=Window.fn_parallel_count, name='test_name', args=(q, rtwndstate.fn_read_sample, indices, witer_copy, None, None))
                p.start()

            elif threads > 0:
                witer_copy = WindowIter(window_iter.sizes, window_iter.strides, window_iter.mode)
                t = threading.Thread(target=Window.fn_parallel_count, name='test_name', args=(q, rtwndstate.fn_read_sample, indices, witer_copy, None, None))
                t.start()

            else:
                assert(concurrency_count == 1)
                count = Window.fn_parallel_count(None, rtwndstate.fn_read_sample, indices, window_iter, None, None)

        if q is not None:
            sum = 0
            for i in range(concurrency_count):
                count = q.get()
                q.task_done()
                sum += count

            # Block until all tasks are done
            q.join()
            count = sum

        duration = timeit.default_timer() - start
        return count, duration

    @staticmethod
    def fn_parallel_count(q, fn_read_sample, indices, window_iter, db, cls_lbl):
        """"Work method for threads or process. Invoked in a seperate thread or process."""
        count = Window.__get_count(window_iter, fn_read_sample, db, cls_lbl, indices=indices)

        if q is not None:
            q.put(count)
            return None
        else:
            return count

    @staticmethod
    def get_total_wnd_counts(window_iter, iter_params, nndb=None, sel=None, nndbs_tup=None):
        """Fetch the total window counts for each dataset (TR|VAL|TE).

         Notes
         -----
         This method follows the same logic written in `NNF` when calculating the total window counts.
         """
        if (nndbs_tup is None):
            from nnf.db.DbSlice import DbSlice  # to avoid circular imports
            nndbs_tup = DbSlice.slice(nndb, sel)

        counts = [0] * (len(nndbs_tup) -1)

        start = timeit.default_timer()

        for ti in range(len(nndbs_tup) - 1):
            nndb = nndbs_tup[ti]
            edataset = nndbs_tup[-1][ti]

            from nnf.core.NNFramework import NNFramework  # to avoid circular imports
            iter_param_map = NNFramework.get_param_map(iter_params)
            ds_iter_param = iter_param_map[edataset]

            # Get the performance cache configuration
            cache_size, processes, threads = Iterator.get_pcache_config(ds_iter_param)
            if processes > 0:
                concurrency_count = processes
            elif threads > 0:
                concurrency_count = threads
            else:  # Default option
                concurrency_count = 1

            if nndb is not None:
                # Create a queue to collect the counts (used only in parallel operation)
                q = None
                if processes > 0:
                    q = JoinableQueue()
                elif threads > 0:
                    q = queue.Queue()

                rng_st = 0
                for i in range(concurrency_count):
                    indices = np.arange(rng_st, nndb.n, concurrency_count)
                    rng_st += 1

                    if processes > 0:
                        witer_copy = WindowIter(window_iter.sizes, window_iter.strides, window_iter.mode)
                        p = Process(target=Window.fn_parallel_count, name='test_name', args=(q, Window._fn_read_sample, indices, witer_copy, nndb.db_scipy, nndb.cls_lbl))
                        p.start()

                    elif threads > 0:
                        witer_copy = WindowIter(window_iter.sizes, window_iter.strides, window_iter.mode)
                        t = threading.Thread(target=Window.fn_parallel_count, name='test_name', args=(q, Window._fn_read_sample, indices, witer_copy, nndb.db_scipy, nndb.cls_lbl))
                        t.start()

                    else:
                        assert(concurrency_count == 1)
                        counts[ti] = Window.fn_parallel_count(None, Window._fn_read_sample, indices, window_iter, nndb.db_scipy, nndb.cls_lbl)

                if q is not None:
                    sum = 0
                    for i in range(concurrency_count):
                        count = q.get()
                        q.task_done()
                        sum += count

                    # Block until all tasks are done
                    q.join()
                    counts[ti] = sum

        duration = timeit.default_timer() - start
        return counts, duration

    @staticmethod
    def get_total_wnd_counts_tp(window_iters, nndb=None, sel=None, nndbs_tup=None, nb_samples=None):
        """Dry runs the `WindowIter`, fetch the counts. (parallel implementation)."""

        if (nndbs_tup is None):
            from nnf.db.DbSlice import DbSlice  # to avoid circular imports
            nndbs_tup = DbSlice.slice(nndb, sel)

        counts = [0] * (len(nndbs_tup) -1)

        # Using thread pool
        executor = ThreadPoolExecutor(max_workers=len(window_iters))

        start = timeit.default_timer()
        for ti in range(len(nndbs_tup) - 1):
            nndb = nndbs_tup[ti]

            if nndb is not None:
                futures = [None] * len(window_iters)
                for wi, window_iters in enumerate(window_iters):
                    futures[wi] = executor.submit(Window.__get_count, window_iters.clone(),
                                                  Window._fn_read_sample, nndb.db_scipy, nndb.cls_lbl)
                sum = 0
                for wi, _ in enumerate(window_iters):
                    sum += int(futures[wi].result())

                counts[ti] = sum
        duration = timeit.default_timer() - start
        executor.shutdown()
        return counts, duration

    ##########################################################################
    # Protected Interface
    ##########################################################################
    # ERROR: _pickle.PicklingError: Can't pickle <function Window.__fn_read_sample at 0x000001F010BF71E0>: attribute lookup Window.__fn_read_sample on nnf.core.iters.custom.WindowIter failed
    # Protected by not private, for the serialization through sel.nnpatches => window_iter
    # See also: https://stackoverflow.com/questions/1914261/pickling-a-staticmethod-in-python
    # Ref: https://stackoverflow.com/questions/12718187/calling-class-staticmethod-within-the-class-body
    # TODO: Study in depth
    @staticmethod
    def _fn_read_sample(X, y, index, reshape):
        return X[index], y[index]

    ##########################################################################
    # Private Interface
    ##########################################################################
    @staticmethod
    def __get_count(window_iter, fn_read_sample, db, cls_lbl, indices=None):
        window_iter.init(None, None, None, fn_read_sample, None,
                    X=db, y=cls_lbl, frecords=None, indices=indices, loop_data=False)

        counts = 0
        for _, _ in window_iter:
            counts += 1

        return counts

