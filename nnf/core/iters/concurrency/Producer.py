# -*- coding: utf-8 -*-
"""
.. module:: Producer
   :platform: Unix, Windows
   :synopsis: Represent the base and related classes for wrappers of python threads or processes
                to provide the data reading capability.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import os
import queue
import threading
from multiprocessing import Process
from abc import ABCMeta, abstractmethod

# Local Imports
from nnf.core.iters.concurrency.Util import *


class Producer(object):
    """Producer is the base class for wrappers of python threads or processes to provide the data reading capability."""

    __metaclass__ = ABCMeta

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, name, iterator, dqueue, squeue):
        self.name = name
        self.stop = False
        self.data_q = dqueue
        self.signal_q = squeue
        self.iterator = iterator

        # Create a python thread or process in python
        self.py_concurrency = self.create(self.work, name)

    # Called by the consumer
    @abstractmethod
    def create(self, fn_work, name):
        """Create a producer thread or process."""
        pass

    # Called by the producer
    @abstractmethod
    def id(self):
        """Python generated internal id of the thread or process."""
        pass

    # Called by the consumer
    def start(self):
        """Start the python thread or process."""
        self.py_concurrency.start()

    # Called by the consumer
    def join(self):
        """Join the consumer. (Will wait till this thread or process will die)."""
        self.py_concurrency.join()

    # Called by the producer
    def work(self):
        """Worker work method. (Will be invoked in a separate thread or process)."""

        while True:
            index_array, current_index, current_batch_size = next(self.iterator.index_generator)

            # Read one sample at a time
            assert(self.iterator.batch_size == 1 and len(index_array) == 1)

            j = index_array[0]
            print_plog("PRODUCER CONCURRENCY:" + self.name + " READING J:" + str(j))

            ############################################################################################################
            # # 1. Add data into the data queue
            # dt_list_x = [None] * len(self.iterator.input_generators)
            # dt_cls_lbl = None
            # dt_list_xt = [None] * len(self.iterator._target_generators)
            #
            # # Iterate input generators
            # for ig, generator in enumerate(self.iterator.input_generators):
            #     dt_list_x[ig], cls_lbl = generator._get_data(generator.frecords, j)
            #     if ig == 0 and cls_lbl is not None:  # Primary generator is responsible for setting the class
            #         dt_cls_lbl = cls_lbl
            #
            # # Iterate target generators
            # for ig, generator in enumerate(self.iterator._target_generators):
            #     dt_list_xt[ig], _ = generator._get_data(generator.frecords, j)
            ############################################################################################################
            list_x, cls_lbl, list_xt = self.iterator.data_tuple(j)

            # Create the data item to be put into the queue
            dataset_flag = self.iterator.is_dataset_traversed()
            ditem = (j, list_x, cls_lbl, list_xt, dataset_flag)
            self.__put(ditem)

            # If the producer has been set to stop
            if self.stop:
                print("PRODUCER DS:" + str(self.iterator.edataset) + " CONCURRENCY:" + self.name + " EXIT @ main !")
                self.release()
                break

    # Called by the producer
    def release(self):
        """Release the resources of this thread or process."""

        self.name = '<RELEASED>'
        self.stop = False
        self.data_q = None
        self.signal_q = None

        # self.iterator.release()  # will be invoked in NNFramework.release()
        self.iterator = None


    ##########################################################################
    # Private Interface
    ##########################################################################
    # Called by the producer
    def __check_signal(self):
        """Read signal from the shared signal queue."""

        signal = None
        try:
            # Blocking call with a timeout
            print_plog("PRODUCER CONCURRENCY:" + self.name + " WAIT FOR SIGNAL")

            signal = self.signal_q.get(True, 5)
            self.signal_q.task_done()

        except queue.Empty:
            pass

        return signal
    
    # Called by the producer
    def __wait_for_signal(self):
        """Read signal from the shared signal queue."""

        # Wait for the signal for further processing of the dataset
        while True:
            signal = self.__check_signal()
            if signal is not None:
                break

        return signal

    # Called by the producer
    def __put(self, ditem):
        """Put data into the shared data queue."""

        # Data feeding to the queue
        while True:
            try:
                # Blocking call with a timeout, if data_q is full
                self.data_q.put(ditem, timeout=5)

                # If data put is successful and if the dataset is already fully traversed once,
                # Wait for a signal from the consumer  (PERF: Memory)
                if self.iterator.is_dataset_traversed():
                    print_plog("PRODUCER THREAD:" + self.name + " DATASET FULL ITERATION: DONE")
                    signal = self.__wait_for_signal()
                    if signal == CSignal.EXIT:
                        self.stop = True
                break

            except queue.Full:
                # If the producer has been set to stop while waiting at data_q
                signal = self.__check_signal()
                if signal == CSignal.EXIT:
                    self.stop = True
                    break

    # Called by the producer
    # Delete (kept only for reference)
    def __read_signal_LEGACY(self):
        """Read signal from the shared signal queue."""

        # Wait for the signal for further processing of the dataset
        while True:
            try:
                # Blocking call with a timeout
                print_plog("PRODUCER THREAD:" + self.name + " WAIT FOR SIGNAL")

                signal = self.signal_q.get(True, 5)
                self.signal_q.task_done()

                if signal == CSignal.READ_FROM_BEGINING:
                    break

            except queue.Empty:
                # If the producer has been set to stop while waiting at above blocking call, exit the thread
                if self.stop:
                    break

    # Called by the producer
    # Delete (kept only for reference)
    def __put_LEGACY(self, ditem):
        """Put data into the shared data queue."""

        # Data feeding to the queue
        while True:
            try:

                # Blocking call with a timeout, if data_q is full
                self.data_q.put(ditem, timeout=5)

                # If data put is successful and if the dataset is already fully traversed once,
                # Wait for a signal from the consumer  (PERF: Memory)
                if self.iterator.is_dataset_traversed():
                    print_plog("PRODUCER THREAD:" + self.name + " DATASET FULL ITERATION: DONE")
                    self.__read_signal()

                break

            except queue.Full:
                pass

            # If the producer has been set to stop, while being blocked at data_q
            if self.stop:
                break


class ProducerThread(Producer):
    """ProducerThread wraps the python worker thread to provide the data reading capability."""

    # Called by the consumer
    def create(self, fn_work, name):
        """Create a producer thread."""
        return threading.Thread(target=fn_work, name=name)  # kwargs={"str_id": "thread_" + str(i)})

    # Called by the producer
    def id(self):
        """Python generated internal id of the thread."""
        return self.py_concurrency.name


class ProducerProcess(Producer):
    """Producer wraps the python worker process to provide the data reading capability."""

    def __init__(self, name, iterator, dqueue, squeue):
        super().__init__(name, iterator, dqueue, squeue)

        # LIMITATION: https://docs.python.org/3/library/multiprocessing.html
        # When process.start() is called, the minimum related resources are loaded at the child process to run.
        # Hence this process object will be pickled as well.
        # Lock objects and index generator objects cannot be pickled. Therefore,
        # Set it to none before the process being spawned. Later, load them back again at work().
        self.iterator.lock = None  # TypeError: can't pickle _thread.lock objects
        self.iterator.index_generator = None  # TypeError: can't pickle generator objects

        # Save the pre-assigned data indices for iterator attached to this process
        # Load it back work() method
        self.indices = self.iterator.indices

        # Process input generator list of the parent iterator
        for ig, generator in enumerate(self.iterator.input_generators):
            if ig == 0:  # Parent generator already fixed above
                continue
            generator.lock = None
            generator.index_generator = None

        # Process target generator list of the parent iterator
        for _, generator in enumerate(self.iterator.target_generators):
            generator.lock = None
            generator.index_generator = None

    # Called by the consumer
    def create(self, fn_work, name):
        """Create a producer process."""
        return Process(target=fn_work, name=name)

    # Called by the producer
    def id(self):
        """Python generated internal id of the process."""
        return self.py_concurrency.pid

    # Called by the producer
    def work(self):
        """Worker work method. (Will be invoked in a separate process)."""

        # Reinitialize the iterator
        # Will construct the non-pickable objects that were set None in the __init__().
        # i.e self.iterator.lock, self.iterator.index_generator
        self.iterator.reinit(batch_size=1, indices=self.indices)

        # Process input generator list of the parent iterator
        for ig, generator in enumerate(self.iterator.input_generators):
            if ig == 0:  # Parent generator already fixed above
                continue
            generator.reinit()

        # Process target generator list of the parent iterator
        for _, generator in enumerate(self.iterator.target_generators):
            generator.reinit()

        super().work()

    # Called by the producer
    def release(self):
        """Release the resources of this thread or process."""
        super().release()

        # https://docs.python.org/3/library/multiprocessing.html  # multiprocessing-programming
        # self.data_q.cancel_join_thread()
        # self.signal_q.cancel_join_thread()
        os._exit(0)  # Exit the child process
