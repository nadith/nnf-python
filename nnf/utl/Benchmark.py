# -*- coding: utf-8 -*-
"""
.. module:: Benchmark
   :platform: Unix, Windows
   :synopsis: Represent Benchmark class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from timeit import default_timer as timer

# Local Imports


class Benchmark(object):
    """Latency benchmark.

    Example
    -------
    >>> with Benchmark("Test 1"): 
            statment_1
            statment_2
            statment_3
    """
    def __init__(self, msg, fmt="%0.3g", threshold=None):
        self.msg = msg
        self.fmt = fmt
        self.threshold = threshold

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start

        if self.threshold is None:
            print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        elif self.threshold is not None and t > self.threshold:
            print(("%s : " + self.fmt + " seconds") % (self.msg, t))

        self.time = t



