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
    def __init__(self, msg, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start
        print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t



