# -*- coding: utf-8 -*-
"""
.. module:: PerformanceCache
   :platform: Unix, Windows
   :synopsis: Represent multi-producer consumer, memory cache for high performance disk data reading.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from enum import Enum

# Local Imports
from nnf.core.Globals import Globals


# Util functions to print consumer logs
def print_clog(str):
    if Globals.CONSUMER_DEBUG_LOGS:
        print(str)

# Util functions to print producer logs
def print_plog(str):
    if Globals.PRODUCER_DEBUG_LOGS:
        print(str)

class CSignal(Enum):
    """Concurrency (thread or process) Signal Enumeration."""

    # Signals
    READ_FROM_BEGINING = 0
    EXIT = 1

    def int(self):
        return self.value