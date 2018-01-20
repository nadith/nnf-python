# -*- coding: utf-8 -*-
"""
.. module:: Globals
   :platform: Unix, Windows
   :synopsis: Represent globals for testing the framework.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from enum import Enum

# Local Imports


class TestIterMode(Enum):
    ITER_NO = 0
    ITER_MEM = 1
    ITER_DSK = 2
    ITER_MEM_DSK = 3

    def int(self):
        return self.value

class Globals:
    TEST_ITERMODE = TestIterMode.ITER_NO