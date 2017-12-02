# -*- coding: utf-8 -*-
"""
.. module:: Format
   :platform: Unix, Windows
   :synopsis: Represent Format enumeration.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from enum import Enum

# Local Imports


class Format(Enum):
    """FORMAT Enumeration describes the db_format of the NNdb database."""

    H_W_CH_N = 1,     # =1 Height x Width x Channels x Samples (image db_format)
    H_N = 2,          # Height x Samples
    N_H_W_CH = 3,     # Samples x Height x Width x Channels x Samples (image db_format)
    N_H = 4,          # Samples x Height
