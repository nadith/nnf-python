# -*- coding: utf-8 -*-
"""
.. module:: FileFormat
   :platform: Unix, Windows
   :synopsis: Represent File Format enumeration.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from enum import Enum

# Local Imports


class FileFormat(Enum):
    """FORMAT Enumeration describes the file format. Each row indicates a sample"""

    ASCII = 1,      # ASCII file format (readable via notepad)
    BINARY = 2,     # Binary file format (not readable via notepad)
    MATLAB = 3,     # Matlab file format (not readable via notepad, but readable via Matlab)
