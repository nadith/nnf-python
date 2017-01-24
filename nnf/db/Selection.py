"""Selection Module to represent Selection class."""
# -*- coding: utf-8 -*-
# Global Imports
from enum import Enum

# Local Imports

class Selection:
    """Denote the selection structure."""

    def __init__(self, **kwds):
        """Initialize a selection structure with given field-values."""
        self.__dict__.update(kwds)

class Select(Enum):
    """SELECT Enumeration describes the special constants for Selection structure."""

    ALL = -999

    def int(self):
        return self.value