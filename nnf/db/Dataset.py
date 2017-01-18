"""Dataset Module to represent Dataset class."""
# -*- coding: utf-8 -*-
# Global Imports
from enum import Enum

# Local Imports


class Dataset(Enum):
    # Do not change the order of the values
    # Refer NNDiskMan.process() for more details
    TR = 0 
    VAL = 1
    TE = 2
    TR_OUT = 3
    VAL_OUT = 4

    def int(self):
        return self.value

    @staticmethod
    def enum(int_value):
        if (int_value == 0):
            return Dataset.TR

        if (int_value == 1):
            return Dataset.VAL

        if (int_value == 2):
            return Dataset.TE

        if (int_value == 3):
            return Dataset.TR_OUT

        if (int_value == 4):
            return Dataset.VAL_OUT

        return None

    def __str__(self):
        if (self.value == 0):
            return "Training"

        if (self.value == 1):
            return "Validation"

        if (self.value == 2):
            return "Testing"

        if (self.value == 3):
            return "TrTarget"

        if (self.value == 4):
            return "ValTarget"

        return "Unknown"