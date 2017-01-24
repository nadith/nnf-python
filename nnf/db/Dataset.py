"""Dataset Module to represent Dataset class."""
# -*- coding: utf-8 -*-
# Global Imports
from enum import Enum

# Local Imports


class Dataset(Enum):
    """Describe Dateset.
    # Do not change the order of the values
    # Refer NNDiskMan.process() for more details

    Attributes
    ----------
    value : describe
        describe.

    Methods
    -------
    int()
        describe.

    get_enum_list()
        staticmethod : describe.

    enum()
        staticmethod : describe.

    __str__()
        describe.

    Examples
    --------
    describe
    >>> nndb = Dataset(object)
    """
    TR = 0 
    VAL = 1
    TE = 2
    TR_OUT = 3
    VAL_OUT = 4
    TE_OUT = 5

    def int(self):
        """describe.

        Parameters
        ----------
        sel : selection structure
            Information to split the dataset.

        Returns
        -------
        self.value : describe
        """
        return self.value

    @staticmethod
    def get_enum_list():
        """describe.

        Parameters
        ----------
        sel : selection structure
            Information to split the dataset.

        Returns
        -------
        Dataset.TR : describe

        Dataset.VAL : describe

        Dataset.TE : describe

        Dataset.TR_OUT : describe

        Dataset.VAL_OUT : describe
        """
        return [Dataset.TR, Dataset.VAL, Dataset.TE, Dataset.TR_OUT, Dataset.VAL_OUT, Dataset.TE_OUT]

    @staticmethod
    def enum(int_value):
        """describe.

        Parameters
        ----------
        int_value : describe
            descriebe.

        Returns
        -------
        None : describe
        """
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

        if (int_value == 5):
            return Dataset.TE_OUT

        return None

    def __str__(self):
        """describe.

        Parameters
        ----------
        sel : selection structure
            Information to split the dataset.

        Returns
        -------
        "Unknown" : describe
        """
        if (self.value == 0):
            return "tr"

        if (self.value == 1):
            return "val"

        if (self.value == 2):
            return "te"

        if (self.value == 3):
            return "tr_out"

        if (self.value == 4):
            return "val_out"

        if (self.value == 5):
            return "te_out"

        return "Unknown"