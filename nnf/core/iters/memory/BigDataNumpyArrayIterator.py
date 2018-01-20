# -*- coding: utf-8 -*-
"""
.. module:: BigDataNumpyArrayIterator
   :platform: Unix, Windows
   :synopsis: Represent BigDataNumpyArrayIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports

# Local Imports
from nnf.core.iters.memory.NumpyArrayIterator import NumpyArrayIterator


class BigDataNumpyArrayIterator(NumpyArrayIterator):
    """BigDataNumpyArrayIterator iterates the raw data in the memory for :obj:`NNModel'."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, X, y, nb_class, imdata_pp, params=None):
        """Construct a :obj:`BigDataNumpyArrayIterator` instance.

        Parameters
        ----------
        X : ndarray
            Data in 2D matrix. Format: Samples x Features.

        y : ndarray
            Vector indicating the class labels.

        imdata_pp : :obj:`ImageDataPreProcessor`, sub class of :obj:`ImageDataGenerator`
            Image data pre-processor.

        params : :obj:`dict`
            Core iterator parameters. 
        """
        super().__init__(X, y, nb_class, imdata_pp, params)

    def _get_data(self, X, y, j):
        """Load raw data item from in memory database, pre-process and return.

        Parameters
        ----------
        X : ndarray
            Data matrix. Format Samples x ...

        j : int
            Index of the data item to be fetched.
        """
        return super()._get_data(X, y, j)
