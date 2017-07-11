# -*- coding: utf-8 -*-
"""
.. module:: BigDataNumpyArrayIterator
   :platform: Unix, Windows
   :synopsis: Represent BigDataNumpyArrayIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np

# Local Imports
from nnf.core.iters.memory.NumpyArrayIterator import NumpyArrayIterator

class BigDataNumpyArrayIterator(NumpyArrayIterator):
    """BigDataNumpyArrayIterator iterates the raw data in the memory for :obj:`NNModel'."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, X, y, nb_class, image_data_pp, params=None):
        """Construct a :obj:`BigDataNumpyArrayIterator` instance.

        Parameters
        ----------
        X : `array_like`
            Data in 2D matrix. Format: Samples x Features.

        y : `array_like`
            Vector indicating the class labels.

        image_data_pp : :obj:`ImageDataPreProcessor`, sub class of :obj:`ImageDataGenerator`
            Image data pre-processor.

        params : :obj:`dict`
            Core iterator parameters. 
        """
        super().__init__(X, y, nb_class, image_data_pp, params)

    def _get_data(self, X, j):
        """Load raw data item from in memory database, pre-process and return.

        Parameters
        ----------
        X : `array_like`
            Data matrix. Format Samples x ...

        j : int
            Index of the data item to be featched. 
        """
        x = self.X[j]

        # TODO: Apply necessary transofmraiton
        # image_data_generator set to image_data_pp in the constructor        
        #x = self.image_data_generator.random_transform(x)
        x = self.image_data_generator.standardize(x)
        return x