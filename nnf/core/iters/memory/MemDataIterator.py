# -*- coding: utf-8 -*-
"""
.. module:: MemDataIterator
   :platform: Unix, Windows
   :synopsis: Represent MemDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from warnings import warn as warning

# Local Imports
from nnf.core.iters.DataIterator import DataIterator


class MemDataIterator(DataIterator):
    """MemDataIterator iterates the data in memory for :obj:`NNModel'.

    Attributes
    ----------
    nndb : :obj:`NNdb`
        Database to iterate.
    """
    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, edataset, nndb, pp_params=None, fn_gen_coreiter=None):
        """Construct a MemDataIterator instance.

        Parameters
        ----------
        edataset : :obj:`Dataset`
            Dataset enumeration key.

        nndb : :obj:`NNdb`
            In memory database to iterate.

        nb_class : int
            Number of classes.

        pp_params : :obj:`dict`, optional
            Pre-processing parameters for :obj:`ImageDataPreProcessor`. 
            (Default value = None). 

        fn_gen_coreiter : `function`, optional
            Factory method to create the core iterator.
            (Default value = None).
        """
        super().__init__(pp_params, fn_gen_coreiter, edataset)

        # NNdb database to create the iterator
        self.nndb = nndb

    def init_ex(self, params=None, setting=None):
        """Initialize the instance.

        Parameters
        ----------
        params : :obj:`dict`
            Core iterator parameters.

        setting : :obj:`dict`
            Pre-process setting (ImageDataPreProcessor.setting) to apply.

        Note
        ----
        `DataIterator` class implements an init() method that accepts different params.
        Hence PEP8 warns in overriding init() method but not for init_ex() method.
        """
        params = {} if (params is None) else params
        params['_edataset'] = self.edataset  # Useful for RTNumpyIterator

        # Required for featurewise_center, featurewise_std_normalization and 
        # zca_whitening. Currently supported only for in memory datasets.
        if (self._imdata_pp.featurewise_center or
            self._imdata_pp.featurewise_std_normalization or
            self._imdata_pp.zca_whitening or 
            (self._pp_params is not None and 'mapminmax' in self._pp_params)):

            # Issue a warning if the given setting is not utilized
            if setting is not None:
                warning('Pre-process setting is provided but ignored in ' +
                        str(self.edataset).upper() + ' dataset fit() processing.')

            # Fit the dataset to calculate the general pre-process setting
            self._imdata_pp.fit(self.nndb,
                                self._imdata_pp.augment, 
                                self._imdata_pp.rounds,
                                self._imdata_pp.seed)
        else:
            self._imdata_pp.apply(setting)

        # Release the core iterator if already exists
        super()._release_core_iter()

        # Create a core-iterator object
        gen_next = self._imdata_pp.flow_ex(self.nndb, params=params)

        # Window max normalization for window iterator
        if (self._imdata_pp.wnd_max_normalize):

            # Issue a warning if the given setting is not utilized
            if setting is not None:
                warning('Pre-process setting is provided but ignored in ' +
                        str(self.edataset).upper() + ' dataset fit_window() processing.')

            # Fit the dataset to calculate the pre-process settings related to window_iter
            self._imdata_pp.fit_window(gen_next.window_iter)

        # Invoke super class init
        super().init(gen_next, params)

        # Returns pre-process setting
        return self._imdata_pp.setting

    def clone(self):
        """Create a copy of this object."""
        new_obj = MemDataIterator(self.edataset, self.nndb, self._pp_params, self._fn_gen_coreiter)

        # IMPORTANT: init_ex() is costly since it fits the data again
        # new_obj.init_ex(self._params)

        # Instead apply the settings which is already calculated
        new_obj._imdata_pp.apply(self._imdata_pp.setting)

        # Clone the core-iterator object along with input|output_generators
        new_gen_next = self.core_iter.clone()

        # Invoke super class init
        new_obj.init(new_gen_next, self.params)

        return new_obj

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def release(self):
        """Release internal resources used by the iterator."""
        super().release()
        del self.nndb
        self.nndb = None
