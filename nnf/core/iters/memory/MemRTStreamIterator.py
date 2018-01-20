# -*- coding: utf-8 -*-
"""
.. module:: MemDataIterator
   :platform: Unix, Windows
   :synopsis: Represent MemDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
from keras import backend as K

# Local Imports
from nnf.core.iters.DataIterator import DataIterator


class MemRTStreamIterator(DataIterator):
    """MemDataIterator iterates the data in memory for :obj:`NNModel'.

    Attributes
    ----------
    nndb : :obj:`NNdb`
        Database to iterate.
    """
    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, edataset, rt_stream, pp_params=None, fn_gen_coreiter=None):
        """Construct a MemDataIterator instance.

        Parameters
        ----------
        edataset : :obj:`Dataset`
            Dataset enumeration key.

        nndb : :obj:`NNdb`
            In memory database to iterate.

        nb_class : int | None
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
        self.rt_stream = rt_stream

    def init_ex(self, params=None, y=None, setting=None):
        """Initialize the instance.

        Parameters
        ----------
        params : :obj:`dict`
            Core iterator parameters.

            TODO: document the params
        """
        params = {} if (params is None) else params
        params['_edataset'] = self.edataset

        # # Required for featurewise_center, featurewise_std_normalization and
        # # zca_whitening. Currently supported only for in memory datasets.
        # if (self._imdata_pp.featurewise_center or
        #     self._imdata_pp.featurewise_std_normalization or
        #     self._imdata_pp.zca_whitening or
        #     (self._pp_params is not None and 'mapminmax' in self._pp_params)):
        #     self._imdata_pp.fit(db,
        #                         self._imdata_pp.augment,
        #                         self._imdata_pp.rounds,
        #                         self._imdata_pp.seed)
        # else:
        #     self._imdata_pp.apply(setting)


        # TODO: Validation for params['input_shape'] for being a mandatary parameters
        gen_next = self._imdata_pp.flow_realtime(self.rt_stream, params=params)

        # if (self._imdata_pp.wnd_max_normalize):
        #     self._imdata_pp.fit_window(gen_next.window)

        # setting = self._imdata_pp.setting
        super().init(gen_next, params)
        # return setting

    def clone(self):
        """Create a copy of this object."""
        new_obj = MemRTStreamIterator(self.edataset, self.rt_stream, self._pp_params, self._fn_gen_coreiter)
        #new_obj.init_ex(self._params)

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
        del self.rt_stream