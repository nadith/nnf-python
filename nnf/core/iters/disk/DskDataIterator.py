# -*- coding: utf-8 -*-
"""
.. module:: DskDataIterator
   :platform: Unix, Windows
   :synopsis: Represent DskDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np

# Local Imports
from nnf.core.iters.DataIterator import DataIterator
from nnf.db.Dataset import Dataset

class DskDataIterator(DataIterator):
    """DskDataIterator iterates the data in the disk for :obj:`NNModel'.

    Attributes
    ----------
    frecords : :obj:`list`
        List of file records. frecord = [fpath, fpos, cls_lbl]
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, edataset, frecords, nb_class, pp_params=None, 
                                                        fn_gen_coreiter=None):
        """Construct a DskmanDskDataIterator instance.

        Parameters
        ----------
        edataset : :obj:`Dataset`
            Dataset enumeration key.

        frecords : :obj:`list`
            List of file records. frecord = [fpath or class_path, fpos or filename, cls_lbl]

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
        self.frecords = frecords
        self.nb_class = nb_class

    def init_ex(self, params=None, setting=None):
        """Initialize the instance.

        Parameters
        ----------
        params : :obj:`dict`
            Core iterator parameters.

        setting : :obj:`dict` (YET TO IMPLEMENT)
            Pre-process setting (ImageDataPreProcessor.setting) to apply.

        Note
        ----
        `DataIterator` class implements an init() method that accepts different params.
        Hence PEP8 warns in overriding init() method but not for init_ex() method.
        """
        params = {} if (params is None) else params
        params['_edataset'] = self.edataset  # For debugging purpose

        # Release the core iterator if already exists
        super()._release_core_iter()

        # Create a core-iterator object
        gen_next = self._imdata_pp.flow_from_directory_ex(self.frecords, self.nb_class, params)

        # Invoke super class init
        super().init(gen_next, params)

    def clone(self):
        """Create a copy of this DskDataIterator object."""
        new_obj = DskDataIterator(self.edataset, self.frecords,
                    self.nb_class, self._pp_params, self._fn_gen_coreiter)

        # IMPORTANT: init_ex() is costly since it fits the data again (fitting data yet to be implemented)
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
        del self.frecords