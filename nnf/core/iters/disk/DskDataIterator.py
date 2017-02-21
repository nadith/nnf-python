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
            List of file records. frecord = [fpath, fpos, cls_lbl].

        nb_class : int
            Number of classes.

        pp_params : :obj:`dict`, optional
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.
            (Default value = None).

        fn_gen_coreiter : `function`, optional
            Factory method to create the core iterator.
            (Default value = None).
        """
        super().__init__(pp_params, fn_gen_coreiter, edataset, nb_class)        
        self.frecords = frecords

    def init(self, params=None): 
        """Initialize the instance.

        Parameters
        ----------
        params : :obj:`dict`
            Core iterator parameters.

        Note
        ----
        `params` may have different set of fields depending on the 
        custom extensions to the class DskDataIterator.
        i.e params.binary_data is unique and only used in
            BigDataDirectoryIterator(DskDataIterator)
        """
        gen_next = self._imdata_pp.flow_from_directory(self.frecords,
                                                        self._nb_class, 
                                                        params)          
        super().init(gen_next, params)

    def clone(self):
        """Create a copy of this NNdb object."""
        new_obj = DskDataIterator(self.edataset, self.frecords,
                    self._nb_class, self._pp_params, self._fn_gen_coreiter)
        new_obj.init(self._params)
        new_obj.sync(self._sync_gen_next)
        return new_obj

    def sync_generator(self, iter):
        """Sync the secondary iterator with this iterator.

        Sync the secondary core iterator with this core itarator internally.
        Used when data needs to be generated with its matching target.

        Parameters
        ----------
        iter : :obj:`DskDataIterator`
            Iterator that needs to be synced with this iterator.

        Note
        ----
        Current supports only 1 iterator to be synced.
        """
        if (iter is None): return False
        self.sync(iter._gen_next)
        return True

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _release(self):
        """Release internal resources used by the iterator."""
        super()._release()
        del self.frecords
        del self._nb_class
        del self._pp_params
        del self._params
        del self._fn_gen_coreiter
        del self._sync_iter