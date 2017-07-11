# -*- coding: utf-8 -*-
"""
.. module:: MemDataIterator
   :platform: Unix, Windows
   :synopsis: Represent MemDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from keras import backend as K

# Local Imports
from nnf.core.iters.DataIterator import DataIterator
from nnf.db.Dataset import Dataset

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
    def __init__(self, edataset, nndb, nb_class, pp_params=None, 
                                                        fn_gen_coreiter=None):
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
        super().__init__(pp_params, fn_gen_coreiter, edataset, nb_class)

        # NNdb database to create the iteartor
        self.nndb = nndb

    def init(self, params=None, y=None, setting=None):
        """Initialize the instance.

        Parameters
        ----------
        params : :obj:`dict`
            Core iterator parameters.
        """
        params = {} if (params is None) else params

        # Set db and _image_shape (important for convonets, and fit() method below)
        data_format = params['data_format'] if ('data_format' in params) else None
        data_format = K.image_data_format() if (data_format is None) else data_format
        target_size = (self.nndb.h, self.nndb.w)

        db = None
        if (self.nndb.ch == 3):  # 'rgb'
            if data_format == 'channels_last':
                params['_image_shape'] = target_size + (3,)
                db = self.nndb.db_convo_tf
            else:
                params['_image_shape'] = (3,) + target_size
                db = self.nndb.db_convo_th

        else:
            if data_format == 'channels_last':
                params['_image_shape'] = target_size + (1,)
                db = self.nndb.db_convo_tf
            else:
                params['_image_shape'] = (1,) + target_size
                db = self.nndb.db_convo_th

        # Required for featurewise_center, featurewise_std_normalization and 
        # zca_whitening. Currently supported only for in memory datasets.
        if (self._imdata_pp.featurewise_center or
            self._imdata_pp.featurewise_std_normalization or
            self._imdata_pp.zca_whitening): 
            self._imdata_pp.fit(db, 
                                self._imdata_pp.augment, 
                                self._imdata_pp.rounds,
                                self._imdata_pp.seed)
        else:
            self.imdata_pp_.apply(setting)

        gen_next = self._imdata_pp.flow(db, self.nndb.cls_lbl, 
                                            self._nb_class, params=params)
        settings = self.imdata_pp_.settings
        super().init(gen_next, params)
        return settings

    def clone(self):
        """Create a copy of this object."""
        new_obj = MemDataIterator(self.edataset, self.nndb, self._nb_class,
                                    self._pp_params, self._fn_gen_coreiter)
        new_obj.init(self._params)
        new_obj.sync(self._sync_gen_next)
        return new_obj

    def sync_generator(self, iter):
        """Sync the secondary iterator with this iterator.

        Sync the secondary core iterator with this core itarator internally.
        Used when data needs to be generated with its matching target.

        Parameters
        ----------
        iter : :obj:`MemDataIterator`
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
        del self.nndb