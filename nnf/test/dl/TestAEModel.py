# -*- coding: utf-8 -*- TODO: Revisit the comments
"""
.. module:: TestAEModel
   :platform: Unix, Windows
   :synopsis: Represent TestAEModel and related classes.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
import scipy.io
import os

# Local Imports
from nnf.db.NNdb import NNdb
from nnf.db.Format import Format
from nnf.db.Dataset import Dataset
from nnf.db.NNPatch import NNPatch
from nnf.db.DbSlice import DbSlice
from nnf.db.Selection import Select
from nnf.db.Selection import Selection
from nnf.core.NNCfg import AECfg
from nnf.core.NNPatchMan import NNPatchMan
from nnf.core.models.Autoencoder import Autoencoder
from nnf.core.models.NNModelPhase import NNModelPhase
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator

class AEPatch(NNPatch):
    def generate_nnmodels(self):
        return Autoencoder({'predict':TestAEModel._fn_predict})

class AEPatchGen(NNPatchGenerator):
    def __init__(self): 
        super().__init__()

    def new_nnpatch(self, h, w, offset):
        return AEPatch(h, w, offset, True)

class TestAEModel(object):
    """TestAEModel to test AE model."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def Test(self):
        # Get the current working directory, define a `DataFolder`
        cwd = os.getcwd()
        data_folder = os.path.join(cwd, "DataFolder")

        # Load image database `AR`
        matStruct = scipy.io.loadmat(os.path.join(data_folder, 'IMDB_66_66_AR_8.mat'),
                                    struct_as_record=False, squeeze_me=True)
        imdb_obj = matStruct['imdb_obj']

        # Training, Validation, Testing databases
        nndb = NNdb('original', imdb_obj.db, 8, True)
        sel = Selection()
        sel.use_rgb = False
        sel.scale = (33, 33)
        sel.tr_col_indices = np.uint8(np.array([0, 1, 2, 3, 4, 5]))
        sel.val_col_indices = np.uint8(np.array([6]))
        sel.te_col_indices = np.uint8(np.array([7]))
        sel.class_range = np.uint8(np.arange(0, 100))
        #[nndb_tr, nndb_val, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)

        # define a data workspace to work with disk database
        db_dir = os.path.join(data_folder, "disk_db")

        # Use nndb, iterators read from the memory
        #list_dbparams = self.__inmem_dbparams(nndb, sel)

        # Use database at db_dir, write the processed data on to the disk, iterators read from the disk
        # list_dbparams = self.__indsk_dbparams(os.path.join(db_dir, "_processed_DB1"), sel)

        # Use nndb, write the processed data on to the disk, iterators read from the disk.
        list_dbparams = self.__mem_to_dsk_indsk_dbparams(nndb, db_dir, sel)

        # Use nndb, write the processed data on to the disk, but iterators read from the memory.
        #list_dbparams = self.__mem_to_dsk_inmem_dbparams(nndb, db_dir, sel)

        nnpatchman = NNPatchMan(AEPatchGen(), list_dbparams)

        aecfg = AECfg()
        aecfg.numepochs = 20
        aecfg.nb_val_samples = 800
        aecfg.samples_per_epoch = 600
        nnpatchman.train(aecfg)
        #nnpatchman.predict(daecfg)

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __inmem_dbparams(self, nndb, sel):
        """Use nndb, iterators read from the memory"""
        dbparam1 = {'alias': "DB1",
                'nndb': nndb, 'selection': sel,
                'iter_param': {'class_mode':None, 
                                'input_vectorized':True, 
                                'batch_size':1
                                },
                'iter_pp_param': {'rescale':1./255},
                'iter_in_mem': True}

        return dbparam1

    def __indsk_dbparams(self, db_dir, sel):
        """Use database at db_dir, write the processed data on to the disk, iterators read from the disk."""
        dbparam1 = {'alias': "DB1",
                'db_dir': db_dir, 'selection': sel,
                'iter_param': {'class_mode':None, 
                                'input_vectorized':True,
                                'batch_size':1,
                                'target_size':(33,33), 
                                'color_mode':'grayscale'
                                },
                'iter_pp_param': {'rescale':1./255},
                'iter_in_mem': False}

        return dbparam1

    def __mem_to_dsk_indsk_dbparams(self, nndb, db_dir, sel):
        """Use nndb, write the processed data on to the disk, iterators read from the disk."""
        dbparam1 = {'alias': "DB1",
                'nndb': nndb, 'db_dir': db_dir, 'selection': sel, 
                'iter_param': {'class_mode':None, 
                                'input_vectorized':True,
                                'batch_size':1,
                                'target_size':(33,33), 
                                'color_mode':'grayscale'
                                },
                'iter_pp_param': {'rescale':1./255},
                'iter_in_mem': False}

        return dbparam1

    def __mem_to_dsk_inmem_dbparams(self, nndb, db_dir, sel):
        """Use nndb, write the processed data on to the disk, iterators read from the memory."""
        dbparam1 = {'alias': "DB1",
                'nndb': nndb, 'db_dir': db_dir, 'selection': sel, 
                'iter_param': {'class_mode':None, 
                                'input_vectorized':True, 
                                'batch_size':1
                                },
                'iter_pp_param': {'rescale':1./255},
                'iter_in_mem': True}
        
        return dbparam1

    ##########################################################################
    # NNF: Callbacks
    ##########################################################################
    @staticmethod
    def _fn_predict(nnmodel, nnpatch, prediction, labels):
        pass