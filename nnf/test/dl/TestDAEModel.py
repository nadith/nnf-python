# -*- coding: utf-8 -*-
"""
.. module:: TestDAEModel
   :platform: Unix, Windows
   :synopsis: Represent TestDAEModel and related classes.

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
from nnf.db.preloaded.MnistDb import MnistDb
from nnf.core.NNCfg import DAEPreCfg, DAECfg
from nnf.core.NNPatchMan import NNPatchMan
from nnf.core.models.DAEModel import DAEModel
from nnf.core.models.NNModelPhase import NNModelPhase
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator

class DAEPatch(NNPatch):
    def generate_nnmodels(self):
        return DAEModel({'predict':TestDAEModel._fn_predict})

class DAEPatchGen(NNPatchGenerator):
    def new_nnpatch(self, h, w, offset):
        return DAEPatch(h, w, offset, True)

class TestDAEModel(object):
    """TestAEModel to test AE model."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def Test_preloaded_db(self, pretrain=True):
        cwd = os.getcwd()
        model_folder = os.path.join(cwd, "ModelFolder")

        # Get the file path for mnist database
        db_file_path = os.path.join(cwd, "DataFolder", "keras", "mnist.npz")
        
        nnpatchman = NNPatchMan(DAEPatchGen())
        daepcfgs = []
        daecfg = DAECfg([784, 512, 400, 512, 784], 
                        act_fns=['input', 'relu', 'relu', 'sigmoid', 'sigmoid'],
                        preloaded_db=MnistDb(db_file_path, debug=True))

        if (pretrain):
            pre_cfg = DAEPreCfg([784, 512, 784], preloaded_db=MnistDb(db_file_path, debug=True))

            # To save the model & weights
            # pre_cfg.model_dir = model_folder

            # To save the weights only
            # pre_cfg.weights_dir = model_folder

            # Append `pre_cfg` to `daepcfgs` list
            daepcfgs.append(pre_cfg)
                
            pre_cfg = DAEPreCfg([512, 400, 512])

            # To save the model & weights
            # pre_cfg.model_dir = model_folder

            # To save the weights only  
            # pre_cfg.weights_dir = model_folder

            # Append `pre_cfg` to `daepcfgs` list
            daepcfgs.append(pre_cfg)

            # Peform pre-training
            nnpatchman.pre_train(daepcfgs, daecfg)

        # To save the model & weights
        # daecfg.model_dir = model_folder

        # To save the weights only      
        # daecfg.weights_dir = model_folder

        nnpatchman.train(daecfg)
        #nnpatchman.test(daecfg)
        nnpatchman.predict(daecfg)

    def Test(self, pretrain=True):
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
        list_dbparams = self.__inmem_dbparams(nndb, sel)

        # Use database at db_dir, write the processed data on to the disk, iterators read from the disk
        # list_dbparams = self.__indsk_dbparams(os.path.join(db_dir, "_processed_DB1"), sel)

        # Use nndb, write the processed data on to the disk, iterators read from the disk.
        #list_dbparams = self.__mem_to_dsk_indsk_dbparams(nndb, db_dir, sel)

        # Use nndb, write the processed data on to the disk, but iterators read from the memory.
        #list_dbparams = self.__mem_to_dsk_inmem_dbparams(nndb, db_dir, sel)

        nnpatchman = NNPatchMan(DAEPatchGen(), list_dbparams)

        daepcfgs = []
        daecfg = DAECfg([1089, 784, 500, 784, 1089], 
                        act_fns=['input', 'relu', 'relu', 'sigmoid', 'sigmoid'])

        if (pretrain):
            daepcfgs.append(DAEPreCfg([1089, 784, 1089]))
            daepcfgs.append(DAEPreCfg([784, 500, 784]))        
            nnpatchman.pre_train(daepcfgs, daecfg)

        nnpatchman.train(daecfg)
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
    # NNModel: Callbacks
    ##########################################################################
    @staticmethod
    def _fn_predict(nnmodel, nnpatch, predictions, true_output):
        pass
