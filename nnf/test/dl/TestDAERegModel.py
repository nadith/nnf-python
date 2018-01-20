# -*- coding: utf-8 -*-
"""
.. module:: TestDAERegModel
   :platform: Unix, Windows
   :synopsis: Represent TestDAERegModel and related classes.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
import scipy.io
import os

# Local Imports
from nnf.test.dl.Globals import *
from nnf.core.NNCfg import DAEPreCfg, DAERegCfg
from nnf.core.NNFramework import NNFramework
from nnf.core.models.DAERegModel import DAERegModel
from nnf.core.callbacks.TensorBoardEx import TensorBoardEx
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
from nnf.db.NNdb import NNdb
from nnf.db.NNPatch import NNPatch
from nnf.db.Selection import Selection
from nnf.db.preloaded.MnistDb import MnistDb
from nnf.db.preloaded.Cifar10Db import Cifar10Db


class Params(object):
    """Params provides the parameters for databases and data iterators. (both disk/memory)."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    @staticmethod
    def inmem_dbparams(nndb, sel):
        """Use nndb, iterators should read from the memory"""
        dbparam1 = {'alias': "DB1",
                    'memdb_param': {'nndb': nndb},
                    'selection': sel}
        return dbparam1

    @staticmethod
    def inmem_iterparams():
        """Iterators read from the memory"""
        iter_params = {'input_vectorized': True,
                       'batch_size': 32,
                       'in_mem': True}
        return iter_params

    @staticmethod
    def indsk_dbparams(db_dir, sel):
        """Use database at db_dir, write the processed data on to the disk, iterators should read from the disk."""
        dbparam1 = {'alias': "DB1",

                    # `target_size` describes the original data size.
                    # Used when reading data at db_dir via core iterators, but not `DskmanDskDataIterator` iterators.
                    # Exception: Used in `DskmanDskBigDataIterator` when reading binary files.
                    'dskdb_param': {'db_dir': db_dir, 'target_size': (33, 33)},
                    'selection': sel}
        return dbparam1

    @staticmethod
    def indsk_iterparams():
        """Iterators read from the disk"""
        iter_params = {'input_vectorized': True,
                       'batch_size': 32,
                       'in_mem': False}
        return iter_params

    @staticmethod
    def mem_to_dsk_dbparams(nndb, db_dir, sel):
        """Use nndb, write the processed data on to the disk, iterators may read from the memory or disk."""
        dbparam1 = {'alias': "DB1",
                    'memdb_param': {'nndb': nndb},

                    # `target_size` describes the original data size.
                    # Used when reading data at db_dir via core iterators, but not `DskmanDskDataIterator` iterators.
                    # Exception: Used in `DskmanDskBigDataIterator` when reading binary files.
                    'dskdb_param': {'db_dir': db_dir, 'target_size': (33, 33)},
                    'selection': sel}
        return dbparam1

class DAERegPatch(NNPatch):
    def _generate_nnmodels(self, nnpatch):
        """Extend this method to implement custom generation of `nnmodels`."""
        iter_params = None
        iter_pp_params = {'rescale': 1. / 255}

        if Globals.TEST_ITERMODE == TestIterMode.ITER_NO:
            # Pre-loaded database is used
            iter_params = None
            iter_pp_params = None

        elif Globals.TEST_ITERMODE == TestIterMode.ITER_MEM:
            iter_params = Params.inmem_iterparams()

        elif Globals.TEST_ITERMODE == TestIterMode.ITER_DSK:
            iter_params = Params.indsk_iterparams()

        elif Globals.TEST_ITERMODE == TestIterMode.ITER_MEM_DSK:
            # Can use mem or dsk iter_params
            # iter_params = Params.inmem_iterparams()
            iter_params = Params.indsk_iterparams()

        return DAERegModel(callbacks={'predict':TestDAERegModel._fn_predict},
                        iter_params=iter_params,
                        iter_pp_params=iter_pp_params)

class DAERegPatchGen(NNPatchGenerator):
    def _new_nnpatch(self, h, w, offset):
        return DAERegPatch(h, w, offset, True)

class TestDAERegModel(object):
    """TestDAERegModel to test TestDAEReg model."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def Test_preloaded_db(self, pretrain=True):

        if not Globals.TEST_ITERMODE == TestIterMode.ITER_NO:
            raise Exception('Globals.TEST_ITERMODE != TestIterMode.ITER_NO: usage is invalid. Invoke Test(...) instead.')

        # Get the file path for mnist database
        cwd = os.getcwd()
        db_file_path = os.path.join(cwd, "DataFolder", "keras", "mnist.npz")

        # Get the file path for Cifar10 database
        # db_file_path = os.path.join(cwd, "DataFolder", "keras", "cifar-10-batches-py")

        # Initialize the Framework
        nnf = NNFramework.init(DAERegPatchGen())

        daepcfgs = []
        daecfg = DAERegCfg([784, 512, 400, 320, 200, 784], 
                            act_fns=['input', 'relu', 'relu', 'sigmoid', 'sigmoid', 'sigmoid'],
                            lp=['input', 'dr', 'dr', 'rg', 'rg', 'output'],
                            preloaded_db=MnistDb(db_file_path, debug=True))
        # daecfg = DAERegCfg([3072, 512, 400, 320, 200, 3072],
        #                     act_fns=['input', 'relu', 'relu', 'sigmoid', 'sigmoid', 'sigmoid'],
        #                     lp=['input', 'dr', 'dr', 'rg', 'rg', 'output'],
        #                     preloaded_db=Cifar10Db(db_file_path, debug=True))

        if (pretrain):
            daepcfgs.append(DAEPreCfg([784, 512, 784], preloaded_db=MnistDb(db_file_path, debug=True)))
            daepcfgs.append(DAEPreCfg([512, 400, 512]))
            daepcfgs.append(DAEPreCfg([400, 320, 784])) 
            daepcfgs.append(DAEPreCfg([320, 200, 784]))
            nnf.pre_train(daepcfgs, daecfg)

        nnf.train(daecfg)
        nnf.predict(daecfg)

    def Test(self, pretrain=True):

        if Globals.TEST_ITERMODE == TestIterMode.ITER_NO:
            raise Exception('Globals.TEST_ITERMODE == TestIterMode.ITER_NO: usage is invalid. Invoke Test_preloaded_db(...) instead.')

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

        # Must mention targets since this is a regression model.
        sel.tr_out_col_indices = np.uint8(np.zeros(len(sel.tr_col_indices)))
        sel.val_out_col_indices = np.uint8(np.zeros(len(sel.val_col_indices)))
        sel.te_out_col_indices = np.uint8(np.zeros(len(sel.te_col_indices)))

        # define a data workspace to work with disk database
        db_dir = os.path.join(data_folder, "disk_db")

        list_dbparams = None
        if Globals.TEST_ITERMODE == TestIterMode.ITER_MEM:
            # Use nndb, iterators should read from the memory
            list_dbparams = Params.inmem_dbparams(nndb, sel)

        elif Globals.TEST_ITERMODE == TestIterMode.ITER_DSK:
            # Use database at db_dir, write the processed data on to the disk, iterators should read from the disk
            list_dbparams = Params.indsk_dbparams(db_dir, sel)

        elif Globals.TEST_ITERMODE == TestIterMode.ITER_MEM_DSK:
            # Use nndb, write the processed data on to the disk, iterators may read from the memory or disk.
            list_dbparams = Params.mem_to_dsk_dbparams(nndb, db_dir, sel)

        # Initialize the Framework
        nnf = NNFramework.init(DAERegPatchGen(), list_dbparams)

        daepcfgs = []
        daecfg = DAERegCfg([1089, 784, 500, 300, 200, 1089], 
                            act_fns=['input', 'relu', 'relu', 'sigmoid', 'sigmoid', 'sigmoid'],
                            lp=['input', 'dr', 'dr', 'rg', 'rg', 'output'])
        daecfg.callbacks = [TensorBoardEx(log_dir='D:\\TF\\DAEReg')]

        if (pretrain):
            daepcfgs.append(DAEPreCfg([1089, 784, 1089]))
            daepcfgs.append(DAEPreCfg([784, 500, 784]))
            daepcfgs.append(DAEPreCfg([500, 300, 1089])) 
            daepcfgs.append(DAEPreCfg([300, 200, 1089]))
            nnf.pre_train(daepcfgs, daecfg)

        # Train with cpu (default is gpu0)
        import tensorflow as tf
        with tf.device('/cpu:0'):
            nnf.train(daecfg)

        nnf.test(daecfg)
        nnf.predict(daecfg)

    ##########################################################################
    # NNModel: Callbacks
    ##########################################################################
    @staticmethod
    def _fn_predict(nnmodel, nnpatch, predictions, true_output):
        pass