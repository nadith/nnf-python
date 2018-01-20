# -*- coding: utf-8 -*-
"""
.. module:: TestCNNModel
   :platform: Unix, Windows
   :synopsis: Represent TestCNNModel and related classes.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
import scipy.io
import os

# Local Imports
from nnf.test.dl.Globals import *
from nnf.core.NNCfg import CNNCfg
from nnf.core.Metric import Metric
from nnf.core.NNFramework import NNFramework
from nnf.core.models.CNN2DRegModel import CNN2DRegModel
from nnf.core.callbacks.TensorBoardEx import TensorBoardEx
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
from nnf.core.iters.custom.WindowIter import WindowIter, Window, WndIterMode
from nnf.db.NNdb import NNdb
from nnf.db.NNPatch import NNPatch
from nnf.db.Dataset import Dataset
from nnf.db.Selection import Selection
from nnf.db.preloaded.MnistDb import MnistDb
from nnf.db.preloaded.Cifar10Db import Cifar10Db


class Params(object):
    """Params provides the parameters for databases and data iterators. (both disk/memory)."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    @staticmethod
    def inmem_dbparams(nndb, sel, nndb2, sel2):
        """Use nndb, iterators should read from the memory"""
        dbparam1 = {'alias': "DB1",
                    'memdb_param': {'nndb': nndb},
                    'selection': sel}

        dbparam2 = {'alias': "DB2",
                    'memdb_param': {'nndb': nndb2},
                    'selection': sel2}

        return [dbparam1, dbparam2]

    @staticmethod
    def inmem_iterparams():
        """Iterators read from the memory"""
        # Default behavior:
        # Iterators read from memory, use `fn_reshape_input` to shape it to the `input_shape`
        # expected by the network.
        # fn_reshape_input = <...> (Default value = resize operation)
        # input_shape = <...> (Default value = dimension of nndb (processed by sel))
        iter_params = {'DB1': {'class_mode': None,
                               'batch_size': 32,
                               'in_mem': True},

                       'DB2': {'class_mode': None,
                               'input_vectorized': True,
                               'in_mem': True}}

        # Behavior 2:
        # Iterators read from memory, use `fn_reshape_input` to shape it to the `input_shape`
        # expected by the network.
        # fn_reshape_input = <...> (Default value = resize operation)
        iter_params = {'DB1': {'class_mode': None,
                               'batch_size': 32,
                               # `input_shape`: ch dimension (if unspecified) is added automatically
                               # depending on the nndb(processed by sel)'s ch property
                               'input_shape': (128, 128),  # or 'input_shape': (128, 128, 1),
                               'in_mem': True},

                       'DB2': {'class_mode': None,
                               'input_vectorized': True,
                               'in_mem': True}}
        #
        # # Behavior 3:
        # # Iterators read from memory, use `fn_reshape_input` to shape it to the `input_shape`
        # # expected by the network.
        # iter_params = {'DB1': {'class_mode': None,
        #                        'batch_size': 32,
        #                        # `input_shape`: ch dimension (if unspecified) is added automatically
        #                        # depending on the nndb(processed by sel)'s ch property
        #                        'input_shape': (128, 128, 3),
        #                        'fn_reshape_input': TestCNN2DRegModel._fn_reshape_input,
        #                        'in_mem': True},
        #
        #                'DB2': {'class_mode': None,
        #                        'input_vectorized': True,
        #                        'in_mem': True}}
        return iter_params

    @staticmethod
    def indsk_dbparams(db_dir, sel, db_dir2, sel2):
        """Use database at db_dir, write the processed data on to the disk, iterators should read from the disk."""
        dbparam1 = {'alias': "DB1",

                    # `target_size` describes the original data size.
                    # Used when reading data at db_dir via core iterators, but not `DskmanDskDataIterator` iterators.
                    # Exception: Used in `DskmanDskBigDataIterator` when reading binary files.
                    'dskdb_param': {'db_dir': db_dir, 'target_size': (128, 128)},
                    'selection': sel}

        dbparam2 = {'alias': "DB2",

                    # `target_size` describes the original data size.
                    # Used when reading data at db_dir via core iterators, but not `DskmanDskDataIterator` iterators.
                    # Exception: Used in `DskmanDskBigDataIterator` when reading binary files.
                    'dskdb_param': {'db_dir': db_dir2, 'target_size': (33, 33)},
                    'selection': sel2}

        return [dbparam1, dbparam2]

    @staticmethod
    def indsk_iterparams():
        """Iterators read from the disk"""
        # Default behavior:
        # Iterators read from disk, use `fn_reshape_input` to shape it to the `input_shape`
        # expected by the network.
        # fn_reshape_input = <...> (Default value = resize operation)
        # input_shape = <...> (Default value = dskdb_param[`target_size`] + ch added automatically)
        iter_params = {'DB1': {'class_mode': None,
                               'batch_size': 32,
                               'in_mem': False},

                       'DB2': {'class_mode': None,
                               'input_vectorized': True,
                               'in_mem': False}}

        # # Behavior 2:
        # # Iterators read from disk, use `fn_reshape_input` to shape it to the `input_shape`
        # # expected by the network.
        # # fn_reshape_input = <...> (Default value = resize operation)
        # iter_params = {'DB1': {'class_mode': None,
        #                        'batch_size': 32,
        #                        # `input_shape`: ch dimension (if unspecified) is added automatically
        #                        # depending on the first data sample read
        #                        'input_shape': (128, 128),  # or 'input_shape': (128, 128, 1),
        #                        'in_mem': False},
        #
        #                 'DB2': {'class_mode': None,
        #                         'input_vectorized': True,
        #                         'in_mem': False}}
        #
        # # Behavior 3:
        # # Iterators read from disk, use `fn_reshape_input` to shape it to the `input_shape`
        # # expected by the network.
        # iter_params = {'DB1': {'class_mode': None,
        #                        'batch_size': 32,
        #                        # `input_shape`: ch dimension (if unspecified) is added automatically
        #                        # depending on the first data sample read
        #                        'input_shape': (128, 128, 3),
        #                        'fn_reshape_input': TestCNN2DRegModel._fn_reshape_input,
        #                        'in_mem': False},
        #
        #                'DB2': {'class_mode': None,
        #                        'input_vectorized': True,
        #                        'in_mem': False}}

        return iter_params

    @staticmethod
    def mem_to_dsk_dbparams(nndb, db_dir, sel, nndb2, db_dir2, sel2):
        """Use nndb, write the processed data on to the disk, iterators may read from the memory and/or disk.

        Notes
        -----
        Both nndb and disk will have the same data splits (memory inefficient).
        """
        dbparam1 = {'alias': "DB1",
                    'memdb_param': {'nndb': nndb},

                    # `target_size` describes the original data size.
                    # Used when reading data at db_dir via core iterators, but not `DskmanDskDataIterator` iterators.
                    # Exception: Used in `DskmanDskBigDataIterator` when reading binary files.
                    'dskdb_param': {'db_dir': db_dir, 'target_size': (33, 33)},
                    'selection': sel}

        dbparam2 = {'alias': "DB2",
                    'memdb_param': {'nndb': nndb2},

                    # `target_size` describes the original data size.
                    # Used when reading data at db_dir via core iterators, but not `DskmanDskDataIterator` iterators.
                    # Exception: Used in `DskmanDskBigDataIterator` when reading binary files.
                    'dskdb_param': {'db_dir': db_dir2, 'target_size': (33, 33)},
                    'selection': sel2}

        return [dbparam1, dbparam2]

    @staticmethod
    def inmem_and_indsk_dbparams(nndb, sel, db_dir, sel_dsk):
        """Use both nndb and db_dir, write processed data from db_dir to disk, iterators can read from memory and/or disk.

        Notes
        -----
        Unlike 'mem_to_dsk_dbparams', part of a complete database can be loaded to memory as a nndb and the rest can
        stay at the disk (memory efficient).
        """
        dbparam1 = {'alias': "DB1",
                    'memdb_param': {'nndb': nndb},

                    # `target_size` describes the original data size.
                    # Used when reading data at db_dir via core iterators, but not `DskmanDskDataIterator` iterators.
                    # Exception: Used in `DskmanDskBigDataIterator` when reading binary files.

                    # `sel`: local selection only applied to database at db_dir.
                    # This is useful when only training data split should be stored in the disk but validation and testing
                    # data splits on the memory for fast testing performance.
                    'dskdb_param': {'db_dir': db_dir, 'target_size': (66, 66), 'force_load_db': True, 'selection': sel_dsk},
                    'selection': sel}

        return dbparam1

class CNNPatch(NNPatch):
    def _generate_nnmodels(self, nnpatch):
        """Extend this method to implement custom generation of `nnmodels`."""
        iter_params = None
        iter_pp_params = None  # {'rescale': 1. / 255}

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

        return CNN2DRegModel(callbacks={'predict':TestCNN2DRegModel._fn_predict},
                        iter_params=iter_params,
                        iter_pp_params=iter_pp_params)

class CNNPatchGen(NNPatchGenerator):
    def _new_nnpatch(self, h, w, offset):
        return CNNPatch(h, w, offset, True)

class TestCNN2DRegModel(object):
    """TestCNNModel to test CNN model."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def Test_preloaded_db(self):

        if not Globals.TEST_ITERMODE == TestIterMode.ITER_NO:
            raise Exception('Globals.TEST_ITERMODE != TestIterMode.ITER_NO: usage is invalid. Invoke Test(...) instead.')

        cwd = os.getcwd()
        model_folder = os.path.join(cwd, "ModelFolder")

        # Get the file path for mnist database
        # db_file_path = os.path.join(cwd, "DataFolder", "keras", "mnist.npz")

        # Get the file path for Cifar10 database
        db_file_path = os.path.join(cwd, "DataFolder", "keras", "cifar-10-batches-py")

        # Initialize the Framework
        nnf = NNFramework.init(CNNPatchGen())

        cnncfg = CNNCfg()
        # Utilizing preloaded db
        # cnncfg.preloaded_db = MnistDb(db_file_path, debug=True)
        cnncfg.preloaded_db = Cifar10Db(db_file_path, debug=True)

        # To save the model & weights
        # cnncfg.save_models_dir = model_folder

        # To save the weights only      
        # cnncfg.save_weights_dir = model_folder

        cnncfg.numepochs = 5
        cnncfg.validation_steps = 3 #800
        cnncfg.steps_per_epoch = 5 #600

        nnf.train(cnncfg)
        # nnf.test(cnncfg)
        nnf.predict(cnncfg)

    def Test(self):

        if Globals.TEST_ITERMODE == TestIterMode.ITER_NO:
            raise Exception('Globals.TEST_ITERMODE == TestIterMode.ITER_NO: usage is invalid. Invoke Test_preloaded_db(...) instead.')

        # Get the current working directory, define a `DataFolder`
        cwd = os.getcwd()
        data_folder = os.path.join(cwd, "DataFolder")
        model_folder = os.path.join(cwd, "ModelFolder")

        # Load image database `AR`
        matStruct = scipy.io.loadmat(os.path.join(data_folder, 'IMDB_66_66_AR_8.mat'),
                                    struct_as_record=False, squeeze_me=True)
        imdb_obj = matStruct['imdb_obj']

        # Training, Validation, Testing databases
        nndb = NNdb('original', imdb_obj.db, 8, True)
        sel = Selection()
        # sel.use_rgb = False
        # sel.histeq = True
        # sel.scale = (128, 128) #(150, 150)  #(100, 100) #(224, 224)
        sel.tr_col_indices = np.uint8(np.array([0, 1, 2, 3, 4, 5]))
        sel.val_col_indices = np.uint8(np.array([6]))
        sel.te_col_indices = np.uint8(np.array([7]))
        sel.class_range = np.uint8(np.arange(0, 6))

        # Training target, Validation target, Testing target database
        sel2 = sel.clone()
        sel2.use_rgb = False
        sel2.scale = (33, 33)
        sel2.tr_col_indices = None
        sel2.val_col_indices = None
        sel2.te_col_indices = None

        # Must mention targets since this is a regression model.
        sel2.tr_out_col_indices = np.uint8(np.zeros(len(sel.tr_col_indices)))
        sel2.val_out_col_indices = np.uint8(np.zeros(len(sel.val_col_indices)))
        sel2.te_out_col_indices = np.uint8(np.zeros(len(sel.te_col_indices)))

        # define a data workspace to work with disk database
        db_dir = os.path.join(data_folder, "disk_db")

        list_dbparams = None
        if Globals.TEST_ITERMODE == TestIterMode.ITER_MEM:
            # Use nndb, iterators should read from the memory
            list_dbparams = Params.inmem_dbparams(nndb, sel, nndb, sel2)

        elif Globals.TEST_ITERMODE == TestIterMode.ITER_DSK:
            # Use database at db_dir, write the processed data on to the disk, iterators should read from the disk
            list_dbparams = Params.indsk_dbparams(db_dir, sel, db_dir, sel2)

        elif Globals.TEST_ITERMODE == TestIterMode.ITER_MEM_DSK:
            # Use nndb, write the processed data on to the disk, iterators may read from the memory or disk.
            list_dbparams = Params.mem_to_dsk_dbparams(nndb, db_dir, sel, nndb, db_dir, sel2)

        # Initialize the Framework
        nnf = NNFramework.init(CNNPatchGen(), list_dbparams)

        cnncfg = CNNCfg()

        # To save the model & weights
        # cnncfg.save_models_dir = model_folder

        # To save the weights only
        # cnncfg.save_weights_dir = model_folder

        cnncfg.numepochs = 5
        cnncfg.validation_steps = 3  #800
        cnncfg.steps_per_epoch = 5  #600
        cnncfg.callbacks = [TensorBoardEx(log_dir='D:\\TF\\CNN2DREG')]
        cnncfg.metrics = ['accuracy', Metric.r]

        # Train with cpu (default is gpu0)
        import tensorflow as tf
        with tf.device('/cpu:0'):
            nnf.train(cnncfg)

        # nnf.test(cnncfg)
        nnf.predict(cnncfg)

    ##########################################################################
    # NNModel: Callbacks
    ##########################################################################
    @staticmethod
    def _fn_reshape_input(data, input_shape, data_format):
        # data.ndim == 3:  H x W x CH or CH x H x W
        # len(input_shape) == 3:  H x W x CH or CH x H x W

        assert (data.ndim == 3)
        assert(len(input_shape) == 3)

        if data_format == 'channels_last':
            n_ch = input_shape[2]
            x = np.tile(data, (1, 1, n_ch))
        else:
            n_ch = input_shape[0]
            x = np.tile(data, (n_ch, 1, 1))
        return x

    @staticmethod
    def _fn_predict(nnmodel, nnpatch, predictions, true_output):
        pass
