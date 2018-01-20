# -*- coding: utf-8 -*-
"""
.. module:: TestCNNModel
   :platform: Unix, Windows
   :synopsis: Represent TestCNNModel and related classes.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import os
import math
import numpy as np
from nnf.keras.optimizers import rmsprop

# Local Imports
from nnf.core.iters.custom.WindowIter import *
from nnf.core.NNCfg import CNNCfg
from nnf.core.NNFramework import NNFramework
from nnf.core.models.CNN2DModel import CNN2DModel
from nnf.core.callbacks.TensorBoardEx import TensorBoardEx
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
from nnf.db.NNdb import NNdb
from nnf.db.Format import Format
from nnf.db.DbSlice import DbSlice
from nnf.db.Dataset import Dataset
from nnf.db.NNPatch import NNPatch
from nnf.db.Selection import Selection
from nnf.db.preloaded.Cifar10Db import Cifar10Db


########################################################################################################################
# NNModel: Callbacks
########################################################################################################################
def fn_predict(nnmodel, nnpatch, predictions, true_output):
    pass

def fn_test(nnmodel, nnpatch, metrics):
    print("\nAccuracy: " + str(metrics['acc']))

########################################################################################################################
# NNFramework: Parameters
########################################################################################################################
class Params(object):

    TR_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 16

    ##########################################################################
    # Public Interface
    ##########################################################################
    @staticmethod
    def _fn_reshape_input(self, data, input_shape, data_format):
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
    def inmem_dbparams(nndb, sel, nndbs_tuple=None):
        """Use nndb, iterators should read from the memory"""
        dbparam1 = {'alias': "DB1",
                    'memdb_param': {'nndb': nndb, 'nndbs_tup': nndbs_tuple},
                    'selection': sel}
        return dbparam1

    @staticmethod
    def inmem_iterparams():
        iter_params = [{'class_mode': 'categorical',
                        'batch_size': Params.TR_BATCH_SIZE,
                        'in_mem': True,
                        },
                       ((Dataset.VAL, Dataset.TE),
                        {'batch_size': Params.VAL_BATCH_SIZE})]
        return iter_params

    @staticmethod
    def indsk_dbparams(db_dir, sel):
        dbparam1 = {'alias': "DB1",
                    'dskdb_param': {'db_dir': db_dir, 'target_size': (66, 66)},
                    'selection': sel}
        return dbparam1

    @staticmethod
    def indsk_iterparams():
        iter_params = [{'class_mode': 'categorical',
                        'batch_size': Params.TR_BATCH_SIZE,
                        'in_mem': False,
                        },
                       ((Dataset.VAL, Dataset.TE),
                        {'batch_size': Params.VAL_BATCH_SIZE})]
        return iter_params

    @staticmethod
    def inmem_iterparams_wnd(window):
        """Use nndb, iterators read from the memory, build the window_iter"""
        iter_params = [{'class_mode': 'categorical',
                        'batch_size': Params.TR_BATCH_SIZE,
                        'window_iter': window,
                        'in_mem': True,
                        },
                       ((Dataset.VAL, Dataset.TE),
                        {'batch_size': Params.VAL_BATCH_SIZE})]
        return iter_params

    @staticmethod
    def indsk_iterparams_wnd(window):
        """Use nndb, iterators read from the memory, build the window_iter"""
        iter_params = [{'class_mode': 'categorical',
                        'batch_size': Params.TR_BATCH_SIZE,
                        'window_iter': window,
                        'in_mem': False,
                        },
                       ((Dataset.VAL, Dataset.TE),
                        {'batch_size': Params.VAL_BATCH_SIZE})]
        return iter_params

    @staticmethod
    def mem_to_dsk_dbparams(nndb, db_dir, sel):
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
                    'dskdb_param': {'db_dir': db_dir, 'target_size': (66, 66)},
                    'selection': sel}
        return dbparam1

########################################################################################################################
# NNFramework: Essential Overrides
########################################################################################################################
class CNNPatch(NNPatch):
    def __init__(self, height, width, offset=(None, None), is_holistic=False, window_iter=None):
        super().__init__(height, width, offset=offset, is_holistic=is_holistic)
        self.window_iter = window_iter

    def _generate_nnmodels(self, nnpatch):
        iter_params = Params.inmem_iterparams()
        iter_pp_params = {'rescale': 1. / 255}

        # iter_params = Params.indsk_iterparams()
        # iter_pp_params = {'rescale': 1. / 255}  # 'featurewise_center': True, 'featurewise_std_normalization': True

        # iter_params = Params.inmem_iterparams_wnd(self.window_iter)
        # iter_pp_params = {'rescale': 1. / 255, 'wnd_featurewise_0_1': True}  # 'wnd_max_normalize': True

        # iter_params = Params.indsk_iterparams_wnd(self.window_iter)
        # iter_pp_params = {'rescale': 1. / 255, 'wnd_featurewise_0_1': True}  # 'wnd_max_normalize': True

        return CNN2DModel(callbacks={'predict':fn_predict}, iter_params=iter_params, iter_pp_params=iter_pp_params)

class CNNPatchGen(NNPatchGenerator):
    def __init__(self, im_h=None, im_w=None, pat_h=None, pat_w=None, xstride=None, ystride=None, window_iter=None):
        super().__init__(im_h=im_h, im_w=im_w, pat_h=pat_h, pat_w=pat_w, xstride=xstride, ystride=ystride)
        self.window_iter = window_iter

    def _new_nnpatch(self, h, w, offset):
        return CNNPatch(h, w, offset, True, self.window_iter)

# ########################################### NNDB / SELECTION #########################################################
# Get the current working directory, define a `DataFolder`
data_folder = r'F:\#DL_THEANO\Workspace\DL\DLN\DataFolder'

# Training, Validation, Testing Split Definition
sel = Selection()
# sel.use_rgb = False
# sel.histeq = True
# sel.scale = (128, 128) #(150, 150)  #(100, 100) #(224, 224)
sel.tr_col_indices = np.uint16(np.arange(0, 5000))
sel.val_col_indices = np.uint16(np.arange(5000, 6000))
sel.class_range = np.uint8(np.arange(0, 10))

############################################# Split Data / Save ########################################################
# # Get the file path for Cifar10 database
# db_dir = os.path.join(data_folder, "keras", "cifar-10-batches-py")
#
# cifar10 = Cifar10Db(db_dir)
# cifar10.reinit()
#
# # Combine the training and te dataset
# X = np.concatenate((cifar10.X, cifar10.Xte), axis=0)
# lbl = np.squeeze(np.concatenate((cifar10.X_lbl, cifar10.Xte_lbl), axis=0), axis=1)
#
# # Sort the dataset in incremental class label order
# isorted = np.argsort(lbl)
#
# nndb = NNdb('original', X[isorted, ...], cls_lbl=lbl[isorted], db_format=Format.N_H_W_CH)
# buffer_sizes = {Dataset.TR: 50000, Dataset.VAL: 10000}
#
# # After the data splits save it on the disk to avoid time it takes to split repeatedly running this code
# [nndb_tr, nndb_val, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel, buffer_sizes=buffer_sizes, savepath=os.path.join(data_folder, "MATLAB_CIFAR10"))

########################################################################################################################
# Load the saved nndb splits in the above line
nndb_tr = NNdb.load(os.path.join(data_folder, "MATLAB_CIFAR10_TR.mat"))  # 0_5000
nndb_val = NNdb.load(os.path.join(data_folder, "MATLAB_CIFAR10_VAL.mat"))  # 5000_6000
nndbs_tup = (nndb_tr, nndb_val, [Dataset.TR, Dataset.VAL])

# ############################################# IMAGE AS INPUT #########################################################

# # IMPORTANT: __name__ check is a must when using multi-process concurrency
# if __name__ == "__main__":
#
#     # Use nndb, iterators read from the memory
#     list_dbparams = Params.inmem_dbparams(None, sel, nndbs_tup)
#
#     # Initialize the Model Configuration
#     cnncfg = CNNCfg()
#     cnncfg.numepochs = 50
#     cnncfg.validation_steps = math.ceil(sel.class_range.size * sel.val_col_indices.size / Params.VAL_BATCH_SIZE)
#     cnncfg.steps_per_epoch = math.ceil(sel.class_range.size * sel.tr_col_indices.size / Params.TR_BATCH_SIZE)
#     # cnncfg.callbacks=[TensorBoardEx(log_dir='D:\\TF\\CNN2D_NNF_CIFAR10', histogram_freq=True, write_images=True)]
#
#     # Initiate model's optimizer
#     # cnncfg.optimizer = SGD(lr=0.0001, momentum=0.01)
#     cnncfg.optimizer = rmsprop(lr=0.0001, decay=1e-6)
#
#     # Initialize the Framework
#     nnf = NNFramework.init(CNNPatchGen(), list_dbparams)
#
#     import tensorflow as tf
#     with tf.device('/gpu:1'):
#         nnf.train(cnncfg)
#
#     # nnf.test(cnncfg)
#     # nnf.predict(cnncfg)

########################################################################################################################
############################################### WINDOW AS INPUT ########################################################

# IMPORTANT: __name__ check is a must when using multi-process concurrency
if __name__ == "__main__":

    # Window iterator instantiation
    # wsizes = np.arange(50, 550, 25, dtype=np.uint16)
    # steps = np.uint16(np.ceil(wsizes / 2))
    # window_iter = WindowIter(wsizes, steps, WndIterMode.ITER_MEM)
    window_iter = WindowIter([20, 30, 40], [10, 15, 20], WndIterMode.ITER_MEM)

    # Use nndb, iterators read from the memory
    list_dbparams = Params.inmem_dbparams(None, sel, nndbs_tup)

    # Initialize the Model Configuration
    cnncfg = CNNCfg()
    cnncfg.numepochs = 50
    wnd_counts, duration = Window.get_total_wnd_counts(window_iter, Params.inmem_iterparams(), sel=sel, nndbs_tup=nndbs_tup)
    cnncfg.validation_steps = math.ceil(wnd_counts[Dataset.VAL.int()] / Params.VAL_BATCH_SIZE)
    cnncfg.steps_per_epoch = math.ceil(wnd_counts[Dataset.TR.int()] / Params.TR_BATCH_SIZE)
    cnncfg.callbacks=[TensorBoardEx(log_dir='D:\\TF\\CNN2D_NNF_CIFAR10', histogram_freq=True, write_images=True)]

    # Initiate model's optimizer
    # cnncfg.optimizer = SGD(lr=0.0001, momentum=0.01)
    cnncfg.optimizer = rmsprop(lr=0.0001, decay=1e-6)

    # Initialize the Framework
    nnf = NNFramework.init(CNNPatchGen(window_iter=window_iter), list_dbparams)

    import tensorflow as tf
    with tf.device('/gpu:1'):
        nnf.train(cnncfg)

    # nnf.test(cnncfg)
    # nnf.predict(cnncfg)

# 1563/1563 [==============================] - 23s 15ms/step - loss: 0.6442 - acc: 0.7839 - val_loss: 0.6482 - val_acc: 0.7792