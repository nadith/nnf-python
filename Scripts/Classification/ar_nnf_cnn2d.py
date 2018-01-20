# -*- coding: utf-8 -*-
"""
.. module:: ar_nnf_cnn2d
   :platform: Unix, Windows
   :synopsis: CNN2D model with AR database.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import os
import math
import scipy.io
import numpy as np
from nnf.keras.optimizers import rmsprop

# Local Imports
from nnf.db.NNdb import NNdb
from nnf.db.Dataset import Dataset
from nnf.db.NNPatch import NNPatch
from nnf.db.Selection import Selection
from nnf.core.NNCfg import CNNCfg
from nnf.core.NNFramework import NNFramework
from nnf.core.models.CNN2DModel import CNN2DModel
from nnf.core.callbacks.TensorBoardEx import TensorBoardEx
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
from nnf.core.iters.custom.WindowIter import WindowIter, WndIterMode, Window


########################################################################################################################
# NNModel: Callbacks
########################################################################################################################
def fn_predict(nnmodel, nnpatch, predictions, true_output):
    pass

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
                        'shuffle': False,
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

# Load image database `AR`
matStruct = scipy.io.loadmat(os.path.join(data_folder, 'IMDB_66_66_AR_8.mat'),
                             struct_as_record=False, squeeze_me=True)
imdb_obj = matStruct['imdb_obj']
nndb = NNdb('original', imdb_obj.db, 8, True)

# Training, Validation, Testing Split Definition
sel = Selection()
# sel.use_rgb = False
# sel.histeq = True
# sel.scale = [33, 33] #(128, 128) #(150, 150)  #(100, 100) #(224, 224)
sel.tr_col_indices = np.uint16(np.arange(0, 8))
sel.val_col_indices = np.uint16(np.arange(7, 8))
# sel.te_col_indices = np.uint16(np.array([7]))
sel.class_range = np.uint8(np.arange(0, 100))

# # ############################################# IMAGE AS INPUT #########################################################
db_dir = os.path.join(data_folder, "disk_db")

# IMPORTANT: __name__ check is a must when using multi-process concurrency
if __name__ == "__main__":
    # Use nndb, iterators read from the memory
    list_dbparams = Params.inmem_dbparams(nndb, sel)

    # Use database at db_dir, write the processed data on to the disk, iterators should read from the disk
    # list_dbparams = Params.indsk_dbparams(db_dir, sel)

    # Use nndb, write the processed data on to the disk, iterators may read from the memory or disk.
    # list_dbparams = Params.mem_to_dsk_dbparams(nndb, db_dir, sel)

    # Initialize the Model Configuration
    cnncfg = CNNCfg()
    cnncfg.numepochs = 1
    # cnncfg.validation_steps = math.ceil(sel.class_range.size * sel.val_col_indices.size / Params.VAL_BATCH_SIZE)
    cnncfg.steps_per_epoch = 100  # math.ceil(sel.class_range.size * sel.tr_col_indices.size / Params.TR_BATCH_SIZE)
    cnncfg.callbacks = [TensorBoardEx(log_dir='D:\\TF\\CNN2D_NNF_AR', histogram_freq=True, write_images=True)]

    # Initiate model's optimizer
    # cnncfg.optimizer = SGD(lr=0.0001, momentum=0.01)
    cnncfg.optimizer = rmsprop(lr=0.0001, decay=1e-6)

    # Initialize the Framework
    nnf = NNFramework.init(CNNPatchGen(), list_dbparams)

    import tensorflow as tf
    with tf.device('/gpu:1'):
        nnf.train(cnncfg)

    # nnf.test(cnncfg)
    # nnf.predict(cnncfg)

########################################################################################################################
############################################### WINDOW AS INPUT ########################################################
# db_dir = os.path.join(data_folder, "disk_db")
#
# # IMPORTANT: __name__ check is a must when using multi-process concurrency
# if __name__ == "__main__":
#
#     # Window iterator instantiation
#     window_iter = WindowIter([2, 4, 5, 6, 7, 8], [1, 1, 1, 1, 1, 1], WndIterMode.ITER_MEM)
#
#     # Use nndb, iterators read from the memory
#     list_dbparams = Params.inmem_dbparams(nndb, sel)
#
#     # Use database at db_dir, write the processed data on to the disk, iterators should read from the disk
#     # list_dbparams = Params.indsk_dbparams(db_dir, sel)
#
#     # Use nndb, write the processed data on to the disk, iterators may read from the memory or disk.
#     # list_dbparams = Params.mem_to_dsk_dbparams(nndb, db_dir, sel)
#
#     # Initialize the Model Configuration
#     cnncfg = CNNCfg()
#     cnncfg.numepochs = 50
#     wnd_counts, duration = Window.get_total_wnd_counts(window_iter, Params.inmem_iterparams(), nndb, sel)
#     cnncfg.validation_steps = math.ceil(wnd_counts[Dataset.VAL.int()] / Params.VAL_BATCH_SIZE)
#     cnncfg.steps_per_epoch = math.ceil(wnd_counts[Dataset.TR.int()] / Params.TR_BATCH_SIZE)
#     cnncfg.callbacks=[TensorBoardEx(log_dir='D:\\TF\\CNN2D_NNF_AR', histogram_freq=True, write_images=True)]
#
#     # Initiate model's optimizer
#     # cnncfg.optimizer = SGD(lr=0.0001, momentum=0.01)
#     cnncfg.optimizer = rmsprop(lr=0.0001, decay=1e-6)
#
#     # Initialize the Framework
#     nnf = NNFramework.init(CNNPatchGen(window_iter=window_iter), list_dbparams)
#
#     import tensorflow as tf
#     with tf.device('/gpu:1'):
#         nnf.train(cnncfg)
#
#     # nnf.test(cnncfg)
#     # nnf.predict(cnncfg)

########################################################################################################################