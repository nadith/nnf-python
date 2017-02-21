
__author__ = "Nadith Pathirage"
__copyright__ = "Copyright 2017, DeepPy Framework"
__credits__ = ["Slinda Liu, Jackie Wang"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = ["Nadith Pathirage", "Jackie Wang"]
__email__ = "chathurdara@gmail.com"
__status__ = "Development" # "Prototype", "Development", or "Production".

import numpy as np
import scipy.io
import os

# Get the current working directory, define a `DataFolder`
cwd = os.getcwd()
data_folder = os.path.join(cwd, "DataFolder")

# Load image database `AR`
matStruct = scipy.io.loadmat(os.path.join(data_folder, 'IMDB_66_66_AR_8.mat'),
                            struct_as_record=False, squeeze_me=True)
imdb_obj = matStruct['imdb_obj']


## Visualizing the database ##################################################
##############################################################################
#1: Visualizing 5 identities, each having 8 images using imap.
#from nnf.utl.immap import *
#db = np.rollaxis(imdb_obj.db, 3)
#immap(db, rows=5, cols=8)

#2: Visualizing 5 identities, each having 8 images using NNdb.
#from nnf.db.NNdb import NNdb
#nndb = NNdb('Original', imdb_obj.db, 8, True)
#nndb.show(5, 8)

#2.1:
#from nnf.db.Format import Format
#nndb = NNdb('Original', db, 8, True, format=Format.N_H_W_CH)
#nndb.show(5, 8)


# Database Slicing ##########################################################
#############################################################################
#from nnf.db.NNdb import NNdb
#from nnf.db.DbSlice import DbSlice
#from nnf.db.Selection import Selection
#from nnf.db.Selection import Select

#1:
#nndb = NNdb('original', imdb_obj.db, 8, True)
#sel = Selection()
#sel.tr_col_indices      = np.array([0, 2, 3, 5], dtype='uint8')
#sel.val_col_indices     = np.array([0, 1, 4], dtype='uint8')
#sel.te_col_indices      = np.array([6, 7], dtype='uint8')
#sel.tr_out_col_indices  = np.zeros(4, dtype='uint8')  # [0, 0, 0, 0]
#sel.val_out_col_indices = np.zeros(3, dtype='uint8')  # [0, 0, 0]
#sel.te_out_col_indices  = np.zeros(2, dtype='uint8')  # [0, 0]
#sel.class_range         = np.uint8(np.arange(0, 60))
#sel.val_class_range     = np.uint8(np.arange(60, 80))
#sel.te_class_range      = np.uint8(np.arange(80, 100))
#[nndb_tr, nndb_val, nndb_te, nndb_tr_out, nndb_val_out, nndb_te_out, _] =\
#                        DbSlice.slice(nndb, sel)

## Visualize first 10 identities of all nndb splits
#nndb_tr.show(10, 4)
#nndb_val.show(10, 3)
#nndb_te.show(10, 2)
#nndb_tr_out.show(10, 4)
#nndb_val_out.show(10, 3)
#nndb_te_out.show(10, 2)

#
#1.1: Using special enumeration values
#
#nndb = NNdb('original', imdb_obj.db, 8, True)
#sel = Selection()
#sel.tr_col_indices      = Select.PERCENT_60
#sel.te_col_indices      = Select.PERCENT_40
#sel.class_range         = np.uint8(np.arange(0, 60))
#[nndb_tr, _, nndb_te, _, _, _, _] =\
#                        DbSlice.slice(nndb, sel)
#nndb_tr.show(10, 5)
#nndb_te.show(10, 4)

# **** VISIT DbSlice.examples() method for more examples
# 

## Pre-processing: Histogram Grayscale/Eq/Match ##############################
##############################################################################
#from nnf.db.NNdb import NNdb
#from nnf.db.Selection import Selection
#from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
#from nnf.db.DbSlice import DbSlice

#nndb = NNdb('Original', imdb_obj.db, 8, True)
#sel = Selection()
#sel.tr_col_indices = np.array([0, 1, 3])
#sel.use_rgb = False
#sel.histeq = True
##sel.histmatch_col_index = 0
#sel.class_range    = np.uint8(np.arange(0, 60))
#[nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
#nndb_tr.show(10, 3)


## Adding noise ##############################################################
##############################################################################
#from nnf.db.NNdb import NNdb
#from nnf.db.Selection import Selection
#from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
#from nnf.db.DbSlice import DbSlice

#nndb = NNdb('Original', imdb_obj.db, 8, True)
#sel = Selection()
#sel.tr_col_indices = np.array([0, 1, 3])
#sel.tr_noise_rate  = np.array([0, 0.5, 0])   # percentage of corruption # noqa E501
##sel.tr_noise_rate = [0 0.5 0 0.5 0 Noise.G]   # last index with Gauss noise # noqa E501
#sel.class_range    = np.uint8(np.arange(0, 60))
#[nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
#nndb_tr.show(10, 3)


## Patch division ############################################################
##############################################################################
#from nnf.db.NNdb import NNdb
#from nnf.db.Selection import Selection
#from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
#from nnf.db.DbSlice import DbSlice

#nndb = NNdb('Original', imdb_obj.db, 8, True)
#sel = Selection()
#sel.tr_col_indices = np.array([0, 1, 3])
#sel.class_range    = np.uint8(np.arange(0, 60))

## Instantiate a patch generator
#pgen = NNPatchGenerator(nndb.h, nndb.w, 33, 33, 33, 33)
#sel.nnpatches = pgen.generate_nnpatches()

## List of NNdb objects for each patch
#[nndbs_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)

## Visualize the 4 patches of the first image
#nndbs_tr[0].show()
#nndbs_tr[1].show()
#nndbs_tr[2].show()
#nndbs_tr[3].show()

##############################################################################
## Deep Learning Framework ###################################################
##############################################################################
##1: AE Model
#from nnf.test.dl.TestAEModel import TestAEModel
#TestAEModel().Test()

##1.1: AE Model (pre-loaded database)
#from nnf.test.dl.TestAEModel import TestAEModel
#TestAEModel().Test_preloaded_db()


##2: DAE Model ################################################################
#from nnf.test.dl.TestDAEModel import TestDAEModel
#TestDAEModel().Test()

##2.1: DAE Model (without pre-training)
#from nnf.test.dl.TestDAEModel import TestDAEModel
#TestDAEModel().Test(pretrain=False)

##2.2: DAE Model (pre-loaded database)
#from nnf.test.dl.TestDAEModel import TestDAEModel
#TestDAEModel().Test_preloaded_db()


##3: DAEReg Model #############################################################
#from nnf.test.dl.TestDAERegModel import TestDAERegModel
#TestDAERegModel().Test()

##3.1: DAEReg Model (pre-loaded database)
#from nnf.test.dl.TestDAERegModel import TestDAERegModel
#TestDAERegModel().Test_preloaded_db()


##4: CNN Model ################################################################
#from nnf.test.dl.TestCNNModel import TestCNNModel
#TestCNNModel().Test()

##4.1: CNN Model (pre-loaded database) 
#from nnf.test.dl.TestCNNModel import TestCNNModel
#TestCNNModel().Test_preloaded_db()


#5: TODO: VGG16 Model ########################################################
#from nnf.test.dl.TestVGG16Model import TestVGG16Model
#TestVGG16Model().Test()
