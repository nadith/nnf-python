__author__ = "Nadith Pathirage"
__copyright__ = "Copyright 2017, DeepPy Framework"
__credits__ = ["Slinda Liu, Jackie Wang"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = ["Nadith Pathirage", "Jackie Wang"]
__email__ = "chathurdara@gmail.com"
__status__ = "Development"  # "Prototype", "Development", or "Production".

import os
import scipy.io
import numpy as np
from nnf.test.dl.Globals import *

# Get the current working directory, define a `DataFolder`
cwd = os.getcwd()
data_folder = os.path.join(cwd, "DataFolder")
model_folder = os.path.join(cwd, "ModelFolder")

##############################################################################
# Run this segment of code ONLY once to download all the data files and
# model files from the FTP server.
##############################################################################
# from nnf.utl.FTPDownloader import FTPDownloader
# ftpd = FTPDownloader(host='203.170.82.33', user='buildwac', passwd='<request_author>')
# ftpd.download(data_folder, '/public_html/data/nnf/DataFolder')
# ftpd.download(model_folder, '/public_html/data/nnf/ModelFolder')
##############################################################################

# Load image database `AR`
filepath = os.path.join(data_folder, 'IMDB_66_66_AR_8.mat')
matStruct = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
imdb_obj = matStruct['imdb_obj']

# Visualizing the database ###################################################
##############################################################################
# # 1: Visualizing 5 identities, each having 8 images using imap.
# from nnf.utl.immap import *
# db = np.rollaxis(imdb_obj.db, 3)
# immap(db, rows=5, cols=8, scale=0.5)

# # 2: Visualizing 5 identities, each having 8 images using NNdb.
# from nnf.db.NNdb import NNdb
# nndb = NNdb('Original', imdb_obj.db, 8, True)
# nndb.show(5, 8)
# nndb.show_ws(5, 8) # with whitespaces
# nndb.save('IMDB_AR.mat')

# # 2.1: NNdb with different db_format
# from nnf.db.Format import Format
# nndb = NNdb('Original', db, 8, True, db_format=Format.N_H_W_CH)
# nndb.show(5, 8)

# Augment database with random transformation ################################
##############################################################################
# # 1: Augment with linear transformations.
# # Initialize transformation related params for random transformations
# from nnf.db.NNdb import NNdb
# from nnf.utl.ImageAugment import ImageAugment
# pp_params = {}
# pp_params['rotation_range'] = 2
# pp_params['width_shift_range'] = 0.1
# pp_params['force_horizontal_flip'] = True
# nndb = NNdb('Original', imdb_obj.db, 8, True)
# nndb_aug = ImageAugment.linear_transform(nndb, pp_params, 1, False)
# nndb_aug.show(10, np.int(np.unique(nndb_aug.n_per_class)))
# nndb_aug.save('IMDB_66_66_AR_8_LTRFM_WSHIFT.mat')

# # 1.1: To enforce each image to have the same transformation in both rounds
# from nnf.db.NNdb import NNdb
# from nnf.utl.ImageAugment import ImageAugment
# pp_params = {}
# pp_params['rotation_range'] = 50
# pp_params['random_transform_seed'] = 10
# nndb = NNdb('Original', imdb_obj.db, 8, True)
# nndb_aug = ImageAugment.linear_transform(nndb, pp_params, 2, False)
# nndb_aug.show(10, np.int(np.unique(nndb_aug.n_per_class)))

# # 1.2: To enforce each image to have the same transformation within a round
# from nnf.db.NNdb import NNdb
# from nnf.utl.ImageAugment import ImageAugment
# pp_params = {}
# pp_params['rotation_range'] = 50
# pp_params['random_transform_seed'] = [10, 20] # two rounds, two seeds
# nndb = NNdb('Original', imdb_obj.db, 8, True)
# nndb_aug = ImageAugment.linear_transform(nndb, pp_params, 2, False)
# nndb_aug.show(10, np.int(np.unique(nndb_aug.n_per_class)))

# # 2: Augment with gaussian data generation.
# from nnf.db.NNdb import NNdb
# from nnf.utl.ImageAugment import ImageAugment
# info = {}
# info['noise_ratio'] = 0.05
# info['samples_per_class'] = 8
# nndb = NNdb('Original', imdb_obj.db, 8, True)
# nndb_aug = ImageAugment.gauss_data_gen(nndb, info, False)
# nndb_aug.show(10, np.int(np.unique(nndb_aug.n_per_class)))
# nndb_aug.save('IMDB_66_66_AR_8_GTRFM_0.05.mat')

# Database Slicing ############################################################
###############################################################################
# from nnf.db.NNdb import NNdb
# from nnf.db.DbSlice import DbSlice
# from nnf.db.Selection import Selection
# from nnf.db.Selection import Select
#
# # 1:
# nndb = NNdb('original', imdb_obj.db, 8, True)
# sel = Selection()
# sel.tr_col_indices      = np.array([0, 2, 3, 5], dtype='uint8')
# sel.val_col_indices     = np.array([0, 1, 4], dtype='uint8')
# sel.te_col_indices      = np.array([6, 7], dtype='uint8')
# sel.tr_out_col_indices  = np.zeros(4, dtype='uint8')  # [0, 0, 0, 0]
# sel.val_out_col_indices = np.zeros(3, dtype='uint8')  # [0, 0, 0]
# sel.te_out_col_indices  = np.zeros(2, dtype='uint8')  # [0, 0]
# sel.class_range         = np.uint8(np.arange(0, 60))
# sel.val_class_range     = np.uint8(np.arange(60, 80))
# sel.te_class_range      = np.uint8(np.arange(80, 100))
# [nndb_tr, nndb_val, nndb_te, nndb_tr_out, nndb_val_out, nndb_te_out, _] =\
#                        DbSlice.slice(nndb, sel)
#
# # Visualize first 10 identities of all nndb splits
# nndb_tr.show(10, 4)
# nndb_val.show(10, 3)
# nndb_te.show(10, 2)
# nndb_tr_out.show(10, 4)
# nndb_val_out.show(10, 3)
# nndb_te_out.show(10, 2)

# # 1.1: Using special enumeration values
# nndb = NNdb('original', imdb_obj.db, 8, True)
# sel = Selection()
# sel.tr_col_indices      = Select.PERCENT_60
# sel.te_col_indices      = Select.PERCENT_40
# sel.class_range         = np.uint8(np.arange(0, 60))
# [nndb_tr, _, nndb_te, _, _, _, _] =\
#                        DbSlice.slice(nndb, sel)
# nndb_tr.show(10, 5)
# nndb_te.show(10, 4)

# **** VISIT DbSlice.examples() method for more examples
# 

# Pre-processing: Histogram Grayscale/Eq/Match ###############################
##############################################################################
# from nnf.db.NNdb import NNdb
# from nnf.db.Selection import Selection
# from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
# from nnf.db.DbSlice import DbSlice

# nndb = NNdb('Original', imdb_obj.db, 8, True)
# sel = Selection()
# sel.tr_col_indices = np.array([0, 1, 3])
# sel.use_rgb = False
# sel.histeq = True
# #sel.histmatch_col_index = 0
# sel.class_range    = np.uint8(np.arange(0, 60))
# [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
# nndb_tr.show(10, 3)

# Adding noise ###############################################################
##############################################################################
# from nnf.db.NNdb import NNdb
# from nnf.db.Selection import Selection
# from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
# from nnf.db.DbSlice import DbSlice

# nndb = NNdb('Original', imdb_obj.db, 8, True)
# sel = Selection()
# sel.tr_col_indices = np.array([0, 1, 3])
# sel.tr_noise_rate  = np.array([0, 0.5, 0])   # percentage of corruption # noqa E501
# #sel.tr_noise_rate = [0 0.5 0 0.5 0 Noise.G]   # last index with Gauss noise # noqa E501
# sel.class_range    = np.uint8(np.arange(0, 60))
# [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
# nndb_tr.show(10, 3)

# Patch division #############################################################
##############################################################################
# from nnf.db.NNdb import NNdb
# from nnf.db.Selection import Selection
# from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
# from nnf.db.DbSlice import DbSlice

# nndb = NNdb('Original', imdb_obj.db, 8, True)
# sel = Selection()
# sel.tr_col_indices = np.array([0, 1, 3])
# sel.class_range    = np.uint8(np.arange(0, 60))

# # Instantiate a patch generator
# pgen = NNPatchGenerator(nndb.h, nndb.w, 33, 33, 33, 33)
# sel.nnpatches = pgen.generate_nnpatches()

# # List of NNdb objects for each patch
# [nndbs_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)

# # Visualize the 4 patches of the first image
# nndbs_tr[0].show()
# nndbs_tr[1].show()
# nndbs_tr[2].show()
# nndbs_tr[3].show()

##############################################################################
# ML Algorithms ##############################################################
##############################################################################
# from nnf.alg.LDA import LDA
# from nnf.alg.Util import Util
# from nnf.db.NNdb import NNdb
# from nnf.db.DbSlice import DbSlice
# from nnf.db.Selection import Selection
# from nnf.db.Selection import Select
#
# # 1: LDA
# nndb = NNdb('original', imdb_obj.db, 8, True)
# sel = Selection()
# sel.scale = 0.5
# sel.use_rgb = False
# sel.tr_col_indices      = np.array([0, 2, 3, 5], dtype='uint8')
# sel.te_col_indices      = np.array([6, 7], dtype='uint8')
# sel.class_range         = np.uint8(np.arange(0, 60))
# [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
#
# W, info = LDA.l2(nndb_tr)
# accuracy = Util.test(W, nndb_tr, nndb_te, info)
# print("LDA: Accuracy:" + str(accuracy))

##############################################################################
# Deep Learning Framework ####################################################
##############################################################################
# # 1: AE Model
# Globals.TEST_ITERMODE = TestIterMode.ITER_MEM
# from nnf.test.dl.TestAEModel import TestAEModel
# TestAEModel().Test()
#
# # 1.1: AE Model (pre-loaded database)
# Globals.TEST_ITERMODE = TestIterMode.ITER_NO
# from nnf.test.dl.TestAEModel import TestAEModel
# TestAEModel().Test_preloaded_db()
#
# # 2: DAE Model ###############################################################
# Globals.TEST_ITERMODE = TestIterMode.ITER_MEM
# from nnf.test.dl.TestDAEModel import TestDAEModel
# TestDAEModel().Test()
#
# # 2.1: DAE Model (without pre-training)
# Globals.TEST_ITERMODE = TestIterMode.ITER_DSK
# from nnf.test.dl.TestDAEModel import TestDAEModel
# TestDAEModel().Test(pretrain=False)
#
# # 2.2: DAE Model (pre-loaded database)
# Globals.TEST_ITERMODE = TestIterMode.ITER_NO
# from nnf.test.dl.TestDAEModel import TestDAEModel
# TestDAEModel().Test_preloaded_db()
#
# # 3: DAEReg Model ############################################################
# Globals.TEST_ITERMODE = TestIterMode.ITER_MEM
# from nnf.test.dl.TestDAERegModel import TestDAERegModel
# TestDAERegModel().Test()
#
# # 2.1: DAE Model (without pre-training)
# Globals.TEST_ITERMODE = TestIterMode.ITER_DSK
# from nnf.test.dl.TestDAERegModel import TestDAERegModel
# TestDAERegModel().Test(pretrain=False)
#
# # 3.2: DAEReg Model (pre-loaded database)
# Globals.TEST_ITERMODE = TestIterMode.ITER_NO
# from nnf.test.dl.TestDAERegModel import TestDAERegModel
# TestDAERegModel().Test_preloaded_db()
#
# # 4: CNN Model ###############################################################
# Globals.TEST_ITERMODE = TestIterMode.ITER_DSK
# from nnf.test.dl.TestCNNModel import TestCNNModel
# TestCNNModel().Test()
#
# # 4.1: CNN Model (pre-loaded database)
# Globals.TEST_ITERMODE = TestIterMode.ITER_NO
# from nnf.test.dl.TestCNNModel import TestCNNModel
# TestCNNModel().Test_preloaded_db()
#
# # 5: CNN2DReg Model ##########################################################
# Globals.TEST_ITERMODE = TestIterMode.ITER_DSK
# from nnf.test.dl.TestCNN2DRegModel import TestCNN2DRegModel
# TestCNN2DRegModel().Test()
#
# # 6: VGG16 Model #############################################################
# Globals.TEST_ITERMODE = TestIterMode.ITER_MEM
# from nnf.test.dl.TestVGG16Model import TestVGG16Model
# TestVGG16Model().Test()
#
# # 6.1: VGG16 Model (pre-loaded database)
# Globals.TEST_ITERMODE = TestIterMode.ITER_NO
# from nnf.test.dl.TestVGG16Model import TestVGG16Model
# TestVGG16Model().Test_preloaded_db()
#
# # 7: CNN2DParallel Model #####################################################
# Globals.TEST_ITERMODE = TestIterMode.ITER_DSK
# from nnf.test.dl.TestCNN2DParallelModel import TestCNN2DParallelModel
# TestCNN2DParallelModel().Test()