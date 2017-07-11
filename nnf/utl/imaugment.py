# -*- coding: utf-8 -*-
"""
.. module:: immap
   :platform: Unix, Windows
   :synopsis: Represent immap function.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""
# Global Imports
import numpy as np
import scipy.io
import os
from keras.preprocessing.image import ImageDataGenerator
from nnf.db.NNdb import NNdb
from nnf.db.Format import Format
from nnf.core.iters.ImageDataPreProcessor import ImageDataPreProcessor

# Local Imports

def imaugment(filepath, pp_params, im_per_class_multiples, save_path=None):


    # Load image database
    matStruct = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    imdb_obj = matStruct['imdb_obj']
    nndb = NNdb('Original', imdb_obj.db, imdb_obj.im_per_class, True)

    # Fetch X_train, scipy compatible (N x H x W x CH)
    X_train = nndb.db_scipy

    ## Initialize nndb_all to concatenate the batches
    nndb_aug = NNdb('AUGMENTED', X_train, nndb.n_per_class, cls_lbl=nndb.cls_lbl, format=Format.N_H_W_CH)

    # To enforce each image to have the same transformation within a round
    if ('random_transform_seed' in pp_params and isinstance(pp_params['random_transform_seed'], list)):        
        seeds = pp_params['random_transform_seed'];

        for i in range(im_per_class_multiples):
            if (i >= len(seeds)):
                pp_params['random_transform_seed'] = seeds[-1]
            else:
                pp_params['random_transform_seed'] = seeds[i]
        
            nndb_aug = nndb_aug.merge(__imaugment(nndb, pp_params, 1))
      
    else:
        nndb_aug = nndb_aug.merge(__imaugment(nndb, pp_params, im_per_class_multiples))


    # Matlab compatible format
    mat_db = np.transpose(nndb_aug.db, (1, 2, 3, 0))
    nndb_aug = NNdb('FORMAT_MATLAB', mat_db, nndb_aug.n_per_class, cls_lbl=nndb_aug.cls_lbl)

    if (save_path is not None):
        matStruct['imdb_obj'].db = mat_db
        matStruct['imdb_obj'].im_per_class = np.unique(nndb_aug.n_per_class)[0]
        scipy.io.savemat(save_path, matStruct)

    return nndb_aug

def __imaugment(nndb, pp_params, im_per_class_multiples, save_folder=''):

    # Image per class (scalar)
    n_per_class = np.unique(nndb.n_per_class)[0]

    ## Initialize nndb_all to concatenate the batches
    nndb_all = None

    if (im_per_class_multiples != 0):
        params = {}
        params['batch_size'] = n_per_class
        params['shuffle'] = False
        
        for i in range(im_per_class_multiples):
            tmp_X = np.zeros((nndb.n, nndb.h, nndb.w, nndb.ch), dtype='uint8')
            cls_i = 1

            for (X_batch, X_batch) in ImageDataPreProcessor(pp_params, None).flow(nndb.db_scipy, None, None, params=params):
                if (cls_i >= nndb.cls_n):
                    break

                tmp_X[(cls_i-1)*n_per_class : cls_i*n_per_class] = X_batch # double to uint8 conversion
                cls_i += 1

            if (nndb_all is None):
                nndb_all = NNdb('BATCH', tmp_X, n_per_class, True, format=Format.N_H_W_CH)
            else:
                nndb_all = nndb_all.merge(NNdb('BATCH', tmp_X, n_per_class, True, format=Format.N_H_W_CH))
    
    return nndb_all