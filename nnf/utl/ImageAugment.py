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
class ImageAugment(object):
    """ImageAugment represents util class to perform image data augmentation."""
    ##########################################################################
    # Public Interface
    ##########################################################################
    @staticmethod
    def gauss_data_gen(nndb, info={}, merge=False):
        """Augment the dataset with centered gaussian at class mean or each image in the class.

        Parameters
        ----------
        nndb : :obj:`NNdb`
            In memory database to iterate.

        info : :obj:`dict`, optional
            Provide additional information to perform guass data generation.
            (Default value = {}).

            Info Params (Defaults)
            ----------------------
            - info['samples_per_class'] = 2     # Samples per class required
            - info['center_cls_mean'] = False   # Use class mean image centered gaussian
            - info['noise_ratio'] = 0.2         # Ratio from class std (ratio * class_std)
            - info['std'] = None                # Explicit defitnition of std ('noise_ratio' is ignored)

        merge : bool
            Merge the augmented database to `nndb` object.
        """

        if (not ('samples_per_class' in info)): info['samples_per_class'] = 2
        if (not ('center_cls_mean' in info)): info['center_cls_mean'] = False        
        if (not ('noise_ratio' in info)): info['noise_ratio'] = 0.2
        if (not ('std' in info)): info['std'] = None # Force standard deviation
        # to avoid the calculation of the std over the samples in the class.
        # 'noise_ratio' will be ignored

        # Initialize necessary variables
        max_samples_per_class = info['samples_per_class']
        use_cls_mean = info['center_cls_mean']        
        noise_ratio = info['noise_ratio']
        force_std = info['std']
        fsize = nndb.h*nndb.w*nndb.ch

        # Create a new database to save the generated data
        nndb_aug = NNdb('augmented', db_format=nndb.format)

        for i in range(0, nndb.cls_n):
    
            # Fetch samples per class i
            n_per_class = nndb.n_per_class[i]
    
            st = nndb.cls_st[i]
            en = st + n_per_class
    
            # Fetch input database for class i
            I = nndb.features[:, st:en]

            # By default, gauss distributions are centered at each image
            M = I

            # Gauss distributions are centered at the mean of each class
            if (use_cls_mean): M = np.mean(I, 1, keepdims=True)
 
            # Initialize a 2D matrix to store augmented data for class
            I_aug = np.zeros((fsize, np.int32(np.ceil(max_samples_per_class/n_per_class))*n_per_class), dtype=nndb.db.dtype)
    
            # Initialize the index variable for above matrix
            j = 0;
    
            # Keep track of current sample count per class
            cur_samples_per_class = 0
            while (cur_samples_per_class < max_samples_per_class):
                r = np.random.normal(size=(fsize, n_per_class)) # r is from gaussian (m=0, std=1)        
                        
                if (force_std is None): # Caculate the std from the samples in the class i
                    I_aug[:, j*n_per_class: (j+1)*n_per_class] = np.asarray(M + noise_ratio * np.std(I, 1, keepdims=True) * r, dtype=nndb.db.dtype)
                else:
                    I_aug[:, j*n_per_class: (j+1)*n_per_class] = np.asarray(M + force_std * r, dtype=nndb.db.dtype)
        
                cur_samples_per_class = cur_samples_per_class + n_per_class
                j = j + 1
             
            I_aug = nndb_aug.features_to_data(I_aug[:, 0:max_samples_per_class], nndb.h, nndb.w, nndb.ch, nndb.db.dtype)
            nndb_aug.add_data(I_aug);
            nndb_aug.update_attr(True, max_samples_per_class)

        if (merge):
            nndb_aug = nndb.merge(nndb_aug)

        return nndb_aug

    @staticmethod
    def linear_transform(nndb, pp_params, im_per_class_multiples, merge=False):
        




        # Fetch X_train, scipy compatible (N x H x W x CH)
        X_train = nndb.db_scipy

        # Create a new database to save the generated data in batches
        nndb_aug = NNdb('augmented', db_format=nndb.format)

        # To enforce each image to have the same transformation within a round
        if ('random_transform_seed' in pp_params and isinstance(pp_params['random_transform_seed'], list)):        
            seeds = pp_params['random_transform_seed'];

            for i in range(im_per_class_multiples):
                if (i >= len(seeds)):
                    pp_params['random_transform_seed'] = seeds[-1]
                else:
                    pp_params['random_transform_seed'] = seeds[i]
        
                nndb_aug = nndb_aug.merge(ImageAugment.__ltransform(nndb, pp_params, 1))
      
        else:
            nndb_aug = nndb_aug.merge(ImageAugment.__ltransform(nndb, pp_params, im_per_class_multiples))

        if (merge):
            nndb_aug = nndb.merge(nndb_aug)

        return nndb_aug

    ##########################################################################
    # Private Interface
    ##########################################################################
    @staticmethod
    def __ltransform(nndb, pp_params, im_per_class_multiples, save_folder=''):

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

                for (X_batch, X_batch) in ImageDataPreProcessor(pp_params, None).flow_ex(nndb.db_scipy, None, None, params=params):
                    if (cls_i >= nndb.cls_n):
                        break

                    tmp_X[(cls_i-1)*n_per_class : cls_i*n_per_class] = X_batch # double to uint8 conversion
                    cls_i += 1

                if (nndb_all is None):
                    nndb_all = NNdb('batch', tmp_X, n_per_class, True, db_format=Format.N_H_W_CH)
                else:
                    nndb_all = nndb_all.merge(NNdb('batch', tmp_X, n_per_class, True, db_format=Format.N_H_W_CH))
    
        return nndb_all