"""DskmanDskDataIterator to represent DskmanDskDataIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os

# Local Imports
from nnf.core.iters.DskmanDataIterator import DskmanDataIterator
import nnf.core.NNDiskMan

class DskmanDskDataIterator(DskmanDataIterator):
    """description of class"""

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self, directory,
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0.,
                width_shift_range=0.,
                height_shift_range=0.,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=False,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None,
                dim_ordering='default'
        ):
        super().__init__(featurewise_center=featurewise_center,
                samplewise_center=samplewise_center,
                featurewise_std_normalization=featurewise_std_normalization,
                samplewise_std_normalization=samplewise_std_normalization,
                zca_whitening=zca_whitening,
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                shear_range=shear_range,
                zoom_range=zoom_range,
                channel_shift_range=channel_shift_range,
                fill_mode=fill_mode,
                cval=cval,
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,
                rescale=rescale,
                preprocessing_function=preprocessing_function,
                dim_ordering=dim_ordering)

        # TODO: expose the pre-processing capability via parent property self._gen_data
        
        # Class count, will be updated below
        self.cls_n = 0 
    
        # Keyed by the cls_idx
        # value = [file_path_1, file_path_2, ...] <= list of file paths
        self.paths = {} # A dictionary that hold lists. self.paths[cls_idx] => list of file paths
        
        # Keyed by the cls_idx
        # value = <int> denoting the images per class
        self.n_per_class = {}
        
        # Future use
        self.cls_idx_to_dir = {}       

        # Inner function: Fetch the files in the disk
        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

        # Assign explicit class index for internal reference
        cls_idx = 0

        # Iterate the directory and populate self.paths dictionary
        for root, dirs, files in _recursive_list(directory):

            # Exclude this directory itself
            if (root == directory):
                continue
            
            # Extract the directory
            dir = root[(root.rindex ('\\')+1):]
        
            # Exclude the internally used data folder
            if (dir == nnf.core.NNDiskMan.NNDiskMan._SAVE_TO_DIR):
                continue

            # Since dir is considered to be a class name, give the explicit internal index
            self.cls_idx_to_dir.setdefault(cls_idx, dir) # Future use

            # Update class count
            self.cls_n += 1

            # Initialize [paths|n_per_class] dictionaries with related cls_idx
            fpaths = self.paths.setdefault(cls_idx, [])            
            n_per_class = self.n_per_class.setdefault(cls_idx, 0)

            # Update paths
            for fname in files:
                fpath = os.path.join(root, fname)
                fpaths.append(fpath)
                n_per_class += 1
 
            # Update n_per_class dictionary
            self.n_per_class[cls_idx] = n_per_class
            cls_idx += 1

    #################################################################
    # Protected Interface
    #################################################################
    def _get_cimg_in_next(self, cls_idx, col_idx):
        """Fetch image @ cls_idx, col_idx"""
        assert(cls_idx < self.cls_n and col_idx < self.n_per_class[cls_idx])

        impath = self.paths[cls_idx][col_idx]            
        img = load_img(impath, grayscale=False, target_size=None)
        cimg = img_to_array(img, dim_ordering='default')
        return cimg

    def _is_valid_cls_idx(self, cls_idx):
        """Check the validity cls_idx"""
        return cls_idx < self.cls_n        

    def _is_valid_col_idx(self, cls_idx, col_idx):
        """Check the validity col_idx of the class denoted by cls_idx"""
        assert(cls_idx < self.cls_n)
        return col_idx < self.n_per_class[cls_idx]
       
# Sample code
#import os

#def _recursive_list(subpath):
#    return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

#for root, dirs, files in _recursive_list('D:\TestImageFolder'):
#    print(root) 

#for root, dirs, files in _recursive_list('D:\TestImageFolder'):
#    print(dirs) 

#for root, dirs, files in _recursive_list('D:\TestImageFolder'):
#    print(files) 