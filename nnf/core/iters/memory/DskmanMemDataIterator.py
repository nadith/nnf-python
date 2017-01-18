"""DskmanMemDataIterator to represent DskmanMemDataIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
import numpy as np

# Local Imports
from nnf.core.iters.DskmanDataIterator import DskmanDataIterator

class DskmanMemDataIterator(DskmanDataIterator):
    """description of class"""

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self, nndb,
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

        self.nndb = nndb
        
    #################################################################
    # Protected Interface
    #################################################################
    def _get_cimg_in_next(self, cls_idx, col_idx):
        """Fetch image @ cls_idx, col_idx"""
        assert(cls_idx < self.nndb.cls_n and col_idx < self.nndb.n_per_class[cls_idx])

        im_idx = self.nndb.cls_st[cls_idx] + col_idx
        cimg = self.nndb.get_data_at(im_idx)
        return cimg

    def _is_valid_cls_idx(self, cls_idx):
        """Check the validity cls_idx"""
        return cls_idx < self.nndb.cls_n        

    def _is_valid_col_idx(self, cls_idx, col_idx):
        """Check the validity col_idx of the class denoted by cls_idx"""
        assert(cls_idx < self.nndb.cls_n)
        return col_idx < self.nndb.n_per_class[cls_idx]
