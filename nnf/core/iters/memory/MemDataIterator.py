"""MemDataIterator to represent MemDataIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
import numpy as np

# Local Imports
from nnf.core.iters.DataIterator import DataIterator

class MemDataIterator(DataIterator):
    """description of class"""

    def __init__(self, nndb, batch_size=1,
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

        self.nndb = nndb

        # Expand to a 4 dimentional database if it is not.
        # flow(...) expect a 4 dimensional database (N x H x W x CH)
        db = self.nndb.db_scipy
        for i in range(self.nndb.db.ndim, 4):
            db = np.expand_dims(db, axis=i)

        self._iter = self._gen_data.flow(db, batch_size=batch_size)

    def reinit(self, batch_size=1, shuffle=True, seed=None,
                save_to_dir=None, save_prefix='', save_format='jpeg', 
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

        super().reinit(featurewise_center=featurewise_center,
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

        self._iter = self._gen_data.flow(self, self.nndb.db_scipy, y=None, 
                            batch_size=batch_size, shuffle=shuffle, seed=seed,
                            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)