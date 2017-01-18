"""DataIterator to represent DataIterator class."""
# -*- coding: utf-8 -*-
# Global Imports

# Local Imports
from nnf.core.iters.ImageDataGeneratorEx import ImageDataGeneratorEx

class DataIterator(object):
    """description of class"""

    def __init__(self, 
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
        self._gen_data = ImageDataGeneratorEx(                 
                    featurewise_center=featurewise_center,
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

        self._iter = None        

    def reinit(self, 
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
                dim_ordering='default'):
        # TODO: set the fields in self._gen_data
        pass

    def __iter__(self):
        return self

    def __next__(self):
        batch_x, batch_y = next(self._iter)
        batch_x = batch_x.reshape((len(batch_x), np.prod(batch_x.shape[1:])))
        batch_y = batch_y.reshape((len(batch_y), np.prod(batch_y.shape[1:])))
        return batch_x, batch_y