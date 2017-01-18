"""DskDataIterator to represent DskDataIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
import numpy as np

# Local Imports
from nnf.core.iters.DataIterator import DataIterator

class DskDataIterator(DataIterator):

    def __init__(self, file_infos, nb_class, batch_size=1,
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

        self.file_infos = file_infos
        self.nb_class = nb_class
        self._iter = self._gen_data.flow_from_directory(file_infos, nb_class, batch_size=batch_size)

    def reinit(self, target_size=(256, 256), color_mode='rgb',
                classes=None, class_mode='categorical',
                batch_size=1, shuffle=True, seed=None,
                save_to_dir=None, save_prefix='', save_format='jpeg',
                follow_links=False, 
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


        self._iter = self._gen_data.flow_from_directory(self.file_infos, self.nb_class,
                        target_size=target_size, color_mode=color_mode,
                        classes=classes, class_mode=class_mode,
                        batch_size=batch_size, shuffle=shuffle, seed=seed,
                        save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
                        follow_links=follow_links)

