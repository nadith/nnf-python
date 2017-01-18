"""NumpyArrayIteratorEx to represent NumpyArrayIteratorEx class."""
# -*- coding: utf-8 -*-
# Global Imports
from keras.preprocessing.image import NumpyArrayIterator

# Local Imports

class NumpyArrayIteratorEx(NumpyArrayIterator):
    def __init__(self, X, y, image_data_generator,
                batch_size=32, shuffle=False, seed=None,
                dim_ordering='default',
                save_to_dir=None, save_prefix='', save_format='jpeg'):
        super().__init__(X, y=y, image_data_generator=image_data_generator,
                batch_size=batch_size, shuffle=shuffle, seed=seed,
                dim_ordering=dim_ordering,
                save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def next(self):
        if self.y == None:
            x = super().next()
            return (x, x)
        return super().next()