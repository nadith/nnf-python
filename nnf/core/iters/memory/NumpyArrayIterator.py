"""NumpyArrayIterator to represent NumpyArrayIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
from warnings import warn as warning
from keras import backend as K
import numpy as np
import os
from keras.preprocessing.image import array_to_img

# Local Imports
from nnf.core.iters.Iterator import Iterator

class NumpyArrayIterator(Iterator):
    """NumpyArrayIterator iterates the image data in the memory for :obj:`NNModel'.

    Attributes
    ----------
    input_vectorized : bool
        Whether the data needs to be returned via the iterator as a batch of data vectors.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, X, y, nb_class, image_data_pp, params=None):
        """Construct a :obj:`NumpyArrayIterator` instance.

        Parameters
        ----------
        X : `array_like`
            Data in 2D matrix. Format: Samples x Features.

        y : `array_like`
            Vector indicating the class labels.

        image_data_pp : :obj:`ImageDataPreProcessor`
            Image data pre-processor.

        params : :obj:`dict`
            Core iterator parameters. 
        """
        # Set defaults for params
        _image_shape = tuple(X.shape[1:])

        if (params is None):
            self.input_vectorized = False            
            data_format = None
            class_mode = None
            batch_size = 32; shuffle = True; seed = None
            save_to_dir = None; save_prefix = ''; save_format = 'jpeg'

        else:
            self.input_vectorized = params['input_vectorized'] if ('input_vectorized' in params) else False
            _image_shape = params.setdefault('_image_shape', _image_shape)  # internal use
            data_format = params['data_format'] if ('data_format' in params) else None
            class_mode = params['class_mode'] if ('class_mode' in params) else None
            batch_size = params['batch_size'] if ('batch_size' in params) else 32
            shuffle = params['shuffle'] if ('shuffle' in params) else True
            seed = params['seed'] if ('seed' in params) else None
            save_to_dir = params['save_to_dir'] if ('save_to_dir' in params) else None
            save_prefix = params['save_prefix'] if ('save_prefix' in params) else ''
            save_format = params['save_format'] if ('save_format' in params) else 'jpeg'

        if y is not None and len(X) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        if data_format is None:
            data_format = K.image_data_format()

        self.X = np.asarray(X)

        if self.X.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.X.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.X.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'data format convention "' + data_format + '" '
                             '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                             'However, it was passed an array with shape ' + str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) + ' channels).')

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None

        self.image_data_generator = image_data_pp
        self.data_format = data_format
        self.image_shape = _image_shape

        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.nb_sample = X.shape[0]   
        self.nb_class = nb_class
        self.sync_gen = None  # Synced generator

        super(NumpyArrayIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def sync(self, gen):
        # Both generators must have same number of images
        assert(len(self.X) == len(gen.X))
        if (self.batch_size != gen.batch_size):
            warning("`batch_size` of synced generators are different.")

        self.sync_gen = gen

    def next(self):
        """Advance to next batch of data.

        Returns
        -------
        `array_like` :
            `batch_x` data matrix. Format: Samples x Features

        `array_like` :
            `batch_y` class label vector or 'batch_x' matrix. refer code.
        """
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype='float32')
        
        sync_gen = self.sync_gen
        batch_xt = None        
        if (sync_gen is not None):
            batch_xt = np.zeros((current_batch_size,) + self.sync_gen.image_shape, dtype='float32')

        for i, j in enumerate(index_array):
            x = self._get_data(self.X, j)
            batch_x[i] = x

            if (sync_gen is not None):
                batch_xt[i] = sync_gen._get_data(sync_gen.X, j)

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        
        # Perform the reshape operation on x if necessary
        if (self.input_vectorized):
            batch_x = batch_x.reshape((len(batch_x), np.prod(batch_x.shape[1:])))

        if (sync_gen is not None and sync_gen.input_vectorized):
            # TODO: use np.ravel or x.ravel()
            batch_xt = batch_xt.reshape((len(batch_xt), np.prod(batch_xt.shape[1:])))

        # Process class_mode
        if (self.class_mode not in {'sparse', 'binary', 'categorical'}):
            if (sync_gen is not None):  
                return (batch_x, batch_xt)
            else:     
                return (batch_x, batch_x)

        assert(self.y is not None)
        classes = self.y[index_array]

        if self.class_mode == 'sparse':
            batch_y = classes
        elif self.class_mode == 'binary':
            batch_y = classes.astype('float32')
        elif self.class_mode == 'categorical':
            assert(self.nb_class is not None)
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(classes):
                batch_y[i, label] = 1.
        else:
            if (sync_gen is not None):  
                return (batch_x, batch_xt)
            else:     
                return (batch_x, batch_x)

        return batch_x, batch_y

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _release(self):
        """Release internal resources used by the iterator."""
        pass

    def _get_data(self, X, j):
        """Load image from in memory database, pre-process and return.

        Parameters
        ----------
        X : `array_like`
            Data matrix. Format Samples x ...

        j : int
            Index of the data item to be featched. 
        """
        x = self.X[j]
        x, _ = self.image_data_generator.random_transform(x.astype('float32'))
        x = self.image_data_generator.standardize(x)
        return x
