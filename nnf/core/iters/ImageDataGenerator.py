# -*- coding: utf-8 -*-
# Global Imports
import numpy as np
from keras import backend as K
from keras.preprocessing.image import transform_matrix_offset_center
from keras.preprocessing.image import apply_transform
from keras.preprocessing.image import random_channel_shift
from keras.preprocessing.image import flip_axis

# Local Imports

class ImageDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument: one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    '''
    def __init__(self, params):

        if (params is None):
            self.featurewise_center = False
            self.samplewise_center = False
            self.featurewise_std_normalization = False
            self.samplewise_std_normalization = False
            self.zca_whitening = False
            self.rotation_range = 0.
            self.width_shift_range = 0.
            self.height_shift_range = 0.
            self.shear_range = 0.
            zoom_range = 0.
            self.channel_shift_range = 0.
            self.fill_mode = 'nearest'
            self.cval = 0.
            self.horizontal_flip = False
            self.vertical_flip = False
            rescale = None
            preprocessing_function = None
            data_format = None
            random_transform_seed = None

            # To invoke fit() function to calculate 
            # featurewise_center, featurewise_std_normalization and zca_whitening
            self.augment = False
            self.rounds = 1
            self.seed = None

        else:
            self.featurewise_center = params['featurewise_center'] if ('featurewise_center' in params) else False
            self.samplewise_center = params['samplewise_center'] if ('samplewise_center' in params) else False
            self.featurewise_std_normalization = params['featurewise_std_normalization'] if ('featurewise_std_normalization' in params) else False
            self.samplewise_std_normalization = params['featurewise_center'] if ('featurewise_center' in params) else False
            self.zca_whitening = params['featurewise_center'] if ('featurewise_center' in params) else False
            self.rotation_range = params['rotation_range'] if ('rotation_range' in params) else 0.
            self.width_shift_range = params['width_shift_range'] if ('width_shift_range' in params) else 0.
            self.height_shift_range = params['height_shift_range'] if ('height_shift_range' in params) else 0.
            self.shear_range = params['shear_range'] if ('shear_range' in params) else 0.
            zoom_range = params['zoom_range'] if ('zoom_range' in params) else 0.
            self.channel_shift_range = params['channel_shift_range'] if ('channel_shift_range' in params) else 0.
            self.fill_mode = params['fill_mode'] if ('fill_mode' in params) else 'nearest'
            self.cval = params['cval'] if ('cval' in params) else 0.
            self.horizontal_flip = params['horizontal_flip'] if ('horizontal_flip' in params) else False
            self.vertical_flip = params['vertical_flip'] if ('vertical_flip' in params) else False
            rescale = params['rescale'] if ('rescale' in params) else None
            preprocessing_function = params['preprocessing_function'] if ('preprocessing_function' in params) else None
            data_format = params['data_format'] if ('data_format' in params) else None
            random_transform_seed = params['random_transform_seed'] if ('random_transform_seed' in params) else None

            # To invoke fit() function to calculate 
            # featurewise_center, featurewise_std_normalization and zca_whitening
            self.augment = params['augment'] if ('augment' in params) else False
            self.rounds = params['rounds'] if ('rounds' in params) else 1
            self.seed = params['seed'] if ('seed' in params) else None           
          
        if data_format is None:
            data_format = K.image_data_format()
        #self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.random_transform_seed = random_transform_seed # NNF: Extended functionality

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format should be "channels_last" (channel after row and '
                             'column) or "channels_first" (channel before row and column). '
                             'Received arg: ', data_format)

        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if data_format == 'channels_last':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg',
                            follow_links=False):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
            follow_links=follow_links)

    def standardize(self, x):
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (x.size))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def random_transform(self, x, xt=None):
        
        if (self.random_transform_seed is not None):
            np.random.seed(self.random_transform_seed)

        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]

        if (xt is not None):
            assert(h==xt.shape[img_row_index] and w==xt.shape[img_col_index])

        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)

        xt = None if (xt is None) else apply_transform(xt, transform_matrix, img_channel_index,
                                                        fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)
            xt = None if (xt is None) else random_channel_shift(xt, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                xt = None if (xt is None) else flip_axis(xt, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                xt = None if (xt is None) else flip_axis(xt, img_row_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, xt

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            X: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        X = np.asarray(X)
        if X.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(X.shape))
        if X.shape[self.channel_index] not in {1, 3, 4}:
            raise ValueError(
                'Expected input to be images (as Numpy array) '
                'following the dimension ordering convention "' + self.data_format + '" '
                '(channels on axis ' + str(self.channel_index) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_index) + '. '
                'However, it was passed an array with shape ' + str(X.shape) +
                ' (' + str(X.shape[self.channel_index]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=(0, self.row_index, self.col_index))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_index - 1] = X.shape[self.channel_index]
            self.mean = np.reshape(self.mean, broadcast_shape)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=(0, self.row_index, self.col_index))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_index - 1] = X.shape[self.channel_index]
            self.std = np.reshape(self.std, broadcast_shape)
            X /= (self.std + K.epsilon())

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[0]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
