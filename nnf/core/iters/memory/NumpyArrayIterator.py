"""NumpyArrayIterator to represent NumpyArrayIterator class."""
# -*- coding: utf-8 -*-
# Global Imports
import numpy as np
from keras import backend as K
from scipy.misc import imresize
from warnings import warn as warning

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
    def __init__(self, X, y, nb_class, imdata_pp, params=None, nb_sample=None):
        """Construct a :obj:`NumpyArrayIterator` instance.

        Parameters
        ----------
        X : ndarray | None
            Data in 4D matrix. Format: N x H x W x CH or N x CH x H x W depending on the params['data_format'].

        y : ndarray | None
            Vector indicating the class labels.

        nb_class : int
            Number of classes.

        imdata_pp : :obj:`ImageDataPreProcessor`
            Image data pre-processor.

        params : :obj:`dict` | None
            Core iterator parameters.

        nb_sample : int | None
            Number of samples. If X is not none, nb_sample will be calculate from X.
        """
        # For warning and error messages in __init__()
        edataset = params['_edataset'] if '_edataset' in params else 'TEMP'

        if (params is None):
            self.input_vectorized = False
            window_iter = None
            input_shape = None
            fn_reshape_input = None
            data_format = None
            class_mode = None
            batch_size = 32; shuffle = True; seed = None
            save_to_dir = None; save_prefix = ''; save_format = 'jpeg'
            _use_rgb = None

        else:
            self.input_vectorized = params['input_vectorized'] if ('input_vectorized' in params) else False
            window_iter = params['window_iter'] if ('window_iter' in params) else None
            input_shape = params['input_shape'] if ('input_shape' in params) else None
            fn_reshape_input = params['fn_reshape_input'] if ('fn_reshape_input' in params) else None
            data_format = params['data_format'] if ('data_format' in params) else None
            class_mode = params['class_mode'] if ('class_mode' in params) else None
            batch_size = params['batch_size'] if ('batch_size' in params) else 32
            shuffle = params['shuffle'] if ('shuffle' in params) else True
            seed = params['seed'] if ('seed' in params) else None
            save_to_dir = params['save_to_dir'] if ('save_to_dir' in params) else None
            save_prefix = params['save_prefix'] if ('save_prefix' in params) else ''
            save_format = params['save_format'] if ('save_format' in params) else 'jpeg'
            _use_rgb = params['_use_rgb'] if ('_use_rgb' in params) else None

        self.imdata_pp = imdata_pp
        self.data_format = K.image_data_format() if data_format is None else data_format
        self.fn_reshape_input = fn_reshape_input

        if input_shape is not None:

            # Convert to a tuple
            input_shape = tuple(input_shape)

            # Input shape validity
            if np.isscalar(input_shape) or len(input_shape) == 1:
                raise Exception("[" + str(edataset).upper() + "] iter_param['input_shape'] must be in the format (H, W) or (H, W, CH).")

        # Use rgb configuration
        if _use_rgb not in {True, False, None}:
            raise ValueError('[' + str(edataset).upper() + '] Invalid internal use_rgb mode:', _use_rgb,
                             '; expected True or False or None.')

        if X is not None:
            self.X = np.asarray(X)
            if self.X.ndim != 4:
                raise ValueError('[' + str(edataset).upper() + '] Input data in `NumpyArrayIterator` '
                                 'should have rank 4. You passed an array '
                                 'with shape', self.X.shape)
            channels_axis = 3 if self.data_format == 'channels_last' else 1
            if self.X.shape[channels_axis] not in {1, 3, 4}:
                raise ValueError('[' + str(edataset).upper() + '] NumpyArrayIterator is set to use the '
                                 'data format convention "' + self.data_format + '" '
                                 '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                                 'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                                 'However, it was passed an array with shape ' + str(self.X.shape) +
                                 ' (' + str(self.X.shape[channels_axis]) + ' channels).')
            self.nb_sample = X.shape[0] if nb_sample is None else nb_sample

            # Case Study:
            # Expected network input shape (input_shape), might be from the data size (X.shape[1:]).
            # i.e single channel data may be expected by the network as multiple channel data.
            if input_shape is None:
                self.input_shape = X.shape[1:]  # H x W x CH or CH x W x H

            elif len(input_shape) == 2:  # H x W (due to explicit definition)

                # Validate images against use_rgb configuration
                ch_idx = -1 if self.data_format == 'channels_last' else 1
                if (_use_rgb is not None and _use_rgb) and not (X.shape[ch_idx] == 3):
                    warning('[' + str(edataset).upper() + '] Images with ch={0} do not support rgb channels.'.format(X.shape[ch_idx]))
                if (_use_rgb is not None and not _use_rgb) and not (X.shape[ch_idx] == 1):
                    warning('[' + str(edataset).upper() + '] Images with ch={0} do not support grayscale processing.'.format(X.shape[ch_idx]))

                self.input_shape = input_shape + (X.shape[ch_idx],) if self.data_format == 'channels_last' else (X.shape[ch_idx],) + input_shape

            elif len(input_shape) == 3:  # H x W x CH (due to explicit definition)
                H, W, CH = input_shape
                self.input_shape = (H, W, CH) if self.data_format == 'channels_last' else (CH, H, W)

        else:
            if input_shape is None:
                raise ValueError("[" + str(edataset).upper() + "]Invalid input_shape. Please specify params['input_shape'] parameter.")

            self.X = None
            self.nb_sample = nb_sample
            self.input_shape = input_shape

        if y is not None:
            if len(X) != len(y):
                raise ValueError('[' + str(edataset).upper() + '] X (images tensor) and y (labels) '
                                 'should have the same length. '
                                 'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
            self.y = np.asarray(y)

        else:
            self.y = None

        # Class mode configuration
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('[' + str(edataset).upper() + '] Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.nb_class = nb_class
        self.input_generators = []   # Input generator list
        self.target_generators = []  # Target generator list
        self.fn_read_sample = self._fn_read_sample

        # IMPORTANT: Take a copy of `window_iter` for this core-iterator that iterates data for params['_edataset']
        from nnf.core.iters.custom.WindowIter import WindowIter  # to avoid circular imports
        self.window_iter = WindowIter(window_iter.sizes, window_iter.strides, window_iter.mode) if window_iter is not None else None

        if (self.window_iter is not None):
            self.window_iter.init(imdata_pp,
                                  self.input_shape,
                                  self.data_format,
                                  self.fn_read_sample,
                                  self.fn_reshape_input,
                                  self.X,
                                  self.y,
                                  indices=np.arange(self.nb_sample))
            if shuffle:
                warning("[" + str(edataset).upper() + "] `window_iter` do not support iter_param['shuffle']=True. Resetting to False")
                shuffle = False

        # DEBUG: Incrementing index used when saving images
        if self.save_to_dir:
            self.save_im_index = 0

        # Use when class label are used for multiple output models
        self.target_duplicator = 1

        super(NumpyArrayIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed, params=params)

        # Must be set last in the constructor after all the instance variables (child, parent) are initialized
        self.input_generators.append(self)  # first element is the primary generator

    def clone(self):
        """Create a copy of this NumpyArrayIterator object."""
        # This will construct
        # - a new window_iter object
        # - a generator object for self.index_generator
        # - matlab engine (meng) will not be None
        new_obj = type(self)(self.X, self.y, self.nb_class, self.imdata_pp, self.params, self.nb_sample)

        # Add the secondary input generators (no need to make a copy for secondary generators)
        # This is because the iterator state is maintained only at the primary input generator
        for i, generator in enumerate(self.input_generators):
            if i==0: continue  # Exclude i==0 (primary generator) since it's already added in the constructor
            new_obj.input_generators.append(generator)

        # Add the target generators (no need to make a copy)
        # This is because the iterator state is maintained only at the primary input generator
        for _, generator in enumerate(self.target_generators):
            new_obj.target_generators.append(generator)

        # Do not copy:
        # self.performance_cache
        return new_obj

    def sync_generator(self, generator):
        """Sync/Combine `generator` with this (primary) generator's input_generators."""

        # For each generator in `generator.input_generators`
        for gen in generator.input_generators:

            # IMPORTANT: Check the compatibility of the gen with this generator
            # Both generators must have same number of images
            assert(len(self.X) == len(gen.X))

            # Warning if batch sizes are different
            if (self.batch_size != gen.batch_size):
                warning("`batch_size` of the sync generator is different from the primary generator.")

            found = next((x for x in self.input_generators if x == gen), None)
            if found:
                warning("Trying to sync an already existing input generator. Skipping.")
                return

            gen.increment_ref_count()
            self.input_generators.append(gen)

    def sync_tgt_generator(self, generator):
        """Sync/Combine `generator` with this (primary) generator's target_generators."""

        # Since parallel generators are added via sync_generator() method above,
        # 'generator.input_generators' must be exhausted for additional target generators.

        # For each generator in `generator.input_generators`
        for gen in generator.input_generators:

            # IMPORTANT: Check the compatibility of the gen with this generator
            # Both generators must have same number of images
            assert(len(self.X) == len(gen.X))

            # Warning if batch sizes are different
            if (self.batch_size != gen.batch_size):
                warning("`batch_size` of the sync generator is different from the primary generator.")

            found = next((x for x in self.target_generators if x == gen), None)
            if found:
                warning("Trying to sync an already existing target generator. Skipping.")
                return

            gen.increment_ref_count()
            self.target_generators.append(gen)

    def new_target_container(self):
        """Get an empty container to collect the targets."""

        classes = None
        if self.class_mode is not None:
            # Classes at primary generator
            classes = np.zeros((self.nb_sample, self.nb_class), K.floatx())

        tmp = []
        for generator in self.target_generators:
            tgt_shape = generator.input_shape
            if (generator.input_vectorized):
                tgt_shape = (np.prod(tgt_shape), )
            tmp.append(np.zeros(((self.nb_sample,) + tgt_shape), K.floatx()))

        if classes is not None and len(tmp) > 0:
            targets = [classes] + tmp
        else:
            targets = tmp if len(tmp) > 1 else (tmp[0] if len(tmp) == 1 else None)

        return targets

    def data_tuple(self, j):
        dt_list_x = [None] * len(self.input_generators)
        dt_cls_lbl = None
        dt_list_xt = [None] * len(self.target_generators)

        # Iterate input generators
        for ig, generator in enumerate(self.input_generators):
            dt_list_x[ig], cls_lbl = generator._get_data(generator.X, generator.y, j)
            if ig == 0 and cls_lbl is not None:  # Primary generator is responsible for setting the class
                dt_cls_lbl = cls_lbl

        # Iterate target generators
        for ig, generator in enumerate(self.target_generators):
            dt_list_xt[ig], _ = generator._get_data(generator.X, generator.y, j)

        return dt_list_x, dt_cls_lbl, dt_list_xt

    def next(self):
        """Advance to next batch of data.

        Returns
        -------
        ndarray
            `batch_x` data matrix. Format: Samples x Features

        ndarray
            `batch_y` class label vector or 'batch_x' matrix. refer code.
        """
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        # with self.lock:  # TypeError: can't pickle _thread.lock objects
        index_array, current_index, current_batch_size = next(self.index_generator)

        # The transformation of images is not under thread lock so it can be done in parallel
        # Initialize list_batch_x and list_batch_xt
        classes = None
        list_batch_x = []
        for _, generator in enumerate(self.input_generators):
            batch = np.zeros((current_batch_size,) + generator.input_shape, dtype='float32')
            list_batch_x.append(batch)

        list_batch_xt = []
        for _, generator in enumerate(self.target_generators):
            batch = np.zeros((current_batch_size,) + generator.input_shape, dtype='float32')
            list_batch_xt.append(batch)

        assert(len(list_batch_x) == len(self.input_generators))
        assert (len(list_batch_xt) == len(self.target_generators))

        if self.performance_cache is not None:
            # Build batch of image data and corresponding labels
            si = 0  # Incrementing sample index
            while si <  current_batch_size:
                list_x, cls_lbl, list_xt = next(self.performance_cache_gen_next)

                # Set the input data
                for ig, x in enumerate(list_x):
                    batch = list_batch_x[ig]
                    batch[si] = x

                # Set the class for input data in first iteration
                if si == 0 and cls_lbl is not None and classes is None:  # PERF: Late instantiation
                    classes = np.zeros((current_batch_size), dtype='uint16')

                if classes is not None:
                    classes[si] = cls_lbl

                # Set the target data
                for ig, xt in enumerate(list_xt):
                    batch = list_batch_xt[ig]
                    batch[si] = xt

                # Increment the sample index
                si += 1

        else:
            # Build batch of image data and corresponding labels
            for i, j in enumerate(index_array):

                list_x, cls_lbl, list_xt = self.data_tuple(j)

                # Set the input data
                for ig, x in enumerate(list_x):
                    batch = list_batch_x[ig]
                    batch[i] = x

                # Set the class for input data in first iteration
                if i == 0 and cls_lbl is not None and classes is None:  # PERF: Late instantiation
                    classes = np.zeros((current_batch_size), dtype='uint16')

                if classes is not None:
                    classes[i] = cls_lbl

                # Set the target data
                for ig, xt in enumerate(list_xt):
                    batch = list_batch_xt[ig]
                    batch[i] = xt

        # Optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            self._save_to_disk(current_batch_size, current_index, self.input_generators, list_batch_x, classes)
            self._save_to_disk(current_batch_size, current_index, self.target_generators, list_batch_xt, classes)

        # Perform the reshape operation on x if necessary
        for ig, generator in enumerate(self.input_generators):
            if (generator.input_vectorized):
                batch = list_batch_x[ig]
                list_batch_x[ig] = batch.reshape((len(batch), np.prod(batch.shape[1:])))

        for ig, generator in enumerate(self.target_generators):
            if (generator.input_vectorized):
                batch = list_batch_xt[ig]
                list_batch_xt[ig] = batch.reshape((len(batch), np.prod(batch.shape[1:])))

        n_ingenrators = len(list_batch_x)
        n_tgtgenrators = len(list_batch_xt)

        # Process primary generator's class_mode & prepare the returns
        ret_x = list_batch_x[0] if n_ingenrators == 1 else list_batch_x

        # When class_mode is None
        if (self.class_mode not in {'sparse', 'binary', 'categorical'}):

            if (n_tgtgenrators == 0):
                ret_xt = ret_x
            else:
                ret_xt = list_batch_xt[0] if n_tgtgenrators == 1 else list_batch_xt

            # ig1 is the primary generator
            # return ([ig1_batch_x, ig2_batch_x], [og1_batch_xt, og2_batch_xt])
            ret = (ret_x, ret_xt)

        else:
            assert (classes is not None)
            assert (self.class_mode in {'sparse', 'binary', 'categorical'})

            batch_y = None

            # Build batch of labels
            if self.class_mode == 'sparse':
                batch_y = classes
            elif self.class_mode == 'binary':
                batch_y = classes.astype('float32')
            elif self.class_mode == 'categorical':
                assert (self.nb_class is not None)
                batch_y = np.zeros((len(list_batch_x[0]), self.nb_class), dtype='float32')
                for i, label in enumerate(classes):
                    batch_y[i, label] = 1.

            if self.target_duplicator > 1:
                ret_y = [batch_y] * self.target_duplicator
            else:
                if (n_tgtgenrators > 0):
                    ret_y = [batch_y]  # since ret_y will be appended below
                else:
                    ret_y = batch_y

            if (n_tgtgenrators > 0):
                ret_y = ret_y + ([list_batch_xt[0]] if n_tgtgenrators == 1 else list_batch_xt)

            # ig1 is the primary generator
            # return ([ig1_batch_x, ig2_batch_x], [og1_classes, og1_classes... <target_duplicator times>, og1_batch_xt, og2_batch_xt])
            ret = (ret_x, ret_y)

        return ret

    def release(self):
        """Release internal resources used by the iterator."""

        super().release()
        assert self.ref_count >= 0

        if self.ref_count == 0:
            self.input_vectorized = None
            self.imdata_pp = None
            self.fn_reshape_input = None
            self.data_format = None
            self.X = None
            self.nb_sample = None
            self.input_shape = None
            self.y = None
            self.class_mode = None
            self.save_to_dir = None
            self.save_prefix = None
            self.save_format = None
            self.nb_class = None
            self.input_generators = None
            self.target_generators =None
            self.fn_read_sample = None
            if self.window_iter: del self.window_iter
            self.window_iter = None
            self.save_im_index = None
            self.target_duplicator = None

    ##########################################################################
    # Dependant Properties (public)
    ##########################################################################
    @property
    def input_shapes(self):
        """List: shapes of the inputs expected."""

        input_shapes = []
        for _, generator in enumerate(self.input_generators):
            if (generator.input_vectorized):
                input_shapes.append(np.prod(generator.input_shape))
            else:
                input_shapes.append(generator.input_shape)

        return input_shapes

    @property
    def output_shapes(self):
        """List: shapes of the outputs expected."""

        output_shapes = []
        if self.class_mode is not None:
            # Classes at primary generator
            primary_generator = next(iter(self.input_generators))
            output_shapes.append(primary_generator.nb_class)

        for generator in self.target_generators:
            if (generator.input_vectorized):
                output_shapes.append(np.prod(generator.input_shape))
            else:
                output_shapes.append(generator.input_shape)

        return output_shapes

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _fn_read_sample(self, X, y, j, reshape=True):
        img = X[j]
        cls_lbl = y[j] if (y is not None) else None

        # If network input shape is not the actual image shape
        if (reshape and img.shape != self.input_shape):

            # Default behavior or invoke `fn_reshape_input`
            if self.fn_reshape_input is None:

                assert img.ndim == 3
                channels_axis = 2 if self.data_format == 'channels_last' else 0

                # Check channel compatibility for `imresize` operation
                if self.input_shape[channels_axis] > img.shape[channels_axis]:
                    raise ValueError("input_shape channel count {0} > image channel count {1}. " +
                        " Specify params['fn_reshape_input'] to perform the transformation."
                                     .format(self.input_shape[channels_axis], img.shape[channels_axis]))

                # For grayscale images or numerical value columns extended with CH = 1
                # `imresize` cannot resize arrays with CH = 1 (105, 1, 1) or (33, 33, 1) => (128, 128, 1)
                # Thus squeezing the channel singleton dimension
                img = np.squeeze(img, axis=channels_axis) if img.shape[channels_axis] == 1 else img

                # Perform imresize
                img = imresize(img, self.input_shape, 'bicubic')

                # if (105, 1) is resize to (128, 128, 1), imresize will return (128, 128)
                # hence need to add the channel dimension explicitly.
                if (img.ndim == 2):
                    img = np.expand_dims(img, axis=2) if self.data_format == 'channels_last' else np.expand_dims(img, axis=0)

            else:
                self.fn_reshape_input(img, self.input_shape, self.data_format)

        img, _ = self.imdata_pp.random_transform(img.astype(K.floatx()))
        img = self.imdata_pp.standardize(img)
        return img, cls_lbl

    def _get_data(self, X, y, j):
        """Load image from in memory database, pre-process and return.

        Parameters
        ----------
        X : ndarray
            4D data array. Format = Samples x H X W x CH or Samples x CH X H X W

        y : ndarray
            1D class label array. Format = (cls_lbls, )

        j : int
            Index of the data item to be fetched.
        """
        if (self.window_iter is None):
            x, cls_lbl = self.fn_read_sample(X, y, j)
        else:
            x, cls_lbl = next(self.window_iter)

        return x, cls_lbl