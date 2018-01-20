# -*- coding: utf-8 -*-
"""
.. module:: ImageDataPreProcessor
   :platform: Unix, Windows
   :synopsis: Represent ImageDataPreProcessor class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# Local Imports
from nnf.core.iters.ImageDataGenerator import ImageDataGenerator
from nnf.core.iters.memory.NumpyArrayIterator import NumpyArrayIterator
from nnf.core.iters.disk.DirectoryIterator import DirectoryIterator


class ImageDataPreProcessor(ImageDataGenerator):
    """ImageDataPreProcessor represents the pre-processor for image data.

    Attributes
    ----------
    _fn_gen_coreiter : `function`, optional
        Factory method to create the core iterator. (Default value = None).
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, pp_params=None, fn_gen_coreiter=None):
        """Construct a :obj:`ImageDataPreProcessor` instance.

        Parameters
        ----------
        pp_params : :obj:`dict`, optional
            Pre-processing parameters. (Default value = None).

        fn_gen_coreiter : `function`, optional
            Factory method to create the core iterator. (Default value = None). 
        """
        super().__init__(pp_params)

        if (pp_params is None):
            self.wnd_max_normalize = False
        else:
            self.wnd_max_normalize = pp_params['wnd_max_normalize'] if ('wnd_max_normalize' in pp_params) else False

        self._fn_gen_coreiter = fn_gen_coreiter
        self._pp_params = pp_params
        self.whiten_components = None
        self.mapminmax_setting = None
        self.wnd_max = None

    def apply(self, setting):
        """Apply setting on `self`."""

        if setting is None:
            return

        self.wnd_max = setting['wnd_max']
        self.mean = setting['mean']
        self.std = setting['std']
        self.principal_components = setting['principal_components']
        self.whiten_components = setting['whiten_components']
        self.mapminmax_setting = setting['mapminmax_setting']

    def standardize(self, x):
        """Standardize data sample."""
        x = super().standardize(x)

        # VGG16Model specific pre-processing param
        if (self._pp_params is not None
                and ('normalize_vgg16' in self._pp_params)
                and self._pp_params['normalize_vgg16']):
            # x /=  255., use iter_pp_param = {'rescale':1./255},
            x -= 0.5
            x *= 2.

        # if (self.mapminmax_setting is not None):
        #    x = mapminmax('apply', x, self.mapminmax_setting)
        return x

    def standardize_window(self, x):
        """Standardize window data sample when :obj:`WindowIter` is utilized."""
        if (self._pp_params is not None
                and ('wnd_featurewise_0_1' in self._pp_params)
                and self._pp_params['wnd_featurewise_0_1']):
            max = x.max()
            if max != 0:
                x = x / max

        # VGG16Model specific pre-processing param
        if (self._pp_params is not None
                and ('wnd_normalize_vgg16' in self._pp_params)
                and self._pp_params['wnd_normalize_vgg16']):

            # x = x / x.max(), use iter_pp_param = {'wnd_featurewise_0_1':True},
            x -= 0.5
            x *= 2.

        # If maximum over all windows is calculated via `fit_window()`
        if (self.wnd_max is not None):
            x = x / self.wnd_max
        return x

    def dop_default(self, data, p_norm):
        """Distance operator default. (Euclidean distance matrix)."""
        return squareform(pdist(data, 'euclidean'))

    def fit_window(self, witer):
        """Required for wnd_max_normalize.

        Parameters
        ----------
        witer : :obj:`WindowIter`
            Window iterator to provide distance matrix feature as the input to the network.
        """

        # Save
        tmp = witer.loop_data

        # Reinitialize
        witer.reinit(loop_data=False)

        # Track the maximum over all windows
        self.wnd_max = None

        # Iterate over all windows
        for X, _ in witer:
            cur_max = X.max()

            if (self.wnd_max is None):
                self.wnd_max = cur_max

            else:
                if (cur_max > self.wnd_max):
                    self.wnd_max = cur_max

        # Restore
        witer.reinit(loop_data=tmp)

    def fit(self, nndb,
            augment=False,
            rounds=1,
            seed=None):
        """Required for featurewise_center, featurewise_std_normalization
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
        """
        # Get database `X` according to data_format
        X = self.__get_db(nndb, self.data_format)

        # Invoke parent fit method
        super().fit(X, augment, rounds, seed)

        # Perform whitening/mapminmax/etc
        if (self._pp_params is not None
                and ('mapminmax' in self._pp_params)):
            minmax_range = self._pp_params['mapminmax']
            # [~, self.mapminmax_setting] = mapminmax(X, minmax_range(1), minmax_range(2))

    def flow_realtime(self, rt_stream, params=None):
        """Construct a core iterator instance for in memory database traversal.

        Parameters
        ----------
        X : ndarray
            Data in tensor. Format: Samples x dim1 x dim2 ...

        y : ndarray, optional
            Vector indicating the class labels.

        nb_class : int, optional
            Number of classes.

        params : :obj:`dict`, optional
            Core iterator parameters.

        Returns
        -------
        :obj:`NumpyArrayIterator`
            Core iterator.
        """
        if self._fn_gen_coreiter is None:
            raise Exception("Default RealTimeStreamIterator (core-iterator) is not available. Please specify params['fn_coreiter'].")
            # return SHMStreamIterator(rt_stream, None, nb_class, self, params)

        else:
            core_iter = None
            if callable(self._fn_gen_coreiter):
                core_iter = self._fn_gen_coreiter(rt_stream, None, rt_stream.nb_class, self, params)

            if not isinstance(core_iter, NumpyArrayIterator):
                raise Exception("`fn_gen_coreiter` is not a child of `NumpyArrayIterator` class")

            return core_iter

    def flow_ex(self, nndb, params=None):
        """Construct a core iterator instance for in memory database traversal.

        Parameters
        ----------
        X : ndarray
            Data in tensor. Format: Samples x dim1 x dim2 ...

        y : ndarray, optional
            Vector indicating the class labels.

        nb_class : int, optional
            Number of classes.

        params : :obj:`dict`, optional
            Core iterator parameters. 

        Returns
        -------
        :obj:`NumpyArrayIterator`
            Core iterator.
        """
        # Get database `X` according to data_format
        X = self.__get_db(nndb, self.data_format)

        if self._fn_gen_coreiter is None:
            return NumpyArrayIterator(X, nndb.cls_lbl, nndb.cls_n, self, params)

        else:
            core_iter = None
            if callable(self._fn_gen_coreiter):
                core_iter = self._fn_gen_coreiter(X, nndb.cls_lbl, nndb.cls_lbl, self, params)

            if not isinstance(core_iter, NumpyArrayIterator):
                raise Exception("`fn_gen_coreiter` is not a child of `NumpyArrayIterator` class")

            return core_iter

    def flow_from_directory_ex(self, frecords, nb_class, params=None):
        """Construct a core iterator instance for disk database traversal.

        Parameters
        ----------
        frecords : :obj:`list`
            List of file records. frecord = [fpath or class_path, fpos or filename, cls_lbl]

        nb_class : int
            Number of classes.

        params : :obj:`dict`
            Core iterator parameters.

        Returns
        -------
        :obj:`DirectoryIterator`
            Core iterator.
        """
        if self._fn_gen_coreiter is None:
            core_iter = DirectoryIterator(frecords, nb_class, self, params)

        else:
            core_iter = None
            if callable(self._fn_gen_coreiter):
                core_iter = self._fn_gen_coreiter(frecords, nb_class, self, params)

            if not isinstance(core_iter, DirectoryIterator):
                raise Exception("`fn_gen_coreiter` is not a child of `DirectoryIterator` class")

        return core_iter

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __get_db(self, nndb, data_format):
        """Fetch the database and class labels."""

        # Fix the input_shape respectively => (H, W, CH) or (CH, H, W)
        if data_format == 'channels_last':
            db = nndb.db_convo_tf
        else:
            db = nndb.db_convo_th

        return db


    ##########################################################################
    # Dependant Properties
    ##########################################################################
    @property
    def setting(self):
        """Calculated setting to use on val/te/etc datasets."""
        value = {'wnd_max': self.wnd_max,
                 'mean': self.mean,
                 'std': self.std,
                 'principal_components': self.principal_components,
                 'whiten_components': self.whiten_components,
                 'mapminmax_setting': self.mapminmax_setting
                 }
        return value
