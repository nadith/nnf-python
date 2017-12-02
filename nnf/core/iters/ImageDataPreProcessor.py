# -*- coding: utf-8 -*-
"""
.. module:: ImageDataPreProcessor
   :platform: Unix, Windows
   :synopsis: Represent ImageDataPreProcessor class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports

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
        self._fn_gen_coreiter = fn_gen_coreiter
        self._pp_params = pp_params
        self.whiten_components = None
        self.mapminmax_setting = None

    def apply(self, setting):
        """Apply setting on `self`."""

        if setting is None:
            return

        self.mean = setting['mean']
        self.std = setting['std']
        self.principal_components = setting['principal_components']
        self.whiten_components = setting['whiten_components']
        self.mapminmax_setting = setting['mapminmax_setting']

    def standardize(self, x):
        """Standardize data sample."""

        # VGG16Model specific pre-processing param
        if (self._pp_params is not None
                and ('normalize_vgg16' in self._pp_params)
                and self._pp_params['normalize_vgg16']):
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68

        x = super().standardize(x)

        # if (self.mapminmax_setting is not None):
        #    x = mapminmax('apply', x, self.mapminmax_setting)
        return x

    def fit(self, X,
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
        super().fit(X, augment, rounds, seed)

        # Perform whitening/mapminmax/etc
        if (self._pp_params is not None
                and ('mapminmax' in self._pp_params)):
            minmax_range = self._pp_params['mapminmax']
            # [~, self.mapminmax_setting] = mapminmax(X, minmax_range(1), minmax_range(2))

    def flow_ex(self, X, y=None, nb_class=None, params=None):
        """Construct a core iterator instance for in memory database traversal.

        Parameters
        ----------
        X : ndarray
            Data in tensor. Format: Samples x dim1 x dim2 ...

        y : ndarray
            Vector indicating the class labels.

        nb_class : int
            Number of classes.

        params : :obj:`dict`
            Core iterator parameters. 

        Returns
        -------
        :obj:`NumpyArrayIterator`
            Core iterator.
        """
        if self._fn_gen_coreiter is None:
            return NumpyArrayIterator(X, y, nb_class, self, params)

        else:
            core_iter = None
            if callable(self._fn_gen_coreiter):
                core_iter = self._fn_gen_coreiter(X, y, nb_class, self, params)

            if not isinstance(core_iter, NumpyArrayIterator):
                raise Exception("`fn_gen_coreiter` is not a child of `NumpyArrayIterator` class")

            return core_iter

    def flow_from_directory_ex(self, frecords, nb_class, params=None):
        """Construct a core iterator instance for disk database traversal.

        Parameters
        ----------
        frecords : :obj:`list`
            List of file records. frecord = [fpath, fpos, cls_lbl] 

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
            return DirectoryIterator(frecords, nb_class, self, params)

        else:
            core_iter = None
            if callable(self._fn_gen_coreiter):
                core_iter = self._fn_gen_coreiter(frecords, nb_class, self, params)

            if not isinstance(core_iter, DirectoryIterator):
                raise Exception("`fn_gen_coreiter` is not a child of `DirectoryIterator` class")

            return core_iter

    ##########################################################################
    # Dependant Properties
    ##########################################################################
    @property
    def setting(self):
        """Calculated setting to use on val/te/etc datasets."""
        value = {'mean': self.mean,
                 'std': self.std,
                 'principal_components': self.principal_components,
                 'whiten_components': self.whiten_components,
                 'mapminmax_setting': self.mapminmax_setting
                 }
        return value
