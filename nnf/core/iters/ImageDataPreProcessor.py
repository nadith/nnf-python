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
        """Apply settings on `self`."""
        if (settings is None): return

        self.mean = settings.mean
        self.std = settings.std
        self.principal_components = settings.principal_components
        self.whiten_components = settings.whiten_components
        self.mapminmax_setting = settings.mapminmax_setting

    def standardize(self, x):
        """Standardize data sample."""

        # VGG16Model specific pre-processing param
        if (self._pp_params is not None
            and ('normalize_vgg16' in self._pp_params) and
            self._pp_params['normalize_vgg16']):
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68

        x = super().standardize(x)

        #if (self.mapminmax_setting is not None):
        #    x = mapminmax('apply', x, self.mapminmax_setting)
        return x

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
        super().fit(self, X, augment, rounds, seed)
            
        # Perform whitening/mapminmax/etc
        if (self._pp_params is not None
            and ('mapminmax' in self._pp_params)):
            minmax_range = self._pp_params['mapminmax']
            #[~, self.mapminmax_setting] = mapminmax(X, minmax_range(1), minmax_range(2))

    def flow(self, X, y=None, nb_class=None, params=None):
        """Construct a core iterator instancef for in memory database traversal.

        Parameters
        ----------
        X : `array_like`
            Data in tensor. Format: Samples x dim1 x dim2 ...

        y : `array_like`
            Vector indicating the class labels.

        params : :obj:`dict`
            Core iterator parameters. 

        Returns
        -------
        :obj:`NumpyArrayIterator`
            Core iterator.
        """
        if (self._fn_gen_coreiter is None):
            return NumpyArrayIterator(X, y, nb_class, self, params)

        else:
            try:
                core_iter = self._fn_gen_coreiter(X, y, nb_class, self, params)
                if (not isinstance(core_iter, NumpyArrayIterator)): raise Exception()  # Error handlings               
                return core_iter

            except:
                raise Exception("`fn_gen_coreiter` is not a child of `NumpyArrayIterator` class")

    def flow_from_directory(self, frecords, nb_class, params=None):
        """Construct a core iterator instancef for disk database traversal.

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
        if (self._fn_gen_coreiter is None):
            return DirectoryIterator(frecords, nb_class, self, params)    
    
        else:
            try:
                core_iter = self._fn_gen_coreiter(frecords, nb_class, self, params)
                if (not isinstance(core_iter, DirectoryIterator)): raise Exception()  # Error handling
                return core_iter

            except:
                raise Exception("`fn_gen_coreiter` is not a child of `DirectoryIterator` class")

    ##########################################################################
    # Dependant Properties
    ##########################################################################
    @property
    def settings(self):
        """Calculated settings to use on val/te/etc datasets."""
        value = {}
        value['mean'] = self.mean
        value['std'] = self.std
        value['principal_components'] = self.principal_components
        value['whiten_components'] = self.whiten_components
        value['mapminmax_setting'] = self.mapminmax_setting