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

    _nrm_vgg16 : bool, optional
        Whether to perform unique normalization for VGG16Model. (Default value = False).
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
        self._fn_gen_coreiter = fn_gen_coreiter

        # VGG16Model specific pre-processing param
        if (pp_params is None): 
            self._nrm_vgg16 = False
        else:        
            self._nrm_vgg16 = pp_params['normalize_vgg16']\
                                if ('normalize_vgg16' in pp_params) else False

        super().__init__(pp_params)

    def apply(self, setting):
        """Apply settings on `self`."""
        if (settings is None): return

        self.mean = settings.mean
        self.std = settings.std
        #self.map_min_max = settings.map_min_max
        #self.whiten = settings.whiten

    def standardize(self, x):
        """Standardize data sample."""
        if (self._nrm_vgg16):
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68

        return super().standardize(x)

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
        value['map_min_max'] = self.map_min_max
        value['whiten'] = self.whiten