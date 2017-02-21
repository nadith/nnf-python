# -*- coding: utf-8 -*-
"""
.. module:: DataIterator
   :platform: Unix, Windows
   :synopsis: Represent DataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from abc import ABCMeta, abstractmethod
import numpy as np
import types

# Local Imports
from nnf.core.iters.ImageDataPreProcessor import ImageDataPreProcessor

class DataIterator(object):
    """DataIterator represents the base class for all iterators in the
        Neural Network Framework.

    .. warning:: abstract class and must not be instantiated.

    Attributes
    ----------
    _imdata_pp : :obj:`ImageDataPreProcessor`
        Image data pre-processor for all iterators.

    _gen_next : `function`
        Core iterator/generator that provide data.

    _sync_gen_next : :obj:`DirectoryIterator` or :obj:`NumpyArrayIterator`
        Core iterator that needs to be synced with `_gen_next`.
        (Default value = None)

    _params : :obj:`dict`
        Core iterator/generator parameters. (Default value = None)

    _pp_params : :obj:`dict`, optional
        Pre-processing parameters for :obj:`ImageDataPreProcessor`.
        (Default value = None).

    _fn_gen_coreiter : `function`, optional
        Factory method to create the core iterator. (Default value = None).

    edataset : :obj:`Dataset`
        Dataset enumeration key.

    _nb_class : int
        Number of classes.

    Notes
    -----
    Disman data iterators are utilzing a generator function in _gen_next 
    while the `nnmodel` data iterators are utilizing an core iterator.
    """
    __metaclass__ = ABCMeta

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, pp_params=None, fn_gen_coreiter=None, edataset=None, 
                                                                nb_class=None):
        """Constructor of the abstract class :obj:`DataIterator`.

        Parameters
        ----------
        pp_params : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.

        fn_gen_coreiter : `function`, optional
            Factory method to create the core iterator. (Default value = None). 
        """
        # Initialize the image data pre-processor with pre-processing params
        # Used by Diskman data iterators and `nnmodel` data iterators
        self._imdata_pp = ImageDataPreProcessor(pp_params, fn_gen_coreiter)
        
        # Core iterator or generator (initilaized in init())
        # Diskman data iterators are utilzing a generator function in _gen_next
        # while the `nnmodel` data iterators are utilizing an core iterator.
        self._gen_next = None

        # :obj:`DirectoryIterator` or :obj:`NumpyArrayIterator`
        # Core iterator that needs to be synced with `_gen_next`.
        self._sync_gen_next = None

        # Iterator params
        # Used by `nnmodel` data iterators only.
        # Can utilize in Diskman data iterators safely in the future.
        self._params = None

        # All the parameters are saved in instance variables to support the 
        # clone() method implementaiton of the child classes.
        self._pp_params = pp_params
        self._fn_gen_coreiter = fn_gen_coreiter

        # Used by `nnmodel` data iterators only.
        self.edataset = edataset
        self._nb_class = nb_class

    def init(self, gen_next=None, params=None):
        """Initialize the instance.

        Parameters
        ----------
        _gen_next : `function`
            Core iterator/generator that provide data.
        """        
        # Set the core iterator or generator 
        self._gen_next = gen_next

        # Set iterator params
        self._params = params

    ##########################################################################
    # Public: Core Iterator Only Operations/Dependant Properties
    ##########################################################################
    def set_shuffle(self, shuffle):
        """Set shuffle property."""
        # This property is only supported by the core iterator
        if (self._gen_next is None or
            isinstance(self._gen_next , types.GeneratorType)):
            return False

        if (self._gen_next.batch_index != 0 and 
            self._gen_next.shuffle != shuffle):
            raise Exception("Iterator is already active and failed to set shuffle")

        # Update iterator params if available
        if (self._params is not None):
            self._params['shuffle'] = shuffle

        self._gen_next.shuffle = shuffle
        return True

    def sync(self, gen_next):
        """Sync the secondary iterator with this iterator.

        Sync the secondary core iterator with this core itarator internally.
        Used when data needs to be generated with its matching target.

        Parameters
        ----------
        gen : :obj:`DirectoryIterator` or :obj:`NumpyArrayIterator` 
            Core iterator that needs to be synced with this core iterator
            `_gen_next`.

        Note
        ----
        Current supports only 1 iterator to be synced.
        """
        # This method is only supported by the core iterator      
        if (self._gen_next is None or
            isinstance(self._gen_next , types.GeneratorType)):
            return False       

        if (gen_next is None or
            isinstance(gen_next , types.GeneratorType)): 
            return False

        self._gen_next.sync(gen_next)
        self._sync_gen_next = gen_next
        return True

    @property
    def is_synced(self):
        """bool : whether this generator is synced with another generator."""
        return (self._sync_gen_next is not None)

    @property
    def input_vectorized(self):
        """bool: whether the input needs to be vectorized via the core iterator."""
        # This property is only supported by the core iterator
        if (self._gen_next is None or
            isinstance(self._gen_next , types.GeneratorType)):
            return None     
        return self._gen_next.input_vectorized

    @property
    def batch_size(self):
        """int: batch size to be read by the core iterator."""
        # This property is only supported by the core iterator
        if (self._gen_next is None or
            isinstance(self._gen_next , types.GeneratorType)):
            return None     
        return self._gen_next.batch_size

    @property
    def class_mode(self):
        """str: class mode at core iterator."""
        # This property is only supported by the core iterator
        if (self._gen_next is None or
            isinstance(self._gen_next , types.GeneratorType)):
            return None     
        return self._gen_next.class_mode

    @property
    def nb_sample(self):
        """int: number of samples registered at core iterator/generator."""
        # This property is only supported by the core iterator
        if (self._gen_next is None or
            isinstance(self._gen_next , types.GeneratorType)):
            return None
        return self._gen_next.nb_sample

    @property
    def nb_class(self):
        """int: number of classes registered at core iterator/generator."""
        # This property is only supported by the core iterator
        if (self._gen_next is None or
            isinstance(self._gen_next , types.GeneratorType)):
            return None     
        return self._gen_next.nb_class

    @property
    def image_shape(self):
        """:obj:`tuple` : shape of the image that is natively producted by this iterator."""
        # This property is only supported by the core iterator      
        if (self._gen_next is None or
            isinstance(self._gen_next , types.GeneratorType)):
            return None  
        return self._gen_next.image_shape

    @property
    def dim_ordering(self):
        """:obj:`tuple` : shape of the image that is natively producted by this iterator."""
        # This property is only supported by the core iterator      
        if (self._gen_next is None or
            isinstance(self._gen_next , types.GeneratorType)):
            return None  
        return self._gen_next.dim_ordering

    ##########################################################################
    # Protected Interface
    ##########################################################################
    @abstractmethod
    def clone(self):
        """Create a copy of this object."""
        pass

    def _release(self):
        """Release internal resources used by the iterator."""
        self._imdata_pp = None
        self._gen_next = None

    ##########################################################################
    # Special Interface
    ##########################################################################
    def __iter__(self):
        """Python iterator interface required method."""
        return self

    def __next__(self):
        """Python iterator interface required method."""
        return next(self._gen_next)

    ##########################################################################
    # Dependant Properties
    ##########################################################################
    @property
    def params(self):
        """:obj:`dict`: Core iterator parameters."""
        if (self._params is None):
            return {}
        return self._params