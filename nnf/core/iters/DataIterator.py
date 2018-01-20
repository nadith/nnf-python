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
from nnf.core.iters.ImageDataPreProcessor import *

class DataIterator(object):
    """DataIterator represents the base class for all iterators in the
        Neural Network Framework.

    .. warning:: abstract class and must not be instantiated.

    Attributes
    ----------
    _imdata_pp : :obj:`ImageDataPreProcessor`
        Image data pre-processor for all iterators.

    _gen_next : `function`
        Core iterator for non DskmanIterator(s) / Generator for DskmanIterator(s), that provide data.

    _params : :obj:`dict`
        Core iterator/generator parameters. (Default value = None)

    _pp_params : :obj:`dict`, optional
        Pre-processing parameters for :obj:`ImageDataPreProcessor`.
        (Default value = None).

    _fn_gen_coreiter : `function`, optional
        Factory method to create the core iterator. (Default value = None).

    edataset : :obj:`Dataset`
        Dataset enumeration key.

    Notes
    -----
    Diskman data iterators are utilizing a generator function in _gen_next
    while the `nnmodel` data iterators are utilizing an core iterator.
    """
    __metaclass__ = ABCMeta

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, pp_params=None, fn_gen_coreiter=None, edataset=None):
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
        
        # Core iterator or generator (initialized in init_ex())
        # Diskman data iterators are utilizing a generator function with yield in _gen_next
        # while the `nnmodel` data iterators are utilizing an core iterator.
        self._gen_next = None

        # Iterator params
        # Used by `nnmodel` data iterators only.
        # Can utilize in Diskman data iterators safely in the future.
        self._params = None

        # All the parameters are saved in instance variables to support the 
        # clone() method implementation of the child classes.
        self._pp_params = pp_params
        self._fn_gen_coreiter = fn_gen_coreiter

        # Used by `nnmodel` data iterators only.
        self.edataset = edataset

    def init(self, gen_next=None, params=None):
        """Initialize the instance.

        Parameters
        ----------
        _gen_next : `function`
            Core iterator/generator that provides data.
        """        
        # Set the core iterator or generator 
        self._gen_next = gen_next

        # Set iterator params
        self._params = params

    ##########################################################################
    # Public: Core Iterator Only Operations/Dependant Properties
    ##########################################################################
    def initiate_parallel_operations(self):
        # This property is only supported by the core iterator
        if not self.__is_core_iter: return False
        self._gen_next.initiate_parallel_operations()

    def terminate_parallel_operations(self):
        # This property is only supported by the core iterator
        if not self.__is_core_iter: return False
        self._gen_next.terminate_parallel_operations()

    def set_shuffle(self, shuffle):
        """Set shuffle property."""
        # This method is only supported by the core iterator
        if not self.__is_core_iter: return False

        if (self._gen_next.batch_index != 0 and 
            self._gen_next.shuffle != shuffle):
            raise Exception("Iterator is already active and failed to set shuffle")

        # Update iterator params if available
        if (self._params is not None):
            self._params['shuffle'] = shuffle

        self._gen_next.shuffle = shuffle
        return True

    def set_target_duplicator(self, duplicator):
        """Set target duplicator property."""
        # This method is only supported by the core iterator
        if not self.__is_core_iter: return False

        self._gen_next.target_duplicator = duplicator
        return True

    def sync_generator(self, gen_next):
        """Sync the secondary iterator with this iterator.

        Sync the secondary core iterator with this core iterator internally.
        Used when data needs to be generated with its matching target.

        Parameters
        ----------
        gen : :obj:`DirectoryIterator` or :obj:`NumpyArrayIterator` 
            Core iterator that needs to be synced with this core iterator
            `_gen_next`.
        """
        # This method is only supported by the core iterator      
        if not self.__is_core_iter: return False

        if (gen_next is None or
            isinstance(gen_next , types.GeneratorType)): 
            return False

        self._gen_next.sync_generator(gen_next._gen_next)
        return True

    def sync_tgt_generator(self, tgt_gen_next):
        """Sync the target iterator with this iterator.

        Sync the secondary core iterator with this core iterator internally.
        Used when data needs to be generated with its matching target.

        Parameters
        ----------
        gen : :obj:`DirectoryIterator` or :obj:`NumpyArrayIterator`
            Core iterator that needs to be synced with this core iterator
            `_gen_next`.
        """
        # This method is only supported by the core iterator
        if not self.__is_core_iter: return False

        if (tgt_gen_next is None or
            isinstance(tgt_gen_next , types.GeneratorType)):
            return False

        self._gen_next.sync_tgt_generator(tgt_gen_next._gen_next)
        return True

    def reset(self, gen_next):
        """Reset the iterator to the beginning."""
        # This method is only supported by the core iterator      
        if not self.__is_core_iter: return False
        self._gen_next.reset()
        return True

    def set_batch_size(self, batch_size):
        """Set the batch size."""
        # This method is only supported by the core iterator
        if not self.__is_core_iter: return False
        self._gen_next.batch_size = batch_size
        return True

    def release(self):
        """Release internal resources used by the iterator."""
        self._imdata_pp = None
        self._release_core_iter()
        del self._params
        del self._pp_params
        self._fn_gen_coreiter = None
        self.edataset = None

    def new_target_container(self):
        """Get an empty container to collect the targets. Used in NNModel.Predict"""
        return self._gen_next.new_target_container() if self.__is_core_iter else None

    ##########################################################################
    # Protected Interface
    ##########################################################################
    @abstractmethod
    def clone(self):
        """Create a copy of this DataIterator object."""
        pass

    def _release_core_iter(self):
        if self.__is_core_iter:
            self._gen_next.release()
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
    # Dependant Properties (public)
    ##########################################################################
    @property
    def has_multiple_inputs(self):
        """bool: whether this iterator provides multiple inputs."""
        return len(self._input_generators) > 1 if self.__is_core_iter else None

    @property
    def has_multiple_targets(self):
        """bool: whether this iterator provides multiple targets."""
        return len(self._target_generators) > 1 if self.__is_core_iter else None

    @property
    def input_vectorized(self):
        """bool: whether the input needs to be vectorized via the core iterator."""
        return self._gen_next.input_vectorized if self.__is_core_iter else None

    @property
    def batch_size(self):
        """int: batch size to be read by the core iterator."""
        return self._gen_next.batch_size if self.__is_core_iter else None

    @property
    def class_mode(self):
        """str: class mode at core iterator."""
        return self._gen_next.class_mode if self.__is_core_iter else None

    @property
    def nb_sample(self):
        """int: number of samples registered at core iterator/generator."""
        return self._gen_next.nb_sample if self.__is_core_iter else None

    # @property
    # def nb_class(self):
    #     """int: number of classes registered at core iterator/generator."""
    #     return self._gen_next.nb_class if self.__is_core_iter else None

    @property
    def input_shapes(self):
        """:obj:`tuple` : shape of the image that is requested by the user. (Default = input_shape)."""
        return self._gen_next.input_shapes if self.__is_core_iter else None

    @property
    def output_shapes(self):
        """List: shapes of the outputs expected."""
        return self._gen_next.output_shapes if self.__is_core_iter else None

    @property
    def data_format(self):
        """:obj:`tuple` : shape of the image that is natively producted by this iterator."""
        return self._gen_next.data_format if self.__is_core_iter else None

    @property
    def params(self):
        """:obj:`dict`: Core iterator parameters."""
        if (self._params is None):
            return {}
        return self._params

    @property
    def core_iter(self):
        """:obj:`Iterator` : core iterator object."""
        if self.__is_core_iter:
            assert self._gen_next == self._input_generators[0]
            return self._gen_next
        else:
            return None

    ##########################################################################
    # Dependant Properties (protected)
    ##########################################################################
    @property
    def _input_generators(self):
        return self._gen_next.input_generators if self.__is_core_iter else None

    @property
    def _target_generators(self):
        return self._gen_next.target_generators if self.__is_core_iter else None

    ##########################################################################
    # Dependant Properties (private)
    ##########################################################################
    @property
    def __is_core_iter(self):
        # Whether this is a core iterator or not.
        if (self._gen_next is None or
            isinstance(self._gen_next , types.GeneratorType)):
            return False
        return True
