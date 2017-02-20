# -*- coding: utf-8 -*- TODO: CHECK COMMENTS
"""
.. module:: PreLoadedDb
   :platform: Unix, Windows
   :synopsis: Represent PreLoadedDb class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from abc import ABCMeta, abstractmethod
from keras import backend as K

# Local Imports

class PreLoadedDb (object):
    """PreLoadedDb represents base class for preloaded databases.

    .. warning:: abstract class and must not be instantiated.
    """
    __metaclass__ = ABCMeta

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self):
        self.dim_ordering = None

    def reinit(self, dim_ordering='default'):
        """Initialize the `PreLoadedDb` instance.
    
            Invoked by the NNF framework
        """
        if (self.dim_ordering is not None): return True
        self.dim_ordering = K.image_dim_ordering()\
                                if (dim_ordering == 'default')\
                                            else dim_ordering
        return False

    @abstractmethod
    def get_input_shape(self):
        pass

    @abstractmethod
    def get_nb_class(self):
        pass

    @abstractmethod
    def LoadPreTrDb(self, nnmodel):
        pass

    @abstractmethod
    def LoadTrDb(self, nnmodel):
        pass

    @abstractmethod
    def LoadTeDb(self, nnmodel):
        pass

    @abstractmethod
    def LoadPredDb(self, nnmodel):
        pass