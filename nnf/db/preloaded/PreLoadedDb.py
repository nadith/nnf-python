# -*- coding: utf-8 -*-
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

    Attributes
    ----------
    dim_ordering : str
        'th' or 'tf'. In 'th' mode, the channels dimension
        (the depth) is at index 1, in 'tf' mode it is at index 3.
        It defaults to the `image_dim_ordering` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "th".
    """
    __metaclass__ = ABCMeta

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self):
        """Constructor of the abstract class :obj:`PreLoadedDb`."""
        self.dim_ordering = None

    def reinit(self, dim_ordering='default'):
        """Initialize the `PreLoadedDb` instance.

        Parameters
        ----------
        dim_ordering : str
            'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".

        Returns
        -------
        bool
            True if already initialized. False otherwise.

        Note
        ----
        This method may be invoked multiple times by the NNF.
        """
        if (self.dim_ordering is not None): return True
        self.dim_ordering = K.image_dim_ordering()\
                                if (dim_ordering == 'default')\
                                            else dim_ordering
        return False

    @abstractmethod
    def get_input_shape(self, nnmodel):
        """Fetch the size of a single image/data sample

        Parameters
        ----------
        nnmodel : :obj:`NNModel`
            The `NNModel` that invokes this method.

        Returns
        -------
        :obj:`tuple` :
            Indicates (height, width, ch) or (height, width, ch)
            depending on the `self.dim_ordering` property.            
        """
        pass

    @abstractmethod
    def get_nb_class(self):
        """Fetch the number of classes for the database.
    
        Returns
        -------
        int :
            Number of classes.
        """
        pass

    @abstractmethod
    def LoadPreTrDb(self, nnmodel):
        """Load the pre-training dataset.

        Parameters
        ----------
        nnmodel : :obj:`NNModel`
            The `NNModel` that invokes this method.

        Returns
        -------
        :obj:`tuple`
            X_L = (`array_like` data tensor, labels)
            If the `nnmodel` is not expecting labels, set it to None.
        
        `array_like`
            Xt = target data tensor
            If the `nnmodel` is not expecting a target data tensor, 
            set it to None.

        :obj:`tuple`
            X_L_val = (`array_like` validation data tensor, labels)
            If the `nnmodel` is not expecting labels, set it to None.

        `array_like`
            Xt_val = validation target data tensor
            If the `nnmodel` is not expecting a validation target data tensor, 
            set it to None.

        Notes
        -----
        Depending on the model, the labels need to be categorial or in a vector 
        arrangement.
        """
        pass

    @abstractmethod
    def LoadTrDb(self, nnmodel):
        """Load the training dataset.

        Parameters
        ----------
        nnmodel : :obj:`NNModel`
            The `NNModel` that invokes this method.

        Returns
        -------
        :obj:`tuple`
            X_L = (`array_like` data tensor, labels)
            If the `nnmodel` is not expecting labels, set it to None.
        
        `array_like`
            Xt = target data tensor
            If the `nnmodel` is not expecting a target data tensor, 
            set it to None.

        :obj:`tuple`
            X_L_val = (`array_like` validation data tensor, labels)
            If the `nnmodel` is not expecting labels, set it to None.

        `array_like`
            Xt_val = validation target data tensor
            If the `nnmodel` is not expecting a validation target data tensor, 
            set it to None.

        Notes
        -----
        Depending on the model, the labels need to be categorial or in a vector 
        arrangement.
        """
        pass

    @abstractmethod
    def LoadTeDb(self, nnmodel):
        """Load the testing dataset.

        Parameters
        ----------
        nnmodel : :obj:`NNModel`
            The `NNModel` that invokes this method.

        Returns
        -------
        :obj:`tuple`
            X_L = (`array_like` data tensor, labels)
            If the `nnmodel` is not expecting labels, set it to None.
        
        `array_like`
            Xt = target data tensor
            If the `nnmodel` is not expecting a target data tensor, 
            set it to None.

        :obj:`tuple`
            X_L_val = (`array_like` validation data tensor, labels)
            If the `nnmodel` is not expecting labels, set it to None.

        `array_like`
            Xt_val = validation target data tensor
            If the `nnmodel` is not expecting a validation target data tensor, 
            set it to None.

        Notes
        -----
        Depending on the model, the labels need to be categorial or in a vector 
        arrangement.
        """
        pass

    @abstractmethod
    def LoadPredDb(self, nnmodel):
        """Load the dataset for predictions.

        Parameters
        ----------
        nnmodel : :obj:`NNModel`
            The `NNModel` that invokes this method.

        Returns
        -------
        :obj:`tuple`
            X_L = (`array_like` data tensor, labels)
            If the `nnmodel` is not expecting labels, set it to None.
        
        `array_like`
            Xt = target data tensor
            If the `nnmodel` is not expecting a target data tensor, 
            set it to None.

        :obj:`tuple`
            X_L_val = (`array_like` validation data tensor, labels)
            If the `nnmodel` is not expecting labels, set it to None.

        `array_like`
            Xt_val = validation target data tensor
            If the `nnmodel` is not expecting a validation target data tensor, 
            set it to None.

        Notes
        -----
        Depending on the model, the labels need to be categorial or in a vector 
        arrangement.
        """
        pass