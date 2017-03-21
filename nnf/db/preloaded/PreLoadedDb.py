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
    data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
        (the depth) is at index 1, in 'channels_last' mode it is at index 3.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
    """
    __metaclass__ = ABCMeta

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self):
        """Constructor of the abstract class :obj:`PreLoadedDb`."""
        self.data_format = None

    def reinit(self, data_format=None):
        """Initialize the `PreLoadedDb` instance.

        Parameters
        ----------
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

        Returns
        -------
        bool
            True if already initialized. False otherwise.

        Note
        ----
        This method may be invoked multiple times by the NNF.
        """
        if (self.data_format is not None): return True
        self.data_format = K.image_data_format()\
                                if (data_format is None)\
                                            else data_format
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
            depending on the `self.data_format` property.            
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