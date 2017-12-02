# -*- coding: utf-8 -*-
"""
.. module:: MnistDb
   :platform: Unix, Windows
   :synopsis: Represent MnistDb class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from abc import ABCMeta, abstractmethod
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# Local Imports
from nnf.db.preloaded.PreLoadedDb import PreLoadedDb
from nnf.core.models.Autoencoder import Autoencoder
from nnf.core.models.CNNModel import CNNModel
from nnf.core.models.DAEModel import DAEModel
from nnf.core.models.VGG16Model import VGG16Model
from nnf.core.models.DAERegModel import DAERegModel

class MnistDb(PreLoadedDb):
    """MnistDb represents MNIST handwritten digit database."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, filepath, debug=False):
        """Initialize the `PreLoadedDb` instance.
    
            Invoked by the NNF.
        """
        super().__init__()
        self.filepath = filepath
        self.debug = debug

        self.tr_n = 0
        self.X = None
        self.Xte = None

        self.X_lbl = None
        self.Xte_lbl = None

    ##########################################################################
    # Public: PreLoadedDb Overrides
    ##########################################################################
    def reinit(self, data_format=None):
        """Initialize the `PreLoadedDb` instance.

        Parameters
        ----------
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

        Note
        ----
        This method may be invoked multiple times by the NNF.
        """
        already_init = super().reinit(data_format)
        if (already_init): return  # PERF

        (X, X_lbl), (Xte, Xte_lbl) = mnist.load_data(self.filepath)

        X = X.astype('float32') / 255.        
        Xte = Xte.astype('float32') / 255.
        X_lbl = X_lbl.astype('float32')
        Xte_lbl = Xte_lbl.astype('float32')

        self.tr_n = len(X)
        self.X = X
        self.Xte = Xte

        self.X_lbl = X_lbl
        self.Xte_lbl = Xte_lbl

        print ("MNIST Database Loaded: Tr.Shape: " + str(X.shape) +
                                                " Te.Shape: " + str(Xte.shape))

        # Debug data extract
        if (self.debug):
            self.tr_n = 10
            self.X = self.X[0:10]
            self.Xte = self.Xte[0:10]
            self.X_lbl = self.X_lbl[0:10]
            self.Xte_lbl = self.Xte_lbl[0:10]

    def get_input_shape(self, nnmodel):
        """Fetch the size of a single image/data sample
    
        Returns
        -------
        :obj:`tuple` :
            Indicates (height, width, ch) or (height, width, ch)
            depending on the `self.data_format` property.            
        """
        target_shape = (28, 28)
        if self.data_format == 'channels_last':
            input_shape =  target_shape + (1,)
        else:
            input_shape = (1,) + target_shape

        return input_shape

    def get_nb_class(self):
        return 10        

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
        tr_offset = int(0.8*self.tr_n)
        val_offset = int(0.2*self.tr_n)
        X = self.X[:tr_offset]
        Xval = self.X[tr_offset:tr_offset + val_offset]
        X_lbl = self.X_lbl[:tr_offset]
        Xval_lbl= self.X_lbl[tr_offset:tr_offset + val_offset]

        if (isinstance(nnmodel, DAERegModel) or
            isinstance(nnmodel, DAEModel)):

            # Vectorize the tr. input/target
            db = np.reshape(X, (len(X), np.prod(X.shape[1:])))
            X_L = (db, None)  # Model expects no label information

            # Vectorize the val. input/target
            db_val = np.reshape(Xval, (len(Xval), np.prod(Xval.shape[1:])))
            X_L_val = (db_val, None)  # Model expects no label information

            return X_L, db, X_L_val, db_val

        else:
            raise Exception("NNModel is not supported")

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
        Depending on the model, the labels need to be categorical or in a vector
        arrangement.
        """
        tr_offset = int(0.8*self.tr_n)
        val_offset = int(0.2*self.tr_n)
        X = self.X[:tr_offset]
        Xval = self.X[tr_offset:tr_offset + val_offset]
        X_lbl = self.X_lbl[:tr_offset]
        Xval_lbl= self.X_lbl[tr_offset:tr_offset + val_offset]

        if (isinstance(nnmodel, Autoencoder) or 
            isinstance(nnmodel, DAERegModel) or 
            isinstance(nnmodel, DAEModel)):

            # Vectorize the tr. input/target
            db = np.reshape(X, (len(X), np.prod(X.shape[1:])))
            X_L = (db, None)  # Model expects no label information

            # Vectorize the val. input/target
            db_val = np.reshape(Xval, (len(Xval), np.prod(Xval.shape[1:])))
            X_L_val = (db_val, None)  # Model expects no label information

            return X_L, db, X_L_val, db_val

        # Child model should be checked first before parent model
        elif (isinstance(nnmodel, VGG16Model)):
            # Original VGG expects 1000 class, color (ch=3) database
            assert(False)  

        elif (isinstance(nnmodel, CNNModel)):
            # Fix categorical labels for CNN
            nb_class = self.get_nb_class()
            X_L = (self.__process(X), np_utils.to_categorical(X_lbl, nb_class).astype('float32'))
            X_L_val = (self.__process(Xval), np_utils.to_categorical(Xval_lbl, nb_class).astype('float32'))

            # Model expects no target information
            return X_L, None, X_L_val, None

    def LoadTeDb(self, nnmodel):
        """Load the testing dataset.

        Parameters
        ----------
        nnmodel : :obj:`NNModel`
            The `NNModel` that invokes this method.

        Returns
        -------
        :obj:`tuple` :
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
        Xte = self.Xte
        Xte_lbl = self.Xte_lbl

        if (isinstance(nnmodel, Autoencoder) or 
            isinstance(nnmodel, DAERegModel) or 
            isinstance(nnmodel, DAEModel)):

            # Vectorize the pred. input/target
            db = np.reshape(Xte, (len(Xte), np.prod(Xte.shape[1:])))            
            X_L_te = (db, None)  # Model expects no label information
            return X_L_te, db

        # Child model should be checked first before parent model
        elif (isinstance(nnmodel, VGG16Model)):
            # Original VGG expects 1000 class, color (ch=3) database
            assert(False)  

        elif (isinstance(nnmodel, CNNModel)):
            # Fix categorical labels for CNN
            nb_class = self.get_nb_class()
            X_L_te = (self.__process(Xte), np_utils.to_categorical(Xte_lbl, nb_class).astype('float32'))

            # Model expects no target information
            return X_L_te, None

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
        Xte = self.Xte
        Xte_lbl = self.Xte_lbl

        if (isinstance(nnmodel, Autoencoder) or 
            isinstance(nnmodel, DAERegModel) or 
            isinstance(nnmodel, DAEModel)):

            # Vectorize the pred. input/target
            db = np.reshape(Xte, (len(Xte), np.prod(Xte.shape[1:])))

            # Model expects no label information
            X_L_te = (db, None)
            return X_L_te, db

        # Child model should be checked first before parent model
        elif (isinstance(nnmodel, VGG16Model)):
            # Original VGG expects 1000 class, color (ch=3) database
            assert(False)  

        # Child model should be checked first before parent model
        elif (isinstance(nnmodel, VGG16Model)):
            # Original VGG expects 1000 class, color (ch=3) database
            assert(False)  

        elif (isinstance(nnmodel, CNNModel)):
            # Fix categorical labels for CNN
            nb_class = self.get_nb_class()
            X_L_te = (self.__process(Xte), np_utils.to_categorical(Xte_lbl, nb_class).astype('float32'))

            # Model expects no target information
            return X_L_te, None

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __process(self, X):
        """Process the data tensor according to `data_format` specified."""
        if self.data_format == 'channels_last':
            X = X[:, :, :, np.newaxis]
        else:
            X = X[:, np.newaxis, :, :]

        return X