# -*- coding: utf-8 -*-
"""
.. module:: Cifar10Db
   :platform: Unix, Windows
   :synopsis: Represent Cifar10Db class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from abc import ABCMeta, abstractmethod
import os
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file

# Local Imports
from nnf.db.preloaded.PreLoadedDb import PreLoadedDb
from nnf.core.models.Autoencoder import Autoencoder
from nnf.core.models.CNNModel import CNNModel
from nnf.core.models.DAEModel import DAEModel
from nnf.core.models.DAERegModel import DAERegModel

class Cifar10Db(PreLoadedDb):
    """Cifar10Db represents CIFAR10 object database with 10 classes."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, dirpath=r'F:\#Research Data\keras\cifar-10-batches-py', 
                                                                debug=False):
        """Initialize the `PreLoadedDb` instance.
    
            Invoked by the NNF.
        """
        super().__init__()
        self.dirpath = dirpath
        self.debug = debug

        self.tr_n = 0
        self.X = None
        self.Xte = None

        self.X_lbl = None
        self.Xte_lbl = None

    ##########################################################################
    # Public: PreLoadedDb Overrides
    ##########################################################################
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

        Note
        ----
        This method may be invoked multiple times by the NNF.
        """
        already_init = super().reinit(dim_ordering)
        if (already_init): return  # PERF

        (X, X_lbl), (Xte, Xte_lbl) = self.__load_data(self.dirpath)

        X = X.astype('float32') / 255.        
        Xte = Xte.astype('float32') / 255.
        X_lbl = X_lbl.astype('float32')
        Xte_lbl = Xte_lbl.astype('float32')

        self.tr_n = len(X)
        self.X = X
        self.Xte = Xte

        self.X_lbl = X_lbl
        self.Xte_lbl = Xte_lbl

        print ("CIFAR10 Database Loaded: Tr.Shape: " + str(X.shape) +
                                            " Te.Shape: " + str(Xte.shape))

        # Debug data extract
        if (self.debug):
            self.tr_n = 10
            self.X = self.X[0:10]
            self.Xte = self.Xte[0:10]
            self.X_lbl = self.X_lbl[0:10]
            self.Xte_lbl = self.Xte_lbl[0:10]

    def get_input_shape(self):
        """Fetch the size of a single image/data sample.
    
        Returns
        -------
        :obj:`tuple` :
            Indicates (height, width, ch) or (height, width, ch)
            depending on the `self.dim_ordering` property.
        """
        target_shape = (32, 32)
        if self.dim_ordering == 'tf':
            input_shape =  target_shape + (3,)
        else:
            input_shape = (3,) + target_shape

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

        if (isinstance(nnmodel, DAEModel) or
            isinstance(nnmodel, DAERegModel)):

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
        Depending on the model, the labels need to be categorial or in a vector 
        arrangement.
        """
        tr_offset = int(0.8*self.tr_n)
        val_offset = int(0.2*self.tr_n)
        X = self.X[:tr_offset]
        Xval = self.X[tr_offset:tr_offset + val_offset]
        X_lbl = self.X_lbl[:tr_offset]
        Xval_lbl= self.X_lbl[tr_offset:tr_offset + val_offset]

        if (isinstance(nnmodel, Autoencoder) or 
            isinstance(nnmodel, DAEModel) or 
            isinstance(nnmodel, DAERegModel)):

            # Vectorize the tr. input/target
            db = np.reshape(X, (len(X), np.prod(X.shape[1:])))
            X_L = (db, None)  # Model expects no label information

            # Vectorize the val. input/target
            db_val = np.reshape(Xval, (len(Xval), np.prod(Xval.shape[1:])))
            X_L_val = (db_val, None)  # Model expects no label information

            return X_L, db, X_L_val, db_val

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
            isinstance(nnmodel, DAEModel) or 
            isinstance(nnmodel, DAERegModel)):

            # Vectorize the pred. input/target
            db = np.reshape(Xte, (len(Xte), np.prod(Xte.shape[1:])))            
            X_L_te = (db, None)  # Model expects no label information
            return X_L_te, db

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
            isinstance(nnmodel, DAEModel) or 
            isinstance(nnmodel, DAERegModel)):

            # Vectorize the pred. input/target
            db = np.reshape(Xte, (len(Xte), np.prod(Xte.shape[1:])))

            # Model expects no label information
            X_L_te = (db, None)
            return X_L_te, db

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
        """Process the data tensor according to `dim_ordering` specified."""
        
        # No need of further processing for CIFAR10, since it's already
        # processed in __load_data() method.
        return X

    def __load_data(self, dirname):
        """Refactored from keras.datasets.cifar10.load_data() method."""
        #dirname = "cifar-10-batches-py"
        origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        path = get_file(dirname, origin=origin, untar=True)

        nb_train_samples = 50000

        X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
        y_train = np.zeros((nb_train_samples,), dtype="uint8")

        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            data, labels = load_batch(fpath)
            X_train[(i - 1) * 10000: i * 10000, :, :, :] = data
            y_train[(i - 1) * 10000: i * 10000] = labels

        fpath = os.path.join(path, 'test_batch')
        X_test, y_test = load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        if K.image_dim_ordering() == 'tf':
            X_train = X_train.transpose(0, 2, 3, 1)
            X_test = X_test.transpose(0, 2, 3, 1)

        return (X_train, y_train), (X_test, y_test)