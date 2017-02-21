# -*- coding: utf-8 -*-
"""
.. module:: CNNModel
   :platform: Unix, Windows
   :synopsis: Represent CNNModel class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from warnings import warn as warning
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
#from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,RMSprop,adam

# Local Imports
from nnf.db.NNdb import NNdb
from nnf.db.Format import Format
from nnf.db.Dataset import Dataset
from nnf.core.models.NNModel import NNModel
from nnf.core.models.Autoencoder import Autoencoder
from nnf.core.models.NNModelPhase import NNModelPhase
from nnf.core.iters.DataIterator import DataIterator
from nnf.core.iters.memory.MemDataIterator import MemDataIterator
from nnf.core.iters.disk.DskDataIterator import DskDataIterator

class CNNModel(NNModel):
    """Generic Convolutional Neural Network Model.

    Attributes
    ----------
    callbacks : :obj:`dict`
        Callback dictionary. Supported callbacks.
        {`test`, `predict`, `get_data_generators`}

    X_L : :obj:`tuple`
        In the format (`array_like` data tensor, labels).
        If the `nnmodel` is not expecting labels, set it to None.

    Xt : `array_like`
        Target data tensor.
        If the `nnmodel` is not expecting a target data tensor,
        set it to None.

    X_L_val : :obj:`tuple`
        In the format (`array_like` validation data tensor, labels).
        If the `nnmodel` is not expecting labels, set it to None.

    Xt_val : `array_like`
        Validation target data tensor.
        If the `nnmodel` is not expecting a validation target data tensor,
        set it to None.

    Note
    ----
    Extend this class to implement custom CNN models.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, X_L=None, Xt=None, X_L_val=None, Xt_val=None,
                                                            callbacks=None):
        """Constructs :obj:`CNNModel` instance."""
        super().__init__()

        # Set defaults for arguments
        self.callbacks = {} if (callbacks is None) else callbacks
        self.callbacks.setdefault('test', None)
        self.callbacks.setdefault('predict', None)
        get_dat_gen = self.callbacks.setdefault('get_data_generators', None)
        if (get_dat_gen is None):
            self.callbacks['get_data_generators'] = self.get_data_generators

        # Used when data is fetched from no iterators
        self.X_L = X_L          # (X, labels)
        self.Xt = Xt            # X target
        self.X_L_val = X_L_val  # (X_val, labels_val)
        self.Xt_val = Xt_val    # X_val target
        pass      

    def pre_train(self, precfgs, cfg, patch_idx=None):
        """Pre-train the :obj:`CNNModel`.

        .. warning:: Not supported.

        Parameters
        ----------
        precfgs : list of :obj:`NNCfg`
            List of Neural Network configurations. Useful for layer-wise pre-training.

        cfg : :obj:`NNCfg`
            Neural Network configuration that will be used in training. 
            Useful to build the deep stacked network after layer-wise pre-trianing.

        patch_idx : int
            Patch's index in this model.
        """
        warning('Pre-training is not supported in CNN')

    def get_data_generators(self, ephase, list_iterstore, dict_iterstore):
        """Get data generators for pre-training, training, testing, prediction.

        Parameters
        ----------
        ephase : :obj:`NNModelPhase`
            Phase of which the data generators are required.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictonary of iterstores for :obj:`DataIterator`.

        Returns
        -------
        :obj:`tuple`
            When ephase == NNModelPhase.PRE_TRAIN or NNModelPhase.TRAIN
            then 
                Generators for training and validation.
                Refer https://keras.io/preprocessing/image/

            When ephase == NNModelPhase.TEST or NNModelPhase.PREDICT
            then
                Generators for testing and testing target.
        """
        if (ephase == NNModelPhase.TRAIN):
            # Iteratorstore for dbparam1
            X1_gen = list_iterstore[0].setdefault(Dataset.TR, None)
            X2_gen = list_iterstore[0].setdefault(Dataset.VAL, None)
    
        elif (ephase == NNModelPhase.TEST or ephase == NNModelPhase.PREDICT):
            # Iteratorstore for dbparam1
            X1_gen = list_iterstore[0].setdefault(Dataset.TE, None)
            X2_gen = list_iterstore[0].setdefault(Dataset.TE_OUT, None)

        else:
            raise Exception('Unsupported NNModelPhase')

        return X1_gen, X2_gen

    ##########################################################################
    # Protected: NNModel Overrides
    ##########################################################################
    def _train(self, cfg, patch_idx=None, dbparam_save_dirs=None, 
                                    list_iterstore=None, dict_iterstore=None):
        """Train the :obj:`CNNModel`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int
            Patch's index in this model.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of 
            each `nnpatch`.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictonary of iterstores for :obj:`DataIterator`.
        """    
        # Initialize data generators
        X_gen, X_val_gen = self._init_data_generators(NNModelPhase.TRAIN,
                                                        list_iterstore,
                                                        dict_iterstore)

        # Pre-condition asserts
        assert((cfg.preloaded_db is None and X_gen is not None) or 
                (cfg.preloaded_db is not None))

        if (X_val_gen is not None):
            # No. of training and validation classes must be equal
            assert(X_gen.nb_class == X_val_gen.nb_class)

            # Should be labeled samples
            assert(not (X_gen.nb_class is None or X_val_gen.nb_class is None))

        # Build the CNN
        self._build(cfg, X_gen)

        # Preloaded databases for quick deployment
        if ((cfg.preloaded_db is not None) and
            (self.X_L is None and self.X_L_val is None and
            self.Xt is None and self.Xt_val is None)):
            self.X_L, self.Xt, self.X_L_val, self.Xt_val = cfg.preloaded_db.LoadTrDb(self)

        # Training without generators
        if (cfg.preloaded_db is not None):  
            super()._start_train(cfg, self.X_L, self.Xt, self.X_L_val, self.Xt_val)
            return (self.X_L, self.Xt, self.X_L_val, self.Xt_val)

        # Training with generators
        super()._start_train(cfg, X_gen=X_gen, X_val_gen=X_val_gen)

        # Save the trained model
        self._try_save(cfg, patch_idx, "CNN")

        return (None, None, None, None)

    def _test(self, cfg, patch_idx=None, dbparam_save_dirs=None, 
                                    list_iterstore=None, dict_iterstore=None):
        """Test the :obj:`CNNModel`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int
            Patch's index in this model.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of 
            each `nnpatch`.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictonary of iterstores for :obj:`DataIterator`.
        """    
        # Initialize data generators
        Xte_gen, Xte_target_gen = self._init_data_generators(
                                                    NNModelPhase.TEST,
                                                    list_iterstore,
                                                    dict_iterstore)
        # Pre-condition asserts
        assert((cfg.preloaded_db is None and Xte_gen is not None) or 
                (cfg.preloaded_db is not None))

        if (Xte_target_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(Xte_gen.nb_sample == Xte_target_gen.nb_sample)

        # 1st Priority, Load the saved model with weights
        is_loaded = self._try_load(cfg, patch_idx, "CNN")

        # 2nd Priority, Load the saved weights but not the model
        if (not is_loaded and cfg.weights_path is not None):
            assert(False)
            #self._build(cfg, Xte_gen) # TODO: X_gen.nb_class is used. Since it needs to follow training phase
            #self.net.load_weights(daecfg.weights_path) 

        assert(self.net is not None)

        # Preloaded databases for quick deployment
        if (cfg.preloaded_db is not None):
            cfg.preloaded_db.reinit('default')
            X_L_te, Xt_te = cfg.preloaded_db.LoadTeDb(self)     

        # Test without generators
        if (cfg.preloaded_db is not None):
            super()._start_test(patch_idx, X_L_te, Xt_te)
            return

        # Test with generators
        super()._start_test(patch_idx,
                            Xte_gen=Xte_gen,
                            Xte_target_gen=Xte_target_gen)

    def _predict(self, cfg, patch_idx=None, dbparam_save_dirs=None, 
                                    list_iterstore=None, dict_iterstore=None):
        """Predict using :obj:`CNNModel`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int
            Patch's index in this model.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of
            each `nnpatch`.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictonary of iterstores for :obj:`DataIterator`.
        """
        # Initialize data generators
        Xte_gen, Xte_target_gen = self._init_data_generators(
                                                NNModelPhase.PREDICT, 
                                                list_iterstore, 
                                                dict_iterstore)
        # Pre-condition asserts
        assert((cfg.preloaded_db is None and Xte_gen is not None) or 
                (cfg.preloaded_db is not None))

        if (Xte_target_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(Xte_gen.nb_sample == Xte_target_gen.nb_sample)

        # 1st Priority, Load the saved model with weights
        is_loaded = self._try_load(cfg, patch_idx, "CNN")

        # 2nd Priority, Load the saved weights but not the model
        if (not is_loaded and cfg.weights_path is not None):
            assert(False)
            #self._build(cfg, Xte_gen) # TODO: X_gen.nb_class is used. Since it needs to follow training phase
            #self.net.load_weights(daecfg.weights_path)

        assert(self.net is not None)

        # Preloaded databases for quick deployment
        if (cfg.preloaded_db is not None):
            cfg.preloaded_db.reinit('default')
            X_L_te, Xt_te = cfg.preloaded_db.LoadPredDb(self)    

        # Predict without generators
        if (cfg.preloaded_db is not None):
            super()._start_predict(patch_idx, X_L_te, Xt_te)
            return

        # Predict with generators
        super()._start_predict(patch_idx, 
                                Xte_gen=Xte_gen, 
                                Xte_target_gen=Xte_target_gen)

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _init_data_generators(self, ephase, list_iterstore, dict_iterstore):
        """Initialize data generators for pre-training, training, testing,
            prediction.

        Parameters
        ----------
        ephase : :obj:`NNModelPhase`
            Phase of which the data generators are required.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictonary of iterstores for :obj:`DataIterator`.

        Returns
        -------
        :obj:`tuple`
            When ephase == NNModelPhase.PRE_TRAIN or NNModelPhase.TRAIN
            then 
                Generators for training and validation.

            When ephase == NNModelPhase.TEST or NNModelPhase.PREDICT
            then
                Generators for testing and testing target.
        """
        X_gen = None; X_val_gen = None
        if (list_iterstore is not None):
            X_gen, X_val_gen = self.callbacks['get_data_generators'](ephase, list_iterstore, dict_iterstore)

        return X_gen, X_val_gen

    def _build(self, cfg, X_gen):
        """Build the keras CNN."""
        if (cfg.preloaded_db is not None):
            cfg.preloaded_db.reinit('default')
            input_shape = cfg.preloaded_db.get_input_shape()
            nb_class = cfg.preloaded_db.get_nb_class()
        else:
            input_shape = X_gen.image_shape # <- 'tf' format (default)
            nb_class = X_gen.nb_class

        self.net = Sequential()

        ## input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
        ## this applies 32 convolution filters of size 3x3 each.
        #self.net.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
        #self.net.add(Activation('relu'))
        #self.net.add(Convolution2D(32, 3, 3))
        #self.net.add(Activation('relu'))
        #self.net.add(MaxPooling2D(pool_size=(2, 2)))
        #self.net.add(Dropout(0.25))

        #self.net.add(Convolution2D(64, 3, 3, border_mode='valid'))
        #self.net.add(Activation('relu'))
        #self.net.add(Convolution2D(64, 3, 3))
        #self.net.add(Activation('relu'))
        #self.net.add(MaxPooling2D(pool_size=(2, 2)))
        #self.net.add(Dropout(0.25))

        #self.net.add(Flatten())
        ## Note: Keras does automatic shape inference.
        #self.net.add(Dense(256))
        #self.net.add(Activation('relu'))
        #self.net.add(Dropout(0.5))

        #self.net.add(Dense(nb_class))
        #self.net.add(Activation('softmax'))

        #sgd = SGD(lr=0.01, decay=1e-8, momentum=0.01, nesterov=True)
        #self.net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        #self.fn_predict_feature = K.function([self.net.layers[0].input, K.learning_phase()],
        #                          [self.net.layers[13].output]) 


        # input: 150x150 images with 3 channels -> (3, 100, 100) tensors.

        self.net = Sequential()
        self.net.add(Convolution2D(32, 3, 3, input_shape=input_shape))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))

        self.net.add(Convolution2D(32, 3, 3))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))

        self.net.add(Convolution2D(64, 3, 3))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))

        self.net.add(Flatten())
        #self.net.add(Dense(64))
        self.net.add(Dense(256))
        self.net.add(Activation('relu'))
        self.net.add(Dropout(0.5))
        self.net.add(Dense(nb_class))
        self.net.add(Activation('softmax'))

        print(self.net.summary())
        self.net.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        self._init_fns_predict_feature(cfg)      

        #rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        #self.net.compile(loss='categorical_crossentropy',
        #              optimizer=rms,
        #              metrics=['accuracy'])

        #sgd = SGD(lr=0.5, decay=1e-6, momentum=0.1, nesterov=True)
        #self.net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

