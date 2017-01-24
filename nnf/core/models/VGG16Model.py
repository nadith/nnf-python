# -*- coding: utf-8 -*-
# Global Imports
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from warnings import warn as warning

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
#from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,RMSprop,adam


# Local Imports
from nnf.core.models.CNNModel import CNNModel
from nnf.core.models.NNModelPhase import NNModelPhase



class VGG16Model(CNNModel):
    """VGGNet16 Convolutional Neural Network Model.

    Notes
    -----
    ref: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, callbaks=None):
        """describe"""
        super().__init__(callbaks=callbaks)

    ##########################################################################
    # Protected: CNNModel Overrides
    ##########################################################################
    def _test(self, cfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
        """Test the :obj:`Autoencoder`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int
            Patch's index in this model.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of each `nnpatch`.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictonary of iterstores for :obj:`DataIterator`.
        """    
        # Initialize data generators
        Xte_gen, Xte_target_gen = self._init_data_generators(NNModelPhase.TEST, list_iterstore, dict_iterstore)

        # Pre-condition asserts
        assert((cfg.use_db is None and Xte_gen is not None) or 
                (cfg.use_db is not None))

        if (Xte_target_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(Xte_gen.nb_sample == Xte_target_gen.nb_sample)

        # 1st Priority, Load the saved model with weights
        if (cfg.load_dir is not None):
            fpath = self._get_saved_model_name(patch_idx, cfg.load_dir)
            self.net = load_model(fpath)
            if (cfg.weights_path is not None):
                warning('ARG_CONFLICT: Model weights will not be loaded from cfg.weights_path')

        # 2nd Priority, Load the saved weights but not the model
        elif (cfg.weights_path is not None):
            self._build(cfg, Xte_gen)
            self.net.load_weights(cfg.weights_path)

        assert(self.net is not None)

        # Preloaded databases for quick deployment
        if (cfg.use_db == 'mnist'):
            Xte_L, Xte_target = MnistAE.LoadTeDb() 

        # Test without generators
        if (cfg.use_db is not None):
            super()._start_test(patch_idx, Xte_L, Xte_target) 

        # Test with generators
        super()._start_test(patch_idx, Xte_gen=Xte_gen, Xte_target_gen=Xte_target_gen)

    def _predict(self, cfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
        """Predict the :obj:`Autoencoder`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int
            Patch's index in this model.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of each `nnpatch`.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictonary of iterstores for :obj:`DataIterator`.
        """    
        # Initialize data generators
        Xte_gen, Xte_target_gen = self._init_data_generators(NNModelPhase.PREDICT, list_iterstore, dict_iterstore)

        # Pre-condition asserts
        assert((cfg.use_db is None and Xte_gen is not None) or 
                (cfg.use_db is not None))

        if (Xte_target_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(Xte_gen.nb_sample == Xte_target_gen.nb_sample)

        # 1st Priority, Load the saved model with weights
        if (cfg.load_dir is not None):
            fpath = self._get_saved_model_name(patch_idx, cfg.load_dir)
            self.net = load_model(fpath)
            if (cfg.weights_path is not None):
                warning('ARG_CONFLICT: Model weights will not be loaded from cfg.weights_path')

        # 2nd Priority, Load the saved weights but not the model
        elif (cfg.weights_path is not None):
            self._build(cfg, Xte_gen)
            self.net.load_weights(cfg.weights_path)

        assert(self.net is not None)

        # Preloaded databases for quick deployment
        if (cfg.use_db == 'mnist'):
            Xte_L, _ = MnistAE.LoadTeDb() 

        # Test without generators
        if (cfg.use_db is not None):
            Xte =  Xte_L[0]
            super()._start_predict(patch_idx, Xte) 

        # Test with generators
        super()._start_predict(patch_idx, Xte_gen=Xte_gen)

    def _predict_feature_size(self):
        return 4096

    def _predict_feature(self, Xte):
        return self.fn_predict_feature([Xte, 0])[0]

    def _build(self, cfg, X_gen):

        if (cfg.use_db is not None):
            input_shape = MnistAE.image_shape
            nb_class = MnistAE.nb_class
        else:
            input_shape = X_gen.image_shape # <- 'tf' format (default)
            nb_class = X_gen.nb_class

        assert(X_gen.dim_ordering == 'th')  
        assert(input_shape == (3,224,224)) # and nb_class == 1000)                 

        self.net = Sequential()
        self.net.add(ZeroPadding2D((1,1),input_shape=input_shape))
        self.net.add(Convolution2D(64, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(64, 3, 3, activation='relu'))
        self.net.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(128, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(128, 3, 3, activation='relu'))
        self.net.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(256, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(256, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(256, 3, 3, activation='relu'))
        self.net.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.net.add(Flatten())
        self.net.add(Dense(4096, activation='relu'))
        self.net.add(Dropout(0.5))
        self.net.add(Dense(4096, activation='relu'))
        self.net.add(Dropout(0.5))
        self.net.add(Dense(1000, activation='softmax'))   
       
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.net.compile(optimizer=sgd, loss='categorical_crossentropy')

        self.fn_predict_feature = K.function([self.net.layers[0].input, K.learning_phase()],
                                  [self.net.layers[34].output]) 
      
