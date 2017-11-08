# -*- coding: utf-8 -*-
"""
.. module:: NNCfg
   :platform: Unix, Windows
   :synopsis: Represent NNPreCfg, NNCfg, DAEPreCfg, DAECfg classes.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from enum import Enum
from keras.callbacks import EarlyStopping

# Local Imports

class BaseCfg(object):
    """BaseCfg describes the pre-training configuration for neural network."""

    def __init__(self):
        self.loss_fn = 'mean_squared_error'
        self.optimizer = 'adadelta'

        # Only for pre-loaded dbs, when no data generators are used.
        # When data generators are used, batch_size is expressed in 
        # `iter_param`.
        self.batch_size = 1  
        self.shuffle = True
        
        # Only when data generators are used in training.
        self.validation_steps = 3
        self.steps_per_epoch = 5

        # For both pre-loaded dbs + data generators
        self.numepochs = 2
        self.callbacks = None
        #self.callbacks = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
        self.metrics = None

        self.preloaded_db = None        # `PreLoadedDb` instance
        self.feature_layers = [-1, -2]  # Used to predict features
        
        # PERF:
        self.model_dir = None   # Location to save/load compiled models
        self.weights_dir = None # Model is not saved, but weights

class CNNCfg(BaseCfg):
    """Training configuration for convolutional neural network."""

    def __init__(self):
        super().__init__()
        self.loss_fn = 'categorical_crossentropy'
        self.optimizer = 'rmsprop'
        #self.optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        #self.optimizer = SGD(lr=0.5, decay=1e-6, momentum=0.1, nesterov=True)
        self.metrics=['accuracy']

class VGG16Cfg(CNNCfg):
    """Training configuration for vgg16 convolutional neural network."""

    def __init__(self):
        super().__init__()

class DAEPreCfg(BaseCfg):
    """Pre-training configuration for each layer of the deep autoencoder network.

    Attributes
    ----------
    arch : list of int
        Architecture of the autoencoder.

    act_fns : list of :obj:`str`
        Activation functions for each layer at `arch`.

    Notes
    -----
    Some of the layers may not be pre-trianed. Hence precfgs itself is
    not sufficient to determine the architecture of the final 
    stacked network.
    """

    def __init__(self, arch=[1089, 784, 1089],
                        act_fns=['input', 'sigmoid', 'sigmoid'],
                        preloaded_db=None):
        super().__init__()
        self.arch = arch
        self.act_fns = act_fns
        self.preloaded_db = preloaded_db
     
class AECfg(DAEPreCfg):
    """Training configuration for simple autoencoder network."""

    def __init__(self, arch=[1089, 784, 1089],
                        act_fns=['input', 'sigmoid', 'sigmoid'],
                        preloaded_db=None):
        super().__init__(arch, act_fns, preloaded_db)

class DAECfg(DAEPreCfg):
    """Training configuration for deep autoencoder network."""

    def __init__(self, arch=[1089, 784, 512, 784, 1089], 
                        act_fns=['input', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],
                        preloaded_db=None):
        super().__init__(arch, act_fns, preloaded_db)

class DAERegCfg(DAEPreCfg):
    """Training configuration for deep regr. autoencoder network."""

    def __init__(self, arch=[1089, 784, 512, 256, 128, 1089], 
                        act_fns=['input', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'linear'],
                        lp=['input', 'dr', 'dr', 'dr', 'rg', 'output'],
                        preloaded_db=None):
        super().__init__(arch, act_fns, preloaded_db)
          
        # Layer purpose
        # 'dr' => dimension reduction layer
        # 'rg' => regression layer
        self.lp = lp


