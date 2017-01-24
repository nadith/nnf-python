# -*- coding: utf-8 -*-
"""
.. module:: NNCfg
   :platform: Unix, Windows
   :synopsis: Represent NNPreCfg, NNCfg, DAEPreCfg, DAECfg classes.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from enum import Enum

# Local Imports

class BaseCfg(object):
    """NNPreCfg describes the pre-training configuration for neural network.

    Attributes
    ----------
    opt_fn : str
        Optimization function.

    lr : double
        Learning rate.

    mr : double
        Momentum rate.

    non_sparse_penalty : double
        Non sparse penalty.

    weight_decay_L2 : double
        L2 weight decay.

    batch_size : int
        Batch size.

    numepochs : int
        Number of epochs.

    loss_fn : str
        Loss function.

    use_db : str
        Whther to use explicit database.
    """

    def __init__(self):
        self.opt_fn = 'adadelta'
        self.lr = 0
        self.mr = 0  
        self.non_sparse_penalty = 0
        self.weight_decay_L2 = 0
        self.batch_size = 1  # only used in explcit database test cases
        self.numepochs = 2
        self.loss_fn = 'mean_squared_error'
        self.use_db  = None #'mnist'

        # When data generators are used to train
        self.nb_val_samples = 100
        self.samples_per_epoch = 300

        # PERF:
        self.model_dir = None  # Location to save/load compiled models
        self.weights_path = None  # Model is not saved, but weights

class CNNCfg(BaseCfg):
    """Training configuration for convolutional neural network."""

    def __init__(self):
        super().__init__()

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
                        act_fns=['input', 'sigmoid', 'sigmoid']):
        super().__init__()
        self.arch = arch
        self.act_fns = act_fns
     
class AECfg(DAEPreCfg):
    """Training configuration for simple autoencoder network."""

    def __init__(self, arch=[1089, 784, 1089],
                        act_fns=['input', 'sigmoid', 'sigmoid']):
        super().__init__(arch)
        self.act_fns = act_fns

class DAECfg(DAEPreCfg):
    """Training configuration for deep autoencoder network."""

    def __init__(self, arch=[1089, 784, 512, 784, 1089], 
                        act_fns=['input', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']):
        super().__init__(arch)
        self.act_fns = act_fns

class DAERegCfg(DAEPreCfg):
    """Training configuration for deep regr. autoencoder network."""

    def __init__(self, arch=[1089, 784, 512, 256, 128, 1089], 
                        act_fns=['input', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],
                        lp=['input', 'dr', 'dr', 'dr', 'rg', 'output']):
        super().__init__(arch)
        self.act_fns = act_fns
          
        # Layer purpose
        # 'dr' => dimension reduction layer
        # 'rg' => regression layer
        self.lp = lp


