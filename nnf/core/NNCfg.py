# -*- coding: utf-8 -*-
"""
.. module:: NNCfg
   :platform: Unix, Windows
   :synopsis: Represent NNPreCfg, NNCfg, DAEPreCfg, DAECfg classes.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from nnf.core.Metric import Metric

# Local Imports


class NNCfg(object):
    """NNCfg describes base model configuration for neural network."""

    def __init__(self, optimizer=None, metrics=None):
        """Constructor for NNCfg.

        Parameters
        ----------
        optimizer: str | :obj:

        metrics: :obj:`list`
            List of metrics to be evaluated against outputs.
            i.e
            For single output:      ['accuracy']
            For multiple outputs:   [['accuracy'], [Metric.r]]
        """
        self.loss_fn = 'mean_squared_error'
        self.optimizer = 'adadelta' if optimizer is None else optimizer

        # Only for pre-loaded dbs when no data generators are used.
        # When data generators are used, batch_size, shuffle is expressed in
        # `iter_param`.
        self.pdb_batch_size = 1
        self.pdb_shuffle = True
        
        # Only when data generators are used in training.
        self.steps_per_epoch = 5
        self.validation_steps = 3

        # For both pre-loaded dbs + data generators
        self.numepochs = 2
        self.callbacks = None
        # self.callbacks = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
        self.metrics = metrics

        self.preloaded_db = None        # `PreLoadedDb` instance
        self.feature_layers = -1        # [-1, -2]  # Used to predict features
        
        # PERF:
        self.load_models_dir = None     # Location to save/load compiled models
        self.load_weights_dir = None    # Model is not saved, but weights

        self.save_models_dir = None     # Location to save/load compiled models with weights
        self.save_weights_dir = None    # Model is not saved, but weights


class CNNCfg(NNCfg):
    """Training configuration for convolutional neural network."""

    def __init__(self):
        super().__init__()
        self.loss_fn = 'categorical_crossentropy'
        self.optimizer = 'rmsprop'
        # self.optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # self.optimizer = SGD(lr=0.5, decay=1e-6, momentum=0.1, nesterov=True)
        self.metrics = ['accuracy']


class VGG16Cfg(CNNCfg):
    """Training configuration for vgg16 convolutional neural network."""

    def __init__(self):
        super().__init__()


class DAEPreCfg(NNCfg):
    """Pre-training configuration for each layer of the deep autoencoder network.

    Attributes
    ----------
    arch : list of int
        Architecture of the autoencoder.

    act_fns : list of :obj:`str`
        Activation functions for each layer at `arch`.

    Notes
    -----
    Some of the layers may not be pre-trained. Hence precfgs itself is
    not sufficient to determine the architecture of the final 
    stacked network.
    """

    def __init__(self,
                 arch=(1089, 784, 1089),
                 act_fns=('input', 'sigmoid', 'sigmoid'),
                 preloaded_db=None,
                 batch_size=32,
                 shuffle=True,
                 steps_per_epoch=1,
                 validation_steps=1,
                 numepochs=1,
                 callbacks=None,
                 optimizer=None,
                 metrics=None
                 ):
        super().__init__(optimizer, metrics)
        self.arch = arch
        self.act_fns = act_fns
        self.preloaded_db = preloaded_db
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.numepochs = numepochs
        self.callbacks = callbacks


class AECfg(DAEPreCfg):
    """Training configuration for simple autoencoder network."""

    def __init__(self, arch=(1089, 784, 1089),
                 act_fns=('input', 'sigmoid', 'sigmoid'),
                 preloaded_db=None):
        super().__init__(arch, act_fns, preloaded_db)


class DAECfg(DAEPreCfg):
    """Training configuration for deep autoencoder network."""

    def __init__(self, arch=(1089, 784, 512, 784, 1089),
                 act_fns=('input', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'),
                 preloaded_db=None):
        super().__init__(arch, act_fns, preloaded_db)


class DAERegCfg(DAEPreCfg):
    """Training configuration for deep regression autoencoder network."""

    def __init__(self, arch=(1089, 784, 512, 256, 128, 1089),
                 act_fns=('input', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'linear'),
                 lp=('input', 'dr', 'dr', 'dr', 'rg', 'output'),
                 preloaded_db=None):
        super().__init__(arch, act_fns, preloaded_db)
          
        # Layer purpose
        # 'dr' => dimension reduction layer
        # 'rg' => regression layer
        self.lp = lp
