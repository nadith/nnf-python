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
from keras.layers.convolutional import Conv2D, MaxPooling2D,ZeroPadding2D
#from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,RMSprop,adam

# Local Imports
from nnf.db.NNdb import NNdb
from nnf.db.Format import Format
from nnf.db.Dataset import Dataset
from nnf.core.models.NNModel import NNModel
from nnf.core.models.NNModelPhase import NNModelPhase
from nnf.core.iters.DataIterator import DataIterator
from nnf.core.iters.memory.MemDataIterator import MemDataIterator
from nnf.core.iters.disk.DskDataIterator import DskDataIterator

class CNNModel(NNModel):
    """Generic Convolutional Neural Network Model.

    Attributes
    ----------
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
        super().__init__(callbacks=callbacks)

        # Used when data is fetched from no iterators
        self.X_L = X_L          # (X, labels)
        self.Xt = Xt            # X target
        self.X_L_val = X_L_val  # (X_val, labels_val)
        self.Xt_val = Xt_val    # X_val target              

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
        self.__build(cfg, X_gen)

        # Preloaded databases for quick deployment
        if ((cfg.preloaded_db is not None) and
            (self.X_L is None and self.X_L_val is None and
            self.Xt is None and self.Xt_val is None)):
            self.X_L, self.Xt, self.X_L_val, self.Xt_val = cfg.preloaded_db.LoadTrDb(self)

        # Training without generators
        if (cfg.preloaded_db is not None):
            super()._start_train(cfg, self.X_L, self.Xt, self.X_L_val, self.Xt_val)
            ret = (self.X_L, self.Xt, self.X_L_val, self.Xt_val)

        # Training with generators
        else:
            super()._start_train(cfg, X_gen=X_gen, X_val_gen=X_val_gen)
            ret = (None, None, None, None)

        # Save the trained model
        self._try_save(cfg, patch_idx, self._model_prefix())
        return ret

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
        X_te_gen, Xt_te_gen = self._init_data_generators(
                                                    NNModelPhase.TEST,
                                                    list_iterstore,
                                                    dict_iterstore)

        # Pre-condition asserts
        assert((cfg.preloaded_db is None and X_te_gen is not None) or 
                (cfg.preloaded_db is not None))

        if (Xt_te_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(X_te_gen.nb_sample == Xt_te_gen.nb_sample)

        # Model prefix
        prefix = self._model_prefix()

        # Checks whether keras net should be pre-built to
        # load weights or the net itself
        if (self._need_prebuild(cfg, patch_idx, prefix)):
            self.__build(cfg, X_te_gen)

        # Try to load the saved model or weights
        self._try_load(cfg, patch_idx, prefix)

        assert(self.net is not None)

        # Preloaded databases for quick deployment
        if (cfg.preloaded_db is not None):
            cfg.preloaded_db.reinit()
            X_L_te, Xt_te = cfg.preloaded_db.LoadTeDb(self)

        # Test without generators
        if (cfg.preloaded_db is not None):
            super()._start_test(patch_idx, X_L_te, Xt_te)
            return

        # Test with generators
        super()._start_test(patch_idx,
                            X_te_gen=X_te_gen,
                            Xt_te_gen=Xt_te_gen)

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
        X_te_gen, Xt_te_gen = self._init_data_generators(
                                                NNModelPhase.PREDICT, 
                                                list_iterstore, 
                                                dict_iterstore)
        # Pre-condition asserts
        assert((cfg.preloaded_db is None and X_te_gen is not None) or 
                (cfg.preloaded_db is not None))

        if (Xt_te_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(Xte_gen.nb_sample == Xt_te_gen.nb_sample)

        # Model prefix
        prefix = self._model_prefix()

        # Checks whether keras net should be pre-built to
        # load weights or the net itself
        if (self._need_prebuild(cfg, patch_idx, prefix)):
            self.__build(cfg, X_te_gen)

        # Try to load the saved model or weights
        self._try_load(cfg, patch_idx, prefix)
        assert(self.net is not None)

        # After self.net configure, for predict functionality, 
        # initialize theano sub functions
        self._init_fns_predict_feature(cfg)

        # Preloaded databases for quick deployment
        if (cfg.preloaded_db is not None):
            cfg.preloaded_db.reinit()
            X_L_te, Xt_te = cfg.preloaded_db.LoadPredDb(self)    

        # Predict without generators
        if (cfg.preloaded_db is not None):
            super()._start_predict(patch_idx, X_L_te, Xt_te)
            return

        # Predict with generators
        super()._start_predict(patch_idx, 
                                X_te_gen=X_te_gen, 
                                Xt_te_gen=Xt_te_gen)

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _model_prefix(self):
        """Fetch the prefix for the file to be saved/loaded.

        Note
        ----
        Override this method for custom prefix.
        """
        return "CNN"

    def _build(self, input_shape, nb_class, data_format):
        """Build the keras CNN.

        Note
        ----
        Override this method for custom CNN builds.
        """
        # input: 150x150 images with 3 channels -> (3, 100, 100) tensors.

        self.net = Sequential()
        self.net.add(Conv2D(32, (3, 3), input_shape=input_shape))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))

        self.net.add(Conv2D(32, (3, 3)))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))

        self.net.add(Conv2D(64, (3, 3)))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))

        self.net.add(Flatten())
        #self.net.add(Dense(64))
        self.net.add(Dense(256))
        self.net.add(Activation('relu'))
        self.net.add(Dropout(0.5))
        self.net.add(Dense(nb_class))
        self.net.add(Activation('softmax'))

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __build(self, cfg, X_gen):
        """Build the keras CNN."""
        if (cfg.preloaded_db is not None):
            cfg.preloaded_db.reinit()
            input_shape = cfg.preloaded_db.get_input_shape(self)
            nb_class = cfg.preloaded_db.get_nb_class()
            data_format = cfg.preloaded_db.data_format

        else:
            input_shape = X_gen.image_shape # <- 'tf' format (default)
            nb_class = X_gen.nb_class
            data_format = X_gen.data_format

        self._build(input_shape, nb_class, data_format)
        print(self.net.summary())
        self.net.compile(loss=cfg.loss_fn,
                            optimizer=cfg.optimizer,
                            metrics=cfg.metrics)


