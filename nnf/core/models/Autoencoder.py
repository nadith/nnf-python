# -*- coding: utf-8 -*-
"""
.. module:: Autoencoder
   :platform: Unix, Windows
   :synopsis: Represent Autoencoder class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from warnings import warn as warning
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
import numpy as np

# Local Imports
from nnf.db.Dataset import Dataset
from nnf.core.models.NNModel import NNModel
from nnf.core.models.NNModelPhase import NNModelPhase

class Autoencoder(NNModel):
    """Generic Autoencoder Model.

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

    encoder : :obj:`Sequential`
        keras model that maps an input to its encoded representation.

    Note
    ----
    Extend this class to implement custom Autoencoder models.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, uid=None, X_L=None, Xt=None, X_L_val=None, Xt_val=None,
                                                            callbacks=None):
        super().__init__(uid)

        # Initialize variables
        self.encoder = None
        #self.decoder = None

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

    def pre_train(self, precfgs, cfg, patch_idx=None):
        """Pre-train the :obj:`Autoencoder`.

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
        warning('Pre-training is not supported in Autoencoder')

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
        """Train the :obj:`Autoencoder`.

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
                (cfg.preloaded_db is not None) or
                (self.X_L is not None))

        # Build the Autoencoder
        self._build(cfg)

        # Preloaded databases for quick deployment
        if ((cfg.preloaded_db is not None) and
            (self.X_L is None and self.X_L_val is None and
            self.Xt is None and self.Xt_val is None)):
            cfg.preloaded_db.reinit('default')
            self.X_L, self.Xt, self.X_L_val, self.Xt_val =\
                                            cfg.preloaded_db.LoadTrDb(self)

        # Training without generators
        if (cfg.preloaded_db is not None or 
            (self.X_L is not None)):
            super()._start_train(cfg, self.X_L, self.Xt, self.X_L_val, self.Xt_val)
            ret = (self.X_L, self.Xt, self.X_L_val, self.Xt_val)

        # Training with generators
        else:        
            super()._start_train(cfg, X_gen=X_gen, X_val_gen=X_val_gen)
            ret = (None, None, None, None)

        # Model prefix
        prefix = self._model_prefix()

        # Save the trained model
        self._try_save(cfg, patch_idx, prefix)
        return ret

    def _test(self, cfg, patch_idx=None, dbparam_save_dirs=None, 
                                    list_iterstore=None, dict_iterstore=None):
        """Test the :obj:`Autoencoder`.

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
            self._build(cfg)

        # Try to load the saved model or weights
        self._try_load(daecfg, patch_idx, prefix)

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
                            X_te_gen=X_te_gen,
                            Xt_te_gen=Xt_te_gen)

    def _predict(self, cfg, patch_idx=None, dbparam_save_dirs=None, 
                                    list_iterstore=None, dict_iterstore=None):
        """Predict using :obj:`Autoencoder`.

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
            assert(X_te_gen.nb_sample == Xt_te_gen.nb_sample)

        # Model prefix
        prefix = self._model_prefix()

        # Checks whether keras net should be pre-built to
        # load weights or the net itself
        if (self._need_prebuild(cfg, patch_idx, prefix)):
            self._build(cfg)

        # Try to load the saved model or weights
        self._try_load(cfg, patch_idx, prefix)

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
        return "AE"

    def _encode(self, X):
        """Encode the data.

        Parameters
        ----------
        X : `array_like`
            Input data tensor.

        Returns
        -------
        `array_like`
            Encoded representation.
        """
        return self.encoder.predict(X)

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

    def _build(self, cfg, use_early_stop=False, 
                cb_early_stop=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')):
        """Build the keras Autoencoder."""
        # Set early stop for later use
        self.cb_early_stop = cb_early_stop

        # this is our input placeholder
        input_img = Input(shape=(cfg.arch[0],))

        # "encoded" is the encoded representation of the input
        encoded = Dense(cfg.arch[1], activation=cfg.act_fns[1])(input_img)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(cfg.arch[2], activation=cfg.act_fns[2])(encoded)

        # this model maps an input to its reconstruction
        self.net = Model(input=input_img, output=decoded)
    
        #
        # this model maps an input to its encoded representation
        self.encoder = Model(input=input_img, output=encoded)
    
        ##
        ## create a placeholder for an encoded (32-dimensional) input
        #encoded_input = Input(shape=(hidden_dim,))

        ## retrieve the last layer of the net model
        #decoder_layer = self.net.layers[-1]            

        ## create the decoder model, will create another inbound node to `decoder_layer`
        ## hence decoder_layer.output cannot be used.
        ## Ref: NNModel._init_predict_feature_fns()
        ## self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        print(self.net.summary())

        self.net.compile(optimizer='adadelta', loss=cfg.loss_fn)
        self._init_fns_predict_feature(cfg)  

    ##########################################################################
    # Dependant Properties
    ##########################################################################
    @property
    def enc_size(self):
        """Size of the encoding layer."""
        enc_layer = self.net.layers[1] 
        assert enc_layer == self.net.layers[-2]
        return enc_layer.output_dim

    @property
    def _encoder_weights(self):
        """Encoding weights."""
        enc_layer = self.net.layers[1] 
        assert enc_layer == self.net.layers[-2]
        return enc_layer.get_weights()

    @property
    def _decoder_weights(self):
        """Decoding weights."""
        dec_layer = self.net.layers[-1] 
        assert dec_layer == self.net.layers[2] 
        return dec_layer.get_weights()