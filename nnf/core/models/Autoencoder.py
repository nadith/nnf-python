"""Autoencoder to represent Autoencoder class."""
# -*- coding: utf-8 -*-
# Global Imports
from warnings import warn as warning
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
import numpy as np

# Local Imports
from nnf.core.models.NNModel import NNModel
from nnf.db.Dataset import Dataset
from nnf.core.models.NNModelPhase import NNModelPhase

class Autoencoder(NNModel):
    """description of class

    Attributes
    ----------
    encoder : describe
        describe.

    decoder : describe
        describe.

    cb_early_stop : describe
        describe.

    list_iterstore : describe
        describe.

    X_gen : describe
        describe.

    X_val_gen : describe
        describe.

    Xte_gen : describe
        describe.

    X : bool
        describe.

    X_val : describe
        describe.

    cb_early_stop : describe
        describe.

    Methods
    -------
    init_iterstores()
        describe.

    build()
        describe.

    pre_train()
        describe.

    train()
        describe.

    _train()
        describe.

    _test()
        describe.

    _encode()
        describe.

    _encoder_weights()
        Property : describe.

    _decoder_weights()
        Property : describe.

    Examples
    --------
    describe
    >>> nndb = Autoencoder(NNModel)
    >>> nndb = init_iterstores(NNModel)
    >>> nndb = build(NNModel)
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, uid=None, X_L=None, Xt=None, X_L_val=None, Xt_val=None, callbacks=None):
        super().__init__(uid)

        # Initialize variables
        self.encoder = None
        self.decoder = None

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
        """describe"""

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
    def _train(self, cfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
        """Train the :obj:`Autoencoder`.

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
        X_gen, X_val_gen = self._init_data_generators(NNModelPhase.TRAIN, list_iterstore, dict_iterstore)  

        # Pre-condition asserts
        assert((cfg.preloaded_db is None and X_gen is not None) or 
                (cfg.preloaded_db is not None) or
                (self.X_L is not None))

        in_dim = cfg.arch[0]
        hid_dim = cfg.arch[1]
        out_dim = cfg.arch[2]

        # Build the Autoencoder
        self.__build(cfg, in_dim, hid_dim, out_dim, 
                    cfg.act_fns[1], None, 
                    cfg.act_fns[2], None,
                    cfg.loss_fn)

        # Preloaded databases for quick deployment
        if ((cfg.preloaded_db is not None) and
            (self.X_L is None and self.X_L_val is None and
            self.Xt is None and self.Xt_val is None)):
            cfg.preloaded_db.reinit('default')
            self.X_L, self.Xt, self.X_L_val, self.Xt_val = cfg.preloaded_db.LoadTrDb(self)

        # Training without generators
        if (cfg.preloaded_db is not None or 
            (self.X_L is not None)):
            super()._start_train(cfg, self.X_L, self.Xt, self.X_L_val, self.Xt_val)
            return (self.X_L, self.Xt, self.X_L_val, self.Xt_val)

        # Training with generators
        super()._start_train(cfg, X_gen=X_gen, X_val_gen=X_val_gen)

        # Save the trained model
        self._try_save(cfg, patch_idx, "AE")

        return (None, None, None, None)

    def _test(self, cfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
        """describe

        Parameters
        ----------
        X : Describe
            describe.

        visualize : Describe
            describe.

        sel : selection structure
            Information to split the dataset.

        Returns
        -------
        decoded_imgs : describe
        """
        # Initialize data generators
        Xte_gen, Xte_target_gen = self._init_data_generators(NNModelPhase.TEST, list_iterstore, dict_iterstore)

        # Pre-condition asserts
        assert((cfg.preloaded_db is None and Xte_gen is not None) or 
                (cfg.preloaded_db is not None))

        if (Xte_target_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(Xte_gen.nb_sample == Xte_target_gen.nb_sample)

        # 1st Priority, Load the saved model with weights
        is_loaded = self._try_load(cfg, patch_idx, "AE")

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
        assert((cfg.preloaded_db is None and Xte_gen is not None) or 
                (cfg.preloaded_db is not None))

        if (Xte_target_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(Xte_gen.nb_sample == Xte_target_gen.nb_sample)

        # 1st Priority, Load the saved model with weights
        is_loaded = self._try_load(cfg, patch_idx, "AE")

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
        super()._start_predict(patch_idx, Xte_gen=Xte_gen, Xte_target_gen=Xte_target_gen)

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _encode(self, X):
        """describe

        Parameters
        ----------
        X : Describe
            describe.

        sel : selection structure
            Information to split the dataset.

        Returns
        -------
        self.encoder.predict(X) : describe
        """
        return self.encoder.predict(X)

    def _init_data_generators(self, ephase, list_iterstore, dict_iterstore):
        """describe"""
        X_gen = None; X_val_gen = None
        if (list_iterstore is not None):
            X_gen, X_val_gen = self.callbacks['get_data_generators'](ephase, list_iterstore, dict_iterstore)

        return X_gen, X_val_gen

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __build(self, cfg, input_dim=784,  hidden_dim=32, output_dim=784, 
                enc_activation='sigmoid', enc_weights=None, 
                dec_activation='sigmoid', dec_weights=None,
                loss_fn='mean_squared_error', use_early_stop=False, 
                cb_early_stop=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')):
        """describe
        Parameters
        ----------
        nndb : NNdb
            NNdb object that represents the dataset.

        sel : selection structure
            Information to split the dataset.
        """   
        # Set early stop for later use
        self.cb_early_stop = cb_early_stop

        # this is our input placeholder
        input_img = Input(shape=(input_dim,))

        # "encoded" is the encoded representation of the input
        encoded = Dense(hidden_dim, activation=enc_activation, weights=enc_weights)(input_img)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(output_dim, activation=dec_activation, weights=dec_weights)(encoded)

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

        print('Autoencoder: {0}, {1}, {2}, {3}, {4}'.format(input_dim, hidden_dim, enc_activation, dec_activation, loss_fn))
        print(self.net.summary())

        self.net.compile(optimizer='adadelta', loss=loss_fn)
        self._init_fns_predict_feature(cfg)  

    ##########################################################################
    # Dependant Properties
    ##########################################################################
    @property
    def enc_size(self):
        enc_layer = self.net.layers[1] 
        assert enc_layer == self.net.layers[-2]
        return enc_layer.output_dim

    @property
    def _encoder_weights(self):
        enc_layer = self.net.layers[1] 
        assert enc_layer == self.net.layers[-2]
        return enc_layer.get_weights()

    @property
    def _decoder_weights(self):
        dec_layer = self.net.layers[-1] 
        assert dec_layer == self.net.layers[2] 
        return dec_layer.get_weights()