# -*- coding: utf-8 -*-
"""
.. module:: Autoencoder
   :platform: Unix, Windows
   :synopsis: Represent Autoencoder class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from warnings import warn as warning
from nnf.keras.layers import Input, Dense
from nnf.keras.models import Model

# Local Imports
from nnf.core.models.NNModel import NNModel
from nnf.core.models.NNModelPhase import NNModelPhase

class Autoencoder(NNModel):
    """Generic Autoencoder Model.

    Attributes
    ----------
    X_L : :obj:`tuple`
        In the format (ndarray data tensor, labels).
        If the `nnmodel` is not expecting labels, set it to None.

    Xt : ndarray
        Target data tensor.
        If the `nnmodel` is not expecting a target data tensor,
        set it to None.

    X_L_val : :obj:`tuple`
        In the format (ndarray validation data tensor, labels).
        If the `nnmodel` is not expecting labels, set it to None.

    Xt_val : ndarray
        Validation target data tensor.
        If the `nnmodel` is not expecting a validation target data tensor,
        set it to None.

    __encoder : :obj:`Sequential`
        keras model that maps an input to its encoded representation.

    Note
    ----
    Extend this class to implement custom Autoencoder models.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, uid=None, X_L=None, Xt=None, X_L_val=None, Xt_val=None, callbacks=None,
                 iter_params=None, iter_pp_params=None, nncfgs=None, managed=True):
        super().__init__(uid, callbacks, iter_params, iter_pp_params, nncfgs)

        # Initialize variables
        self.__encoder = None
        #self.decoder = None
        
        # Used when data is fetched from no iterators
        self.X_L = X_L          # (X, labels)
        self.Xt = Xt            # X target
        self.X_L_val = X_L_val  # (X_val, labels_val)
        self.Xt_val = Xt_val    # X_val target

        # Whether `Autoencoder` itself manages the resources for itself.
        # i.e When `Autoencoder` is created in DAE, self.managed=False
        self.managed = managed

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

    def _encode(self, X):
            """Encode the data.

            Parameters
            ----------
            X : ndarray
                Input data tensor.

            Returns
            -------
            ndarray
                Encoded representation.
            """
            return self.__encoder.predict(X)

    ##########################################################################
    # Protected: NNModel Overrides
    ##########################################################################
    def _model_prefix(self):
        """Fetch the _prefix for the file to be saved/loaded."""
        return "AE"

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
            Dictionary of iterstores for :obj:`DataIterator`.
        """

        # Initialize data generators
        X_gen, X_val_gen = (None, None)
        if cfg.preloaded_db is None and self.X_L is None:
            X_gen, X_val_gen = self._init_data_generators(NNModelPhase.TRAIN, list_iterstore, dict_iterstore)
            assert X_gen is not None

        # Model _prefix
        prefix = self._model_prefix()
        self._init_net(cfg, patch_idx, prefix, None)
        assert (self.net is not None)

        # Preloaded databases for quick deployment
        if ((cfg.preloaded_db is not None) and
            (self.X_L is None and self.X_L_val is None and
            self.Xt is None and self.Xt_val is None)):
            cfg.preloaded_db.reinit()
            self.X_L, self.Xt, self.X_L_val, self.Xt_val =\
                                            cfg.preloaded_db.LoadTrDb(self)

        # Training without generators
        if (cfg.preloaded_db is not None or 
            (self.X_L is not None)):
            super()._start_train(cfg, self.X_L, self.Xt, self.X_L_val, self.Xt_val)
            ret = (self.X_L, self.Xt, self.X_L_val, self.Xt_val)

        # Training with generators
        else:
            # DAE is constructed using this model, hence managed=False in that case.
            super()._start_train(cfg, X_gen=X_gen, X_val_gen=X_val_gen, managed=self.managed)
            ret = (None, None, None, None)

        # Model _prefix
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
            Dictionary of iterstores for :obj:`DataIterator`.
        """  
        # Initialize data generators
        X_te_gen, Xt_te_gen = (None, None)
        if cfg.preloaded_db is None:
            X_te_gen, Xt_te_gen = self._init_data_generators(NNModelPhase.TEST, list_iterstore, dict_iterstore)
            assert X_te_gen is not None

        if (Xt_te_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(X_te_gen.nb_sample == Xt_te_gen.nb_sample)

        # Model _prefix
        prefix = self._model_prefix()

        # Build the Autoencoder if not already built during training
        if self.net is None:
            self._init_net(cfg, patch_idx, prefix, None)

        assert (self.net is not None)

        # Preloaded databases for quick deployment
        if (cfg.preloaded_db is not None):
            cfg.preloaded_db.reinit()
            X_L_te, Xt_te = cfg.preloaded_db.LoadTeDb(self)     

            # Test without generators
            super()._start_test(patch_idx, X_L_te, Xt_te)
            return

        # Test with generators
        super()._start_test(patch_idx, X_te_gen=X_te_gen)

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
            Dictionary of iterstores for :obj:`DataIterator`.
        """    
        # Initialize data generators
        X_te_gen, Xt_te_gen = (None, None)
        if cfg.preloaded_db is None:
            X_te_gen, Xt_te_gen = self._init_data_generators(NNModelPhase.PREDICT, list_iterstore, dict_iterstore)
            assert X_te_gen is not None

        if (Xt_te_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(X_te_gen.nb_sample == Xt_te_gen.nb_sample)

        # Model _prefix
        prefix = self._model_prefix()

        # Build the Autoencoder if not already built during training
        if self.net is None:
            self._init_net(cfg, patch_idx, prefix, None)

        assert (self.net is not None)

        # After self.net configure, for predict functionality, 
        # initialize theano sub functions
        self._init_fns_predict_feature(cfg)

        # Preloaded databases for quick deployment
        if (cfg.preloaded_db is not None):
            cfg.preloaded_db.reinit()
            X_L_te, Xt_te = cfg.preloaded_db.LoadPredDb(self)

            # Predict without generators
            super()._start_predict(patch_idx, X_L_te, Xt_te)
            return

        # Predict with generators
        super()._start_predict(patch_idx, X_te_gen=X_te_gen)

    ##########################################################################
    # Protected Private: NNModel Overrides (Build Related)
    ##########################################################################
    def _internal__build(self, cfg, X_gen):
        """Build the keras Autoencoder."""
        # this is our input placeholder
        input_img = Input(shape=(cfg.arch[0],))

        # "encoded" is the encoded representation of the input
        encoded = Dense(cfg.arch[1], activation=cfg.act_fns[1])(input_img)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(cfg.arch[2], activation=cfg.act_fns[2])(encoded)

        # this model maps an input to its reconstruction
        self.net = Model(inputs=input_img, outputs=decoded)

        #
        # this model maps an input to its encoded representation
        self.__encoder = Model(inputs=input_img, outputs=encoded)
    
        ##
        ## create a placeholder for an encoded (32-dimensional) input
        #encoded_input = Input(shape=(hidden_dim,))

        ## retrieve the last layer of the net model
        #decoder_layer = self.net.layers[-1]            

        ## create the decoder model, will create another inbound node to `decoder_layer`
        ## hence decoder_layer.output cannot be used.
        ## Ref: NNModel._init_predict_feature_fns()
        ## self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        # TODO: Validate X_gen.input_shapes, X_gen.output_shapes against the cfg.arch[0], cfg.arch[end]

    def _internal__pre_compile(self, cfg, X_gen):
        pass

    def _internal__compile(self, cfg):
        """Compile the keras DAE."""
        self.net.compile(optimizer=cfg.optimizer,
                            loss=cfg.loss_fn,
                            metrics=cfg.metrics)
        print(self.net.summary())

    ##########################################################################
    # Private Interface
    ##########################################################################
    # def __init_net(self, cfg, patch_idx, prefix):
    #     # Checks whether keras net is already prebuilt
    #     if not self._is_prebuilt(cfg, patch_idx, prefix):
    #         self._build(cfg)
    #
    #     # Try to load the saved model or weights
    #     self._try_load(cfg, patch_idx, prefix)
    #
    #     # PERF: Avoid compiling before loading saved weights/model
    #     # Pre-compile callback
    #     self._pre_compile(cfg)
    #
    #     # PERF: Avoid compiling before the callback
    #     self.__compile(cfg)

    ##########################################################################
    # Dependant Properties
    ##########################################################################
    @property
    def enc_size(self):
        """Size of the encoding layer."""
        enc_layer = self.net.layers[1] 
        assert enc_layer == self.net.layers[-2]
        return enc_layer.output_shape[1]

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