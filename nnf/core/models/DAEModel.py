# -*- coding: utf-8 -*-
"""
.. module:: DAEModel
   :platform: Unix, Windows
   :synopsis: Represent DAEModel class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import os
import math
import numpy as np
from nnf.keras.engine import Model
from nnf.keras.layers import Input, Dense

# Local Imports
from nnf.db.NNdb import NNdb
from nnf.db.Format import Format
from nnf.db.Dataset import Dataset
from nnf.core.FileFormat import FileFormat
from nnf.core.models.NNModel import NNModel
from nnf.core.models.Autoencoder import Autoencoder
from nnf.core.models.NNModelPhase import NNModelPhase
from nnf.core.iters.disk.DskDataIterator import DskDataIterator
from nnf.core.iters.memory.MemDataIterator import MemDataIterator
from nnf.core.iters.disk.BigDataDirectoryIterator import BigDataDirectoryIterator
from nnf.core.iters.memory.BigDataNumpyArrayIterator import BigDataNumpyArrayIterator

class DAEModel(NNModel):
    """Generic deep autoencoder model.

    Note
    ----
    Extend this class to implement custom deep autoencoder models.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, callbacks=None,
                 iter_params=None, iter_pp_params=None, nncfgs=None):
        """Constructs :obj:`DAEModel` instance."""
        super().__init__(callbacks=callbacks,
                         iter_params=iter_params,
                         iter_pp_params=iter_pp_params,
                         nncfgs=nncfgs)

    ##########################################################################
    # Protected: NNModel Overrides
    ##########################################################################
    def _model_prefix(self):
        """Fetch the _prefix for the file to be saved/loaded."""
        return "DAE"

    def _pre_train(self, daeprecfgs, daecfg, patch_idx=None,
                    dbparam_save_dirs=None, 
                    list_iterstore=None, dict_iterstore=None):
        """Pre-train the :obj:`DAEModel`.

        Parameters
        ----------
        daeprecfgs : list of :obj:`NNCfg`
            List of Neural Network configurations. Useful for layer-wise 
            pre-training.

        daecfg : :obj:`NNCfg`
            Neural Network configuration that will be used in training. 
            Useful to build the deep stacked network after layer-wise 
            pre-training.

        patch_idx : int
            Patch's index in this model.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of each `nnpatch`.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictionary of iterstores for :obj:`DataIterator`.
        """
        # Validate nncfgs
        for _, daeprecfg in enumerate(daeprecfgs):
            self._validate_cfg(NNModelPhase.PRE_TRAIN, daeprecfg)
        self._validate_cfg(NNModelPhase.TRAIN, daecfg)

        # Initialize parameters
        save_to_dir = None
        if (dbparam_save_dirs is not None):
            # For all dbparams use dbparam1 since it 
            # overwrites the temporary data before use.
            save_to_dir = dbparam_save_dirs[0]  

        # Initialize data generators
        X_gen, X_val_gen = (None, None)
        if daecfg.preloaded_db is None:
            X_gen, X_val_gen = self._init_data_generators(NNModelPhase.PRE_TRAIN, list_iterstore, dict_iterstore)
            assert X_gen is not None

        # Track layers to build Deep/Stacked Auto Encoder (DAE)
        layers = [Input(shape=(daecfg.arch[0],))]

        # Initialize weights and transpose of weights
        w_T = w = None       

        # For Preloaded Databases
        X_L = Xt = X_L_val = Xt_val = None
        is_preloaded_db = (daecfg.preloaded_db is not None)
        if (is_preloaded_db): daecfg.preloaded_db.reinit()

        # map of iterators for TR|VAL|TE|... datasets in Autoencoder
        ae_iterstore = {}

        # Model _prefix
        prefix = self._model_prefix()

        # Perform layer-wise pre-training.
        for ipcfg, daeprecfg in enumerate(daeprecfgs):

            # Set the layer index (Input Layer Index = 0)
            layer_idx = ipcfg + 1

            # Explicit uid definition for simple AE
            ae_uid = 'dae_' + str(self.uid) + '_ae_' + str(layer_idx)

            # Create a simple AE
            # Using non-generators for external databases
            if (is_preloaded_db):
                # Create a simple AE
                if (X_L is None and Xt is None and 
                    X_L_val is None and Xt_val is None):
                    daeprecfg.preloaded_db.reinit()
                    X_L, Xt, X_L_val, Xt_val =\
                                    daeprecfg.preloaded_db.LoadPreTrDb(self)

                ae = Autoencoder(ae_uid, X_L, Xt, X_L_val, Xt_val)

            # Using generators for NNDb or disk databases
            else:
                # Callback for pre-training pre-processing
                X_gen, X_val_gen = self._pre_train_preprocess(layer_idx,
                                                              daeprecfgs,
                                                              daecfg,
                                                              X_gen, X_val_gen)
                # Set data for simple AE
                ae_iterstore[Dataset.TR] = X_gen
                ae_iterstore[Dataset.VAL] = X_val_gen

                # The resources of the simple AE is managed by this object (DAEModel)
                ae = Autoencoder(ae_uid, managed=False)
                ae.add_iterstores(list_iterstore=[ae_iterstore])

            ##################################################################
            # Checks whether keras net is already prebuilt
            ae_isprebuilt = ae._is_prebuilt(daeprecfg, patch_idx, ae._model_prefix())

            # This will build self.__encoder
            if (not ae_isprebuilt):
                ae._internal__build(daeprecfg, None)

            # Try to load the saved model or weights
            is_loaded = ae._try_load(daeprecfg, patch_idx, ae._model_prefix())

            # If a trained model is not loaded, train the simple AE
            if (not is_loaded):

                # Initiate the data flow for the simple AE
                if not is_preloaded_db:
                    X_gen.initiate_parallel_operations()
                    if (X_val_gen is not None): X_val_gen.initiate_parallel_operations()

                # Train the simple AE
                ae.train(daeprecfg, patch_idx)

            # Stack the trained layers
            layers = self._stack_layers(ae, layer_idx, layers, daecfg)

            # Last iteration
            if layer_idx == len(daeprecfgs):
                # Release before break
                if not is_preloaded_db:
                    X_gen.release()
                    if X_val_gen is not None: X_val_gen.release()
                break

            ##################################################################
            # Prepare for next layer
            if (is_preloaded_db):

                # Post process datasets and prepare for next round                
                X_L, Xt, X_L_val, Xt_val =\
                    self._pre_train_postprocess_preloaded_db(
                                                    layer_idx, daecfg, ae, 
                                                    X_L, Xt, X_L_val, Xt_val)
            else:
                # Release current generators, fetch generator for next layer
                X_gen, X_val_gen = self.__get_genrators_at(
                                                    patch_idx, save_to_dir,
                                                    layer_idx, ae,
                                                    X_gen, X_val_gen)

        # Build DAE
        self.__build_with_layers(daecfg, layers)

        # Compile the model
        self._internal__compile(daecfg)
    
    def _train(self, daecfg, patch_idx=None, dbparam_save_dirs=None, 
                                    list_iterstore=None, dict_iterstore=None):
        """Train the :obj:`DAEModel`.

        Parameters
        ----------
        daecfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int
            Patch's index in this model.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of each `nnpatch`.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictionary of iterstores for :obj:`DataIterator`.
        """
        # Validate nncfg
        self._validate_cfg(NNModelPhase.TRAIN, daecfg)

        # Initialize data generators
        X_gen, X_val_gen = (None, None)
        if daecfg.preloaded_db is None:
            X_gen, X_val_gen = self._init_data_generators(NNModelPhase.TRAIN, list_iterstore, dict_iterstore)
            assert X_gen is not None

        # Model _prefix
        prefix = self._model_prefix()

        # Build the DAE if not already built during pre-training
        if self.net is None:
            self._init_net(daecfg, patch_idx, prefix, None)

        assert (self.net is not None)

        # Preloaded databases for quick deployment
        if (daecfg.preloaded_db is not None):
            daecfg.preloaded_db.reinit()
            X_L, Xt, X_L_val, Xt_val = daecfg.preloaded_db.LoadTrDb(self)
 
            # Training without generators
            super()._start_train(daecfg, X_L, Xt, X_L_val, Xt_val)
            ret = (X_L, Xt, X_L_val, Xt_val)
        
        # Training with generators
        else:
            super()._start_train(daecfg, X_gen=X_gen, X_val_gen=X_val_gen)
            ret = (None, None, None, None)

        # Save the trained model
        self._try_save(daecfg, patch_idx, prefix)
        return ret

    def _test(self, daecfg, patch_idx=None, dbparam_save_dirs=None,
                                    list_iterstore=None, dict_iterstore=None):
        """Test the :obj:`Autoencoder`.

        Parameters
        ----------
        daecfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int
            Patch's index in this model.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of each `nnpatch`.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictionary of iterstores for :obj:`DataIterator`.
        """    
        # Initialize data generators
        X_te_gen, Xt_te_gen = (None, None)
        if daecfg.preloaded_db is None:
            X_te_gen, Xt_te_gen = self._init_data_generators(NNModelPhase.TEST, list_iterstore, dict_iterstore)
            assert X_te_gen is not None

        if (Xt_te_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(X_te_gen.nb_sample == Xt_te_gen.nb_sample)

        # Model _prefix
        prefix = self._model_prefix()

        # Build the DAE if not already built during training
        if self.net is None:
            self._init_net(daecfg, patch_idx, prefix, None)

        assert (self.net is not None)

        # Preloaded databases for quick deployment
        if (daecfg.preloaded_db is not None):
            daecfg.preloaded_db.reinit()
            X_L_te, Xt_te = daecfg.preloaded_db.LoadTeDb(self)     

            # Test without generators
            super()._start_test(patch_idx, X_L_te, Xt_te)
            return

        # Test with generators
        super()._start_test(patch_idx, X_te_gen=X_te_gen)

    def _predict(self, daecfg, patch_idx=None, dbparam_save_dirs=None, 
                                    list_iterstore=None, dict_iterstore=None):
        """Predict the :obj:`Autoencoder`.

        Parameters
        ----------
        daecfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int
            Patch's index in this model.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of each `nnpatch`.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictionary of iterstores for :obj:`DataIterator`.
        """
        # Initialize data generators
        X_te_gen, Xt_te_gen = (None, None)
        if daecfg.preloaded_db is None:
            X_te_gen, Xt_te_gen = self._init_data_generators(NNModelPhase.PREDICT, list_iterstore, dict_iterstore)
            assert X_te_gen is not None

        if (Xt_te_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(X_te_gen.nb_sample == Xt_te_gen.nb_sample)

        # Model _prefix
        prefix = self._model_prefix()

        # Build the DAE if not already built during training
        if self.net is None:
            self._init_net(daecfg, patch_idx, prefix, None)

        assert (self.net is not None)

        # After self.net configure, for predict functionality, 
        # initialize theano sub functions
        self._init_fns_predict_feature(daecfg)

        # Preloaded databases for quick deployment
        if (daecfg.preloaded_db is not None):
            daecfg.preloaded_db.reinit()
            X_L_te, Xt_te = daecfg.preloaded_db.LoadPredDb(self)

            # Predict without generators
            super()._start_predict(patch_idx, X_L_te, Xt_te)
            return

        # Predict with generators
        super()._start_predict(patch_idx, X_te_gen=X_te_gen)

    def _get_input_data_generators(self, ephase, list_iterstore, dict_iterstore):
        """Get input data generators for training, testing, prediction.

        Parameters
        ----------
        ephase : :obj:`NNModelPhase`
            Phase of which the data generators are required.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictionary of iterstores for :obj:`DataIterator`.

        Returns
        -------
        :obj:`tuple`
            When ephase == NNModelPhase.TRAIN
            then
                Generators for training and validation.

            When ephase == NNModelPhase.TEST or NNModelPhase.PREDICT
            then
                Generators for testing and testing target.
        """
        X1_gen = None
        X2_gen = None
        if (ephase == NNModelPhase.PRE_TRAIN):
            # Iterstore for dbparam1, TR, VAL:
            # Take a copy since the iterator state will be altered in pre-training,
            # thus the originals can be used again in training phase.
            X1_gen = self._clone_iter(list_iterstore[0].setdefault(Dataset.TR, None))
            X2_gen = self._clone_iter(list_iterstore[0].setdefault(Dataset.VAL, None))

        elif (ephase == NNModelPhase.TRAIN):
            # Iterstore for dbparam1, TR, VAL
            X1_gen = list_iterstore[0].setdefault(Dataset.TR, None)
            X2_gen = list_iterstore[0].setdefault(Dataset.VAL, None)

        elif (ephase == NNModelPhase.PREDICT or ephase == NNModelPhase.TEST):
            # Iterstore for dbparam1, TE
            X1_gen = list_iterstore[0].setdefault(Dataset.TE, None)

        return X1_gen, X2_gen

    def _get_target_data_generators(self, ephase, list_iterstore, dict_iterstore):
        """Get target data generators for training, testing, prediction.

        Parameters
        ----------
        ephase : :obj:`NNModelPhase`
            Phase of which the data generators are required.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictionary of iterstores for :obj:`DataIterator`.

        Returns
        -------
        :obj:`tuple`
            When ephase == NNModelPhase.TRAIN
            then
                Generators for training (X1_gen) and validation (X2_gen).

            When ephase == NNModelPhase.TEST or NNModelPhase.PREDICT
            then
                Generators for testing target (X1_gen) and (X2_gen) is unused.
        """
        X1_gen = None
        X2_gen = None
        if (ephase == NNModelPhase.PREDICT or ephase == NNModelPhase.TEST):
            # Iterstore for dbparam1, TE_OUT
            X1_gen = list_iterstore[0].setdefault(Dataset.TE_OUT, None)

        return X1_gen, X2_gen

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _validate_cfg(self, ephase, cfg):
        """Validate NN Configuration for this model"""
        if (len(cfg.arch) != len(cfg.act_fns)):
            raise Exception('Activation function for each layer is not' + 
            ' specified. (length of `cfg.arch` != length of `cfg.act_fns`).')       

    def _pre_train_preprocess(self, layer_idx, daeprecfgs, daecfg,
                                                            X_gen, X_val_gen):
        """Pre-processing invocation for pre-training with generators
            in pre-training loop.

            May override for custom functionality.

        Returns
        -------
        :obj:`DataIterator`
            X_gen = Data iterator that generates data in 
            (ndarray, labels or ndarray) format, depending on
            the `nnmodel` architecture.

        :obj:`DataIterator`
            X_val_gen = Validation data iterator that generates data in 
            (ndarray, labels or ndarray) format, depending on
            the `nnmodel` architecture.
        """
        return X_gen, X_val_gen

    def _pre_train_postprocess_preloaded_db(self, layer_idx, daecfg, ae,
                                                    X_L, Xt, X_L_val, Xt_val):
        """Post-processing invocation for pre-training with preloaded dbs
            in pre-training loop.

            May override for custom functionality.

        Returns
        -------
        :obj:`tuple`
            X_L = (ndarray data tensor, labels)
            If the `nnmodel` is not expecting labels, set it to None.
        
        ndarray
            Xt = target data tensor
            If the `nnmodel` is not expecting a target data tensor, 
            set it to None.

        :obj:`tuple`
            X_L_val = (ndarray validation data tensor, labels)
            If the `nnmodel` is not expecting labels, set it to None.

        ndarray
            Xt_val = validation target data tensor
            If the `nnmodel` is not expecting a validation target data tensor, 
            set it to None.
        """
        # Fix variables for the next layer pre-training
        X_L = (ae._encode(X_L[0]), None)
        Xt = X_L[0]
        X_L_val = (ae._encode(X_L_val[0]), None)
        Xt_val = X_L_val[0]

        return  X_L, Xt, X_L_val, Xt_val

    def _stack_layers(self, ae, layer_idx, layers, daecfg):
        """Stack the layers.
        
            Encoding and correponding decoder layer is stacked accordingly.
            May override for custom stacking mechanism.

        Parameters
        ----------
        ae : :obj:`Autoencoder`
            `Autoencoder model to fetch the encoding/decoding weights.

        layer_idx : int
            Layer index. (0 = Input layer)

        layers : :obj:`list`
            List of keras layers.

        daecfg : :obj:`NNCfg`
            Neural Network configuration that will be used in training. 
            Useful to build the deep stacked network after layer-wise 
            pre-trianing.  
        """
        # Tied weights
        w = ae._encoder_weights
        w_T = [np.transpose(w[0]), np.zeros(w[0].shape[0])]

        # Adding encoding layer for DAE
        layers.insert(layer_idx, 
                        Dense(daecfg.arch[layer_idx], 
                                activation=daecfg.act_fns[layer_idx],
                                weights=w, 
                                name="enc_" + str(layer_idx)))

        # Adding decoding layer for DAE
        dec_i = len(daecfg.arch)-layer_idx
        layers.insert(layer_idx+1, 
                        Dense(daecfg.arch[dec_i], 
                                activation=daecfg.act_fns[dec_i], 
                                weights=w_T, 
                                name="dec_" + str(layer_idx)))
        return layers

    def _build(self, daecfg):
        """Build the keras DAE directly from the NN configuration.
        
            Invoked when the network needs to be built without pre-training.
        """
        layers = [Input(shape=(daecfg.arch[0],))]
        mid = (len(daecfg.arch) + 1) // 2
        for i, dim in enumerate(daecfg.arch[1:mid]):
            layer_idx = i + 1
            layers.insert(layer_idx, 
                            Dense(daecfg.arch[layer_idx], 
                                    activation=daecfg.act_fns[layer_idx],
                                    name="enc_" + str(layer_idx)))

            dec_i = len(daecfg.arch)-layer_idx
            layers.insert(layer_idx+1, 
                            Dense(daecfg.arch[dec_i], 
                                    activation=daecfg.act_fns[dec_i],
                                    name="dec_" + str(layer_idx)))

        # Build DAE
        self.__build_with_layers(daecfg, layers)

    ##########################################################################
    # Protected Private: NNModel Overrides (Build Related)
    ##########################################################################
    def _internal__build(self, daecfg, X_gen):
        self._build(daecfg)

    def _internal__pre_compile(self, daecfg, X_gen):
        pass

    def _internal__compile(self, daecfg):
        """Compile the keras DAE."""
        self.net.compile(optimizer=daecfg.optimizer,
                            loss=daecfg.loss_fn,
                            metrics=daecfg.metrics)
        print(self.net.summary())

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __build_with_layers(self, daecfg, layers):
        """Build the keras DAE with layers."""
        in_tensor = layers[0]
        out_tensor = layers[0]

        for i in range(1, len(layers)):
            out_tensor = layers[i](out_tensor)
        
        self.net = Model(inputs=in_tensor, outputs=out_tensor)

    def __get_genrators_at(self, patch_idx, save_to_dir, layer_idx, ae, X_gen, X_val_gen):
        """Fetch the generators for next layer pre-training in the pre-training loop.

        Returns
        -------
        :obj:`DataIterator`
            X_gen = Data iterator that generates data in 
            (ndarray, labels or ndarray) format, depending on
            the `nnmodel` architecture.

        :obj:`DataIterator`
            X_val_gen = Validation data iterator that generates data in 
            (ndarray, labels or ndarray) format, depending on
            the `nnmodel` architecture.
        """
        # If the current generators are pointing to a temporary location, mark it for deletion
        # predict from current X_gen and X_val_gen and save the data in a temporary location 
        # for index i (layer i). Construct new generators for the data stored in this temporary location.
        # if the previous data locations are marked for deletion (temporary location), proceed the deletion

        new_X_gen, new_X_val_gen = (None, None)
        if (isinstance(X_gen, MemDataIterator)):
            new_X_gen = self.__save_encoded_data_and_create_mem_generator(layer_idx, ae, X_gen)

            # Create a generator for validation
            if (X_val_gen is not None):
                new_X_val_gen = self.__save_encoded_data_and_create_mem_generator(layer_idx, ae, X_val_gen)

        elif (isinstance(X_gen, DskDataIterator)):

            patch_dir = os.path.join(save_to_dir, "ptmp_" + str(patch_idx))
            if not os.path.exists(patch_dir):
                os.makedirs(patch_dir)

            fname = "tmp_m_" + str(self.uid) + ".l_" + str(layer_idx) + ".TR.dat"
            fpath = os.path.join(patch_dir, fname)
            new_X_gen = self.__save_encoded_data_and_create_dsk_generator(fpath, ae, X_gen)

            if (X_val_gen is not None):
                fname = "tmp_m_" + str(self.uid) + ".l_" + str(layer_idx) + ".VAL.dat"
                fpath = os.path.join(patch_dir, fname)
                new_X_val_gen = self.__save_encoded_data_and_create_dsk_generator(fpath, ae, X_val_gen)

        if layer_idx > 1:  # Skip the iterator created by NNF
            # Release the iterators created temporary
            X_gen.release()
            if (X_val_gen is not None): X_val_gen.release()

        return (new_X_gen, new_X_val_gen)

    def __save_encoded_data_and_create_mem_generator(self, layer_idx, ae, memgen):
        """Save the encoded data and create a memory dsk generator for next layer pre-training.

        Returns
        -------
        :obj:`MemDataIterator`
            New memory data iterator that generates data from the memory.

        Notes
        -----
        This method is only compatible with single input generator but not multiple input generators that generates
        multiple inputs in one loop.
        """
        if memgen.has_multiple_inputs or memgen.has_multiple_targets:
            raise Exception("Models with multiple inputs or targets are not supported.")

        # Iterate with the current generator and encode, save data
        params = memgen.params.copy()
        params['batch_size'] = memgen.nb_sample
        params['shuffle'] = False

        # Reinit current dsk generator
        memgen.init_ex(params)

        X_batch, _ = next(memgen)
        enc_data = ae._encode(X_batch)

        # Create a new core iterator
        fn_gen_coreiter = lambda X, y, nb_class, image_data_generator, params: \
            BigDataNumpyArrayIterator(X, y, nb_class, image_data_generator, params)

        # Params to create a new mem generator
        params['class_mode'] = None  # X -> h - > X, reconstructing the input itself
        params['shuffle'] = memgen.params['shuffle']
        params['_use_rgb'] = None
        params['window_iter'] = None
        params['input_shape'] = None

        nndb = NNdb('Temp_Layer:' + str(layer_idx), enc_data, cls_lbl=memgen.nndb.cls_lbl, db_format=Format.N_H)
        memgen = MemDataIterator(memgen.edataset, nndb, None, fn_gen_coreiter)
        memgen.init_ex(params)

        return memgen

    def __save_encoded_data_and_create_dsk_generator(self, fpath, ae, dskgen):
        """Save the encoded data and create a new dsk generator for next layer pre-training.

        Returns
        -------
        :obj:`DskDataIterator`
            New disk data iterator that generates data from the disk.

        Notes
        -----
        This method is only compatible with single input generator but not multiple input generators that generates
        multiple inputs in one loop.
        """
        if dskgen.has_multiple_inputs or dskgen.has_multiple_targets:
            raise Exception("Models with multiple inputs or targets are not supported.")

        # Iterate with the current generator and encode, save data
        params = dskgen.params.copy()
        params['class_mode'] = 'sparse'  # This is enforce to return the class labels of the primary generator
        params['shuffle'] = False

        # Reinit current dsk generator
        dskgen.init_ex(params)

        # Fetch new target size (the encoder layer size)
        enc_layer_size = (ae.enc_size, 1)

        # Calculate when to stop
        nloops = math.ceil(dskgen.nb_sample / dskgen.batch_size)

        # Open a file for writing (binary mode)
        f = open(fpath, "wb")

        # file records array
        frecords = []
        fpos = 0

        # Breaks when one round of iteration on dataset is performed
        # Default behavior: infinite looping over and over again on the dataset
        for i, batch in enumerate(dskgen):
            X_batch, Y_batch = batch[0], batch[1]
            enc_data = ae._encode(X_batch)
            enc_data.tofile(f)

            # Create new frecords for the new generator
            for isample in range(len(X_batch)):  # No of samples

                # Preserve the class labels of the primary generator while progressing layer-wise
                # Useful for the models where the last layer need to be pre-trained with class labels
                cls_lbl = None if Y_batch is None else Y_batch[isample]

                # Is first frecord
                if len(frecords) == 0:
                    frecords.append([fpath, np.float32(fpos), cls_lbl])
                else:
                    # PERF: Memory efficient implementation (only first frecord of the class has the path to the file)
                    frecords.append([0, np.float32(fpos), cls_lbl])

                fpos += (enc_layer_size[0] * enc_data.dtype.itemsize)

            # Break when full dataset is traversed once
            if (i + 1 >= nloops):
                break
        f.close()

        # Create a new core generator
        fn_gen_coreiter = lambda frecords, nb_class, image_data_pp, params: \
            BigDataDirectoryIterator(frecords,
                                     nb_class,
                                     image_data_pp,
                                     params)
        # Params to create a new dsk generator
        params['class_mode'] = None  # X -> h - > X, reconstructing the input itself
        params['shuffle'] = dskgen.params['shuffle']
        params['_use_rgb'] = None
        params['window_iter'] = None
        params['input_shape'] = None
        params['target_size'] = enc_layer_size
        params['file_format'] = FileFormat.BINARY

        dskgen = DskDataIterator(dskgen.edataset, frecords, dskgen.nb_class, fn_gen_coreiter=fn_gen_coreiter)
        dskgen.init_ex(params=params)

        return dskgen