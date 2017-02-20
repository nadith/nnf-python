# -*- coding: utf-8 -*- TODO: COmment
# Global Imports
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import os
import math

# Local Imports
from nnf.core.models.NNModel import NNModel
from nnf.core.models.Autoencoder import Autoencoder
from nnf.core.iters.DataIterator import DataIterator
from nnf.db.NNdb import NNdb
from nnf.db.Format import Format
from nnf.db.Dataset import Dataset
from nnf.core.iters.memory.MemDataIterator import MemDataIterator
from nnf.core.iters.memory.BigDataNumpyArrayIterator import BigDataNumpyArrayIterator
from nnf.core.iters.disk.DskDataIterator import DskDataIterator
from nnf.core.iters.disk.BigDataDirectoryIterator import BigDataDirectoryIterator
from nnf.core.models.NNModelPhase import NNModelPhase

class DAEModel(NNModel):
    """Generic deep autoencoder model.

    Note
    ----
    Extend this class to implement custom deep autoencoder models.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, callbacks=None):
        super().__init__()

        # Set defaults for arguments
        self.callbacks = {} if (callbacks is None) else callbacks
        self.callbacks.setdefault('test', None)
        self.callbacks.setdefault('predict', None)
        get_dat_gen = self.callbacks.setdefault('get_data_generators', None)
        if (get_dat_gen is None):
            self.callbacks['get_data_generators'] = self.get_data_generators

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
        if (ephase == NNModelPhase.PRE_TRAIN or ephase == NNModelPhase.TRAIN):
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
    def _pre_train(self, daeprecfgs, daecfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
        """Pre-train the :obj:`DAEModel`.

        Parameters
        ----------
        daeprecfgs : list of :obj:`NNCfg`
            List of Neural Network configurations. Useful for layer-wise pre-training.

        daecfg : :obj:`NNCfg`
            Neural Network configuration that will be used in training. 
            Useful to build the deep stacked network after layer-wise pre-trianing.

        patch_idx : int
            Patch's index in this model.

        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user db-param of each `nnpatch`.

        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictonary of iterstores for :obj:`DataIterator`.
        """
        # Validate nncfgs
        for _, daeprecfg in enumerate(daeprecfgs):
            self._validate_cfg(NNModelPhase.PRE_TRAIN, daeprecfg)
        self._validate_cfg(NNModelPhase.TRAIN, daecfg)

        # Initialize parameters
        save_to_dir = None
        if (dbparam_save_dirs is not None):
            save_to_dir = dbparam_save_dirs[0]  # save_to_dir for dbparam1

        # Initialize data generators
        X_gen, X_val_gen = self._init_data_generators(NNModelPhase.PRE_TRAIN, list_iterstore, dict_iterstore)

        # Pre-condition asserts
        assert((daecfg.preloaded_db is None and X_gen is not None) or 
                (daecfg.preloaded_db is not None))

        # Track layers to build Deep/Stacked Auto Encoder (DAE)
        layers = [Input(shape=(daecfg.arch[0],))]

        # Initialize weights and transpose of weights
        w_T = w = None       

        # For Preloaded Databases
        X_L = Xt = X_L_val = Xt_val = None
        is_preloaded_db = (daecfg.preloaded_db is not None)
        if (is_preloaded_db): daecfg.preloaded_db.reinit('default')

        # map of iterators for TR|VAL|TE|... datasets in Autoencoder
        ae_iterstore = {}

        # Iterate through pre-trianing configs and 
        # perform layer-wise pre-training.
        layer_idx = 1
        for daeprecfg in daeprecfgs:

            # Explicit uid definition for simple AE
            ae_uid = 'dae_' + str(self.uid) + '_ae_' + str(layer_idx)
    
            # Using non-generators for external databases
            if (is_preloaded_db):
                # Create a simple AE
                if (X_L is None and Xt is None and X_L_val is None and Xt_val is None):
                    daeprecfg.preloaded_db.reinit('default')
                    X_L, Xt, X_L_val, Xt_val = daeprecfg.preloaded_db.LoadPreTrDb(self)

                # Create a simple AE
                ae = Autoencoder(ae_uid, X_L, Xt, X_L_val, Xt_val)

                # Train the simple AE
                ae.train(daeprecfg, patch_idx)

                # Post process datasets and prepare for next round                
                X_L, Xt, X_L_val, Xt_val =\
                    self._pre_train_postprocess_preloaded_db(layer_idx, daecfg, ae, X_L, Xt, X_L_val, Xt_val)

            else:
                # Callback for pre-training pre-processing
                X_gen, X_val_gen = self._pre_train_preprocess(layer_idx, daeprecfgs, daecfg, X_gen, X_val_gen)

                # Set data for simple AE
                ae_iterstore[Dataset.TR] = X_gen
                ae_iterstore[Dataset.VAL] = X_val_gen

                if (ae_iterstore == [(None, None)]):
                    raise Exception("No data iterators. Check daecfg.use_db option !")   

                # Create a simple AE
                ae = Autoencoder(ae_name)
                ae._add_iterstores(list_iterstore=[ae_iterstore])

                print("\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< AUTOENCODER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # Using generators for NNDb or disk databases            
                ae.train(daeprecfg)

                # Fetch generator for next layer (i+1)
                X_gen, X_val_gen = self.__get_genrators_at(patch_idx, save_to_dir, layer_idx, ae, X_gen, X_val_gen)

            layers = self._stack_layers(ae, layer_idx, layers, daeprecfgs, daecfg)

            # Increment layer index
            layer_idx = layer_idx + 1

        # Build DAE
        self.__build(layers, daecfg)
    
    def _train(self, daecfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
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
            Dictonary of iterstores for :obj:`DataIterator`.
        """
        # Validate nncfg
        self._validate_cfg(NNModelPhase.TRAIN, daecfg)

        # Initialize data generators
        X_gen, X_val_gen = self._init_data_generators(NNModelPhase.TRAIN, list_iterstore, dict_iterstore)

        # Pre-condition asserts
        assert((daecfg.preloaded_db is None and X_gen is not None) or 
                (daecfg.preloaded_db is not None))

        # Build the DAE if not already built (no-pretraining)
        if (self.net is None):
            self._build(daecfg)

        # Preloaded databases for quick deployment
        if (daecfg.preloaded_db is not None):
            daecfg.preloaded_db.reinit('default')
            X_L, Xt, X_L_val, Xt_val = daecfg.preloaded_db.LoadTrDb(self)
 
        # Training without generators
        if (daecfg.preloaded_db is not None):
            super()._start_train(daecfg, X_L, Xt, X_L_val, Xt_val)
            return (X_L, Xt, X_L_val, Xt_val)

        # Training with generators
        super()._start_train(daecfg, X_gen=X_gen, X_val_gen=X_val_gen)

        # Save the trained model
        self._try_save(daecfg, patch_idx, "DAE")

        return (None, None, None, None)

    def _test(self, daecfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
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
            Dictonary of iterstores for :obj:`DataIterator`.
        """    
        # Initialize data generators
        Xte_gen, Xte_target_gen = self._init_data_generators(NNModelPhase.TEST, list_iterstore, dict_iterstore)

        # Pre-condition asserts
        assert((daecfg.preloaded_db is None and Xte_gen is not None) or 
                (daecfg.preloaded_db is not None))

        if (Xte_target_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(Xte_gen.nb_sample == Xte_target_gen.nb_sample)

        # 1st Priority, Load the saved model with weights
        is_loaded = self._try_load(daecfg, patch_idx, "DAE")

        # 2nd Priority, Load the saved weights but not the model
        if (not is_loaded and daecfg.weights_path is not None):
            assert(False)  # TODO: Pretraining must be skipped and it should only build the model with daecfg
            #self._build(daecfg, Xte_gen)
            #self.net.load_weights(daecfg.weights_path)

        assert(self.net is not None)

        # Preloaded databases for quick deployment
        if (daecfg.preloaded_db is not None):
            daecfg.preloaded_db.reinit('default')
            X_L_te, Xt_te = daecfg.preloaded_db.LoadTeDb(self)     

        # Test without generators
        if (daecfg.preloaded_db is not None):
            super()._start_test(patch_idx, X_L_te, Xt_te)
            return

        # Test with generators
        super()._start_test(patch_idx, Xte_gen=Xte_gen, Xte_target_gen=Xte_target_gen)

    def _predict(self, daecfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
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
            Dictonary of iterstores for :obj:`DataIterator`.
        """
        # Initialize data generators
        Xte_gen, Xte_target_gen = self._init_data_generators(NNModelPhase.PREDICT, list_iterstore, dict_iterstore)

        # Pre-condition asserts
        assert((daecfg.preloaded_db is None and Xte_gen is not None) or 
                (daecfg.preloaded_db is not None))

        if (Xte_target_gen is not None):
            # No. of testing and testing target samples must be equal
            assert(Xte_gen.nb_sample == Xte_target_gen.nb_sample)

        # 1st Priority, Load the saved model with weights
        is_loaded = self._try_load(daecfg, patch_idx, "DAE")

        # 2nd Priority, Load the saved weights but not the model
        if (not is_loaded and daecfg.weights_path is not None):
            assert(False)  # TODO: Pretraining must be skipped and it should only build the model with daecfg
            #self._build(daecfg, Xte_gen)
            #self.net.load_weights(daecfg.weights_path)

        assert(self.net is not None)

        # Preloaded databases for quick deployment
        if (daecfg.preloaded_db is not None):
            daecfg.preloaded_db.reinit('default')
            X_L_te, Xt_te = daecfg.preloaded_db.LoadPredDb(self)

        # Predict without generators
        if (daecfg.preloaded_db is not None):
            super()._start_predict(patch_idx, X_L_te, Xt_te)
            return

        # Predict with generators
        super()._start_predict(patch_idx, Xte_gen=Xte_gen, Xte_target_gen=Xte_target_gen)

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _validate_cfg(self, ephase, cfg):
        if (len(cfg.arch) != len(cfg.act_fns)):
            raise Exception('Activation function for each layer is not' + 
            ' specified. (length of `cfg.arch` != length of `cfg.act_fns`).')       

    def _init_data_generators(self, ephase, list_iterstore, dict_iterstore):
        """Initialize data generators for pre-training, training, testing, prediction.

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
            When ephase == NNModelPhase.PRE_TRAIN
            then 
                Generators for pre-training and validation. 
                But cloned before use.

            When ephase == NNModelPhase.TRAIN
            then 
                Generators for training and validation.

            When ephase == NNModelPhase.TEST or NNModelPhase.PREDICT
            then
                Generators for testing and testing target.
        """
        X_gen = None; X_val_gen = None
        if (list_iterstore is not None):
            X_gen, X_val_gen = self.callbacks['get_data_generators'](ephase, list_iterstore, dict_iterstore)

            if (ephase == NNModelPhase.PRE_TRAIN):  # Pre-training stage
                # Take a copy since we are going to manipulate it                
                X_gen = self._clone_iter(X_gen)
                X_val_gen = self._clone_iter(X_val_gen)

        return X_gen, X_val_gen

    def _pre_train_preprocess(self, layer_idx, daeprecfgs, daecfg, X_gen, X_val_gen):
        """describe"""
        return X_gen, X_val_gen

    def _pre_train_postprocess_preloaded_db(self, layer_idx, daecfg, ae, X_L, Xt, X_L_val, Xt_val):

        # Fix variables for the next layer pre-training
        X_L = (ae._encode(X_L[0]), None)
        Xt = X_L[0]
        X_L_val = (ae._encode(X_L_val[0]), None)
        Xt_val = X_L_val[0]

        return  X_L, Xt, X_L_val, Xt_val

    def _stack_layers(self, ae, layer_idx, layers, daeprecfgs, daecfg):
        # Tied weights
        w = ae._encoder_weights
        w_T = [np.transpose(w[0]), np.zeros(w[0].shape[0])]

        # Adding encoding layer for DAE
        layers.insert(layer_idx, 
                        Dense(daecfg.arch[layer_idx], 
                                activation=daecfg.act_fns[layer_idx],
                                weights=w, 
                                name="enc: " + str(layer_idx)))

        # Adding decoding layer for DAE
        dec_i = len(daecfg.arch)-layer_idx
        layers.insert(layer_idx+1, 
                        Dense(daecfg.arch[dec_i], 
                                activation=daecfg.act_fns[dec_i], 
                                weights=w_T, 
                                name="dec: " + str(layer_idx)))
        return layers

    def _build(self, daecfg):
        """Build the DAE.
        
            Used when the network needs to be build without pretraining.
        """
        layers = [Input(shape=(daecfg.arch[0],))]
        mid = (len(daecfg.arch) + 1) // 2
        for i, dim in enumerate(daecfg.arch[1:mid]):
            layer_idx = i + 1
            layers.insert(layer_idx, 
                            Dense(daecfg.arch[layer_idx], 
                                    activation=daecfg.act_fns[layer_idx],
                                    name="enc: " + str(layer_idx)))

            dec_i = len(daecfg.arch)-layer_idx
            layers.insert(layer_idx+1, 
                            Dense(daecfg.arch[dec_i], 
                                    activation=daecfg.act_fns[dec_i],
                                    name="dec: " + str(layer_idx)))

        # Build DAE
        self.__build(layers, daecfg)

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __build(self, layers, daecfg):

        in_tensor = layers[0]
        out_tensor = layers[0]

        for i in range(1, len(layers)):
            out_tensor = layers[i](out_tensor)
        
        self.net = Model(input=in_tensor, output=out_tensor)
        print(self.net.summary())

        self.net.compile(optimizer=daecfg.opt_fn, loss=daecfg.loss_fn)
        self._init_fns_predict_feature(daecfg)  

    def __get_genrators_at(self, patch_idx, save_to_dir, layer_idx, ae, X_gen, X_val_gen):
        """describe

        Parameters
        ----------
        i : Describe
            describe.

        ae : Describe
            describe.

        X_gen : Describe
            describe.

        X_val_gen : Describe
            describe.

        sel : selection structure
            Information to split the dataset.

        Returns
        -------
        X_gen : describe

        X_val_gen : describe
        """
        # If the current generators are pointing to a temporary location, mark it for deletion
        # predict from current X_gen and X_val_gen and save the data in a temporary location 
        # for index i (layer i). Construct new generators for the data stored in this temporary location.
        # if the previous data locations are marked for deletion (temporary location), proceed the deletion
        
        # TODO: release X_gen._release() resources

        if (isinstance(X_gen, MemDataIterator)):

            # Create a new core iterator
            fn_gen_coreiter = lambda X, y, nb_class, image_data_generator, params:\
                                BigDataNumpyArrayIterator(X, y, nb_class, image_data_generator, params)

            # Create a generator for training
            enc_data = ae._encode(X_gen.nndb.features_scipy)     
            params = X_gen.params
            X_gen = MemDataIterator(X_gen.edataset, NNdb('Temp_Layer:'+str(layer_idx), enc_data, format=Format.N_H), X_gen.nndb.cls_n, None, fn_gen_coreiter)
            X_gen.init(params)

            # Create a generator for validation
            if (X_val_gen is not None):
                enc_data_val = ae._encode(X_val_gen.nndb.features_scipy)
                params = X_val_gen.params
                X_val_gen = MemDataIterator(X_val_gen.edataset, NNdb('TempVal_Layer:'+str(layer_idx), enc_data_val, format=Format.N_H), X_val_gen.nndb.cls_n, None, fn_gen_coreiter)
                X_val_gen.init(params)

        elif (isinstance(X_gen, DskDataIterator)):
            
            patch_dir = os.path.join(save_to_dir, "ptmp_" + str(patch_idx))
            if not os.path.exists(patch_dir):
                os.makedirs(patch_dir)

            fname = "tmp_m_" + str(self.uid) + ".l_" + str(layer_idx) + ".TR.dat"
            fpath = os.path.join(patch_dir, fname)
            X_gen = self.__save_encoded_data_and_create_dsk_generator(fpath, ae, X_gen)

            if (X_val_gen is not None):
                fname = "tmp_m_" + str(self.uid) + ".l_" + str(layer_idx) + ".VAL.dat"
                fpath = os.path.join(patch_dir, fname)
                X_val_gen = self.__save_encoded_data_and_create_dsk_generator(fpath, ae, X_val_gen)

        return (X_gen, X_val_gen)

    def __save_encoded_data_and_create_dsk_generator(self, fpath, ae, cur_dskgen):
        """Save the encoded data and create a new dsk generator for next iteration"""

        # Iterate with the current generator and encode, save data        
        params = cur_dskgen.params.copy()
        params['class_mode'] = 'sparse'
        params['shuffle'] = False

        # Reinit current dsk generator
        cur_dskgen.init(params)  

        # Fetch new target size (the encoder layer size)
        enc_layer_size = (ae.enc_size, )

        # Calculate when to stop
        nloops = math.ceil(cur_dskgen.nb_sample / cur_dskgen.batch_size)
            
        # Open a file for writing (binary mode)
        f = open(fpath, "wb")

        # file records array
        frecords = []
        fpos = 0

        # Breaks when one round of iteration on dataset is performed
        # Default bhavior: inifnite looping over and over again on the dataset
        for i, batch in enumerate(cur_dskgen):
            X_batch, Y_batch = batch[0], batch[1]
            enc_data = ae._encode(X_batch)
            enc_data.tofile(f)
                
            # Update frecords
            if (not cur_dskgen.is_synced):
                for cls_lbl in Y_batch:
                    frecords.append([fpath, np.float32(fpos), np.uint16(cls_lbl)])
                    fpos = enc_layer_size[0] * enc_data.dtype.itemsize

            else:
                for _ in range(len(X_batch)):  # No of samples
                    frecords.append([fpath, np.float32(fpos), np.uint16(0)])
                    fpos = enc_layer_size[0] * enc_data.dtype.itemsize
                        
            # Break when full dataset is traversed once
            if (i + 1 >= nloops):
                break

        f.close()

        # Create a new core generator
        fn_gen_coreiter = lambda frecords, nb_class, image_data_generator, params:\
                            BigDataDirectoryIterator(frecords,
                                                    nb_class,
                                                    image_data_generator,
                                                    params)
        # Params for create a new dsk generator  
        params['class_mode'] = None
        params['shuffle'] = True
        params['color_mode'] = None
        params['target_size'] = enc_layer_size
        params['binary_data'] = True

        new_dskgen = DskDataIterator(cur_dskgen.edataset, frecords, cur_dskgen.nb_class, fn_gen_coreiter=fn_gen_coreiter)
        new_dskgen.init(params)
    
        return new_dskgen