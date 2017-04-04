# -*- coding: utf-8 -*-
"""
.. module:: NNModel
   :platform: Unix, Windows
   :synopsis: Represent NNModel class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from abc import ABCMeta, abstractmethod
import os
import math
import numpy as np
from warnings import warn as warning
from keras.models import load_model
from warnings import warn as warning
from keras import backend as K

# Local Imports
from nnf.db.Dataset import Dataset
from nnf.core.models.NNModelPhase import NNModelPhase

# Circular Imports
# ref:http://stackoverflow.com/questions/22187279/python-circular-importing
import nnf.db.NNPatch

class NNModel(object):
    """NNModel represents base class for Neural Network Models.

    .. warning:: abstract class and must not be instantiated.

    Maintain assoiated 'nnpatches' along with the 'iteratorstores' and 
    the related paths to the directories to save temporary data both
    user dbparam-wise and patch-wise. 
    (See also DAEModel)

    Attributes
    ----------
    uid : int or str
        Unique id of this model across the framework. Can also set a custom
        id for models created inside another model. 
        Ref:`DAEModel` creating `Autoencoder`

    nnpatches : list of :obj:`NNPatch`
        Associated `nnpatches` with this model.

    _iteratorstores : list of :obj:`tuple`
        Each tuple consists of `dict_iterstore` and `list_iterstore' for each `nnpatch`.

    _list_save_to_dirs : list of :obj:`list`
        List of paths to temporary directories for each user db-param of each `nnpatch`.

    net : :obj:`keras.Model`
        Core network model (keras).

    fns_predict_feature :
        Keras/teano sub functions to predict each feature.

    feature_sizes :
        Feature size for each prediction.

    callbacks : :obj:`dict`
        Callback dictionary. Supported callbacks.
        {`test`, `predict`, `get_data_generators`}
    """
    __metaclass__ = ABCMeta

    ##########################################################################
    # Public Interface
    ##########################################################################
    
    #  [STATIC] Constants
    _W_FILE_EXT = ".weights.h5"
    _M_FILE_EXT = ".model.h5"

    # [STATIC] Unique id dynamic base value
    _UID_BASE = -1

    @staticmethod
    def get_uid():
        """[STATIC] Get an unique id to this `nnmodel` across the framework.
 
        .. warning:: Must be invoked during the instance creation to preserve thread safety.

        Returns
        -------
        int
            Unique id starting from 0.
        """
        NNModel._UID_BASE = NNModel._UID_BASE + 1
        return NNModel._UID_BASE

    @staticmethod
    def reset_uid():
        """[STATIC] Reset the static uid in each test case in the test suit."""
        NNModel._UID_BASE = -1

    def __init__(self, uid=None, callbacks=None):
        """Constructor of the abstract class :obj:`NNModel`.

        Notes
        -----
        `uid` is not None for nnmodels that are created inside a nnmodel. 
        i.e Autoencoder in DAEModel
        """
        # Assign unique id
        if (uid is None):
            self.uid = NNModel.get_uid()
        else:
            self.uid = uid

        # Initialize instance variables
        # Iteartorstores format = [ (dict_iterstore, list_iterstore) for nnpatch_1, 
        #                           (dict_iterstore, list_iterstore) for nnpatch_2
        #                           (dict_iterstore, list_iterstore) for nnpatch_3
        #                           (dict_iterstore, list_iterstore) for nnpatch_4
        #                           ...
        #                           ]
        self._iteratorstores = []

        # To save temporary encoded data
        # [ [folder_for_param_1_db, folder_for_param_2_db] for nnpatch_1]
        #   [folder_for_param_1_db, folder_for_param_2_db] for nnpatch_2]
        #   ...
        #   ]
        self._list_save_to_dirs = []

        # Associated nnpatches with this model
        # len(self._iteratorstores) == len(self.nnpatches)
        # len(self._list_save_to_dirs) == len(self.nnpatches)
        self.nnpatches = [] 

        # Core network model (keras).
        self.net = None

        # Keras/teano sub functions to predict each feature
        self.fns_predict_feature = []

        # Feature sizes for each prediction
        self.feature_sizes = []

        # Set defaults for general callbacks
        self.callbacks = {} if (callbacks is None) else callbacks
        self.callbacks.setdefault('test', None)
        self.callbacks.setdefault('predict', None)
        get_dat_gen = self.callbacks.setdefault('get_data_generators', None)

        # Use `_get_data_generators` method as default to fetch data generators
        if (get_dat_gen is None):
            self.callbacks['get_data_generators'] = self._get_data_generators
    
    def pre_train(self, precfgs, cfg, patch_idx=None):
        """Pre-train the :obj:`NNModel`.

        Parameters
        ----------
        precfgs : list of :obj:`NNCfg`
            List of Neural Network configurations. Useful for layer-wise pre-training.

        cfg : :obj:`NNCfg`
            Neural Network configuration that will be used in training. 
            Useful to build the deep stacked network after layer-wise pre-trianing.

        patch_idx : int, optional
            Patch's index in this model. (Default value = None).

        Notes
        -----
        Some of the layers may not be pre-trianed. Hence precfgs itself is
        not sufficient to determine the architecture of the final 
        stacked network.
        """
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PRE-TRAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.__common_train_test_predict_routine(self._pre_train, cfg, patch_idx, True, precfgs=precfgs)

    def train(self, cfg, patch_idx=None):
        """Train the :obj:`NNModel`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int, optional
            Patch's index in this model. (Default value = None).
        """
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TRAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        # For models created inside another model and using preloaded dbs.
        # Ref:`DAEModel` creating `Autoencoder`
        if (len(self._iteratorstores) == 0):            
            return self._train(cfg, patch_idx)

        self.__common_train_test_predict_routine(self._train, cfg, patch_idx, True)

    def test(self, cfg, patch_idx=None):
        """Train the :obj:`NNModel`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int, optional
            Patch's index in this model. (Default value = None).
        """        
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TEST >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.__common_train_test_predict_routine(self._test, cfg, patch_idx)

    def predict(self, cfg, patch_idx=None):
        """Train the :obj:`NNModel`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int
            Patch's index in this model.
        """
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PREDICT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.__common_train_test_predict_routine(self._predict, cfg, patch_idx)

    @abstractmethod
    def generate_nnpatches(self):
        """Generate list of :obj:`NNPatch` for Neural Network Model Based Framework.

        Returns
        -------
        List of :obj:`NNPatch`
            `nnpatches` for Neural Network Model Based Framework.

        Notes
        -----
        Invoked by :obj:`NNModelMan`. 

        Note
        ----
        Used only in Model Based Framework. Extend this method to implement custom 
        generation of `nnpatches`.    
        """
        nnpatches = []
        nnpatches.append(nnf.db.NNPatch.NNPatch(33, 33, (0, 0)))
        nnpatches.append(nnf.db.NNPatch.NNPatch(33, 33, (10, 10)))
        return nnpatches

    ##########################################################################
    # Protected Interface
    ##########################################################################
    @abstractmethod
    def _model_prefix(self):
        """Fetch the prefix for the file to be saved/loaded.

        Note
        ----
        Override this method for custom prefix.
        """
        return "PFX"

    @abstractmethod
    def _pre_train(self, precfgs, cfg, patch_idx=None, dbparam_save_dirs=None,
                                    list_iterstore=None, dict_iterstore=None):
        """Pre-train the :obj:`NNModel`.

        Parameters
        ----------
        precfgs : list of :obj:`NNCfg`
            List of Neural Network configurations. Useful for layer-wise pre-training.

        cfg : :obj:`NNCfg`
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
        pass

    @abstractmethod
    def _train(self, cfg, patch_idx=None, dbparam_save_dirs=None,
                                    list_iterstore=None, dict_iterstore=None):
        """Train the :obj:`NNModel`.

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
        pass

    @abstractmethod
    def _test(self, cfg, patch_idx=None, dbparam_save_dirs=None,
                                    list_iterstore=None, dict_iterstore=None):
        """Test the :obj:`NNModel`.

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
        pass

    @abstractmethod
    def _predict(self, cfg, patch_idx=None, dbparam_save_dirs=None,
                                    list_iterstore=None, dict_iterstore=None):
        """Predict the :obj:`NNModel`.

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
        pass

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _clone_iter(self, iter):
        """Clone the iterator.

        Parameters
        ----------
        iter : :obj:`DskDataIterator` or :obj:`MemDataIterator`
            Clone is supported only on those two iterators.

        Note
        ----
        If the original iterators need to be preserved for the second phase 
        of training.
        i.e 
        In pretraing, clone the iterators, in training, use the originals
        """
        return iter.clone() if (iter is not None) else None

    def _debug_print(self, list_iterstore):
        """Print information for each iterator store in `list_iterstore`.

            The iterator params and pre-processor params of iterator store
            for each dataset. 
            i.e (Dataset.TR | VAL | ...).

        Parameters
        ----------
        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.
        """
        def __print_params(iterstore, edataset):
            if (edataset not in iterstore): return
            gen = iterstore[edataset]
            if (gen is not None):
                if (gen._params is not None):
                    print('{} : {}'.format(edataset, gen))
                    print("\tIterator Parameters: (iter_param, iter_pp_param)")
                    print("\t-------------------------------------------------")
                    for key, val in gen._params.items():
                        if (key.startswith('_')): continue
                        print('\t{} : {}'.format(key, val))
                
                if (gen._pp_params is not None):
                    print("\t-------------------------------------------------")
                    for key, val in gen._pp_params.items():
                        if (key.startswith('_')): continue
                        print('\t{} : {}'.format(key, val))
                print("")

        if (list_iterstore is None):
            return

        for i, iterstore in enumerate(list_iterstore):
            print("\nIterator Store:{}".format(i))
            print("=================")
            __print_params(iterstore, Dataset.TR)
            __print_params(iterstore, Dataset.VAL)
            __print_params(iterstore, Dataset.TE)
            __print_params(iterstore, Dataset.TR_OUT)
            __print_params(iterstore, Dataset.VAL_OUT)
            __print_params(iterstore, Dataset.TE_OUT)

    def _init_data_generators(self, ephase, list_iterstore, dict_iterstore):
        """Initialize data generators for [pre-training,] training, testing, prediction.

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

    def _get_data_generators(self, ephase, list_iterstore, dict_iterstore):
        """Get data generators for [pre-training,] training, testing, prediction.

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
    # Protected: Train, Test, Predict Common Routines Of Keras
    ##########################################################################
    def _start_train(self, cfg, X_L=None, Xt=None, X_L_val=None, Xt_val=None, 
                                                X_gen=None, X_val_gen=None):
        """Common routine to start the training phase of `NNModel`.

        Parameters
        ----------
        patch_idx : int, optional
            Patch's index in this model. (Default value = None).

        X_L : :obj:`tuple`
            (`array_like` data tensor, labels).
            If the `nnmodel` is not expecting labels, set it to None.

        Xt : `array_like`
            Target data tensor. If the `nnmodel` is not expecting a 
            target data tensor, set it to None.

        X_L_val : :obj:`tuple`
            (`array_like` validation data tensor, labels).
            If the `nnmodel` is not expecting labels, set it to None.

        Xt_val : `array_like`
            Target validation data tensor. If the `nnmodel` is not expecting a
            target validation data tensor, set it to None.

        X_gen : :obj:`DataIterator`
            Data iterator that generates data in 
            (`array_like`, labels or `array_like`) format, depending on
            the `nnmodel` architecture.

        X_val_gen : :obj:`DataIterator`
            Validation data iterator that generates data in 
            (`array_like`, labels or `array_like`) format, depending on
            the `nnmodel` architecture.
        """
        assert((X_L is not None) or (X_gen is not None))

        # Train from preloaded database
        if (X_L is not None):
            if (X_L_val is not None):
                
                X, lbl = X_L
                X_val, lbl_val = X_L_val

                # Train with labels
                if (lbl is not None):
                    self.net.fit(X, lbl, epochs=cfg.numepochs, batch_size=cfg.batch_size, callbacks=cfg.callbacks, shuffle=True, validation_data=(X_val, lbl_val))  #, callbacks=[self.cb_early_stop])

                # Train with targets
                elif (lbl is None):
                    self.net.fit(X, Xt, epochs=cfg.numepochs, batch_size=cfg.batch_size, callbacks=cfg.callbacks, shuffle=True, validation_data=(X_val, Xt_val))  #, callbacks=[self.cb_early_stop])

            else:
                X, lbl = X_L
                X_val, lbl_val = X_L_val

                # Train with labels
                if (lbl is not None):
                    self.net.fit(X, lbl, epochs=cfg.numepochs, batch_size=cfg.batch_size, callbacks=cfg.callbacks, shuffle=True) 

                # Train without targets
                elif (lbl is None):
                    self.net.fit(X, Xt, epochs=cfg.numepochs, batch_size=cfg.batch_size, callbacks=cfg.callbacks, shuffle=True) 
                  
        # Train from data generators
        else:
            if (X_val_gen is not None):
                self.net.fit_generator(
                        X_gen, steps_per_epoch=cfg.steps_per_epoch,
                        epochs=cfg.numepochs, callbacks=cfg.callbacks,
                        validation_data=X_val_gen, validation_steps=cfg.nb_val_samples) # callbacks=[self.cb_early_stop]
            else:
                self.net.fit_generator(
                        X_gen, steps_per_epoch=cfg.steps_per_epoch,
                        epochs=cfg.numepochs, callbacks=cfg.callbacks)

    def _start_test(self, patch_idx=None, X_L_te=None, Xt_te=None, 
                                            X_te_gen=None, Xt_te_gen=None):
        """Common routine to start the testing phase of `NNModel`.

        Parameters
        ----------
        patch_idx : int, optional
            Patch's index in this model. (Default value = None).

        X_L_te : :obj:`tuple`
            (`array_like` test data tensor, labels).
            If the `nnmodel` is not expecting labels, set it to None.

        Xt_te : `array_like`
            Target test data tensor. If the `nnmodel` is not expecting a
            target test data tensor, set it to None.

        X_te_gen : :obj:`DataIterator`
            Test data iterator that generates data in 
            (`array_like`, labels or `array_like`) format, depending on
            the `nnmodel` architecture.

        Xt_te_gen : :obj:`DataIterator`
            Target test data iterator that generates data in 
            (`array_like`, labels or `array_like`) format, depending on
            the `nnmodel` architecture.
        """
        assert((X_L_te is not None) or (X_te_gen is not None))
        assert(self.net is not None)

        # Test from preloaded database
        if (X_L_te is not None):

            Xte, lbl = X_L_te

            # Dictionary to collect loss and accuracy for batches
            metrics = {}
            for mname in self.net.metrics_names:
                metrics.setdefault(mname, [])

            # Test with labels
            if (lbl is not None):
                eval_res = self.net.evaluate(Xte, lbl, verbose=1)

                # Accumilate metrics into a list 
                for mi, mname in enumerate(self.net.metrics_names):
                    metrics[mname].append(eval_res if np.isscalar(eval_res) else eval_res[mi])

            # Train with targets
            elif (Xt_te is not None):
                eval_res = self.net.evaluate(Xte, Xt_te, verbose=1)

                # Accumilate metrics into a list 
                for mi, mname in enumerate(self.net.metrics_names):
                    metrics[mname].append(eval_res if np.isscalar(eval_res) else eval_res[mi])

            else:
                raise Exception("Unsupported mode in testing...")

        # Test from data generators
        else:  
            # Test with labels or target
            if (Xt_te_gen is not None):
                X_te_gen.sync_generator(Xt_te_gen)

            # Calculate when to stop
            nloops = math.ceil(X_te_gen.nb_sample / X_te_gen.batch_size)

            # Dictionary to collect loss and accuracy for batches
            metrics = {}
            for mname in self.net.metrics_names:
                metrics.setdefault(mname, [])

            for i, batch in enumerate(X_te_gen):
                X_te_batch, Y_te_batch = batch[0], batch[1]

                # Y_te_batch=X_te_batch when X_te_gen is sycned with Xt_te_gen
                eval_res = self.net.evaluate(X_te_batch, Y_te_batch, verbose=1)
                
                # Accumilate metrics into a list 
                for mi, mname in enumerate(self.net.metrics_names):
                    metrics[mname].append(eval_res if np.isscalar(eval_res) else eval_res[mi])

                # Break when full dataset is traversed once
                if (i + 1 > nloops):
                    break

            # Calcualte the mean of the accumilated figures
            for mi, mname in enumerate(self.net.metrics_names):
                metrics[mname] = np.mean(metrics[mname])

        if (self.callbacks['test'] is not None):
            self.callbacks['test'](self, self.nnpatches[patch_idx], metrics)

    def _start_predict(self, patch_idx=None, X_L_te=None, Xt_te=None, 
                                            X_te_gen=None, Xt_te_gen=None):
        """Common routine to start the prediction phase of `NNModel`.

        Parameters
        ----------
        patch_idx : int, optional
            Patch's index in this model. (Default value = None).

        X_L_te : :obj:`tuple`
            (`array_like` test data tensor, labels).
            If the `nnmodel` is not expecting labels, set it to None.

        Xt_te : `array_like`
            Target test data tensor. If the `nnmodel` is not expecting a 
            target test data tensor, set it to None.

        X_te_gen : :obj:`DataIterator`
            Test data iterator that generates data in 
            (`array_like`, labels or `array_like`) format, depending on
            the `nnmodel` architecture.

        Xt_te_gen : :obj:`DataIterator`
            Target test data iterator that generates data in 
            (`array_like`, labels or `array_like`) format, depending on
            the `nnmodel` architecture.
        """
        assert((X_L_te is not None) or (X_te_gen is not None))
        assert(self.net is not None)

        # Predict from preloaded database
        if (X_L_te is not None):

            # true_output=labels or other
            Xte, true_output = X_L_te

            if (true_output is None):
                true_output = Xt_te

            predictions = self._predict_features(Xte)

        # Predict from data generators
        else:
            # Turn off shuffling: the predictions will be in original order
            X_te_gen.set_shuffle(False)

            # Labels or other
            true_output = None

            # Test with target of true_output
            if (Xt_te_gen is not None):
                X_te_gen.sync_generator(Xt_te_gen)

                tshape = Xt_te_gen.image_shape
                if (X_te_gen.input_vectorized):
                    tshape = (np.prod(np.array(tshape)), )

                true_output = np.zeros((X_te_gen.nb_sample, ) + tshape, 'float32')
            else:
                # Array to collect true labels for batches
                if (X_te_gen.class_mode is not None):                    
                    true_output = np.zeros(X_te_gen.nb_sample, 'float32')

            # Calculate when to stop
            nloops = math.ceil(X_te_gen.nb_sample / X_te_gen.batch_size)

            # Array to collect prediction from various feature layers in batches
            predictions = []
            predict_feature_sizes = self._predict_feature_sizes()
            for i, predict_feature_size in enumerate(predict_feature_sizes):
                predictions.append(np.zeros((X_te_gen.nb_sample, predict_feature_size), 'float32'))
    
            for i, batch in enumerate(X_te_gen):
                X_te_batch, Y_te_batch = batch[0], batch[1]
                # Y_te_batch=X_te_batch when X_te_gen is sycned with Xt_te_gen

                # Set the range
                np_sample_batch = X_te_batch.shape[0]
                rng = range(i*np_sample_batch, (i+1)*np_sample_batch)

                # Predictions for this batch
                batch_predictions = self._predict_features(X_te_batch)
                for j, batch_prediction in enumerate(batch_predictions):
                    predictions[j][rng, :] = batch_prediction

                # true_output for this batch
                if (true_output is not None):
                    true_output[rng] = Y_te_batch

                # Break when full dataset is traversed once
                if (i + 1 >= nloops):
                    break

        if (self.callbacks['predict'] is not None):
            self.callbacks['predict'](self, self.nnpatches[patch_idx], predictions, true_output)

    def _predict_feature_sizes(self):
        """Get the list of feature sizes to be used in the predictions.
            
            Each feature size must corresponds to the size of a hidden 
            layer of the `nnmodel`.
            See also: self._predict_features()

        Returns
        -------
        :obj:`list` :
            Feature size for each prediction.
        """
        return self.feature_sizes

    def _predict_features(self, Xte):
        """Get the list of predicted features.
            
            Each predicted feature must be fetched via a hidden layer of 
            the `nnmodel`.
            See also: self._predict_feature_sizes()

        Parameters
        ----------
        Xte : `array_like`
            Test data tensor to be fed into the keras model.

        Returns
        -------
        :obj:`list` :
            Predicted features.
        """
        predictions = []
        for _, fn_predict_feature in enumerate(self.fns_predict_feature):
            predictions.append(fn_predict_feature([Xte, 0])[0])

        # return [self.net.predict(Xte, verbose=1)]
        return predictions

    def _init_fns_predict_feature(self, cfg):
        """Initialize keras/teano sub functions to predict each feature.

        .. warning:: Currently compatible with output layers that have 
                    exactly one inbound node.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            NN Configuration.
        """
        self.fns_predict_feature = []
        self.feature_sizes = []
        if (cfg.feature_layers is None): 
            return

        for i, f_idx in enumerate(cfg.feature_layers):
            f_layer = self.net.layers[f_idx]

            if (hasattr(f_layer, 'output_dim')):
                self.feature_sizes.append(f_layer.output_dim)

            elif (hasattr(f_layer, 'output_shape')):
                self.feature_sizes.append(f_layer.output_shape[1])

            else:
                raise Exception("Feature layers chosen are invalid. " +
                                                    "`cfg.feature_layers`")

            # IMPORTANT: 
            # Retrieves the output tensor(s) of a layer at a given node.
            # f_layer.get_output_at(node_index): 

            # Retrieves the output tensor(s) of a layer (only applicable if
            # the layer has exactly one inbound node, i.e. if it is connected
            # to one incoming layer).
            # f_layer.output
            self.fns_predict_feature.append(
                        K.function([self.net.layers[0].input, K.learning_phase()],
                                    [f_layer.output]))

    ##########################################################################
    # Protected: Save, Load [Weights or Model] Related
    ##########################################################################
    def _try_save(self, cfg, patch_idx, prefix="PREFIX"):
        """Attempt to save keras/teano model or weights in `nnmodel`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            NN Configuration.

        patch_idx : int
            Patch's index in this model.

        prefix : str, optional
            Prefix string for the file to be saved.
            (Default value = 'PREFIX').

        Returns
        -------
        bool
            True if succeeded. False otherwise.
        """
        assert(self.net is not None)

        ext = None
        if (cfg.model_dir is not None):
            ext = NNModel._M_FILE_EXT
            dir  = cfg.model_dir

        elif (cfg.weights_dir is not None):
            ext = NNModel._W_FILE_EXT
            dir  = cfg.weights_dir

        else:
            return False

        # Fetch a unique file path
        fpath = self._get_saved_model_name(patch_idx,
                                            dir,
                                            prefix, ext)

        if (cfg.model_dir is not None):

            # Check the existance of the file
            if (os.path.isfile(fpath)):
                warning("The saved model at " + fpath + " will be overridden.")

            self.net.save(fpath)
            return True

        elif (cfg.weights_dir is not None):

            # Check the existance of the file
            if (os.path.isfile(fpath)):
                warning("The saved weights at " + fpath + " will be overridden.")

            self.net.save_weights(fpath)
            return True

    def _need_prebuild(self, cfg, patch_idx, prefix="PREFIX"):
        """Whether to build the keras net to load the weights from
            `cfg.weights_dir` or the net itself from `cfg.model_dir`.

        Returns
        -------
        bool
            True if `cfg.weights_dir` or `cfg.model_dir` defines a 
            valid path, False otherwise.
        """
        ext = None
        dir = None
        if (cfg.model_dir is not None):
            ext = NNModel._M_FILE_EXT
            dir  = cfg.model_dir

        elif (cfg.weights_dir is not None):
            ext = NNModel._W_FILE_EXT
            dir  = cfg.weights_dir

        else:
            return False

        fpath = self._get_saved_model_name(patch_idx,
                                            dir,
                                            prefix, 
                                            ext)

        # Check the existance of the file
        return os.path.isfile(fpath)

    def _try_load(self, cfg, patch_idx, prefix="PREFIX", raise_err=False):
        """Attempt to load keras/teano model or weights in `nnmodel`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            NN Configuration.

        patch_idx : int
            Patch's index in this model.

        prefix : str, optional
            Prefix string for the file to be loaded.
            (Default value = 'PREFIX').

        Returns
        -------
        bool
            True if succeeded. False otherwise.
        """
        ext = None
        dir = None
        if (cfg.model_dir is not None):
            ext = NNModel._M_FILE_EXT
            dir  = cfg.model_dir

        elif (cfg.weights_dir is not None):
            ext = NNModel._W_FILE_EXT
            dir  = cfg.weights_dir

        else:
            return False

        # Fetch a unique file path
        fpath = self._get_saved_model_name(patch_idx,
                                            dir,
                                            prefix, ext)

        # Check the existance of the file
        if (not os.path.isfile(fpath)):
            if (raise_err):
                raise Exception("File: " + fpath +" does not exist.")
            else:
                return False
        
        if (cfg.model_dir is not None):

            # Load the model and weights
            self.net = load_model(fpath)

            # Error handling
            if (cfg.weights_dir is not None):
                warning('ARG_CONFLICT: Model weights will not be used since a' +
                        ' saved model is already loaded.')
            return True

        elif (cfg.weights_dir is not None):

            # Load only the weights
            self.net.load_weights(fpath)
            return True

    def _get_saved_model_name(self, patch_idx, cfg_save_dir, prefix, ext):
        """Generate a file path for the keras/teano model to be 
            saved or loaded.

        Parameters
        ----------
        patch_idx : int
            Patch's index in this model.

        cfg_save_dir : str
            Path to the folder where the keras/teano model needs to be 
            saved/loaded.

        prefix : str
            Prefix string for the file to be saved/loaded.

        Returns
        -------
        str :
            File path.
        """
        fname = prefix + "_p_" + str(patch_idx) + ".m_" + str(self.uid) + ext
        fpath = os.path.join(cfg_save_dir, fname)
        return fpath

    ##########################################################################
    # Protected: For Neural Network Framework Building
    ##########################################################################
    def _init_nnpatches(self):
        """Generate and register `nnpatches` for this model.

        Notes
        -----
        Invoked by :obj:`NNModelMan`.

        Note
        ----
        Used only in Model Based Framework.
        """
        nnpatches = self.generate_nnpatches()
        self._add_nnpatches(nnpatches)

        # Assign this model to patch
        for nnpatch in nnpatches:
            nnpatch.add_model(self)

    def _add_nnpatches(self, nnpatches):
        """Add `nnpatches` for this nnmodel.

        Parameters
        ----------
        nnpatches : :obj:`NNPatch` or list of :obj:`NNPatch`
            List of :obj:`NNPatch` instances.
        """
        if (isinstance(nnpatches, list)):
            self.nnpatches = self.nnpatches + nnpatches
        else:
            self.nnpatches.append(nnpatches)

    def _add_iterstores(self, list_iterstore, dict_iterstore=None):
        """Add dictionary and list of iterstores into a list indexed by `nnpatch` index.

        Parameters
        ----------
        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictonary of iterstores for :obj:`DataIterator`.
        """
        self._iteratorstores.append((list_iterstore, dict_iterstore))

    def _add_save_to_dirs(self, dbparam_save_dirs):
        """Add directory paths for each user dbparam into a list indexed by `nnpatch` index.

        Parameters
        ----------
        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user dbparam of each `nnpatch`.
        """
        self._list_save_to_dirs.append(dbparam_save_dirs)

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __common_train_test_predict_routine(self, exec_fn, cfg, 
                                patch_idx=None, verbose=False, precfgs=None):
        """Common routine for 'pre-train', `train`, `test`, `predict`(...) of :obj:`NNModel`."""
        # If patch unique id is not given (Serial Processing)
        if (patch_idx is None):

            # Train with itearators that belong to each patch
            for patch_idx, stores_tup in enumerate(self._iteratorstores):
                list_iterstore, dict_iterstore = stores_tup
                
                # For in memory databases, dbparam_save_dirs = None.
                # For disk databases, dbparam_save_dirs can be used as a 
                # temporary directory for each patch to store temporary data.
                # FUTURE_USE: Currently used in DAEModel pre_train(...) only.
                # For i.e Training a DAEModel will need to save temporary
                # data in layerwise pre-training.
                dbparam_save_dirs = None
                if (len(self._list_save_to_dirs) > 0):                    
                    dbparam_save_dirs = self._list_save_to_dirs[patch_idx]

                # Debug print
                if (verbose):
                    self._debug_print(list_iterstore)

                if (precfgs is not None):
                    exec_fn(precfgs, cfg, patch_idx, dbparam_save_dirs, 
                                            list_iterstore, dict_iterstore)
                else:
                    exec_fn(cfg, patch_idx, dbparam_save_dirs, 
                                            list_iterstore, dict_iterstore)
            
        else:
            # If patch unique id is not None (Parallel Processing Level 2 Support) 
            assert(patch_idx < len(self._iteratorstores))
            list_iterstore, dict_iterstore = self._iteratorstores[patch_idx]

            # For in memory databases, dbparam_save_dirs = None
            # For disk databases, dbparam_save_dirs can be used as a 
            # temporary directory for each patch to store temporary data.
            # FUTURE_USE: Currently used in DAEModel pre_train(...) only.
            dbparam_save_dirs = None
            if (len(self._list_save_to_dirs) > 0):                    
                dbparam_save_dirs = self._list_save_to_dirs[patch_idx]

            # Debug print
            if (verbose):
                self._debug_print(list_iterstore)

            if (precfgs is not None):
                exec_fn(precfgs, cfg, patch_idx, dbparam_save_dirs, 
                                        list_iterstore, dict_iterstore)
            else:
                exec_fn(cfg, patch_idx, dbparam_save_dirs, 
                                        list_iterstore, dict_iterstore)      
