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
import collections
from warnings import warn as warning
from nnf.keras.models import load_model
from warnings import warn as warning
from keras import backend as K
from nnf.core.callbacks.TensorBoardEx import TensorBoardEx

# Local Imports
from nnf.db.Dataset import Dataset
from nnf.core.models.NNModelPhase import NNModelPhase

# Circular Imports
# ref:http://stackoverflow.com/questions/22187279/python-circular-importing
import nnf.db.NNPatch

class NNModel(object):
    """NNModel represents base class for Neural Network Models.

    .. warning:: abstract class and must not be instantiated.

    Maintain associated 'nnpatches' along with the 'iteratorstores' and 
    the related paths to the directories to save temporary data both
    user dbparam-wise and patch-wise. 
    (See also DAEModel)

    Attributes
    ----------
    uid : int or str
        Unique id of this model across the framework. Can also set a custom
        id for models created inside another model. 
        Ref:`DAEModel` creating `Autoencoder`

    net : :obj:`keras.Model`
        Core network model (keras).

    nnpatches : list of :obj:`NNPatch`
        Associated `nnpatches` with this model.

    _iterstores : list of :obj:`tuple`
        Each tuple consists of `dict_iterstore` and `list_iterstore' for each `nnpatch`.

    _list_save_dirs : list of :obj:`list`
        List of paths to temporary directories for each user db-param of each `nnpatch`.

    _fns_predict_feature :
        Keras/theano sub functions to predict each feature.

    _feature_sizes :
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

    def __init__(self, uid=None, callbacks=None,
                 iter_params=None, iter_pp_params=None, nncfgs=None):
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
        # iterstores format = [ (dict_iterstore, list_iterstore) for nnpatch_1,
        #                           (dict_iterstore, list_iterstore) for nnpatch_2
        #                           (dict_iterstore, list_iterstore) for nnpatch_3
        #                           (dict_iterstore, list_iterstore) for nnpatch_4
        #                           ...
        #                           ]
        self._iterstores = []

        # To save temporary encoded data
        # [ [folder_for_param_1_db, folder_for_param_2_db] for nnpatch_1]
        #   [folder_for_param_1_db, folder_for_param_2_db] for nnpatch_2]
        #   ...
        #   ]
        self._list_save_dirs = []

        # Associated nnpatches with this model
        # len(self._iterstores) == len(self.nnpatches)
        # len(self._list_save_dirs) == len(self.nnpatches)
        self.nnpatches = [] 

        # Core network model (keras).
        self.net = None

        # Keras/theano sub functions to predict each feature
        self._fns_predict_feature = []

        # Feature sizes for each prediction
        self._feature_sizes = []

        # Set defaults for general callbacks
        self.callbacks = {} if (callbacks is None) else callbacks
        self.callbacks.setdefault('test', None)
        self.callbacks.setdefault('predict', None)

        # Use `_get_input_` methods as defaults
        tmp = self.callbacks.setdefault('get_input_data_generators', None)
        if (tmp is None): self.callbacks['get_input_data_generators'] = self._get_input_data_generators

        # Use `_get_target_` methods as defaults
        tmp = self.callbacks.setdefault('get_target_data_generators', None)
        if (tmp is None): self.callbacks['get_target_data_generators'] = self._get_target_data_generators

        # Public params used when initializing the framework `NNFramework`
        self.iter_params = iter_params
        self.iter_pp_params = iter_pp_params
        self.nncfgs = nncfgs  # Keyed by `NNModelPhase` enumeration
    
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
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " + self._model_prefix() + " (PRE-TRAIN) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
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
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " + self._model_prefix() + " (TRAIN) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        # For models created inside another model and using preloaded dbs.
        # Ref:`DAEModel` creating `Autoencoder`
        if (len(self._iterstores) == 0):
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
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " + self._model_prefix() + " (TEST) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
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
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " + self._model_prefix() + " (PREDICT) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.__common_train_test_predict_routine(self._predict, cfg, patch_idx)

    def _debug_print(self, cfg, list_iterstore):
        """Print information for each iterator store in `list_iterstore`.

            The iterator params and pre-processor params of iterator store
            for each dataset. 
            i.e (Dataset.TR | VAL | ...).

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

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

        print("\nNNCfg")
        print("=====")
        for key, val in cfg.__dict__.items():
            if (key.startswith('_')): continue
            print('\t{} : {}'.format(key, val))

        for i, iterstore in enumerate(list_iterstore):
            print("\nIterator Store:{}".format(i))
            print("=================")
            __print_params(iterstore, Dataset.TR)
            __print_params(iterstore, Dataset.VAL)
            __print_params(iterstore, Dataset.TE)
            __print_params(iterstore, Dataset.TR_OUT)
            __print_params(iterstore, Dataset.VAL_OUT)
            __print_params(iterstore, Dataset.TE_OUT)

    ##########################################################################
    # Public: For Neural Network Framework Building
    ##########################################################################
    def init_nnpatches(self):
        """Initialize `nnpatches` of this model.

        Notes
        -----
        Invoked by :obj:`NNModelMan`.

        Note
        ----
        Used only in Model Based Framework.
        Invoked by :obj:`NNModelMan`.
        """
        # nnpatches = self._generate_nnpatches()

        # Register `nnpatches` to this `nnmodel`
        # Refer: NNFramework._init_model_params(...)

        # # Register this `nnmodel` to `nnpatch`
        # for nnpatch in nnpatches:
        #     nnpatch.add_nnmodels(self)

    def add_nnpatches(self, nnpatches):
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

    def add_iterstores(self, list_iterstore, dict_iterstore=None):
        """Add dictionary and list of iterstores into a list indexed by `nnpatch` index.

        Parameters
        ----------
        list_iterstore : :obj:`list`
            List of iterstores for :obj:`DataIterator`.

        dict_iterstore : :obj:`dict`
            Dictionary of iterstores for :obj:`DataIterator`.
        """
        self._iterstores.append((list_iterstore, dict_iterstore))

    def add_save_dirs(self, dbparam_save_dirs):
        """Add directory paths for each user dbparam into a list indexed by `nnpatch` index.

        Parameters
        ----------
        dbparam_save_dirs : :obj:`list`
            Paths to temporary directories for each user dbparam of each `nnpatch`.
        """
        self._list_save_dirs.append(dbparam_save_dirs)

    def release_iterstores(self):
        """Release the iterators held at iterstores."""
        if self._iterstores is None: return

        for list_iterstore, _ in self._iterstores:
            for iterstore in list_iterstore:
                edatasets = Dataset.get_enum_list()
                for edataset in edatasets:
                    gen = iterstore.setdefault(edataset, None)
                    if gen is not None: gen.release()
                    iterstore[edataset] = None

        del self._iterstores
        self._iterstores = None

    ##########################################################################
    # Protected Interface
    ##########################################################################
    @abstractmethod
    def _generate_nnpatches(self):
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

    @abstractmethod
    def _model_prefix(self):
        """Fetch the _prefix for the file to be saved/loaded.

        Note
        ----
        Override this method for custom _prefix.
        """
        return "<PFX>"

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
            Dictionary of iterstores for :obj:`DataIterator`.
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
            Dictionary of iterstores for :obj:`DataIterator`.
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
            Dictionary of iterstores for :obj:`DataIterator`.
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
            Dictionary of iterstores for :obj:`DataIterator`.
        """
        pass

    ##########################################################################
    # Protected:
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
        In pre-training, clone the iterators, in training, use the originals
        """
        return iter.clone() if (iter is not None) else None

    def _init_data_generators(self, ephase, list_iterstore, dict_iterstore):
        """Initialize data generators for [pre-training,] training, testing, prediction.

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
            When ephase == NNModelPhase.PRE_TRAIN or NNModelPhase.TRAIN
            then 
                Generators for training (Xin_gen) and validation (Xin_val_gen).
                Refer https://keras.io/preprocessing/image/

            When ephase == NNModelPhase.TEST or NNModelPhase.PREDICT
            then
                Generators for testing input and testing target (Xin_gen). (Xin_val_gen) is unused.
        """
        # Fetch input data generators
        Xin_gen, Xin_val_gen = \
            self.callbacks['get_input_data_generators'](ephase, list_iterstore, dict_iterstore)

        # Fetch target data generators
        Xt_gen, Xt_val_gen = \
            self.callbacks['get_target_data_generators'](ephase, list_iterstore, dict_iterstore)

        if Xin_gen is None:
            raise Exception('Required data generators are unavailable')

        if (ephase == NNModelPhase.PREDICT or ephase == NNModelPhase.TEST):
            if Xin_val_gen is not None or Xt_val_gen is not None:
                raise Exception('In NNModelPhase.PREDICT|TEST, `X2_gen` is unused. But it is not None.')

        # Sync the data generators (input, target)
        Xin_gen.sync_tgt_generator(Xt_gen)
        if (Xin_val_gen is not None):
            Xin_val_gen.sync_tgt_generator(Xt_val_gen)

        return Xin_gen, Xin_val_gen

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
                Generators for training input (X1_gen) and validation input (X2_gen).

            When ephase == NNModelPhase.TEST or NNModelPhase.PREDICT
            then
                Generators for testing input (X1_gen). X2_gen is unused.
        """
        X1_gen = None
        X2_gen = None
        if (ephase == NNModelPhase.TRAIN):
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
                Generators for training target (X1_gen) and validation target (X2_gen).

            When ephase == NNModelPhase.TEST or NNModelPhase.PREDICT
            then
                Generators for testing target (X1_gen). X2_gen is unused.
        """
        X1_gen = None
        X2_gen = None
        if (ephase == NNModelPhase.PREDICT or ephase == NNModelPhase.TEST):
            # Iterstore for dbparam1, TE_OUT
            X1_gen = list_iterstore[0].setdefault(Dataset.TE_OUT, None)

        return X1_gen, X2_gen

    ##########################################################################
    # Protected: Train, Test, Predict Common Routines Of Keras
    ##########################################################################
    def _start_train(self, cfg, X_L=None, Xt=None, X_L_val=None, Xt_val=None, 
                                                X_gen=None, X_val_gen=None, managed=True):
        """Common routine to start the training phase of `NNModel`.

        Parameters
        ----------
        patch_idx : int, optional
            Patch's index in this model. (Default value = None).

        X_L : :obj:`tuple`
            (ndarray data tensor, labels).
            If the `nnmodel` is not expecting labels, set it to None.

        Xt : ndarray
            Target data tensor. If the `nnmodel` is not expecting a 
            target data tensor, set it to None.

        X_L_val : :obj:`tuple`
            (ndarray validation data tensor, labels).
            If the `nnmodel` is not expecting labels, set it to None.

        Xt_val : ndarray
            Target validation data tensor. If the `nnmodel` is not expecting a
            target validation data tensor, set it to None.

        X_gen : :obj:`DataIterator`
            Data iterator that generates data in 
            (ndarray, labels or ndarray) format, depending on
            the `nnmodel` architecture.

        X_val_gen : :obj:`DataIterator`
            Validation data iterator that generates data in 
            (ndarray, labels or ndarray) format, depending on
            the `nnmodel` architecture.

        managed : bool
            Whether the iterators are by default managed by NNModel.
        """
        assert((X_L is not None) or (X_gen is not None))

        # Train from preloaded database
        if (X_L is not None):
            if (X_L_val is not None):
                
                X, lbl = X_L
                X_val, lbl_val = X_L_val

                # Train with labels
                if (lbl is not None):
                    self.net.fit(X, lbl, epochs=cfg.numepochs, batch_size=cfg.pdb_batch_size, callbacks=cfg.callbacks, shuffle=cfg.pdb_shuffle, validation_data=(X_val, lbl_val))  #, callbacks=[self.cb_early_stop])

                # Train with targets
                elif (lbl is None):
                    self.net.fit(X, Xt, epochs=cfg.numepochs, batch_size=cfg.pdb_batch_size, callbacks=cfg.callbacks, shuffle=cfg.pdb_shuffle, validation_data=(X_val, Xt_val))  #, callbacks=[self.cb_early_stop])

            else:
                X, lbl = X_L

                # Train with labels
                if (lbl is not None):
                    self.net.fit(X, lbl, epochs=cfg.numepochs, batch_size=cfg.pdb_batch_size, callbacks=cfg.callbacks, shuffle=cfg.pdb_shuffle)

                # Train with targets
                elif (lbl is None):
                    self.net.fit(X, Xt, epochs=cfg.numepochs, batch_size=cfg.pdb_batch_size, callbacks=cfg.callbacks, shuffle=cfg.pdb_shuffle)
                  
        # Train from data generators
        else:
            # Initiate the data flow
            if managed:
                X_gen.initiate_parallel_operations()
                if (X_val_gen is not None): X_val_gen.initiate_parallel_operations()

            if (X_val_gen is not None):
                if cfg.callbacks is not None:
                    for callback in cfg.callbacks:
                        if (isinstance(callback, TensorBoardEx)):
                            callback.init(X_val_gen, cfg.validation_steps)

                self.net.fit_generator(
                        X_gen, steps_per_epoch=cfg.steps_per_epoch,
                        epochs=cfg.numepochs, callbacks=cfg.callbacks,
                        validation_data=X_val_gen, validation_steps=cfg.validation_steps, verbose=1) # callbacks=[self.cb_early_stop]

            else:
                self.net.fit_generator(
                        X_gen, steps_per_epoch=cfg.steps_per_epoch,
                        epochs=cfg.numepochs, callbacks=cfg.callbacks, verbose=1)

            # Only stop the parallel pool, release() will be invoked in NNFramework.release()
            if managed:
                X_gen.terminate_parallel_operations()
                if (X_val_gen is not None): X_val_gen.terminate_parallel_operations()

    def _start_test(self, patch_idx=None, X_L_te=None, Xt_te=None,
                                            X_te_gen=None, managed=True):
        """Common routine to start the testing phase of `NNModel`.

        Parameters
        ----------
        patch_idx : int, optional
            Patch's index in this model. (Default value = None).

        X_L_te : :obj:`tuple`
            (ndarray test data tensor, labels).
            If the `nnmodel` is not expecting labels, set it to None.

        Xt_te : ndarray
            Target test data tensor. If the `nnmodel` is not expecting a
            target test data tensor, set it to None.

        X_te_gen : :obj:`DataIterator`
            Test data iterator that generates data in 
            (ndarray, labels or ndarray) format, depending on
            the `nnmodel` architecture.

        Xt_te_gen : :obj:`DataIterator`
            Target test data iterator that generates data in 
            (ndarray, labels or ndarray) format, depending on
            the `nnmodel` architecture.

        managed : bool
            Whether the iterators are by default managed by NNModel.
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

                # Accumulate metrics into a list
                for mi, mname in enumerate(self.net.metrics_names):
                    metrics[mname].append(eval_res if np.isscalar(eval_res) else eval_res[mi])

            else:
                raise Exception("Unsupported mode in testing...")

        # Test from data generators
        else:  
            # Initiate the data flow
            if managed:
                X_te_gen.initiate_parallel_operations()

            # Calculate when to stop
            nloops = math.ceil(X_te_gen.nb_sample / X_te_gen.batch_size)

            # Dictionary to collect loss and accuracy for batches
            metrics = {}
            for mname in self.net.metrics_names:
                metrics.setdefault(mname, [])

            for i, batch in enumerate(X_te_gen):
                X_te_batch, Y_te_batch = batch[0], batch[1]

                # Y_te_batch=Xt_te_batch when X_te_gen is synced with Xt_te_gen
                eval_res = self.net.evaluate(X_te_batch, Y_te_batch, verbose=1)
                
                # Accumulate metrics into a list
                for mi, mname in enumerate(self.net.metrics_names):
                    metrics[mname].append(eval_res if np.isscalar(eval_res) else eval_res[mi])

                # Break when full dataset is traversed once
                if (i + 1 >= nloops):
                    break

            # # Calculate the mean of the figures collected via each batch in above step
            # for mi, mname in enumerate(self.net.metrics_names):
            #     metrics[mname] = np.mean(metrics[mname])

            # Only stop the parallel pool, release() will be invoked in NNFramework.release()
            if managed:
                X_te_gen.terminate_parallel_operations()

        if (self.callbacks['test'] is not None):
            self.callbacks['test'](self, self.nnpatches[patch_idx], metrics)

    def _start_predict(self, patch_idx=None, X_L_te=None, Xt_te=None,
                                            X_te_gen=None, managed=True):
        """Common routine to start the prediction phase of `NNModel`.

        Parameters
        ----------
        patch_idx : int, optional
            Patch's index in this model. (Default value = None).

        X_L_te : :obj:`tuple`
            (ndarray test data tensor, labels).
            If the `nnmodel` is not expecting labels, set it to None.

        Xt_te : ndarray
            Target test data tensor. If the `nnmodel` is not expecting a 
            target test data tensor, set it to None.

        X_te_gen : :obj:`DataIterator`
            Test data iterator that generates data in 
            (ndarray, labels or ndarray) format, depending on
            the `nnmodel` architecture.

        Xt_te_gen : :obj:`DataIterator`
            Target test data iterator that generates data in 
            (ndarray, labels or ndarray) format, depending on
            the `nnmodel` architecture.

        managed : bool
            Whether the iterators are by default managed by NNModel.
        """
        assert((X_L_te is not None) or (X_te_gen is not None))
        assert(self.net is not None)

        # Predict from pre-loaded database
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

            # New container to collect ground truth
            true_output = X_te_gen.new_target_container()

            # Initiate the data flow
            if managed:
                X_te_gen.initiate_parallel_operations()

            # Calculate when to stop
            nloops = math.ceil(X_te_gen.nb_sample / X_te_gen.batch_size)

            # Array to collect prediction from various feature layers in batches
            predictions = []
            predict_feature_sizes = self._predict_feature_sizes()
            for i, predict_feature_size in enumerate(predict_feature_sizes):
                predictions.append(np.zeros((X_te_gen.nb_sample, predict_feature_size), K.floatx()))
    
            for i, batch in enumerate(X_te_gen):
                X_te_batch, Y_te_batch = batch[0], batch[1]
                # Y_te_batch=Xt_te_batch when X_te_gen is sycned with Xt_te_gen

                # Set the range
                np_sample_per_batch = X_te_batch.shape[0] if not isinstance(X_te_batch, list) else X_te_batch[0].shape[0]
                rng = range(i*np_sample_per_batch, (i+1)*np_sample_per_batch)

                # Predictions for this batch
                batch_predictions = self._predict_features(X_te_batch)
                for j, batch_prediction in enumerate(batch_predictions):
                    predictions[j][rng, :] = batch_prediction

                if true_output is not None:

                    # true_output(s) for this batch
                    if isinstance(Y_te_batch, list):
                        o_idx = 0
                        for Y_te in Y_te_batch:
                            true_output[o_idx][rng] = Y_te
                            o_idx += 1
                    else:
                        true_output[rng] = Y_te_batch

                # Break when full dataset is traversed once
                if (i + 1 >= nloops):
                    break

            # Only stop the parallel pool, release() will be invoked in NNFramework.release()
            if managed:
                X_te_gen.terminate_parallel_operations()

        if (self.callbacks['predict'] is not None):
            #predictions = predictions[0] if len(predictions) == 1 else predictions
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
        return self._feature_sizes

    def _predict_features(self, Xte):
        """Get the list of predicted features.
            
            Each predicted feature must be fetched via a hidden layer of 
            the `nnmodel`.
            See also: self._predict_feature_sizes()

        Parameters
        ----------
        Xte : ndarray
            Test data tensor to be fed into the keras model.

        Returns
        -------
        :obj:`list`
            Predicted features.
        """
        predictions = []
        for _, fn_predict_feature in enumerate(self._fns_predict_feature):
            Xte = Xte if isinstance(Xte, list) else [Xte]
            predictions.append(fn_predict_feature(Xte + [0])[0])
            # Xte + [0]: 0-> `predict` learning phase; 1-> `train` learning phase

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
        self._fns_predict_feature = []
        self._feature_sizes = []
        if (cfg.feature_layers is None): 
            return

        if not isinstance(cfg.feature_layers, collections.Iterable):
            cfg.feature_layers = [cfg.feature_layers]

        for i, f_idx in enumerate(cfg.feature_layers):
            f_layer = self.net.layers[f_idx]

            if (hasattr(f_layer, 'output_dim')):
                self._feature_sizes.append(f_layer.output_dim)

            elif (hasattr(f_layer, 'output_shape')):
                self._feature_sizes.append(f_layer.output_shape[1])

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
            self._fns_predict_feature.append(
                        K.function(self.net.inputs + [K.learning_phase()],
                                    [f_layer.output]))

    ##########################################################################
    # Protected: Build Related
    ##########################################################################
    def _internal__build(self, cfg, X_gen):
        pass

    def _internal__pre_compile(self, cfg, X_gen):
        pass

    def _internal__compile(self, cfg):
        pass

    def _init_net(self, cfg, patch_idx, prefix, X_gen):
        # Checks whether keras net is already prebuilt
        if not self._is_prebuilt(cfg, patch_idx, prefix):
            self._internal__build(cfg, X_gen)

        # Try to load the saved model or weights
        self._try_load(cfg, patch_idx, prefix)

        # PERF: Avoid compiling before loading saved weights/model
        # Pre-compile callback
        self._internal__pre_compile(cfg, X_gen)

        # Compile the model
        self._internal__compile(cfg)

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

        if (cfg.save_models_dir is None and cfg.save_weights_dir is None):
            return False

        # Save the model
        if (cfg.save_models_dir is not None):

            # Get the unique lookup file path
            fpath_m = self._get_unique_lookup_filepath(patch_idx, cfg.save_models_dir, prefix, NNModel._M_FILE_EXT)

            # Check the existence of the file
            if (os.path.isfile(fpath_m)):
                warning("The saved model will be overridden at " + fpath_m)

            self.net.save(fpath_m)

        # Save the weights
        if (cfg.save_weights_dir is not None):

            # Get the unique lookup file path
            fpath_w = self._get_unique_lookup_filepath(patch_idx, cfg.save_weights_dir, prefix, NNModel._W_FILE_EXT)

            # Check the existence of the file
            if (os.path.isfile(fpath_w)):
                warning("The saved weights will be overridden at " + fpath_w)

            self.net.save_weights(fpath_w)

        return True

    def _is_prebuilt(self, cfg, patch_idx, prefix="PREFIX"):
        """Whether the keras net is already prebuilt with `cfg.load_models_dir`.

        Returns
        -------
        bool
            True if `cfg.load_models_dir` defines a valid path, False otherwise.
        """
        ext = None
        dir = None

        if (cfg.load_models_dir is None):
            return False

        # Get the unique lookup file path
        fpath = self._get_unique_lookup_filepath(patch_idx, cfg.load_models_dir, prefix, NNModel._M_FILE_EXT)

        # Check the existence of the file
        if not os.path.isfile(fpath):
            raise Exception('Model file does not exist: {0}'.format(fpath))

        return True

    def _try_load(self, cfg, patch_idx, prefix="PREFIX"):
        """Attempt to load keras/theano model or weights in `nnmodel`.

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
        if (cfg.load_models_dir is None and cfg.load_weights_dir is None):
            return False

        fpath_m = None
        load_model_failed = False

        if (cfg.load_models_dir is not None):
            # Get the unique lookup file path
            fpath_m = self._get_unique_lookup_filepath(patch_idx, cfg.load_models_dir, prefix, NNModel._M_FILE_EXT)

            # Check the existence of the file
            if not os.path.isfile(fpath_m):
                if cfg.load_weights_dir is not None:
                    load_model_failed = True
                else:
                    raise Exception('Model file does not exist: {0}'.format(fpath_m))

            from nnf.core.Metric import Metric

            # Load the model and weights
            self.net = load_model(fpath_m, {'r': Metric.r, 'cov': Metric.cov, 's_acc': Metric.s_acc})

            # Error handling
            if (cfg.load_weights_dir is not None):
                warning('ARG_CONFLICT: Model weights will not be used since a' +
                        ' saved model is already loaded.')

            print(self.net.summary())
            print("---- MODEL LOADED SUCCESSFULLY ----")
            return True

        if cfg.load_weights_dir is not None:

            # Get the unique lookup file path
            fpath_w = self._get_unique_lookup_filepath(patch_idx, cfg.load_weights_dir, prefix, NNModel._W_FILE_EXT)

            if load_model_failed:
                warning('Model file does not exist: {0}. Attempting to load: {1}'.format(fpath_m, fpath_w))

            # Check the existence of the file
            if not os.path.isfile(fpath_w):
                raise Exception('Model weights file does not exist: {0}'.format(fpath_w))

            # Load only the weights
            self.net.load_weights(fpath_w)

            print(self.net.summary())
            print("---- WEIGHTS LOADED SUCCESSFULLY ----")
            return True

    def _get_unique_lookup_filepath(self, patch_idx, cfg_save_dir, prefix, ext):
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
    # Private Interface
    ##########################################################################
    def __common_train_test_predict_routine(self, exec_fn, cfg, 
                                patch_idx=None, verbose=False, precfgs=None):
        """Common routine for 'pre-train', `train`, `test`, `predict`(...) of :obj:`NNModel`."""
        # If patch unique id is not given (Serial Processing)
        if (patch_idx is None):

            # Train with iterators that belong to each patch
            for patch_idx, stores_tup in enumerate(self._iterstores):
                list_iterstore, dict_iterstore = stores_tup
                
                # For in memory databases, dbparam_save_dirs = None.
                # For disk databases, dbparam_save_dirs can be used as a
                # temporary directory for each patch to store temporary data.
                # FUTURE_USE: Currently used in DAEModel pre_train(...) only.
                # For i.e Training a DAEModel will need to save temporary
                # data in layer-wise pre-training.
                dbparam_save_dirs = None
                if (len(self._list_save_dirs) > 0):
                    dbparam_save_dirs = self._list_save_dirs[patch_idx]

                # Debug print
                if (verbose):
                    self._debug_print(cfg, list_iterstore)

                if (precfgs is not None):
                    exec_fn(precfgs, cfg, patch_idx, dbparam_save_dirs,
                                            list_iterstore, dict_iterstore)
                else:
                    exec_fn(cfg, patch_idx, dbparam_save_dirs, list_iterstore, dict_iterstore)
            
        else:
            # If patch unique id is not None (Parallel Processing Level 2 Support) 
            assert(patch_idx < len(self._iterstores))
            list_iterstore, dict_iterstore = self._iterstores[patch_idx]

            # For in memory databases, dbparam_save_dirs = None
            # For disk databases, dbparam_save_dirs can be used as a
            # temporary directory for each patch to store temporary data.
            # FUTURE_USE: Currently used in DAEModel pre_train(...) only.
            dbparam_save_dirs = None
            if (len(self._list_save_dirs) > 0):
                dbparam_save_dirs = self._list_save_dirs[patch_idx]

            # Debug print
            if (verbose):
                self._debug_print(cfg, list_iterstore)

            if (precfgs is not None):
                exec_fn(precfgs, cfg, patch_idx, dbparam_save_dirs, list_iterstore, dict_iterstore)
            else:
                exec_fn(cfg, patch_idx, dbparam_save_dirs,
                                        list_iterstore, dict_iterstore)      
