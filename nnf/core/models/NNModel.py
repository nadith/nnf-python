# -*- coding: utf-8 -*-: TODO: Ccomment
"""
.. module:: NNModel
   :platform: Unix, Windows
   :synopsis: Represent NNModel class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from abc import ABCMeta, abstractmethod
import numpy as np
import math
from keras.models import load_model
from warnings import warn as warning

# Local Imports
from nnf.db.Dataset import Dataset

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
    uid : int
        Unique id of this model across the framework.

    nnpatches : list of :obj:`NNPatch`
        Associated `nnpatches` with this model.

    _iteratorstores : list of :obj:`tuple`
        Each tuple consists of `dict_iterstore` and `list_iterstore' for each `nnpatch`.

    _list_save_to_dirs : list of :obj:`list`
        List of paths to temporary directories for each user db-param of each `nnpatch`.

    net : :obj:`keras.Model`
        Core network model (keras).
    """
    __metaclass__ = ABCMeta

    ##########################################################################
    # Public Interface
    ##########################################################################
    
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

    def __init__(self, uid=None):
        """Constructor of the abstract class :obj:`NNModel`.

        Notes
        -----
        `uid` is not None for nnmodels that are created inside a nnmodel. 
        i.e Autoencoder in DAEModel
        """

        # Assign unique id
        if (uid is None):
            self.uid = NNModel.get_uid()

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
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MODEL PRE-TRAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # If pre-train is called directly without the nnf
        if (len(self._iteratorstores) == 0):
            self._pre_train(daeprecfgs, daecfg)
            return

        # If patch unique id is not given (Serial Processing)
        if (patch_idx is None):

            # Pretrain with itearators that belong to each patch
            for patch_idx, stores_tup in enumerate(self._iteratorstores):
                list_iterstore, dict_iterstore = stores_tup

                # In memory databases will have dbparam_save_dirs = None
                dbparam_save_dirs = None

                # Note: Models created inside other models for temporary purpose
                # i.e (DAEModel creating Autoencoder model) -> _list_save_to_dirs = []
                if (len(self._list_save_to_dirs) > 0):                    
                    dbparam_save_dirs = self._list_save_to_dirs[patch_idx]

                # Debug print
                self._debug_print(list_iterstore)

                self._pre_train(precfgs, cfg, patch_idx, dbparam_save_dirs, list_iterstore, dict_iterstore)
            
        else:
            # If patch unique id is not None (Parallel Processing Level 2 Support) 
            assert(patch_idx < len(self._iteratorstores))
            list_iterstore, dict_iterstore = self._iteratorstores[patch_idx]

            # In memory databases will have dbparam_save_dirs = None
            dbparam_save_dirs = None

            # Note: Models created inside other models for temporary purpose
            # i.e (DAEModel creating Autoencoder model) -> _list_save_to_dirs = []
            if (len(self._list_save_to_dirs) > 0):                    
                dbparam_save_dirs = self._list_save_to_dirs[patch_idx]

            # Debug print
            self._debug_print(list_iterstore)

            self._pre_train(precfgs, cfg, patch_idx,  dbparam_save_dirs, list_iterstore, dict_iterstore)

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

        # If pre-train is called directly without the nnf
        if (len(self._iteratorstores) == 0):
            self._pre_train(daeprecfgs, daecfg)
            return

        # If patch unique id is not given (Serial Processing)
        if (patch_idx is None):

            # Train with itearators that belong to each patch
            for patch_idx, stores_tup in enumerate(self._iteratorstores):
                list_iterstore, dict_iterstore = stores_tup
                
                # In memory databases will have dbparam_save_dirs = None
                dbparam_save_dirs = None

                # Note: Models created inside other models for temporary purpose
                # i.e (DAEModel creating Autoencoder model) -> _list_save_to_dirs = []
                if (len(self._list_save_to_dirs) > 0):                    
                    dbparam_save_dirs = self._list_save_to_dirs[patch_idx]

                # Debug print
                self._debug_print(list_iterstore)

                self._train(cfg, patch_idx, dbparam_save_dirs, list_iterstore, dict_iterstore)
            
        else:
            # If patch unique id is not None (Parallel Processing Level 2 Support) 
            assert(patch_idx < len(self._iteratorstores))
            list_iterstore, dict_iterstore = self._iteratorstores[patch_idx]

            # In memory databases will have dbparam_save_dirs = None
            dbparam_save_dirs = None

            # Note: Models created inside other models for temporary purpose
            # i.e (DAEModel creating Autoencoder model) -> _list_save_to_dirs = []
            if (len(self._list_save_to_dirs) > 0):                    
                dbparam_save_dirs = self._list_save_to_dirs[patch_idx]

            # Debug print
            self._debug_print(list_iterstore)

            self._train(cfg, patch_idx,  dbparam_save_dirs, list_iterstore, dict_iterstore)        

    def test(self, cfg, patch_idx=None):
        """Train the :obj:`NNModel`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int, optional
            Patch's index in this model. (Default value = None).
        """   
        # If pre-train is called directly without the nnf
        if (len(self._iteratorstores) == 0):
            self._pre_train(daeprecfgs, daecfg)
            return

        # If patch unique id is not given (Serial Processing)
        if (patch_idx is None):

            # Test with itearators that belong to each patch
            for patch_idx, stores_tup in enumerate(self._iteratorstores):
                list_iterstore, dict_iterstore = stores_tup
                
                # In memory databases will have dbparam_save_dirs = None
                dbparam_save_dirs = None

                # Note: Models created inside other models for temporary purpose
                # i.e (DAEModel creating Autoencoder model) -> _list_save_to_dirs = []
                if (len(self._list_save_to_dirs) > 0):                    
                    dbparam_save_dirs = self._list_save_to_dirs[patch_idx]

                self._test(cfg, patch_idx, dbparam_save_dirs, list_iterstore, dict_iterstore)
            
        else:
            # If patch unique id is not None (Parallel Processing Level 2 Support) 
            assert(patch_idx < len(self._iteratorstores))
            list_iterstore, dict_iterstore = self._iteratorstores[patch_idx]

            # In memory databases will have dbparam_save_dirs = None
            dbparam_save_dirs = None

            # Note: Models created inside other models for temporary purpose
            # i.e (DAEModel creating Autoencoder model) -> _list_save_to_dirs = []
            if (len(self._list_save_to_dirs) > 0):                    
                dbparam_save_dirs = self._list_save_to_dirs[patch_idx]

            self._test(cfg, patch_idx,  dbparam_save_dirs, list_iterstore, dict_iterstore)

    def predict(self, cfg, patch_idx=None):
        """Train the :obj:`NNModel`.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        patch_idx : int
            Patch's index in this model.
        """   
        # If pre-train is called directly without the nnf
        if (len(self._iteratorstores) == 0):
            self._pre_train(daeprecfgs, daecfg)
            return

        # If patch unique id is not given (Serial Processing)
        if (patch_idx is None):

            # Predict with itearators that belong to each patch
            for patch_idx, stores_tup in enumerate(self._iteratorstores):
                list_iterstore, dict_iterstore = stores_tup

                # In memory databases will have dbparam_save_dirs = None
                dbparam_save_dirs = None

                # Note: Models created inside other models for temporary purpose
                # i.e (DAEModel creating Autoencoder model) -> _list_save_to_dirs = []
                if (len(self._list_save_to_dirs) > 0):                    
                    dbparam_save_dirs = self._list_save_to_dirs[patch_idx]

                self._predict(cfg, patch_idx, dbparam_save_dirs, list_iterstore, dict_iterstore)
            
        else:
            # If patch unique id is not None (Parallel Processing Level 2 Support) 
            assert(patch_idx < len(self._iteratorstores))
            list_iterstore, dict_iterstore = self._iteratorstores[patch_idx]

            # In memory databases will have dbparam_save_dirs = None
            dbparam_save_dirs = None

            # Note: Models created inside other models for temporary purpose
            # i.e (DAEModel creating Autoencoder model) -> _list_save_to_dirs = []
            if (len(self._list_save_to_dirs) > 0):                    
                dbparam_save_dirs = self._list_save_to_dirs[patch_idx]

            self._predict(cfg, patch_idx,  dbparam_save_dirs, list_iterstore, dict_iterstore)

    @abstractmethod
    def _pre_train(self, precfgs, cfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
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
    def _train(self, cfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
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
    def _test(self, cfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
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
    def _predict(self, cfg, patch_idx=None, dbparam_save_dirs=None, list_iterstore=None, dict_iterstore=None):
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

        for i, iterstore in enumerate(list_iterstore):
            print("\nIterator Store:{}".format(i))
            print("=================")
            __print_params(iterstore, Dataset.TR)
            __print_params(iterstore, Dataset.VAL)
            __print_params(iterstore, Dataset.TE)
            __print_params(iterstore, Dataset.TR_OUT)
            __print_params(iterstore, Dataset.VAL_OUT)
            __print_params(iterstore, Dataset.TE_OUT)

    def _start_train(self, cfg, X_L=None, Xt=None, X_L_val=None, Xt_val=None, X_gen=None, X_val_gen=None):
        assert((X_L is not None) or (X_gen is not None))

        # Train from preloaded database
        if (X_L is not None):
            if (X_L_val is not None):
                
                X, lbl = X_L
                X_val, lbl_val = X_L_val

                # Train with labels
                if (lbl is not None):
                    self.net.fit(X, lbl, nb_epoch=50, batch_size=256, shuffle=True, validation_data=(X_val, lbl_val))  #, callbacks=[self.cb_early_stop])

                # Train with targets
                elif (lbl is None):
                    self.net.fit(X, Xt, nb_epoch=50, batch_size=256, shuffle=True, validation_data=(X_val, Xt_val))  #, callbacks=[self.cb_early_stop])

            else:
                X, lbl = X_L
                X_val, lbl_val = X_L_val

                # Train with labels
                if (lbl is not None):
                    self.net.fit(X, lbl, nb_epoch=50, batch_size=256, shuffle=True) 

                # Train without targets
                elif (lbl is None):
                    self.net.fit(X, Xt, nb_epoch=50, batch_size=256, shuffle=True) 
                  
        # Train from data generators
        else:
            if (X_val_gen is not None):
                self.net.fit_generator(
                        X_gen, samples_per_epoch=cfg.samples_per_epoch,
                        nb_epoch=cfg.numepochs,
                        validation_data=X_val_gen, nb_val_samples=cfg.nb_val_samples) # callbacks=[self.cb_early_stop]
            else:
                self.net.fit_generator(
                        X_gen, samples_per_epoch=cfg.samples_per_epoch,
                        nb_epoch=cfg.numepochs)

    def _start_test(self, patch_idx=None, Xte_L=None, Xte_target=None, Xte_gen=None, Xte_target_gen=None):
        assert((Xte_L is not None) or (Xte_gen is not None))
        assert(self.net is not None)

        # Test from preloaded database
        if (Xte_L is not None):

            Xte, lbl = Xte_L

            # Test with labels
            if (lbl is not None):
                loss, accuracy = self.net.evaluate(Xte, lbl, verbose=1)

            # Train with targets
            elif (Xte_target is not None):
                loss, accuracy = self.net.evaluate(Xte, Xte_target, verbose=1)
            else:
                raise Exception("Unsupported mode in testing...")

        # Test from data generators
        else:  
            # Test with labels or target
            if (Xte_target_gen is not None):
                Xte_gen.sync_generator(Xte_target_gen)

            # Calculate when to stop
            nloops = math.ceil(Xte_gen.nb_sample / Xte_gen.batch_size)

            # Dictionary to collect loss and accuracy for batches
            metrics = {}
            for mname in self.net.metrics_names:
                metrics.setdefault(mname, [])

            for i, batch in enumerate(Xte_gen):
                Xte_batch, Yte_batch = batch[0], batch[1]

                # Yte_batch=Xte_target_batch when Xte_gen is sycned with Xte_target_gen
                eval_res = self.net.evaluate(Xte_batch, Yte_batch, verbose=1)
                
                # Accumilate figure into a list 
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

    def _start_predict(self, patch_idx=None, Xte_L=None, Xte_target=None, Xte_gen=None, Xte_target_gen=None):
        assert((Xte_L is not None) or (Xte_gen is not None))
        assert(self.net is not None)

        # Predict from preloaded database
        if (Xte_L is not None):
            prediction = self.net.predict(Xte_L, verbose=1)

        # Predict from data generators
        else:
            # Turn off shuffling: the predictions will be in original order
            Xte_gen.set_shuffle(False)

            labels = None

            # Test with labels or target
            if (Xte_target_gen is not None):
                Xte_gen.sync_generator(Xte_target_gen)

                tshape = Xte_target_gen.image_shape
                if (Xte_gen.input_vectorized):
                    tshape = (np.prod(np.array(tshape)), )

                labels = np.zeros((Xte_gen.nb_sample, ) + tshape, 'float32')
            else:
                # Array to collect true labels for batches
                if (Xte_gen.class_mode is not None):                    
                    labels = np.zeros(Xte_gen.nb_sample, 'float32')

            # Calculate when to stop
            nloops = math.ceil(Xte_gen.nb_sample / Xte_gen.batch_size)

            # Array to collect prediction for batches
            prediction = np.zeros((Xte_gen.nb_sample, self._predict_feature_size()), 'float32')
    
            for i, batch in enumerate(Xte_gen):
                Xte_batch, yte_batch = batch[0], batch[1]

                # Set the range
                np_sample_batch = Xte_batch.shape[0]
                rng = range(i*np_sample_batch, (i+1)*np_sample_batch)

                # Predictions for this batch
                prediction[rng, :] = self._predict_feature(Xte_batch)

                # Labels for this batch
                if (labels is not None):
                    labels[rng] = yte_batch

                # Break when full dataset is traversed once
                if (i + 1 >= nloops):
                    break

        if (self.callbacks['predict'] is not None):
            self.callbacks['predict'](self, self.nnpatches[patch_idx], prediction, labels)

    def _predict_feature_size(self):
        return self.net.output_shape[1]        

    def _predict_feature(self, Xte):
        return self.net.predict(Xte, verbose=1)

    def _try_save(self, cfg, patch_idx, prefix="PREFIX"):
        assert(self.net is not None)
        if (cfg.model_dir is not None):
            fpath = self._get_saved_model_name(patch_idx, cfg.model_dir, prefix)
            self.net.save(fpath)
            return True
        return False
    
    def _try_load(self, cfg, patch_idx, prefix="PREFIX"):
        if (cfg.model_dir is not None):
            fpath = self._get_saved_model_name(patch_idx, cfg.model_dir, prefix)
            self.net = load_model(fpath)
            if (cfg.weights_path is not None):
                warning('ARG_CONFLICT: Model weights will not be used since a saved model is already loaded.')
            return True
        return False

    def _get_saved_model_name(self, patch_idx, cfg_save_dir, prefix):
        fname = prefix + "_p_" + str(patch_idx) + ".m_" + str(self.uid) + ".model.h5"
        fpath = os.path.join(cfg_save_dir, fname)
        return fpath

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