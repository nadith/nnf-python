# -*- coding: utf-8 -*-
# Global Imports
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

# Local Imports
from nnf.core.models.NNModel import NNModel
from nnf.core.models.DAEModel import DAEModel
from nnf.db.Dataset import Dataset
from nnf.core.models.NNModelPhase import NNModelPhase

class DAERegModel(DAEModel):
    """Asymmetric DAE Model. Multiple encoder, single/many decoder model.

        DAE Model will be formed according to the layer purpose 
        attribute vector at :obj:`DAERegCfg`.

        For: i.e
        105 -> 70 -> 56 -> 56 -> 70
        lp = ['Input', 'dr', 'dr', 'rg', 'Ouput']

        .. warning:: Last layer must be a regression ('rg') layer.
    """
    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, callbaks=None):
        super().__init__(callbaks)
        self.Xt_gen = self.Xt_val_gen = None

        # Set defaults for arguments
        get_target_dat_gen = self.callbacks.setdefault('get_target_data_generators', None)
        if (get_target_dat_gen is None):
            self.callbacks['get_target_data_generators'] = self._get_target_data_generators
     
    ##########################################################################
    # Protected: DAEModel Overrides
    ##########################################################################
    def _validate_cfg(self, ephase, cfg):
        super()._validate_cfg(ephase, cfg)

        if (ephase != NNModelPhase.TRAIN): return

        if (len(cfg.arch) != len(cfg.lp)):
            raise Exception('Layer purpose string for each layers is not' + 
            ' specified. (length of `cfg.arch` != length of `cfg.lp`).')

    def _get_target_data_generators(self, ephase, list_iterstore, dict_iterstore):
        """Get target data generators for pre-training, training only.

        .. warning:: Only invoked in PRE_TRAIN and TRAIN phases. For other phases,
                    refer `get_data_generators()` function.

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

        Note
        ----
            When ephase == NNModelPhase.TEST or NNModelPhase.PREDICT
            then this method will not be invoked.
        """
        if (ephase == NNModelPhase.PRE_TRAIN or ephase == NNModelPhase.TRAIN):
            # Iteratorstore for dbparam1, TR_OUT and VAL_OUT
            X1_gen = list_iterstore[0].setdefault(Dataset.TR_OUT, None)
            X2_gen = list_iterstore[0].setdefault(Dataset.VAL_OUT, None)

        return X1_gen, X2_gen

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
        # Fetch generators from parent
        X_gen, X_val_gen =\
            super()._init_data_generators(ephase, list_iterstore, dict_iterstore)

        # PERF: No need to make a copy of the following generators since 
        # they are used as secondary generators to X_gen, X_val_gen with sync_generator(...)
        if (ephase == NNModelPhase.PRE_TRAIN):

            # Refer _pre_train_preprocess() for syncing these data generators
            self.Xt_gen, self.Xt_val_gen =\
                    self.callbacks['get_target_data_generators'](ephase, list_iterstore, dict_iterstore)
        
        elif (ephase == NNModelPhase.TRAIN):
            Xt_gen, Xt_val_gen =\
                    self.callbacks['get_target_data_generators'](ephase, list_iterstore, dict_iterstore)

            # Sync the data generators
            X_gen.sync_generator(Xt_gen)        
            if (X_val_gen is not None):
                X_val_gen.sync_generator(Xt_val_gen)       

        return X_gen, X_val_gen

    def _pre_train_preprocess(self, layer_idx, daeprecfgs, daecfg, X_gen, X_val_gen):
        """describe"""

        # Layer purpose
        lp = daecfg.lp[layer_idx]

        # If this is a regression learning layer
        if (lp == 'rg'):
            X_gen.sync_generator(self.Xt_gen)

            if (X_val_gen is not None):
                X_val_gen.sync_generator(self.Xt_val_gen)  

        return X_gen, X_val_gen

    def _stack_layers(self, ae, layer_idx, layers, daeprecfgs, daecfg):
        """Non tied weights"""

        # Layer purpose
        lp = daecfg.lp[layer_idx]

        # Dimension reduction layer
        if (lp == 'dr'):  
            layer_name = "enc: "

        # Regression layer
        elif (lp == 'rg'):  
            layer_name = "reg: "

        # Layer name
        layer_name += str(layer_idx)

        # Adding encoding or regression layer for DAERegModel
        layers.insert(layer_idx, 
                        Dense(daecfg.arch[layer_idx], 
                                activation='relu', 
                                weights=ae._encoder_weights, 
                                name=layer_name))

        # Last layer
        if (layer_idx == len(daeprecfgs)):
            # REQUIREMENT: Last layer must be a regression layer
            if (lp != 'rg'):
                raise Exception("ARG_ERR: Last layer must be a regression layer.") 

            dec_act = 'sigmoid' # nnprecfg[last_i].output_fn

            # Adding last decoding layer
            last_i = len(daecfg.arch) - 1
            layers.insert(layer_idx + 1, 
                        Dense(daecfg.arch[last_i], 
                                activation=dec_act, 
                                weights=ae._decoder_weights, 
                                name="dec_"+layer_name))
        return layers