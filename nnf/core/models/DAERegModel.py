# -*- coding: utf-8 -*-
# Global Imports
from nnf.keras.layers import Input, Dense

# Local Imports
from nnf.db.Dataset import Dataset
from nnf.core.models.DAEModel import DAEModel
from nnf.core.models.NNModelPhase import NNModelPhase

class DAERegModel(DAEModel):
    """Asymmetric DAE Model. Multiple encoder, single/many decoder model.

        DAEReg Model will be formed according to the layer purpose 
        attribute vector at :obj:`DAERegCfg`.

        For: i.e
        105 -> 70 -> 56 -> 56 -> 70
        lp = ['Input', 'dr', 'dr', 'rg', 'Ouput']

        .. warning:: Last layer must be a regression ('rg') layer.

    Attributes
    ----------
    callbacks : :obj:`dict`
        Callback dictionary. Supported callbacks.
        {common `NNModel` callbacks, `get_target_data_generators`}
    """
    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, callbacks=None,
                 iter_params=None, iter_pp_params=None, nncfgs=None):
        """Constructs :obj:`DAERegModel` instance."""
        super().__init__(callbacks=callbacks,
                         iter_params=iter_params,
                         iter_pp_params=iter_pp_params,
                         nncfgs=nncfgs)
        self.Xt_gen = self.Xt_val_gen = None

        # Set defaults for arguments
        tmp = self.callbacks.setdefault('get_target_data_generators', None)
        if (tmp is None):
            self.callbacks['get_target_data_generators'] = self._get_target_data_generators
     
    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _get_target_data_generators(self, ephase, 
                                            list_iterstore, dict_iterstore):
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
        if (ephase == NNModelPhase.PRE_TRAIN):
            # Iterstore for dbparam1, TR_OUT and VAL_OUT
            self.Xt_gen = list_iterstore[0].setdefault(Dataset.TR_OUT, None)
            self.Xt_val_gen = list_iterstore[0].setdefault(Dataset.VAL_OUT, None)

            # IMPORTANT: Do not return `self.Xt_gen`, `self.Xt_val_gen`
            # since those iterators must be utilized depending on the layer purpose (daecfg.lp) configuration later
            # in the method _pre_train_preprocess()

        elif (ephase == NNModelPhase.TRAIN):
            X1_gen = list_iterstore[0].setdefault(Dataset.TR_OUT, None)
            X2_gen = list_iterstore[0].setdefault(Dataset.VAL_OUT, None)

        elif (ephase == NNModelPhase.PREDICT or ephase == NNModelPhase.TEST):
            # Iterstore for dbparam1, TE_OUT
            X1_gen = list_iterstore[0].setdefault(Dataset.TE_OUT, None)

        return X1_gen, X2_gen

    ##########################################################################
    # Protected: DAEModel Overrides
    ##########################################################################
    def _model_prefix(self):
        """Fetch the _prefix for the file to be saved/loaded."""
        return "DAEReg"

    def _validate_cfg(self, ephase, cfg):
        """Validate NN Configuration for this model"""
        super()._validate_cfg(ephase, cfg)

        if (ephase != NNModelPhase.TRAIN): return

        if (len(cfg.arch) != len(cfg.lp)):
            raise Exception('Layer purpose string for each layers is not' + 
            ' specified. (length of `cfg.arch` != length of `cfg.lp`).')

    def _pre_train_preprocess(self, layer_idx, daeprecfgs, daecfg, 
                                                            X_gen, X_val_gen):
        """describe"""

        # Layer purpose
        lp = daecfg.lp[layer_idx]

        # If this is a regression learning layer
        if (lp == 'rg'):
            X_gen.sync_tgt_generator(self.Xt_gen)

            if (X_val_gen is not None):
                X_val_gen.sync_tgt_generator(self.Xt_val_gen)

        return X_gen, X_val_gen

    def _pre_train_postprocess_preloaded_db(self, layer_idx, daecfg, ae,
                                                    X_L, Xt, X_L_val, Xt_val):
        X_L, Xt, X_L_val, Xt_val =\
            super()._pre_train_postprocess_preloaded_db(
                                                    layer_idx, daecfg, ae, 
                                                    X_L, Xt, X_L_val, Xt_val)
        
        # Layer purpose (for the next layer)
        lp = daecfg.lp[layer_idx + 1]
        
        # Update the targets if it is a regression layer
        if (lp == 'rg'):
            _, Xt, _, Xt_val = daecfg.preloaded_db.LoadPreTrDb(self)

        return  X_L, Xt, X_L_val, Xt_val

    def _stack_layers(self, ae, layer_idx, layers, daecfg):
        """Non tied weights"""

        # Layer purpose
        lp = daecfg.lp[layer_idx]
        layer_name = ''

        # Dimension reduction layer
        if (lp == 'dr'):  
            layer_name = "enc_"

        # Regression layer
        elif (lp == 'rg'):  
            layer_name = "reg_"
        # Layer name
        layer_name += str(layer_idx)

        # Adding encoding or regression layer for DAERegModel
        layers.insert(layer_idx, 
                        Dense(daecfg.arch[layer_idx], 
                                activation=daecfg.act_fns[layer_idx], 
                                weights=ae._encoder_weights, 
                                name=layer_name))

        # Last layer
        if (layer_idx == len(daecfg.arch) - 2): # Pre-training is only for hidden layers
            # REQUIREMENT: Last layer must be a regression layer
            if (lp != 'rg'):
                raise Exception("ARG_ERR: Last layer must be a regression layer.") 

            dec_act = 'sigmoid' # nnprecfg[last_i].output_fn

            # Adding last decoding layer
            last_i = len(daecfg.arch) - 1
            layers.insert(layer_idx + 1, 
                        Dense(daecfg.arch[last_i], 
                                activation=daecfg.act_fns[last_i], 
                                weights=ae._decoder_weights, 
                                name="dec_"+layer_name))
        return layers