"""DAEModel to represent DAEModel class."""
# -*- coding: utf-8 -*-
# Global Imports
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np

# Local Imports
from nnf.core.models.NNModel import NNModel
from nnf.core.models.Autoencoder import Autoencoder
from nnf.core.iters.DataIterator import DataIterator
from nnf.db.NNdb import NNdb
from nnf.db.Format import Format

class DAEModel(NNModel):
    """Generic deep order encoder model
    
       Extend this class to implement specific auto encoder models
    """
    def __init__(self, patch, iterstore):   
        super().__init__(patch, iterstore)

        # Initialize instance variables
        self.X_gen = self.X_val_gen = self.Xte_gen = None

        # Initialize iterators
        if (self._iterstore is not None):
            (self.X_gen, self.X_val_gen, self.Xte_gen) = self._iterstore[0]
    
    # Model based framework will set the iterstore via set_iterstore() 
    # due to iterstores being not created during the model creation.
    # In contrast Patch based framework set the iterator store in the constructor.
    def set_iterstore(self, iterstore):
        super().set_iterstore(iterstore)
               
         # Initialize iterators
        if (self._iterstore is not None):
            (self.X_gen, self.X_val_gen, self.Xte_gen) = self._iterstore[0]

    def pre_train(self, daeprecfgs, daecfg):

        # Track layers to build Deep/Stacked Auto Encoder (DAE)
        layers = [Input(shape=(daecfg.arch[0],))]

        # Initialize weights and transpose of weights
        w_T = w = None 
       
        # Set generator variables
        X_gen = self.X_gen
        X_val_gen = self.X_val_gen

        # Iterate through pre-trianing configs and 
        # perform layer-wise pre-training.
        i = 0
        for daeprecfg in daeprecfgs:

            # Directly feeding external databases            
            X = None
            X_val = None

            # Data from generators    
            _iterstore = [(X_gen, X_val_gen)]
    
            # Create a simple AE
            ae = Autoencoder(self.patches, _iterstore, X, X_val)
     
            # Using non-generators for external databases
            if (daecfg.use_db != 'generator'):    
                X, X_val = ae.train(daeprecfg)            
                X = ae._encode(X)
                X_val = ae._encode(X_val)

            else:                
                if (_iterstore == [(None, None)]):
                    raise Exception("No data iterators. Check daecfg.use_db option !")   

                # Using generators for NNDb or disk databases            
                ae.train(daeprecfg)   

                # Fetch generator for next layer (i+1)
                X_gen, X_val_gen = self._get_genrators_at(i, ae, X_gen, X_val_gen)

            #w = ae._encoder_weights
            #w_T = [np.transpose(w[0]), np.zeros(w[0].shape[0])]
            
            ## Adding encoding layer for DAE
            #layers.insert(i, Dense(enc_arch[i], activation='relu', weights=w, name="encoder" + str(i+1)))

            # Adding decoding layer for DAE
            dec_act = 'relu' # nnprecfg[i].act_fn  
        #    if (i == 1):  # for last decoder
        #        dec_act = 'sigmoid' # nnprecfg[n-1].output_fn      
        #    layers.insert(i+1, Dense(enc_arch[i-1], activation=dec_act, weights=w_T, name="decoder" + str(i+1)))

            i = i + 1

        ## Build DAE
        #in_tensor = layers[0]
        #out_tensor = layers[0]

        #for i in range(1, len(layers)):
        #    out_tensor = layers[i](out_tensor)
        
        #self.net = Model(input=in_tensor, output=out_tensor)
        #print(self.net.summary())

        #self.net.compile(optimizer=daecfg.opt_fn, loss=daecfg.loss_fn)

    def train(self, daecfg):
        if (self.net is None):
            pass

        if (daecfg.use_db == 'mnist'):
            enc_data, enc_data_val = MnistAE.LoadDb()

        if (daecfg.use_db != 'generator'):    
            self.net.fit(enc_data, enc_data,
                    nb_epoch=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(enc_data_val, enc_data_val), callbacks=[cb_early_stop])
        else:
            X_gen = self.data_client
            X_val_gen = self.data_val_client

            self.net.fit_generator(
                    X_gen,
                    samples_per_epoch=2000,
                    nb_epoch=50,
                    validation_data=X_val_gen,
                    nb_val_samples=800)

        pass

    def _get_genrators_at(self, i, ae, X_gen, X_val_gen):
        # If the current generators are pointing to a temporary location, mark it for deletion
        # predict from current X_gen and X_val_gen and save the data in a temporary location 
        # for index i (layer i). Construct new generators for the data stored in this temporary location.
        # if the previous data locations are marked for deletion (temporary location), proceed the deletion

        enc_data = ae._encode(X_gen.nndb.features_scipy)         
        X_gen = DataIterator(NNdb('Temp', enc_data, format=Format.N_H))

        if (X_val_gen is not None):
            enc_data_val = ae._encode(X_val_gen.nndb.features_scipy)
            X_val_gen = DataIterator(NNdb('TempVal', enc_data_val, format=Format.N_H))

        return (X_gen, X_val_gen)

class DAEModelEx(NNModel):
    """Multiple encoder, single decoder model"""
    pass