"""Autoencoder to represent Autoencoder class."""
# -*- coding: utf-8 -*-
# Global Imports
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from warnings import warn as warning
import numpy as np

# Local Imports
from nnf.core.models.NNModel import NNModel

class Autoencoder(NNModel):
    """description of class"""

    def __init__(self, patch, iterstore, X=None, X_val=None):   
        super().__init__(patch, iterstore)

        # Initialize variables
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.cb_early_stop = None

        # Initialize iterators
        if (self._iterstore is not None):
            (self.in_X_gen, self.in_X_val_gen) = self._iterstore[0]
        
        # Used when data is fetched from no iterators
        self.X = X
        self.X_val = X_val

    def build(self, input_dim=784,  hidden_dim=32, output_dim=784, 
                enc_activation='sigmoid', enc_weights=None, 
                dec_activation='sigmoid', dec_weights=None,
                loss_fn='mean_squared_error', use_early_stop=False, 
                cb_early_stop=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')):
   
        # Set early stop for later use
        self.cb_early_stop = cb_early_stop

        # this is our input placeholder
        input_img = Input(shape=(input_dim,))

        # "encoded" is the encoded representation of the input
        encoded = Dense(hidden_dim, activation=enc_activation, weights=enc_weights)(input_img)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(input_dim, activation=dec_activation, weights=dec_weights)(encoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input=input_img, output=decoded)
    
        #
        # this model maps an input to its encoded representation
        self.encoder = Model(input=input_img, output=encoded)
    
        #
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(hidden_dim,))

        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]            

        # create the decoder model
        self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        self.autoencoder.compile(optimizer='adadelta', loss=loss_fn)
        
        print('Autoencoder: {0}, {1}, {2}, {3}, {4}'.format(input_dim, hidden_dim, enc_activation, dec_activation, loss_fn))
        print(self.autoencoder.summary())

    def pre_train(self, daeprecfgs, daecfg):
        warning('Pre-training is not supported in Autoencoder')

    def train(self, cfg):
        in_dim = cfg.arch[0]
        hid_dim = cfg.arch[1]
        out_dim = cfg.arch[2]

        # For generic auto encoder model
        assert(in_dim == out_dim)

        # Set the configutation of the AE acoording to the cfg
        self.build(in_dim, hid_dim, out_dim, 
                    cfg.act_fns[1], None, 
                    cfg.act_fns[2], None,
                    cfg.loss_fn)

        # External databases for quick deployment
        if (cfg.use_db == 'mnist' and 
            self.X is None and self.X_val is None):
            self.X, self.X_val = MnistAE.LoadDb()     

        # Training without generators
        if (cfg.use_db != 'generator'):    
            self._train(self.X, self.X, self.X_val, self.X_val) 
            return (self.X, self.X_val)

        # Training with generators
        self._train(X_gen=self.in_X_gen, X_val_gen=self.in_X_val_gen)
        return (None, None)

    def _train(self, X=None, XT=None, X_val=None, XT_val=None, X_gen=None, X_val_gen=None):
        assert((X is not None) or (X_gen is not None))
        
        # Train from in memory data
        if (X is not None):
            if (X_val is not None):
                self.autoencoder.fit(X, XT,
                        nb_epoch=50,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(XT_val, XT_val), 
                        callbacks=[self.cb_early_stop])
            else:
                self.autoencoder.fit(X, XT,
                        nb_epoch=50,
                        batch_size=256,
                        shuffle=True)   
        
        else: # Train from data generators
            if (X_val_gen is not None):
                self.autoencoder.fit_generator(
                        X_gen, samples_per_epoch=1,
                        nb_epoch=50,
                        validation_data=X_val_gen, nb_val_samples=1, 
                        callbacks=[self.cb_early_stop])
            else:
                self.autoencoder.fit_generator(
                        X_gen, samples_per_epoch=1,
                        nb_epoch=50)

    def _test(self, X, visualize=False):
        decoded_imgs = self.autoencoder.predict(X)
        return decoded_imgs

    def _encode(self, X):
        return self.encoder.predict(X)

    #################################################################
    # Dependant property Implementations
    #################################################################
    @property
    def _encoder_weights(self):
        enoding_layer = self.autoencoder.layers[1] 
        assert enoding_layer == self.autoencoder.layers[-2]
        return enoding_layer.get_weights()

    @property
    def _decoder_weights(self):
        decoding_layer = self.autoencoder.layers[-1] 
        assert decoding_layer == self.autoencoder.layers[2] 
        return decoder_layer.get_weights()