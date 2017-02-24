# -*- coding: utf-8 -*-
# Global Imports
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from warnings import warn as warning

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
#from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,RMSprop,adam


# Local Imports
from nnf.core.models.CNNModel import CNNModel
from nnf.core.models.NNModelPhase import NNModelPhase



class VGG16Model(CNNModel):
    """VGGNet16 Convolutional Neural Network Model.

    Notes
    -----
    ref: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, callbacks=None):
        """Constructs :obj:`VGG16Model` instance."""
        super().__init__(callbacks=callbacks)

    ##########################################################################
    # Protected: CNNModel Overrides
    ##########################################################################
    def _model_prefix(self):
        """Fetch the prefix for the file to be saved/loaded."""

        return "VGG16"

    def _build(self, input_shape, nb_class, dim_ordering):
        """Build the keras VGG16."""
        assert(dim_ordering == 'th')  
        assert(input_shape == (3,224,224)) # and nb_class == 1000)                 

        self.net = Sequential()
        self.net.add(ZeroPadding2D((1,1),input_shape=input_shape))
        self.net.add(Convolution2D(64, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(64, 3, 3, activation='relu'))
        self.net.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(128, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(128, 3, 3, activation='relu'))
        self.net.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(256, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(256, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(256, 3, 3, activation='relu'))
        self.net.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(ZeroPadding2D((1,1)))
        self.net.add(Convolution2D(512, 3, 3, activation='relu'))
        self.net.add(MaxPooling2D((2,2), strides=(2,2)))
    
        self.net.add(Flatten())
        self.net.add(Dense(4096, activation='relu'))
        self.net.add(Dropout(0.5))
        self.net.add(Dense(4096, activation='relu'))
        self.net.add(Dropout(0.5))
        self.net.add(Dense(1000, activation='softmax'))   
       
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.net.compile(optimizer=sgd, loss='categorical_crossentropy') 
      
