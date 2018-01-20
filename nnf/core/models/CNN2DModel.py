# -*- coding: utf-8 -*-
"""
.. module:: CNN2DModel
   :platform: Unix, Windows
   :synopsis: Represent CNN2DModel class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from nnf.keras.models import Sequential
from nnf.keras.layers import Dense, Dropout, Activation, Flatten
from nnf.keras.layers.convolutional import Conv2D, MaxPooling2D

# Local Imports
from nnf.core.models.CNNModel import CNNModel


class CNN2DModel(CNNModel):
    """CNN2DModel Convolutional Neural Network Model."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, X_L=None, Xt=None, X_L_val=None, Xt_val=None, callbacks=None,
                 iter_params=None, iter_pp_params=None, nncfgs=None):
        """Constructs :obj:`CNN2DModel` instance."""
        super().__init__(X_L, Xt, X_L_val, Xt_val, callbacks,
                         iter_params, iter_pp_params, nncfgs)

    ##########################################################################
    # Protected: CNNModel Overrides
    ##########################################################################
    def _model_prefix(self):
        """Fetch the _prefix for the file to be saved/loaded."""
        return "CNN2D"

    def _build(self, input_shapes, output_shapes, data_format):
        """Build the keras CNN2D."""
        self.net = Sequential()
        self.net.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shapes[0]))
        self.net.add(Activation('relu'))
        self.net.add(Conv2D(32, (3, 3)))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))
        self.net.add(Dropout(0.25))

        self.net.add(Conv2D(64, (3, 3), padding='same'))
        self.net.add(Activation('relu'))
        self.net.add(Conv2D(64, (3, 3)))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2)))
        self.net.add(Dropout(0.25))

        self.net.add(Flatten())
        self.net.add(Dense(512))
        self.net.add(Activation('relu'))
        self.net.add(Dropout(0.5))
        self.net.add(Dense(output_shapes[0], activation='softmax'))
        # self.net.add(Activation('softmax'))