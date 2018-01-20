# -*- coding: utf-8 -*-
"""
.. module:: CNN2DRegModel
   :platform: Unix, Windows
   :synopsis: Represent CNN2DRegModel class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
from nnf.keras.models import Sequential
from nnf.keras.layers import Dense, Dropout, Activation, Flatten
from nnf.keras.layers.convolutional import Conv2D, MaxPooling2D

# Local Imports
from nnf.db.Dataset import Dataset
from nnf.core.models.CNNModel import CNNModel
from nnf.core.models.NNModelPhase import NNModelPhase

class CNN2DRegModel(CNNModel):
    """CNN2DRegModel Convolutional Neural Network Model."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, X_L=None, Xt=None, X_L_val=None, Xt_val=None, callbacks=None,
                 iter_params=None, iter_pp_params=None, nncfgs=None):
        """Constructs :obj:`CNN2DRegModel` instance."""

        # For regression output
        callbacks = {} if callbacks is None else callbacks

        tmp = callbacks.setdefault('get_input_data_generators', None)
        callbacks['get_input_data_generators'] = self._get_input_data_generators

        tmp = callbacks.setdefault('get_target_data_generators', None)
        if (tmp is None):
            callbacks['get_target_data_generators'] = self._get_target_data_generators

        super().__init__(X_L, Xt, X_L_val, Xt_val, callbacks, iter_params,
                                                        iter_pp_params, nncfgs)

    ##########################################################################
    # Protected: CNNModel Overrides
    ##########################################################################
    def _model_prefix(self):
        """Fetch the _prefix for the file to be saved/loaded."""
        return "CNN2DR"

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
        if (ephase == NNModelPhase.PRE_TRAIN or ephase == NNModelPhase.TRAIN):
            # Iterstore for dbparam1, TR_OUT and VAL_OUT
            X1_gen = list_iterstore[1].setdefault(Dataset.TR_OUT, None)
            X2_gen = list_iterstore[1].setdefault(Dataset.VAL_OUT, None)

        return X1_gen, X2_gen

    def _build(self, input_shapes, output_shapes, data_format):
        """Build the keras CNN2DParallel."""

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

        self.net.add(Dense(output_shapes[0]))
        self.net.add(Activation('sigmoid'))