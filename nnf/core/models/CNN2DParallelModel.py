# -*- coding: utf-8 -*-
"""
.. module:: CNN2DParallelModel
   :platform: Unix, Windows
   :synopsis: Represent CNN2DParallelModel class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from nnf.keras.engine import Model
from nnf.keras.layers import Input, Dense, Flatten
from nnf.keras.layers.convolutional import Conv2D, MaxPooling2D
from nnf.keras.layers import concatenate

# Local Imports
from nnf.db.Dataset import Dataset
from nnf.core.models.CNNModel import CNNModel
from nnf.core.models.NNModelPhase import NNModelPhase

class CNN2DParallelModel(CNNModel):
    """CNN2DParallelModel Convolutional Neural Network Model."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, X_L=None, Xt=None, X_L_val=None, Xt_val=None, callbacks=None,
                 iter_params=None, iter_pp_params=None, nncfgs=None):
        """Constructs :obj:`CNN2DParallelModel` instance."""
        super().__init__(X_L, Xt, X_L_val, Xt_val, callbacks,
                         iter_params, iter_pp_params, nncfgs)

    ##########################################################################
    # Protected: CNNModel Overrides
    ##########################################################################
    def _model_prefix(self):
        """Fetch the _prefix for the file to be saved/loaded."""
        return "CNN2DP"

    # def _init_data_generators(self, ephase, list_iterstore, dict_iterstore):
    #     """Initialize data generators for training, testing, prediction.
    #
    #     Parameters
    #     ----------
    #     ephase : :obj:`NNModelPhase`
    #         Phase of which the data generators are required.
    #
    #     list_iterstore : :obj:`list`
    #         List of iterstores for :obj:`DataIterator`.
    #
    #     dict_iterstore : :obj:`dict`
    #         Dictionary of iterstores for :obj:`DataIterator`.
    #
    #     Returns
    #     -------
    #     :obj:`tuple`
    #         When ephase == NNModelPhase.TRAIN
    #         then
    #             Generators for training and validation.
    #
    #         When ephase == NNModelPhase.TEST or NNModelPhase.PREDICT
    #         then
    #             Generators for testing and testing target.
    #     """
    #     # Fetch generators from parent
    #     X_gen, X_val_gen = \
    #         super()._init_data_generators(ephase,
    #                                       list_iterstore,
    #                                       dict_iterstore)
    #
    #     # PERF: No need to make a copy of the following generators since
    #     # they are used as secondary generators to X_gen, X_val_gen with
    #     # sync_tgt_generator(...)
    #
    #     if (list_iterstore is not None):
    #         # Fetch input auxiliary input data generators
    #         Xin_gen, Xin_val_gen = \
    #             self.callbacks['get_input_data_generators'](ephase, list_iterstore, dict_iterstore)
    #
    #         # Fetch input auxiliary target data generators
    #         Xt_gen, Xt_val_gen = \
    #             self.callbacks['get_target_data_generators'](ephase, list_iterstore, dict_iterstore)
    #
    #         # Sync the data generators (input, target)
    #         X_gen.sync_generator(Xin_gen)
    #         X_gen.sync_tgt_generator(Xt_gen)
    #         if (X_val_gen is not None):
    #             X_val_gen.sync_generator(Xin_val_gen)
    #             X_val_gen.sync_tgt_generator(Xt_val_gen)
    #
    #     return X_gen, X_val_gen

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
            # Iterstore for dbparam1, TR
            X1_gen = list_iterstore[0].setdefault(Dataset.TR, None)
            X1_gen2 = self._clone_iter(list_iterstore[0].setdefault(Dataset.TR, None))

            # Two inputs to parallel model (training)
            X1_gen.sync_generator(X1_gen2)

            X2_gen = list_iterstore[0].setdefault(Dataset.VAL, None)
            X2_gen2 = self._clone_iter(list_iterstore[0].setdefault(Dataset.VAL, None))

            # Two inputs to parallel model (validation)
            X2_gen.sync_generator(X2_gen2)

        elif (ephase == NNModelPhase.PREDICT or ephase == NNModelPhase.TEST):
            # Iterstore for dbparam1, TE
            X1_gen = list_iterstore[0].setdefault(Dataset.TE, None)
            X1_gen2 = self._clone_iter(list_iterstore[0].setdefault(Dataset.TE, None))

            # Two inputs to parallel model (testing input)
            X1_gen.sync_generator(X1_gen2)

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
        # if (ephase == NNModelPhase.PREDICT or ephase == NNModelPhase.TEST):
            # Iterstore for dbparam1, TE_OUT
            # X1_gen = list_iterstore[0].setdefault(Dataset.TE_OUT, None)

        return X1_gen, X2_gen

    def _build(self, input_shapes, output_shapes, data_format):
        """Build the keras CNN2DParallel."""
        main_input = Input(shape=input_shapes[0], name='main_input')

        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(main_input)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Note: For demonstration purpose `main_input` is used instead of `auxiliary_input`
        auxiliary_input = Input(shape=input_shapes[0], name='auxiliary_input')

        y = Conv2D(32, (3, 3), activation='relu', padding='same', name='aux_block1_conv1')(main_input)
        y = Conv2D(32, (3, 3), activation='relu', padding='same', name='aux_block1_conv2')(y)
        y = MaxPooling2D((2, 2), strides=(2, 2), name='aux_block1_pool')(y)

        x = concatenate([x, y])

        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        main_output = Dense(output_shapes[0], activation='softmax', name='main_output')(x)

        self.net = Model(inputs=[main_input, auxiliary_input], outputs=main_output)

        # For multiple outputs
        # x = Flatten(name='flatten')(x)
        # x = Dense(4096, activation='relu', name='fc1')(x)
        # auxiliary_output = Dense(output_shapes[0], activation='softmax', name='aux_output')(x)
        #
        # x = Dense(2048, activation='relu', name='fc2')(x)
        # main_output = Dense(output_shapes[0], activation='softmax', name='main_output')(x)
        #
        # self.net = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])