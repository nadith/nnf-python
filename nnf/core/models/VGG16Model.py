# -*- coding: utf-8 -*-
# Global Imports
from nnf.keras.engine import Model
from keras.utils import layer_utils
from nnf.keras.layers import Input, Dense, Flatten
from nnf.keras.layers.convolutional import Conv2D, MaxPooling2D

# Local Imports
from nnf.core.models.CNNModel import CNNModel


class VGG16Model(CNNModel):
    """VGGNet16 Convolutional Neural Network Model.

    Notes
    -----
    ref: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, X_L=None, Xt=None, X_L_val=None, Xt_val=None, callbacks=None,
                 iter_params=None, iter_pp_params=None, nncfgs=None):
        """Constructs :obj:`VGG16Model` instance."""
        super().__init__(callbacks=callbacks,
                         iter_params=iter_params,
                         iter_pp_params=iter_pp_params,
                         nncfgs=nncfgs)

        # Used when data is fetched from no iterators
        self.X_L = X_L          # (X, labels)
        self.Xt = Xt            # X target
        self.X_L_val = X_L_val  # (X_val, labels_val)
        self.Xt_val = Xt_val    # X_val target

    ##########################################################################
    # Protected: CNNModel Overrides
    ##########################################################################
    def _model_prefix(self):
        """Fetch the _prefix for the file to be saved/loaded."""
        return "VGG16"

    def _build(self, input_shapes, output_shapes, data_format):
        """Build the keras VGG16.

            This model is a replicate of `keras.applications.vgg16.VGG16`
        """
        if data_format == 'channels_last':
            assert input_shapes[0] == (224, 224, 3)
        else:
            assert input_shapes[0] == (3, 224, 224)

        img_input = Input(shape=input_shapes[0])

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # For a non-classification block
        # if pooling == 'avg':
        #     x = GlobalAveragePooling2D()(x)
        # elif pooling == 'max':
        #     x = GlobalMaxPooling2D()(x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)

        # In order to support pre-trained weight loading, keep the number of class as original
        x = Dense(1000, activation='softmax', name='predictions')(x)

        self.net = Model(img_input, x, name=self._model_prefix())

        if data_format == 'channels_first':
            maxpool = self.net.get_layer(name='block5_pool')
            shape = maxpool.output_shapes[0][1:]
            dense = self.net.get_layer(name='fc1')
            layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

    def _pre_compile(self, cfg, input_shapes, output_shapes, data_format):
        """Pre-compile processing for keras model.

            Notes
            -----
            This callback is invoked after loading weights but before compiling
        """
        # Pop the last layer
        self.net.layers.pop()

        # Add a new last layer with required no. of classes
        x = Dense(output_shapes[0], activation='softmax', name='predictions')(self.net.layers[-1].output)

        # Create a new keras model object
        self.net = Model(self.net.inputs[0], x, name=self._model_prefix())
      
