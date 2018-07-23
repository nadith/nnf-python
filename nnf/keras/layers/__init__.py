from __future__ import absolute_import

from keras.utils.generic_utils import deserialize_keras_object
from ..engine import Layer
from ..engine import Input
from ..engine import InputLayer
from ..engine import InputSpec
from .merge import *
from .core import *
from .convolutional import *
from .pooling import *
from keras.layers.local import *
from keras.layers.recurrent import *
from keras.layers.cudnn_recurrent import *
from keras.layers.normalization import *
from keras.layers.embeddings import *
from keras.layers.noise import *
from keras.layers.advanced_activations import *
from keras.layers.wrappers import *
from keras.layers.convolutional_recurrent import *
from keras.legacy.layers import *


def serialize(layer):
    """Serialize a layer.

    # Arguments
        layer: a Layer object.

    # Returns
        dictionary with config.
    """
    return {'class_name': layer.__class__.__name__,
            'config': layer.get_config()}


def deserialize(config, custom_objects=None):
    """Instantiate a layer from a config dictionary.

    # Arguments
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names)
            of custom (non-Keras) objects to class/functions

    # Returns
        Layer instance (may be Model, Sequential, Layer...)
    """
    from nnf.keras import models
    globs = globals()  # All layers.
    globs['Model'] = models.Model
    globs['Sequential'] = models.Sequential
    return deserialize_keras_object(config,
                                    module_objects=globs,
                                    custom_objects=custom_objects,
                                    printable_module_name='layer')
