# -*- coding: utf-8 -*-
"""
.. module:: internals
   :platform: Unix, Windows
   :synopsis: Represent nnf internal utility functions.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

#  Global Imports
import numpy as np
from keras import backend as K

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

# Local Imports


if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS

def load_img(path, use_rgb=None, target_size=None,
             interpolation='bilinear'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        use_rgb: Boolean, whether to load the image as color or grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "bilinear" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if use_rgb is None:
        pass  # do nothing
    elif use_rgb:
        if img.mode == 'L':
            raise Exception("Data read cannot be converted to rgb.")

        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        if img.mode != 'L':
            img = img.convert('L')

    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())

    # Make the array writable
    if not x.flags['WRITEABLE']:
        x.flags['WRITEABLE'] = True

    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose((2, 0, 1))
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def array_to_img(x, data_format=None, scale=True, keep_max_precision=False):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose((1, 2, 0))

    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255

    if x.shape[2] == 3:
        # RGB
        if keep_max_precision:
            # Does not support RGB floating point images
            # REF: http: // pillow.readthedocs.io / en / 3.1.x / handbook / concepts.html  # concept-modes
            # Hence save the average image across channels
            return pil_image.fromarray(np.average(x, axis=2), 'F')
        else:
            return pil_image.fromarray(x.astype('uint8'), 'RGB')

    elif x.shape[2] == 1:
        # grayscale
        if keep_max_precision:
            return pil_image.fromarray(x[:, :, 0], 'F')
        else:
            return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')

    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])
