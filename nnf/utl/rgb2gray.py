# -*- coding: utf-8 -*-
"""
.. module:: rgb2gray
   :platform: Unix, Windows
   :synopsis: Represent rgb2gray function.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

#  Global Imports
import numpy as np

# Local Imports

    
def rgb2gray(rgb, dtype='uint8', keepDims=False):
    """convert a color image to gray scale image.

    Parameters
    ----------
    rgb : array_like -uint8
        3D Data tensor that contains image.

        Format for color images: (H x W x CH).

    cast : bool, optional
        Cast to uint8 data type. (Default value = False).

    keepDims : bool, optional
        Preserve dimension after conversion. (Default value = False).

    Returns
    -------
    img : array_like -uint8
        2D Data tensor that contains gray scale image if keepDims=False.

        3D Data tensor that contains gray scale image if keepDims=True.

    Examples
    --------
    Color image to gray scale image conversion

    >>> gray = rgb2gray(img)

    Notes
    ------
    Copyright 2015-2016 Nadith Pathirage, Curtin University.
    (chathurdara@gmail.com).
    """
    # rgb: (..., channels)

    # Alternatively
    #r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    gray = np.sum(rgb* [0.2989, 0.5870, 0.1140], axis=len(rgb.shape)-1, keepdims=keepDims)
    gray = gray.astype(dtype, copy=False)

    return gray
