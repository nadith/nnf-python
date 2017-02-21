# -*- coding: utf-8 -*-
"""
.. module:: contrast_strech
   :platform: Unix, Windows
   :synopsis: Represent contrast_strech function.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
from skimage import exposure

# Local Imports

def contrast_strech(img, percentile=(2, 98)):
    """Contrast streching on single channel (grayscale) image."""
    # Ref:http://scikit-image.org/docs/0.9.x/auto_examples/plot_equalize.html

    p1 = np.percentile(img, percentile[0])
    p2 = np.percentile(img, percentile[1])
    return exposure.rescale_intensity(img, in_range=(p1, p2))
