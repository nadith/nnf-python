# -*- coding: utf-8 -*-
"""
.. module:: hist_eq
   :platform: Unix, Windows
   :synopsis: Represent hist_eq function.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np

# Local Imports

def hist_eq(img, number_bins=256):
    """Histogram equalization on single channel (grayscale) image."""
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(img.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(img.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(img.shape), cdf
