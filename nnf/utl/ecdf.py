# -*- coding: utf-8 -*-
"""
.. module:: ecdf
   :platform: Unix, Windows
   :synopsis: Represent ecdf function.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np

# Local Imports

def ecdf(x):
    """Compute the empirical CDF of an image.

    Parameters
    ----------
    x: `array_like`
        Image vector. (1D)

    Returns
    -------
    `array_like` :
        Unique values of image vector provided.
    
    `array_like` :
        Empirical CDF vector computed.
    """
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf
