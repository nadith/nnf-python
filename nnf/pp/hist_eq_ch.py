# -*- coding: utf-8 -*-
"""
.. module:: hist_eq_ch
   :platform: Unix, Windows
   :synopsis: Represent hist_eq_ch function.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np

# Local Imports
from nnf.pp.hist_eq import hist_eq

def hist_eq_ch(cimg, ch_axis, number_bins=256):
    """Histogram equalization on individual color channel."""

    nch = np.size(cimg, ch_axis)
    for ich in range(nch):
        if (ch_axis == 0):
            cimg[ich, :, :], _ = hist_eq(cimg[ich, :, :], number_bins)
        elif (ch_axis == 1):
            cimg[:, ich, :], _ = hist_eq(cimg[:, ich, :], number_bins)
        elif (ch_axis == 2):
            cimg[:, :, ich], _ = hist_eq(cimg[:, :, ich], number_bins)

    return cimg