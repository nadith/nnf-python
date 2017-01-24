# -*- coding: utf-8 -*-
"""
.. module:: hist_match_ch
   :platform: Unix, Windows
   :synopsis: Represent hist_match_ch function.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np

# Local Imports
from nnf.pp.hist_match import hist_match

def hist_match_ch(cimg, cann_cimg, ch_axis):
    """Histogram matching on individual color channel."""

    nch = np.size(cimg, ch_axis)
    for ich in range(nch):
        if (ch_axis == 0):
            cimg[ich, :, :] = hist_match(cimg[ich, :, :], cann_cimg[ich, :, :])
        elif (ch_axis == 1):
            cimg[:, ich, :] = hist_match(cimg[:, ich, :], cann_cimg[:, ich, :])
        elif (ch_axis == 2):
            cimg[:, :, ich] = hist_match(cimg[:, :, ich], cann_cimg[:, :, ich])

    return cimg