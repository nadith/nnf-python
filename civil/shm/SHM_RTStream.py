# -*- coding: utf-8 -*-: TODO: RECHECK COMMENTS
"""
.. module:: SHM_RTStream
   :platform: Unix, Windows
   :synopsis: Represent selection structure for real-time SHM data stream.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
from nnf.db.Selection import RTStream

# Local Imports


class SHM_RTStream(RTStream):
    """SHM_RTStream denotes the selection parameters for a real-time SHM data stream."""

    def __init__(self, **kwds):
        """Constructs :obj:`SHM_RTStream` instance."""
        super().__init__()
        self.uncert_per_sample = 5
        self.mnoise_per_sample = 5
        self.damage_cases = [[1], [1, 2]]
        self.stiffness_range = np.arange(0.95, 0.75, -0.01)
        self.element_count = 70
        self.generate_input = False

    @property
    def nb_class(self):
        total_cls_n = 0
        for damage_case in self.damage_cases:
            total_cls_n += self.stiffness_range.size ** len(damage_case)
        return total_cls_n

    @property
    def nb_sample(self):
        total_n = 0
        val1 = 1 if self.uncert_per_sample == 0 else self.uncert_per_sample
        val2 = 1 if self.mnoise_per_sample == 0 else self.mnoise_per_sample
        n_per_class = val1 * val2
        for damage_case in self.damage_cases:
            total_n += n_per_class * self.stiffness_range.size ** len(damage_case)
        return total_n