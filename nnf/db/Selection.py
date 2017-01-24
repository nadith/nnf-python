# -*- coding: utf-8 -*-: TODO: RECHECK COMMENTS
"""
.. module:: Selection
   :platform: Unix, Windows
   :synopsis: Represent Selection class and Select enumeration.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from enum import Enum
import numpy as np

# Local Imports

class Selection:
    """Selection denotes the selection paramters for a database.

    Attributes
    ----------
    TODO: Put this comment to a python compatible way

    Selection Structure (with defaults)
    -----------------------------------
    sel.tr_col_indices      = None    # Training column indices
    sel.tr_noise_rate       = None    # Rate or noise types for the above field
    sel.tr_out_col_indices  = None    # Training target column indices
    sel.val_col_indices     = None    # Validation column indices
    sel.val_out_col_indices = None    # Validation target column indices
    sel.te_col_indices      = None    # Testing column indices
    sel.te_out_col_indices  = None    # Testing target column indices
    sel.nnpatches           = None    # NNPatch object array
    sel.use_rgb             = None    # Use rgb or convert to grayscale
    sel.color_indices       = None    # Specific color indices (set .use_rgb = false)
    sel.use_real            = False   # Use real valued database TODO: (if .normalize = true, Operations ends in real values)  # noqa E501
    
    sel.scale               = None    # Scaling factor (resize factor)
                                        int - Percentage of current size.
                                        float - Fraction of current size.
                                        tuple - Size of the output image.

    sel.normalize           = False   # Normalize (0 mean, std = 1)
    sel.histeq              = False   # Histogram equalization
    sel.histmatch_col_index = None    # Histogram match reference column index
    sel.class_range         = None    # Class range for training database or all (tr, val, te)
    sel.val_class_range     = None    # Class range for validation database
    sel.te_class_range      = None    # Class range for testing database
    sel.pre_process_script  = None    # Custom preprocessing script
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, **kwds):
        """Constructs :obj:`Selection` instance."""

        self.tr_col_indices      = None    # Training column indices
        self.tr_noise_rate       = None    # Rate or noise types for the above field
        self.tr_out_col_indices  = None    # Training target column indices
        self.val_col_indices     = None    # Validation column indices
        self.val_out_col_indices = None    # Validation target column indices
        self.te_col_indices      = None    # Testing column indices
        self.te_out_col_indices  = None    # Testing target column indices
        self.nnpatches           = None    # NNPatch object array
        self.use_rgb             = None    # Use rgb or convert to grayscale
        self.color_indices       = None    # Specific color indices (set .use_rgb = false)
        self.use_real            = False   # Use real valued database TODO: (if .normalize = true, Operations ends in real values)  # noqa E501
        self.scale               = None    # Scaling factor (resize factor)
        self.normalize           = False   # Normalize (0 mean, std = 1)
        self.histeq              = False   # Histogram equalization
        self.histmatch_col_index = None    # Histogram match reference column index
        self.class_range         = None    # Class range for training database or all (tr, val, te)
        self.val_class_range     = None    # Class range for validation database
        self.te_class_range      = None    # Class range for testing database
        self.pre_process_script  = None    # Custom preprocessing script

    def need_processing(self):
        """Return True only if pre-processing parameters of the selection structure is set"""
        return (self.use_rgb is not None) or\
            (self.color_indices is not None) or\
            (self.scale is not None) or\
            (self.normalize is not False) or\
            (self.histeq is not False) or\
            (self.histmatch_col_index is not None)

    def clone(self, shallow=True):
        """Clone this :obj:`Selection` instance.

        Parameters
        ----------
        shallow : bool
            Whether to perform deep or shallow copy.

        Returns
        -------
        :obj:`Selection`
            Cloned object.
        """
        # deep copy is not performed on nnpatches
        assert(shallow == True)

        sel = Selection()
        sel.tr_col_indices      = self.tr_col_indices        # Training column indices
        sel.tr_noise_rate       = self.tr_noise_rate         # Rate or noise types for the above field
        sel.tr_out_col_indices  = self.tr_out_col_indices    # Training target column indices
        sel.val_col_indices     = self.val_col_indices       # Validation column indices
        sel.val_out_col_indices = self.val_out_col_indices   # Validation target column indices
        sel.te_col_indices      = self.te_col_indices        # Testing column indices
        sel.te_out_col_indices  = self.te_out_col_indices    # Testing target column indices
        sel.nnpatches           = self.nnpatches             # NNPatch object array
        sel.use_rgb             = self.use_rgb               # Use rgb or convert to grayscale
        sel.color_indices       = self.color_indices         # Specific color indices (set .use_rgb = false)
        sel.use_real            = self.use_real              # Use real valued database TODO: (if .normalize = true, Operations ends in real values)  # noqa E501
        sel.scale               = self.scale                 # Scaling factor (resize factor)
        sel.normalize           = self.normalize             # Normalize (0 mean, std = 1)
        sel.histeq              = self.histeq                # Histogram equalization
        sel.histmatch_col_index = self.histmatch_col_index   # Histogram match reference column index
        sel.class_range         = self.class_range           # Class range for training database or all (tr, val, te)
        sel.val_class_range     = self.val_class_range       # Class range for validation database
        sel.te_class_range      = self.te_class_range        # Class range for testing database
        sel.pre_process_script  = self.pre_process_script    # Custom preprocessing script
        return sel
        
    ##########################################################################
    # Special Interface
    ##########################################################################
    def __eq__(self, sel):
        """Equality of two :obj:`ImagePreProcessingParam` instances.

        Parameters
        ----------
        sel : :obj:`Selection`
            The instance to be compared against this instance.

        Returns
        -------
        bool
            True if both instances are the same. False otherwise.
        """
        iseq = False
        if (np.array_equal(self.tr_col_indices, sel.tr_col_indices) and
            np.array_equal(self.tr_noise_rate, sel.tr_noise_rate) and
            np.array_equal(self.tr_out_col_indices, sel.tr_out_col_indices) and
            np.array_equal(self.val_col_indices, sel.val_col_indices) and
            np.array_equal(self.val_out_col_indices, sel.val_out_col_indices) and
            np.array_equal(self.te_col_indices, sel.te_col_indices) and
            np.array_equal(self.te_out_col_indices, sel.te_out_col_indices) and
            (self.use_rgb == sel.use_rgb) and
            np.array_equal(self.color_indices, sel.color_indices) and
            (self.use_real == sel.use_real) and
            (self.normalize == sel.normalize) and
            (self.histeq == sel.histeq) and
            (self.histmatch_col_index == sel.histmatch_col_index) and
            np.array_equal(self.class_range, sel.class_range) and
            np.array_equal(self.val_class_range, sel.val_class_range) and
            np.array_equal(self.te_class_range, sel.te_class_range) and
            len(self.nnpatches) == len(sel.nnpatches)):
            # self.pre_process_script # LIMITATION: Cannot compare for eqaulity (in the context of serilaization)
            iseq = True

        if (not iseq):
            return iseq

        for i, self_patch in enumerate(self.nnpatches):
            self_patch = self.nnpatches[i]
            sel_patch = sel.nnpatches[i]
            iseq = iseq and (self_patch == sel_patch)
            if (not iseq):
                break

        return iseq


class Select(Enum):
    """Select Enumeration describes the special constants for Selection structure.

    Attributes
    ----------
    ALL : int
        -999
    """
    ALL = -999
    PERCENT_40 = -40
    PERCENT_60 = -60

    ##########################################################################
    # Public Interface
    ##########################################################################
    def int(self):
        """Evaluate the enumeration value to its representating integer."""
        return self.value