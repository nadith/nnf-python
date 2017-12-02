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


# noinspection PyPep8
class Selection:
    """Selection denotes the selection parameters for a database.

    Attributes
    ----------
    tr_col_indices : ndarray -int
        Training column indices. (Default value = None).

    tr_noise_rate : ndarray -float
        Noise rate (0-1) or Noise types for `tr_col_indices`. (Default value = None).

    tr_occlusion_rate : ndarray -float
        Occlusion rate (0-1) for `tr_col_indices`. (Default value = None).

    tr_occlusion_type : ndarray -char
        Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right) for `tr_col_indices`. (Default value = None).

    tr_occlusion_offset : ndarray -int
        Occlusion start offset from top/bottom/left/right corner depending on `tr_occlusion_type`. (Default value = None).

    tr_out_col_indices : ndarray -int
        Training target column indices. (Default value = None).

    val_col_indices : ndarray -int
        Validation column indices. (Default value = None).

    val_out_col_indices : ndarray -int
        Validation target column indices. (Default value = None).

    te_col_indices : ndarray -int
        Testing column indices. (Default value = None).

    te_out_col_indices : ndarray -int
        Testing target column indices. (Default value = None).

    nnpatches : ndarray -`nnf.db.NNPatch`
        NNPatch object array. (Default value = None).

    use_rgb : bool
        Use rgb or convert to grayscale. (Default value = None).

    color_indices : ndarray -int
        Specific color indices (set use_rgb = false). (Default value = None).

    use_real : bool
        Use real valued database. (Default value = False).
        TODO: (if .normalize = true, Operations ends in real values)  # noqa E501

    scale : float or `tuple`
        Scaling factor (resize factor). (Default value = None).
        int - Percentage of current size.
        float - Fraction of current size.
        tuple - Size of the output image.

    normalize : bool
        Normalize (0 mean, std = 1). (Default value = False).

    histeq : bool
        Histogram equalization. (Default value = False).

    histmatch_col_index : bool
        Histogram match reference column index. (Default value = None).

    class_range : ndarray -int
        Class range for training database or default for (tr, val, te). (Default value = None).

    val_class_range : ndarray -int
        Class range for validation database. (Default value = None).

    te_class_range : ndarray -int
        Class range for testing database. (Default value = None).

    pre_process_script : ndarray -int
        Custom pre-processing script. (Default value = None).
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, **kwds):
        """Constructs :obj:`Selection` instance."""

        self.tr_col_indices      = None    # Training column indices
        self.tr_noise_rate       = None    # Rate or noise types for column index
        self.tr_occlusion_rate   = None    # Occlusion rate for column index
        self.tr_occlusion_type   = None    # Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right)
        self.tr_occlusion_offset = None    # Occlusion start offset from top/bottom/left/right corner depending on `tr_occlusion_type`
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
        self.pre_process_script  = None    # Custom pre-processing script

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
        sel.tr_occlusion_rate   = self.tr_occlusion_rate     # Occlusion rate for column index
        sel.tr_occlusion_type   = self.tr_occlusion_type     # Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right)
        sel.tr_occlusion_offset = self.tr_occlusion_offset
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
        sel.pre_process_script  = self.pre_process_script    # Custom pre-processing script

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
            np.array_equal(self.tr_occlusion_rate, sel.tr_occlusion_rate) and
            np.array_equal(self.tr_occlusion_type, sel.tr_occlusion_type) and
            np.array_equal(self.tr_occlusion_offset, sel.tr_occlusion_offset) and
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
            # self.pre_process_script # LIMITATION: Cannot compare for equality (in the context of serialization)
            iseq = True

        if not iseq:
            return iseq

        for i, self_patch in enumerate(self.nnpatches):
            self_patch = self.nnpatches[i]
            sel_patch = sel.nnpatches[i]
            iseq = iseq and (self_patch == sel_patch)
            if not iseq:
                break

        return iseq


class Select(Enum):
    """Select Enumeration describes the special constants for Selection structure.

    Attributes
    ----------
    Select.ALL : int
        -999
    """
    ALL = -999
    PERCENT_40 = -40
    PERCENT_60 = -60

    ##########################################################################
    # Public Interface
    ##########################################################################
    def int(self):
        """Evaluate the enumeration value to its representative integer."""
        return self.value
