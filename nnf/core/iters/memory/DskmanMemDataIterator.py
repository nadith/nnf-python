# -*- coding: utf-8 -*-
"""
.. module:: DskmanMemDataIterator
   :platform: Unix, Windows
   :synopsis: Represent DskmanMemDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
from warnings import warn as warning

# Local Imports
from nnf.core.iters.DskmanDataIterator import DskmanDataIterator


class DskmanMemDataIterator(DskmanDataIterator):
    """DskmanMemDataIterator represents the diskman iterator for in memory databases.

    Attributes
    ----------
    nndb : :obj:`NNdb`
        Database to iterate.

    _save_to_dir : str
        Path to directory of processed data.
    """

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self, pp_params):
        """Construct a DskmanMemDataIterator instance.

        Parameters
        ----------
        pp_params : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.
        """
        super().__init__(pp_params)
        self.nndb = None
        self._save_to_dir = None

        # INHERITED: Whether to read the data
        # self._read_data

    def init_params(self, nndb, save_dir=None):
        """Initialize parameters for :obj:`DskmanMemDataIterator` instance.

        Parameters
        ----------
        nndb : :obj:`NNdb`
            Database to iterate.

        save_dir : str, optional
            Path to directory of processed data. (Default value = None)
        """
        self.nndb = nndb
        self._save_to_dir = save_dir

    def clone(self):
        """Create a copy of this DskmanMemDataIterator object."""
        assert(False) # Currently not implemented

    def get_im_ch_axis(self):
        """Image channel axis."""
        return self.nndb.im_ch_axis

    def release(self):
        """Release internal resources used by the iterator."""
        super().release()
        del self.nndb
        del self._save_to_dir

    def get_cimg_frecord(self, cls_idx, col_idx):
        """Get image and file record (frecord) at cls_idx, col_idx.

        Parameters
        ----------
        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        col_idx : int
            Column index. Belongs to `union_col_range`.

        Returns
        -------
        ndarray
            Color image.

        :obj:`list`
            file record. [file_path, file_position, class_label]
        """
        return self._get_cimg_frecord_in_next(cls_idx, col_idx)

    #################################################################
    # Protected Interface
    #################################################################
    def _get_cimg_frecord_in_next(self, cls_idx, col_idx):
        """Get image and file record (frecord) at cls_idx, col_idx.

        Parameters
        ----------
        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        col_idx : int
            Column index. Belongs to `union_col_range`.

        Returns
        -------
        ndarray
            Color image.

        :obj:`list`
            file record. [file_path, file_position, class_label]
        """
        if (cls_idx >= self.nndb.cls_n):
            raise Exception('Class:'+ str(cls_idx) + ' is missing in the database.')

        if (col_idx >= self.nndb.n_per_class[cls_idx]):
            raise Exception('Class:'+ str(cls_idx) + ' ImageIdx:' + str(col_idx) + ' is missing in the database.')

        # Calculate the image index for the database
        im_idx = self.nndb.cls_st[cls_idx] + col_idx

        cimg = None
        if (self._read_data):
            cimg = self.nndb.get_data_at(im_idx)

        filename = None
        if (self._save_to_dir is not None):
            filename = "image_" + str(im_idx) + ".jpg"

        return cimg, [self._save_to_dir, filename, np.uint16(cls_idx)]  # [fpath_to_base, fpos/filename, cls_lbl]

    def _is_valid_cls_idx(self, cls_idx, show_warning=True):
        """Check the validity of class index.

        Parameters
        ----------
        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        Returns
        -------
        bool
            True if valid. False otherwise.
        """
        if (cls_idx >= self.nndb.cls_n and show_warning):
            warning('Class:'+ str(cls_idx) + ' is missing in the database')

        return cls_idx < self.nndb.cls_n        

    def _is_valid_col_idx(self, cls_idx, col_idx, show_warning=True):
        """Check the validity of column index of the class denoted by cls_idx.

        Parameters
        ----------
        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        col_idx : int
            Column index. Belongs to `union_col_range`.

        Returns
        -------
        bool
            True if valid. False otherwise.
        """
        assert(cls_idx < self.nndb.cls_n)

        if (col_idx >= self.nndb.n_per_class[cls_idx] and show_warning):
            warning('Class:'+ str(cls_idx) + ' ImageIdx:' + str(col_idx) + ' is missing in the database')

        return col_idx < self.nndb.n_per_class[cls_idx]

    def _get_n_per_class(self, cls_idx):
        """Get no of images per class.

        Parameters
        ----------
        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        Returns
        -------
        int
            no of samples per class.
        """
        assert(cls_idx < self.nndb.cls_n)
        return self.nndb.n_per_class[cls_idx]