# -*- coding: utf-8 -*-
"""
.. module:: DskmanDskDataIterator
   :platform: Unix, Windows
   :synopsis: Represent DskmanDskDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import os
import numpy as np
from warnings import warn as warning
from keras.preprocessing.image import img_to_array


# Local Imports
from nnf.utl.internals import load_img
from nnf.core.iters.DskmanDataIterator import DskmanDataIterator


class DskmanDskDataIterator(DskmanDataIterator):
    """DskmanDataIterator iterates the data in the disk for :obj:`NNDiskMan'.

        Each data item is assumed to be stored in each file. Subdirectories denote
        the classes.

    Attributes
    ----------
    cls_n : int
        Class count.

    paths : :obj:`dict`
        Track the files for each class.

    n_per_class : :obj:`dict`
        Track images per class for each class.

    cls_idx_to_dir : :obj:`dict`
        Auto assigned class index to user assigned class name mapping.
    """

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self, pp_params):
        """Construct a DskmanDskDataIterator instance.

        Parameters
        ----------
        pp_params : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.
        """
        super().__init__(pp_params)
        
        # Class count
        self.cls_n = 0 
    
        # Keyed by the cls_idx
        # value = [file_path_1, file_path_2, ...] <= list of file paths
        self.paths = {}
        
        # Indexed by cls_idx
        # value = <int> denoting the images per class
        self.n_per_class = np.array([], dtype=np.uint16)
        
        # INHERITED: Whether to read the data
        # self._read_data

        # FUTURE_USE:
        # Auto assigned class index to user assigned class name mapping.
        self.cls_idx_to_dir = {}

    def init_params(self, db_dir, save_dir):
        """Initialize parameters for :obj:`DskmanDskDataIterator` instance.

        .. warning:: DO NOT override init_ex() method, since it is used to initialize the
        :obj:`DskmanDataIterator` with essential information.
    
            Each data item is assumed to be stored in each file. Subdirectories denote
            the classes.

        Parameters
        ----------
        db_dir : str
            Path to the database directory.

        save_dir : str
            Path to directory of processed data.
        """
        # Inner function: Fetch the files in the disk
        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

        # Assign explicit class index for internal reference
        cls_idx = 0

        # Iterate the directory and populate self.paths dictionary
        for root, dirs, files in _recursive_list(db_dir):

            # Exclude this directory itself
            if (root == db_dir):
                continue
            
            # Extract the directory
            dir = root[(root.rindex ('\\')+1):]
        
            # Exclude the internally used data folder
            if (dir == save_dir):
                continue

            # Since dir is considered to be a class name, give the explicit internal index
            self.cls_idx_to_dir.setdefault(cls_idx, dir)  # Future use

            # Initialize [paths|n_per_class] dictionaries with related cls_idx
            fpaths = self.paths.setdefault(cls_idx, [])            
            n_per_class = 0

            # Update paths
            for fname in files:
                fpath = os.path.join(root, fname)
                fpaths.append(fpath)
                n_per_class += 1

            # Update n_per_class list
            self.n_per_class = np.append(self.n_per_class, n_per_class)

            # Increment the class index
            cls_idx = cls_idx + 1

        # Update class count
        self.cls_n  = cls_idx

    def get_im_ch_axis(self):
        """Image channel axis."""
        # Ref: _get_cimg_frecord_in_next(...) method
        return 2

    def release(self):
        """Release internal resources used by the iterator."""
        super().release()
        del self.cls_n
        del self.paths
        del self.n_per_class
        del self.cls_idx_to_dir

    ##########################################################################
    # Protected: DskmanDataIterator Overrides
    ##########################################################################
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
        if (cls_idx >= self.cls_n):
            raise Exception('Class:'+ str(cls_idx) + ' is missing in the database.')

        if (col_idx >= self.n_per_class[cls_idx]):
            raise Exception('Class:'+ str(cls_idx) + ' ImageIdx:' + str(col_idx) + ' is missing in the database.')

        # Fetch the image path to read from the disk
        impath = self.paths[cls_idx][col_idx]

        # Read the actual data only if necessary
        cimg = None
        if (self._read_data):
            cimg = load_img(impath, use_rgb=None, target_size=None)
            cimg = img_to_array(cimg, data_format='channels_last')

        [fpath_to_class, filename] = os.path.split(impath)
        return cimg, [fpath_to_class, filename, np.uint16(cls_idx)]  # [fpath_to_class, fpos/filename, cls_lbl]

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
        if (cls_idx >= self.cls_n and show_warning):
            warning('Class:'+ str(cls_idx) + ' is missing in the database')

        return cls_idx < self.cls_n        

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
        assert(cls_idx < self.cls_n)

        if (col_idx >= self.n_per_class[cls_idx] and show_warning):
            warning('Class:'+ str(cls_idx) + ' ImageIdx:' + str(col_idx) + ' is missing in the database')

        return col_idx < self.n_per_class[cls_idx]