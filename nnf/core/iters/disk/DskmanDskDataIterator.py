# -*- coding: utf-8 -*-
"""
.. module:: DskmanDskDataIterator
   :platform: Unix, Windows
   :synopsis: Represent DskmanDskDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from warnings import warn as warning
import numpy as np
import os

# Local Imports
from nnf.core.iters.DskmanDataIterator import DskmanDataIterator
from nnf.db.Format import Format

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
    def __init__(self, db_pp_params):
        """Construct a DskmanDskDataIterator instance.

        Parameters
        ----------
        db_pp_params : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.
        """
        super().__init__(db_pp_params)
        
        # Class count
        self.cls_n = 0 
    
        # Keyed by the cls_idx
        # value = [file_path_1, file_path_2, ...] <= list of file paths
        self.paths = {}
        
        # Keyed by the cls_idx
        # value = <int> denoting the images per class
        self.n_per_class = {}
        
        # INHERITED: Whether to read the data
        # self._read_data

        # Future use
        # Auto assigned class index to user assigned class name mapping.
        self.cls_idx_to_dir = {}

    def init_params(self, db_dir, save_dir):
        """Initialize parameters for :obj:`DskmanDskDataIterator` instance.

        .. warning:: DO NOT override init() method, since it is used to initialize the
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
            n_per_class = self.n_per_class.setdefault(cls_idx, 0)

            # Update paths
            for fname in files:
                fpath = os.path.join(root, fname)
                fpaths.append(fpath)
                n_per_class += 1
 
            # assert(cls_idx == self.cls_n)

            # Update n_per_class dictionary
            self.n_per_class[cls_idx] = n_per_class
            cls_idx += 1

            # Update class count       
            self.cls_n += 1

    ##########################################################################
    # Protected: DskmanDataIterator Overrides
    ##########################################################################
    def _release(self):
        """Release internal resources used by the iterator."""
        super()._release()
        del self.cls_n
        del self.paths
        del self.n_per_class
        del self.cls_idx_to_dir

    def get_im_ch_axis(self):
        """Image channel axis."""
        # Ref: _get_cimg_frecord_in_next(...) method
        return 2

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
        `array_like`
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
            cimg = load_img(impath, grayscale=False, target_size=None)
            cimg = img_to_array(cimg, data_format='channels_last')

        return cimg, [impath, np.uint8(0), np.uint16(cls_idx)]  # [fpath, fpos, cls_lbl]

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