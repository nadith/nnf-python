# -*- coding: utf-8 -*-
"""
.. module:: DskmanDskBigDataIterator
   :platform: Unix, Windows
   :synopsis: Represent DskmanDskBigDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from warnings import warn as warning
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os

# Local Imports
from nnf.core.iters.disk.DskmanDskDataIterator import DskmanDskDataIterator
import nnf.core.NNDiskMan

class DskmanDskBigDataIterator(DskmanDskDataIterator):
    """DskmanDskBigDataIterator iterates the data files in the disk for :obj:`NNDiskMan'.

        This class is capable of reading data (binary/ASCII) onto the disk. 
        Many data items are assumed to be stored in each file. No 
        subdirectories are allowed.

    Attributes
    ----------
    cls_n : int
        Class count.

    n_per_class : int
        Images per class.

    opened_files : :obj:`dict`
        Track open files for perf.

    binary_data : bool
        Whether binary data or ASCII data.

    target_size : :obj:`tuple`
        Data sample size. Used to load data from binary files.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, db_pp_params, binary_data, target_size):
        """Construct a :obj:`DskmanDskBigDataIterator` instance.

        Parameters
        ----------
        db_pp_params : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.

        binary_data : bool
            Whether binary data or ASCII data.

        target_size : :obj:`tuple`, optional
            Data sample size. Used to load data from binary files.
        """
        super().__init__(db_pp_params)
        
        # INHERITED: [Parent's Attirbutes] ------------------------
        # Class count, will be updated below
        self.cls_n = 0 
        
        # Keyed: by the cls_idx
        # Value: frecord [fpath, fpos]
        # self.paths = {} 
        
        # Parent's 'n_per_class' is a dictionary.
        self.n_per_class = 0

        # Whether to read the data
        # self._read_data
        #------------------------------------------------------------

        # PERF (To save open file handlers)
        self.opened_files = {}

        # Whether the file is in binary format
        self.binary_data = binary_data

        # Data sample size. Used to load data from binary files
        self.target_size = target_size
      
    def init_params(self, db_dir, save_dir):
        """Initialize parameters for :obj:`DskmanDskDataIterator` instance.

        .. warning:: DO NOT override init() method, since it is used to initialize the
        :obj:`DskmanDataIterator` with essential information.
    
            Many data items are assumed to be stored in each file. No subdirectories
            are allowed.

        Parameters
        ----------
        db_dir : str
            Path to the database directory.

        save_dir : str
            [UNUSED] Path to directory of processed data.
        """
        # Inner function: Fetch the files in the disk
        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

        # Assign explicit class index for internal reference
        cls_idx = 0

        # Iterate the directory and populate self.paths dictionary
        for root, dirs, files in _recursive_list(db_dir):

            # Process the files only in 'db_dir' directory
            if (root != db_dir):
                continue

            # Read each file in files
            for fname in files:
                fpath = os.path.join(root, fname)                
                                
                if (not self.binary_data):
                    # Open and read the content of the file 'f'
                    f = self.open_file(fpath)
                    while (True):
                        fpos = f.tell()    

                        # EOF check
                        line = f.readline()
                        if line == '':
                            # either end of file or just a blank line.....
                            # we'll assume EOF, because we don't have a choice with the while loop!
                            break
                        else:
                            # Add frecord
                            # Assumption: Each sample belongs to unique class
                            frecord = self.paths.setdefault(cls_idx, [])
                            cls_idx += 1
                            frecord.append(fpath)
                            frecord.append(fpos)

                else:
                    # Open and read the content of the file 'f'
                    f = self.open_file(fpath, 'rb')
                    while (True):
                        fpos = f.tell()  
                        data = np.fromfile(f, dtype='float32', count=self.target_size[0])

                        # EOF check
                        if (data.size == 0):
                            break
                        else:
                            frecord = self.paths.setdefault(cls_idx, [])
                            cls_idx += 1
                            frecord.append(fpath)
                            frecord.append(fpos)
 
        # Assumption: Each sample belongs to unique class
        self.n_per_class = 1
        self.cls_n = cls_idx

    def open_file(self, fpath, mode='r+'):
        """Open and track the opened file.

        Parameters
        ----------
        mode : str, optional
            File open mode. (Default value = 'r+').
        """
        return self.opened_files.setdefault(fpath, open(fpath, mode))

    def close_all_opened_files(self):
        """Close all tracked files"""
        for _, fhandle in self.opened_files.items():
            fhandle.close()        

    def get_im_ch_axis(self):
        """Image channel axis.

            Since feature is 1 dimentional im_ch_axis=0
            TODO: Revisit to see why im_ch_axis needs to be 0 for 1-dim data.
                Check hist_eq functionality. What if `None` is used instead of 0 ?
        """
        return 0
    ##########################################################################
    # Protected: DskmanDskDataIterator Overrides
    ##########################################################################
    def _release(self):
        """Release internal resources used by the iterator."""
        super()._release()
        self.close_all_opened_files()

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
            Raw data item.

        :obj:`list`
            file record. [file_path, file_position, class_label]
        """
        assert(cls_idx < self.cls_n and col_idx < self.n_per_class)
        fpath, fpos = self.paths[cls_idx]

        # Read the actual data only if necessary
        data = None
        if (self._read_data):
            f = self.open_file(fpath)
            f.seek(fpos, 0)

            if (not self.binary_data):
                data = f.readline().strip()
            else:
                data = np.fromfile(f, dtype='float32', count=self.target_size[0])

        return data, [fpath, np.float32(fpos), np.uint16(cls_idx)]  # [fpath, fpos, cls_lbl]

    def _is_valid_cls_idx(self, cls_idx):
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
        if (cls_idx >= self.cls_n):
            warning('Class:'+ str(cls_idx) + ' is missing in the database')

        return cls_idx < self.cls_n        

    def _is_valid_col_idx(self, cls_idx, col_idx):
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

        if (col_idx >= self.n_per_class):
            warning('Class:'+ str(cls_idx) + ' ImageIdx:' + str(col_idx) + ' is missing in the database')

        return col_idx < self.n_per_class  # Assumption: each sample belongs to unique class