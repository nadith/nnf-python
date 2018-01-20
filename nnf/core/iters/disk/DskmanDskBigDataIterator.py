# -*- coding: utf-8 -*-
"""
.. module:: DskmanDskBigDataIterator
   :platform: Unix, Windows
   :synopsis: Represent DskmanDskBigDataIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
import os
import h5py

# Local Imports
from nnf.core.iters.disk.DskmanDskDataIterator import DskmanDskDataIterator
from nnf.core.FileFormat import FileFormat

class DskmanDskBigDataIterator(DskmanDskDataIterator):
    """DskmanDskBigDataIterator iterates the data files in the disk for :obj:`NNDiskMan'.

        This class is capable of reading data (binary/ASCII) from the disk.
        Many data items are assumed to be stored in each file. No 
        subdirectories are allowed.

    Attributes
    ----------
    fpositions : :obj:`dict`
        Track the file position for each record in the class.

    opened_files : :obj:`dict`
        Track open files for perf.

    file_format : bool
        Whether matlab/binary or ASCII data.

    data_field : bool
        Name of the data field in the file to read data. (used for files with data fields, like matlab).

    target_size : :obj:`tuple`
        Data sample size. Used to load data from binary files.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, pp_params, file_format, data_field, target_size):
        """Construct a :obj:`DskmanDskBigDataIterator` instance.

        Parameters
        ----------
        pp_params : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.

        file_format : bool
            Whether matlab/binary or ASCII data.

        data_field : bool
            Name of the data field in the file to read data. (used for files with data fields, like matlab).

        target_size : :obj:`tuple`, optional
            Data sample size. Used to load data from binary files.
        """
        super().__init__(pp_params)
        
        # INHERITED: [Parent's Attirbutes] ------------------------
        # Class count, will be updated below
        # self.cls_n = 0
        
        # Keyed by the cls_idx
        # value = [file_path_1, file_path_2, ...] <= list of file paths
        # self.paths = {}

        # Indexed by cls_idx
        # value = <int> denoting the images per class
        # self.n_per_class = np.array([], dtype=np.uint16)

        # Whether to read the data
        # self._read_data
        #------------------------------------------------------------

        # Keyed: by the cls_idx
        # value = <file positions for records in class>
        self.fpositions = {}

        # PERF (To save open file handlers)
        self.opened_files = {}

        # Whether the file is in binary format
        self.file_format = file_format

        # Fetch data field name (used for files with data fields, like matlab)
        self.data_field = data_field

        # Data sample size. Used to load data from binary files
        self.target_size = target_size
      
    def init_params(self, db_dir, save_dir):
        """Initialize parameters for :obj:`DskmanDskDataIterator` instance.

        .. warning:: DO NOT override init_ex() method, since it is used to initialize the
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

        # For matlab files
        fpos_offset = 0

        # Iterate the directory and populate self.paths dictionary
        for root, dirs, files in _recursive_list(db_dir):

            # Process the files only in 'db_dir' directory
            if (root != db_dir):
                continue

            # Read each file in files
            for fname in files:
                fpath = os.path.join(root, fname)

                if self.file_format == FileFormat.MATLAB:

                    with h5py.File(fpath, 'r') as file:
                        cls_lbl = np.squeeze(np.array(file['ulbl']))

                    # Starting index of each unique cls label
                    _, iusd, counts = np.unique(cls_lbl, return_index=True, return_counts=True)
                    self.n_per_class = np.append(self.n_per_class, np.uint16(counts))

                    if (not np.array_equal(np.sort(iusd), iusd)):
                        raise Exception('Data belong to same class need to be placed in consecutive blocks. '
                                        'Hence the class labels should be in sorted order.')

                    # Fix the end index
                    iusd = np.append(iusd, np.size(cls_lbl))

                    prev_fpos = iusd[0]
                    for fpos in iusd[1:]:
                        self.paths.setdefault(cls_idx, fpath)
                        fpositions = self.fpositions.setdefault(cls_idx, [])

                        # file position (row index for data accessing)
                        fpos_list = list(range(prev_fpos, fpos))
                        fpositions.extend(np.uint16(fpos_list))

                        cls_idx = cls_idx + 1
                        prev_fpos = fpos

                    continue

                elif self.file_format == FileFormat.BINARY:

                    # Open and read the content of the file 'f'
                    f = self._open_file(fpath, 'rb')
                    while (True):
                        fpos = f.tell()
                        data = np.fromfile(f, dtype='float32', count=self.target_size[0])

                        # EOF check
                        if (data.size == 0):
                            break

                        else:
                            # Assumption: each record belong to a separate class
                            self.paths.setdefault(cls_idx, fpath)
                            fpositions = self.fpositions.setdefault(cls_idx, [])
                            fpositions.append(np.uint16(fpos))
                            cls_idx = cls_idx + 1

                            # Assumption: each sample belongs to unique class
                            self.n_per_class = np.append(self.n_per_class, np.uint16(1))

                elif self.file_format == FileFormat.ASCII:

                    # Open and read the content of the file 'f'
                    f = self._open_file(fpath)
                    while (True):
                        fpos = f.tell()    

                        # EOF check
                        line = f.readline()
                        if line == '':
                            # either end of file or just a blank line.....
                            # we'll assume EOF, because we don't have a choice with the while loop!
                            break
                        else:
                            # Assumption: each record belong to a separate class
                            self.paths.setdefault(cls_idx, fpath)
                            fpositions = self.fpositions.setdefault(cls_idx, [])
                            fpositions.append(np.uint16(fpos))
                            cls_idx = cls_idx + 1

                            # Assumption: each sample belongs to unique class
                            self.n_per_class = np.append(self.n_per_class, np.uint16(1))

                else:
                    raise Exception("Unsupported file format")

        self.cls_n = cls_idx

    def get_im_ch_axis(self):
        """Image channel axis.

            Since feature is 1 dimentional im_ch_axis=0
            TODO: Revisit to see why im_ch_axis needs to be 0 for 1-dim data.
                Check hist_eq functionality. What if `None` is used instead of 0 ?
        """
        return 0

    def release(self):
        """Release internal resources used by the iterator."""
        super().release()
        self._close_all_opened_files()

    ##########################################################################
    # Protected: DskmanDskDataIterator Overrides
    ##########################################################################
    def _open_file(self, fpath, mode='r+'):
        """Open and track the opened file.

        Parameters
        ----------
        mode : str, optional
            File open mode. (Default value = 'r+').
        """
        return self.opened_files.setdefault(fpath, open(fpath, mode))

    def _close_all_opened_files(self):
        """Close all tracked files"""
        for _, fhandle in self.opened_files.items():
            fhandle.close()

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
            Raw data item.

        :obj:`list`
            file record. [file_path, file_position, class_label]
        """
        assert(cls_idx < self.cls_n and col_idx < self.n_per_class[cls_idx])

        fpath = self.paths[cls_idx]
        fpos = self.fpositions[cls_idx][col_idx]

        # Read the actual data only if necessary
        data = None
        if (self._read_data):

            if self.file_format == FileFormat.MATLAB:
                # TODO: Saving processed matlab data is currently not supported.
                raise Exception('Saving processed matlab data is currently not supported.')
                # with h5py.File(fpath, 'r') as file:
                #     input = np.array(file[self.data_field])
                #     data = input[fpos, :]

            else:
                # For other file formats
                # Open the file and seek to fpos
                f = self._open_file(fpath)
                f.seek(fpos, 0)

                # Read the content
                if self.file_format == FileFormat.BINARY:
                    data = np.fromfile(f, dtype='float32', count=self.target_size[0])

                elif self.file_format == FileFormat.ASCII:
                    data = f.readline().strip()

                else:
                    raise Exception("Unsupported file format")

        return data, [fpath, np.uint16(fpos), np.uint16(cls_idx)]  # [fpath, fpos, cls_lbl]