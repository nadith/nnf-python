# -*- coding: utf-8 -*-
"""
.. module:: BigDataDirectoryIterator
   :platform: Unix, Windows
   :synopsis: Represent BigDataDirectoryIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import h5py
import numpy as np
from keras import backend as K
from scipy.misc import imresize

# Local Imports
from nnf.core.FileFormat import FileFormat
from nnf.core.iters.disk.DirectoryIterator import DirectoryIterator


class BigDataDirectoryIterator(DirectoryIterator):
    """BigDataDirectoryIterator iterates the raw data in the disk files for :obj:`NNModel'.

    Attributes
    ----------
    file_format : bool
        Whether matlab/binary or ASCII data.

    data_field : bool
        Name of the data field in the file to read data. (used for files with data fields, like matlab).

    opened_files : :obj:`dict`
        Track open files for perf.  
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, frecords, nb_class, imdata_pp, params):
        """Construct a :obj:`BigDataDirectoryIterator` instance.

        Parameters
        ----------
        frecords : :obj:`list`
            List of file records. frecord = [fpath, fpos, cls_lbl]

        nb_class : int
            Number of classes.

        imdata_pp : :obj:`ImageDataPreProcessor`
            Image data pre-processor.

        params : :obj:`dict`
            Core iterator parameters. 
        """
        # Fetch file format
        self.file_format = None
        if (params is not None):
            self.file_format = params['file_format'] if ('file_format' in params) else None

        # Set defaults
        if (self.file_format is None):
            self.file_format = FileFormat.ASCII

        # Fetch data field name (used for files with data fields, like matlab)
        self.data_field = params['data_field'] if ('data_field' in params) else None

        # Data field is mandatory for matlab files
        if self.file_format == FileFormat.MATLAB: assert self.data_field

        # PERF (To save open file handlers)
        self.opened_files = {}

        # Last step, since super class constructor will invoke `self._fn_preview_sample`
        super().__init__(frecords, nb_class, imdata_pp, params)

    def release(self):
        """Release internal resources used by the iterator."""
        super().release()
        self._close_all_opened_files()

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _open_file(self, fpath):
        """Open and track the file

        Parameters
        ----------
        fpath : str
            File path.
        """
        return self.opened_files.setdefault(fpath, open(fpath, "r+"))

    def _close_all_opened_files(self):
        """Close all tracked files"""
        for _, fhandle in self.opened_files.items():
            fhandle.close()

    def _fn_preview_sample(self, fpath, fpos):
        # For matlab files
        if self.file_format == FileFormat.MATLAB:
            if (self.data_field is None):
                raise Exception("param.data_field is not defined for FileFormat.MATLAB")

            with h5py.File(fpath, 'r') as file:
                input = np.array(file[self.data_field])  # Returns the transpose of the original data
                x = np.float32(input[fpos, :])

        else:
            # For other file formats
            f = self._open_file(fpath)
            f.seek(fpos)

            if self.file_format == FileFormat.BINARY:
                assert(self.target_size[1] == 1)  # Format H x W: but must be W == 1
                x = np.fromfile(f, dtype='float32', count=self.target_size[0])

            elif self.file_format == FileFormat.ASCII:
                x = np.array(f.readline().split(), dtype='float32')

            else:
                raise Exception("Unsupported file format")

            # Make it 2D, compatibility for 1D convo nets
            x = np.expand_dims(x, axis=1)

        # Add channel dimension
        if self.data_format == 'channels_last':
            x = np.expand_dims(x, axis=1)
        else:
            x = np.expand_dims(x, axis=0)
        return x

    def _fn_read_sample(self, fpath, fpos, reshape=True):
        x = self._fn_preview_sample(fpath, fpos)

        # If network input shape is not the actual image shape
        if (reshape and x.shape != self.input_shape):

            # Default behavior or invoke `fn_reshape_input`
            x = imresize(x, self.input_shape, 'bicubic', mode='F')\
                    if (self.fn_reshape_input is None) else \
                        self.fn_reshape_input(x, self.input_shape, self.data_format)
            x = x.astype(K.floatx())

        # TODO: Apply necessary transformations
        #x = self.imdata_pp.random_transform(x)
        x = self.imdata_pp.standardize(x)
        return x

    def _get_data(self, frecords, j):
        """Load image from disk, pre-process and return.

        Parameters
        ----------
        frecords : :obj:`list`
            List of file records. frecord = [fpath, fpos, cls_lbl] 
        """
        if (self.window_iter is None):
            frecord = frecords[j]
            fpos = frecord[1]
            cls_lbl = frecord[2]

            # PERF: Memory efficient implementation (only first frecord of the class has the path to the file)
            fpath = frecord[0] if isinstance(frecord[0], str) else frecords[frecord[0]][0]
            x = self.fn_read_sample(fpath, fpos)

        else:
            x, cls_lbl = next(self.window_iter)

        return x, cls_lbl