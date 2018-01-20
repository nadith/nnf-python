# -*- coding: utf-8 -*-
"""
.. module:: NNBigDataDiskMan
   :platform: Unix, Windows
   :synopsis: Represent NNBigDataDiskMan class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import os

# Local Imports
from nnf.core.iters.disk.DskmanDskBigDataIterator import DskmanDskBigDataIterator
from nnf.core.FileFormat import FileFormat
from nnf.core.NNDiskMan import NNDiskMan


class NNBigDataDiskMan(NNDiskMan):
    """Manage big-data like database in disk.

        This class is capable of writing data (binary/ASCII) onto the disk.

    Attributes
    ----------
    _save_file : file_descriptor
        File descriptor for save file.

    save_file_path : str
        File path for saved file.

    file_format : bool
        Whether matlab/binary or ASCII data.

    data_field : bool, optional
        Name of the data field in the file to read data. (used for files with data fields, like matlab).

    target_size : :obj:`tuple`, optional
        Data sample size. Used to load data from binary files. 
        (Default value = None)
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, sel, dskdb_param, memdb_param=None, nndb=None, save_dir=None):
        """Constructs :obj:`NNDiskMan` instance. 

        Must call init() to initialize the instance.

        Parameters
        ----------
        sel : :obj:`Selection`
            Information to split the dataset.

        dskdb_param : :obj:`dict`
            Disk database related parameters.

        memdb_param : :obj:`dict`, optional
            Memory database related parameters. (Default value = None).

        nndb : :obj:`NNdb`, optional
            Database to be processed against `sel` and `dskdb_param['pp']` or `memdb_param['pp']`.
            Either `nndb` or `dskman_param['db_dir']` must be provided. (Default value = None).

        save_dir : str, optional
            Path to save the processed data. (Default value = None).
        """
        super().__init__(sel, dskdb_param, memdb_param, nndb, save_dir)
        self._save_file = None   
        self.save_file_path = None

        # INHERITED: Iterator parameters
        self.file_format = self._dskdb_param['file_format'] if ('file_format' in self._dskdb_param) else FileFormat.ASCII
        self.data_field = self._dskdb_param['data_field'] if ('data_field' in self._dskdb_param) else None
        self.target_size = self._dskdb_param['target_size'] if ('target_size' in self._dskdb_param) else None

        # Error handling
        if self.file_format == FileFormat.BINARY:
            raise Exception("dskdb_param['target_size'] is mandatory for processing binary files.")

    ##########################################################################
    # Public: NNDiskMan Overrides
    ##########################################################################
    def init(self):
        """Initialize the :obj:`NNBigDataDiskMan` instance."""
        super().init()

        # Close the file if opened for writing
        if (self._save_file is not None):
            self._save_file.close()

    ##########################################################################
    # Protected: NNDiskMan Overrides
    ##########################################################################
    def _create_dskman_dskdataiter(self, pp_params):
        """Create the :obj:`DskmanDskBigDataIterator` instance to iterate the disk.

        Parameters
        ----------
        pp_params : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.

        Returns
        -------
        :obj:`DskmanDskBigDataIterator`
            Diskman data iterator for disk database. 
        """
        return DskmanDskBigDataIterator(pp_params, self.file_format, self.data_field, self.target_size)

    def _extract_patch(self, raw_data, nnpatch):
        """Extract the image patch from the nnpatch.

        Parameters
        ----------
        raw_data : ndarray
            raw data.
 
        nnpatch : :obj:`NNPatch`
            Information about the raw data patch. (dimension and offset).
        """
        if (nnpatch.is_holistic):
            return raw_data
        
        # BUG: extract the patch from raw_data
        pdata = raw_data
        return pdata

    def _save_data(self, pdata, patch_id, cls_idx, col_idx, 
                    data_format=None, scale=False):
        """Save data to the disk.

        Parameters
        ----------
        pdata : ndarray or str
            Numpy array if the pdata is from in memory database or 
            binary file. string line otherwise.

        patch_id : str
            :obj:`NNPatch` identification string. See also `NNPatch.id`.

        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        col_idx : int
            Column index. Belongs to `union_col_range`.

        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

        scale : bool
            Whether to scale the data range to 0-255.

        Returns
        -------
        str :
            File save path.

        int :
            File position where the data is written.
        """
        if (self._save_file is None):
            self.save_file_path = os.path.join(self.save_dir_abspath, 'processed_data.dat')
            self._save_file = open(self.save_file_path, 'wb')

        # Fetch the file position
        fpos = self._save_file.tell()

        if self.file_format == FileFormat.MATLAB:
            raise Exception('Saving the file in matlab file format is not supported')

        elif self.file_format == FileFormat.BINARY:
            # Write to the file in binary format
            pdata.tofile(self._save_file)

        elif self.file_format == FileFormat.ASCII:
            # Write to the file in ASCII format 
            pdata.tofile(self._save_file, sep=" ", format="%s")
            self._save_file.write(str.encode('\n'))

        else:
            raise Exception("Unsupported file format")

        return self.save_file_path, fpos

    def __getstate__(self):
        """Serialization call back with Pickle. 

        Used to Remove the following fields from serialization.
        """
        odict = super().__getstate__()
        del odict['_save_file']  # Opened files cannot be serialized.
        del odict['save_file_path']
        del odict['file_format']
        del odict['data_field']
        del odict['target_size']
        return odict