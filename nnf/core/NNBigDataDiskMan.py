# -*- coding: utf-8 -*-
"""
.. module:: NNBigDataDiskMan
   :platform: Unix, Windows
   :synopsis: Represent NNBigDataDiskMan class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from keras.preprocessing.image import array_to_img
import numpy as np
import pickle
import pprint
import os

# Local Imports
from nnf.core.iters.disk.DskmanDskBigDataIterator import DskmanDskBigDataIterator
from nnf.core.iters.memory.DskmanMemDataIterator import DskmanMemDataIterator
from nnf.db.Dataset import Dataset
from nnf.db.DbSlice import DbSlice
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

    binary_data : bool, optional
        Whether binary data or ASCII data. (Default value = False)

    target_size : :obj:`tuple`, optional
        Data sample size. Used to load data from binary files. 
        (Default value = None)
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, sel, dskman_param, nndb=None, save_dir=None):
        """Constructs :obj:`NNDiskMan` instance. 

        Must call init_ex() to initialize the instance.

        Parameters
        ----------
        sel : :obj:`Selection`
            Information to split the dataset.

        dskman_param : :obj:`dict`, optional
            Iterator parameters and Pre-processing parameters (keras) for iterators.
            Iterator parameters are used in :obj:`NNBigDataDiskMan`
            to handle binary files. (Default value = None).
            See also :obj:`ImageDataPreProcessor`.

        nndb : :obj:`NNdb`, optional
            Database to be processed against `sel` and `_dskman_param['pp']`.
            Either `nndb` or `dskman_param['db_dir']` must be provided. (Default value = None).

        save_dir : str, optional
            Path to save the processed data. (Default value = None).
        """
        super().__init__(sel, dskman_param, nndb, save_dir)
        self._save_file = None   
        self.save_file_path = None

        # INHERITED: Iterator parameters
        self.binary_data = self._dskman_param['binary_data'] if ('binary_data' in self._dskman_param) else False
        self.target_size = self._dskman_param['target_size'] if ('target_size' in self._dskman_param) else None

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
        return DskmanDskBigDataIterator(pp_params, self.binary_data, self.target_size)

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
        
        # TODO: extract the patch from raw_data
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

        if (not self.binary_data):
            # Write to the file in ASCII format 
            pdata.tofile(self._save_file, sep=" ", format="%s")
            self._save_file.write(str.encode('\n'))

        else:
            # Write to the file in binary format
            pdata.tofile(self._save_file)

        return self.save_file_path, fpos

    def __getstate__(self):
        """Serialization call back with Pickle. 

        Used to Remove the following fields from serialization.
        """
        odict = super().__getstate__()
        del odict['_save_file']  # Opened files cannot be serialized.
        del odict['save_file_path']
        del odict['binary_data']
        del odict['target_size']
        return odict