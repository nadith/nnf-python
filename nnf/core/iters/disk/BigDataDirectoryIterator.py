# -*- coding: utf-8 -*-
"""
.. module:: BigDataDirectoryIterator
   :platform: Unix, Windows
   :synopsis: Represent BigDataDirectoryIterator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np

# Local Imports
from nnf.core.iters.disk.DirectoryIterator import DirectoryIterator

class BigDataDirectoryIterator(DirectoryIterator):
    """BigDataDirectoryIterator iterates the raw data in the disk files for :obj:`NNModel'.

    Attributes
    ----------
    binary_data : bool
        Whteher the files contain binary data.

    opened_files : :obj:`dict`
        Track open files for perf.  
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, frecords, nb_class, image_data_pp, params):
        """Construct a :obj:`BigDataDirectoryIterator` instance.

        Parameters
        ----------
        frecords : :obj:`list`
            List of file records. frecord = [fpath, fpos, cls_lbl] 

        nb_class : int
            Number of classes.

        image_data_pp : :obj:`ImageDataPreProcessor`
            Image data pre-processor.

        params : :obj:`dict`
            Core iterator parameters. 
        """
        super().__init__(frecords, nb_class, image_data_pp, params)

        if (params is None):
            self.binary_data = False
        
        else:
            self.binary_data = params['binary_data'] if ('binary_data' in params) else False

        # PERF (To save open file handlers)
        self.opened_files = {}

    def open_file(self, fpath):
        """Open and track the file

        Parameters
        ----------
        fpath : str
            File path.
        """
        return self.opened_files.setdefault(fpath, open(fpath, "r+"))

    def close_all_opened_files(self):
        """Close all tracked files"""
        for _, fhandle in self.opened_files.items():
            fhandle.close()  
      
    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _release(self):
        """Release internal resources used by the iterator."""
        super()._release()
        self.close_all_opened_files()

    def _get_data(self, frecord):
        """Load image from disk, pre-process and return.

        Parameters
        ----------
        frecords : :obj:`list`
            List of file records. frecord = [fpath, fpos, cls_lbl] 
        """
        fpath = frecord[0]
        fpos = frecord[1]
        cls_lbl = frecord[2]

        f = self.open_file(fpath)
        f.seek(fpos)

        if (self.binary_data):
            assert(len(self.target_size) == 1)  # Current support: Target should be 1-dimentional
            x = np.fromfile(f, dtype='float32', count=self.target_size[0])
            if self.data_format == 'channels_last':
                x = np.expand_dims(x, axis=1)
            else:
                x = np.expand_dims(x, axis=0)

        else:
            x = np.array(f.readline().split(), dtype='float32')
            if self.data_format == 'channels_last':
                x = np.expand_dims(x, axis=1)
            else:
                x = np.expand_dims(x, axis=0)

        # TODO: Apply necessary transofmraiton
        #x = self.image_data_generator.random_transform(x)
        x = self.image_data_generator.standardize(x)
        return x, cls_lbl