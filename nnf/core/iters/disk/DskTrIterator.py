"""DskTrIterator to represent DskTrIterator class."""
# -*- coding: utf-8 -*-
# Global Imports

# Local Imports
from nnf.core.iters.disk.DskDataIterator import DskDataIterator
from nnf.db.Dataset import Dataset

class DskTrIterator(DskDataIterator):
    """description of class"""

    def __init__(self, nnpatch, diskman):        
        super().__init__(
                diskman.get_file_infos(nnpatch.id, Dataset.TR),
                diskman.get_nb_class(Dataset.TR))