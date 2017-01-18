"""NNDiskMan to represent NNDiskMan class."""
# -*- coding: utf-8 -*-
# Global Imports
from keras.preprocessing.image import array_to_img
import numpy as np
import os

# Local Imports
from nnf.core.iters.disk.DskmanDskDataIterator import DskmanDskDataIterator
from nnf.core.iters.memory.DskmanMemDataIterator import DskmanMemDataIterator
from nnf.db.Dataset import Dataset
from nnf.db.DbSlice import DbSlice

class NNDiskMan(object):
    """description of class"""

    # Internally used directory to save the processed data
    _SAVE_TO_DIR = "patches"

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self, sel, nndb=None, db_dir=None):
        
        self.sel = sel
        self._save_to_dir = os.path.join(db_dir, NNDiskMan._SAVE_TO_DIR)

        # Keyed by the patch_id, and [tr|val|te|etc] dataset key
        # value = (abs_file_patch, cls_lbl) <= file_info tuple
        self.dict_fregistry = {}

        # Keyed by [tr|val|te|etc] dataset key
        # value = <int> denoting the class count
        self.dict_nb_class = {}      

        # Instantiate an iterator object
        if (nndb is not None):
            self.data_generator = DskmanMemDataIterator(nndb)
        elif (db_dir is not None):
            self.data_generator = DskmanDskDataIterator(db_dir)
        else:
            raise Exception("NNDiskMan(): Unsupported Mode")

    def process(self, nnpatches):
        # Initialize class ranges and column ranges
        # DEPENDANCY -The order must be preserved as per the enum Dataset 
        # REF_ORDER: [TR=0, VAL=1, TE=2, TR_OUT=3, VAL_OUT=4], Refer Dataset enum
        cls_ranges = [self.sel.class_range, self.sel.val_class_range, self.sel.te_class_range]
        col_ranges = [self.sel.tr_col_indices, self.sel.val_col_indices, self.sel.te_col_indices]

        # Set the default range if not specified
        DbSlice._set_default_range(0, cls_ranges, col_ranges)
        self.data_generator.init(cls_ranges, col_ranges)

        # [PERF] Iterate through the choset subset of the disk|nndb database
        for cimg, cls_idx, col_idx, datasets in self.data_generator:             

            # cimg: color/greyscale image 
            # has_cls_changed: bool (Whether the class has changed or not)
            # cls_idx, col_idx: int 
            # range_indices: list - int (index of which the data belongs to [0->tr|1->val|2->te|etc]? data)
            #                        - following the same order defined above.
    
            # Process the patches against cimg
            for nnpatch in nnpatches:
                img = array_to_img(cimg, dim_ordering='default', scale=False)
                patch_id = nnpatch.id
                fname = '{cls_idx}_{col_idx}_{patch_id}.{format}'.format(cls_idx=cls_idx,
                                                                    col_idx=col_idx,
                                                                    patch_id=patch_id,
                                                                    format='jpg')
                fpath = os.path.join(self._save_to_dir, fname)
                fpath = fname # [DEBUG]: TODO: comment
                img.save(fpath)                

                for edataset, _ in datasets:
                    self._add_to_fregistry(patch_id, edataset, fpath, cls_idx)
                
            for edataset, is_new_class in datasets:
                if (is_new_class):
                    self._increment_nb_class(edataset)

        # TODO: PERF: Serialize the Diskman object itself and write it in the same folder
        # Avoid writing the same datasets again and again
        pass

    def get_file_infos(self, patch_id, ekey):
        """Fetch file info tuples by patch id and [tr|val|te|etc] dataset key"""
        value = self.dict_fregistry.setdefault(patch_id, {})
        file_infos = value.setdefault(ekey, [])
        return file_infos

    def get_nb_class(self, ekey):
        """Fetch class count by [tr|val|te|etc] dataset key""" 
        nb_class = self.dict_nb_class.setdefault(ekey, 0)       
        return nb_class

    #################################################################
    # Private Interface
    #################################################################
    def _increment_nb_class(self, ekey):
        """Add a file info tuple to file registry by patch id and [tr|val|te|etc] dataset key"""
        value = self.dict_nb_class.setdefault(ekey, 0)
        self.dict_nb_class[ekey] = value + 1

    def _add_to_fregistry(self, patch_id, ekey, fpath, cls_lbl):
        """Add a file info tuple to file registry by patch id and [tr|val|te|etc] dataset key"""
        value = self.dict_fregistry.setdefault(patch_id, {})
        value = value.setdefault(ekey, [])
        value.append((fpath, cls_lbl))