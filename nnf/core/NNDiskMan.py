# -*- coding: utf-8 -*-
"""
.. module:: NNDiskMan
   :platform: Unix, Windows
   :synopsis: Represent NNDiskMan class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from keras.preprocessing.image import array_to_img
import numpy as np
import pickle
import pprint
import os

# Local Imports
from nnf.core.iters.disk.DskmanDskDataIterator import DskmanDskDataIterator
from nnf.core.iters.memory.DskmanMemDataIterator import DskmanMemDataIterator
from nnf.db.Dataset import Dataset
from nnf.db.DbSlice import DbSlice
from nnf.pp.im_pre_process import im_pre_process

class NNDiskMan(object):
    """Manage database in disk.

    Attributes
    ----------
    sel : :obj:`Selection`
        Information to split the dataset.

    _db_pp_param :obj:`dict`
        Pre-processing parameters (keras) for iterators.
        See also :obj:`ImageDataPreProcessor`.

    _iter_param : :obj:`dict`, optional
        Iterator parameters. Currently used in :obj:`NNBigDataDiskMan`
        to handle binary files. (Default value = None).

    _nndb : :obj:`NNdb`, optional
        Database to be processed against `sel` and `db_pp_params`.
        Either `nndb` or `db_dr` must be provided. (Default value = None).

    _db_dir : str, optional
        Path to the disk database. (Default value = None).
        Either `nndb` or `db_dr` must be provided.

    _save_sub_dir : str, optional
        Path to save the processed data. (Default value = None).

    _save_to_dir : str
        Path to the folder that contains the `diskman.pkl` file.

    _dict_fregistry : :obj:`dict`
        Track the files for each patch and each dataset.

    _dict_cls_lbl : :obj:`dict`
        Counter for auto assigned class label for each patch of each dataset.

    _dict_nb_class : :obj:`dict`
        Track the class counts for each dataset.

    _save_images : bool
        Whether to save the images that are processed via nndiskman.

    _force_save_images : bool
        Whether to save the images forcefully that are processed via nndiskman.

    _update_dicts : bool
        Whether to update the dictionaries 
        (`_dict_fregistry`, `_dict_cls_lbl`, `_dict_nb_class`).

    dskman_dataiter : :obj:`DskmanDataIterator`
        Itearator for the database.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, sel, db_pp_param, nndb=None, db_dir=None, 
                                                save_dir=None, iter_param=None):
        """Constructs :obj:`NNDiskMan` instance.

        Must call init() to intialize the instance.

        Parameters
        ----------
        sel : :obj:`Selection`
            Information to split the dataset.

        db_pp_param : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.

        nndb : :obj:`NNdb`, optional
            Database to be processed against `sel` and `db_pp_params`.
            Either `nndb` or `db_dr` must be provided. (Default value = None).

        db_dir : str, optional
            Path to the disk database. (Default value = None).
            Either `nndb` or `db_dr` must be provided.

        save_dir : str, optional
            Path to save the processed data. (Default value = None).

        iter_param : :obj:`dict`, optional
            Iterator parameters. Currently used in :obj:`NNBigDataDiskMan`
            to handle binary files. (Default value = None).
        """
        self.sel = sel
        self._db_pp_param = db_pp_param
        self._iter_param = iter_param
        self._nndb = nndb
        self._db_dir = db_dir
        self._save_sub_dir = save_dir
        self._save_to_dir = os.path.join(db_dir, save_dir)

        # Create the _save_to_dir if does not exist
        if not os.path.exists(self._save_to_dir):
            os.makedirs(self._save_to_dir)

        # Track the files for each patch and each dataset
        # Keyed by the patch_id, and [Dataset.TR|Dataset.VAL|Dataset.TE|...]
        # value = [file_path, fpos, cls_lbl] <= frecord
        self._dict_fregistry = {}

        # Track the class counts for each dataset
        # Keyed by [Dataset.TR|Dataset.VAL|Dataset.TE|...]
        # value = <int> - class count
        self._dict_nb_class = {}

        # Counter for auto assigned class label for each patch of each dataset
        # Keyed by the patch_id, and [Dataset.TR|Dataset.VAL|Dataset.TE|...]
        # value = <int> - counter for the class labal
        self._dict_cls_lbl = {}

        # PERF: Whether to save the images that are processed via nndiskman
        self._save_images = True

        # PERF: Whether to save the images forcefully that are processed via nndiskman
        self._force_save_images = False

        # PERF: Whether to update the dictionaries 
        # (_dict_fregistry, _dict_nb_class, _dict_cls_lbl)
        self._update_dicts = True

        # Diskman data iterator
        self.dskman_dataiter = None

    def init(self):
        """Initialize the :obj:`NNDiskMan` instance."""

        # Init data iterator
        self.__init_dataiter()

        # PERF: Update the dictionaries if necessary
        if (not self._update_dicts):
            return

        # Initialize class ranges and column ranges
        # DEPENDANCY -The order must be preserved as per the enum Dataset 
        # REF_ORDER: [TR=0, VAL=1, TE=2, TR_OUT=3, VAL_OUT=4], Refer Dataset enum
        cls_ranges = [self.sel.class_range, self.sel.val_class_range, self.sel.te_class_range, 
                        self.sel.class_range, self.sel.val_class_range]
        col_ranges = [self.sel.tr_col_indices, self.sel.val_col_indices, self.sel.te_col_indices, 
                        self.sel.tr_out_col_indices, self.sel.val_out_col_indices]

        # Set the default range if not specified
        DbSlice._set_default_cls_range(0, cls_ranges, col_ranges)

        # Must have atleast 1 patch to represent the whole image
        nnpatch = self.sel.nnpatches[0]

        # PERF: Init diskman data iterator (could be in memory iter or disk iter)
        need_patch_processing = not ((len(self.sel.nnpatches) == 1) and nnpatch.is_holistic)

        self._save_images = self._force_save_images or\
                                (self._save_images and\
                                    (self.sel.need_processing()  or\
                                    need_patch_processing or\
                                    isinstance(self.dskman_dataiter, DskmanMemDataIterator)))
        self.dskman_dataiter.init(cls_ranges, col_ranges, self._save_images)

        # [PERF] Iterate through the choset subset of the disk|nndb database
        for cimg, frecord, cls_idx, col_idx, dataset_tup in self.dskman_dataiter:

            # cimg: array_like data (maybe an image or raw data item) 
            # frecord: [fpath, fpos, cls_lbl], assert(cls_lbl == cls_idx)
            # cls_idx, col_idx: int
            # dataset_tup: list of tuples
            #       [(Dataset.TR, True), (Dataset.TR, False), (Dataset.VAL, True), ...]
    
            # SPECIAL NOTE:  cls_lbl, cls_idx, col_idx are expressed with respect to the
            # cls_ranges, col_ranges defined above and may not consist of continuous
            # indices. i.e cls_lbl=[0 4 6 3]

            # For histogram equalizaion operation (cannonical image)
            cann_cimg = None                
            if (self.sel.histmatch_col_index is not None):
                cann_cimg, _= data_generator._get_cimg_frecord_in_next(cls_idx, self.sel.histmatch_col_index)   
                    
            # Peform image preocessing
            cimg = DbSlice.preprocess_im_with_sel(cimg, cann_cimg, self.sel, self.dskman_dataiter.get_im_ch_axis())

            # Update the class counts for each dataset  
            for edataset, is_new_class in dataset_tup:
                if (is_new_class):
                    self._increment_nb_class(edataset)

            # PERF: If there is only 1 patch and it is the whole image
            if (not need_patch_processing):
                pimg = self._extract_impatch(cimg, nnpatch)
                self._post_process_loop(frecord, pimg, self.sel.nnpatches[0].id, dataset_tup, cls_idx, col_idx)
            
            else:
                # Process the nnpatches against cimg
                for nnpatch in self.sel.nnpatches:
                    pimg = self._extract_impatch(cimg, nnpatch)
                    self._post_process_loop(frecord, pimg, nnpatch.id, dataset_tup, cls_idx, col_idx)    

        # Cleanup the iterator by release the internal resources
        self.dskman_dataiter._release()

        # PERF: Save this nndiskman object
        self.save(self._save_to_dir)

    def load_nndiskman(self, dest_dir):
        """Load diskman from the disk.

        Parameters
        ----------
        dest_dir : str
            Path to the folder that contains the `diskman.pkl` file.

        Returns
        -------
        :obj:`NNDiskMan`
            :obj:`NNDiskMan` if successful, None otherwise.
        """
        pkl_fpath = os.path.join(dest_dir, 'diskman.pkl')
        
        if (not os.path.isfile(pkl_fpath)):
            return None

        pkl_file = open(pkl_fpath, 'rb')
        nndskman = pickle.load(pkl_file)
        pkl_file.close()

        return nndskman

    def save(self, dest_dir):
        """Save this nndiskman to disk.
    
        Parameters
        ----------
        dest_dir : str
            Path to the folder that contains the `diskman.pkl` file.
        """
        pkl_fpath = os.path.join(dest_dir, 'diskman.pkl')
        pkl_file = open(pkl_fpath, 'wb')

        # Pickle the self object using the highest protocol available.
        pickle.dump(self, pkl_file, -1)

        pkl_file.close()

    def get_frecords(self, patch_id, ekey):
        """Get the file record (frecord) by <patch_id> and <dataset_key>.

        Parameters
        ----------
        patch_id : str
            :obj:`NNPatch` identification string. See also `NNPatch.id`

        ekey : `Dataset`
            Enumeration that describe the dataset. 
            i.e Dataset.TR, Dataset.VAL, Dataset.TE, ...

        Returns
        -------
        :obj:`list`
            frecord indicating [fpath, fpos, cls_lbl] if `ekey` is valid, [] otherwise.
        """
        value = self._dict_fregistry.setdefault(patch_id, {})
        frecords = value.setdefault(ekey, [])
        return frecords

    def get_nb_class(self, ekey):
        """Get class count by <dataset_key>.

        Parameters
        ----------
        ekey : `Dataset`
            Enumeration that describe the dataset. 
            i.e Dataset.TR, Dataset.VAL, Dataset.TE, ...

        Returns
        -------
        int
            class count if `ekey` is valid, 0 otherwise.
        """
        nb_class = self._dict_nb_class.setdefault(ekey, 0)       
        return nb_class

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _create_dskman_memdataiter(self, db_pp_param):
        """Create the :obj:`DskmanMemDataIterator` instance to iterate the disk.

        Parameters
        ----------
        db_pp_param : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.

        Returns
        -------
        :obj:`DskmanMemDataIterator`
            Diskman data iterator for in memory database.
        """
        return DskmanMemDataIterator(db_pp_param)

    def _create_dskman_dskdataiter(self, db_pp_param):
        """Create the :obj:`DskmanDskDataIterator` instance to iterate the disk.

        Parameters
        ----------
        db_pp_param : :obj:`dict`
            Pre-processing parameters for :obj:`ImageDataPreProcessor`.

        Returns
        -------
        :obj:`DskmanDskDataIterator`
            Diskman data iterator for disk database.
        """
        return DskmanDskDataIterator(db_pp_param)

    def _increment_nb_class(self, ekey):
        """Increment the class count of <dataset_key>.

        Parameters
        ----------
        ekey : `Dataset`
            Enumeration that describe the dataset. 
            i.e Dataset.TR, Dataset.VAL, Dataset.TE, ...
        """
        value = self._dict_nb_class.setdefault(ekey, 0)
        self._dict_nb_class[ekey] = value + 1

    def _add_to_fregistry(self, patch_id, ekey, frecord):
        """Add the file record (frecord) by <patch_id> and <dataset_key>.

        Parameters
        ----------
        patch_id : str
            :obj:`NNPatch` identification string. See also `NNPatch.id`

        ekey : `Dataset`
            Enumeration that describe the dataset. 
            i.e Dataset.TR, Dataset.VAL, Dataset.TE, ...

        frecord : :obj:`list`
            list of values [file_path, file_position, class_label]
        """
        value = self._dict_fregistry.setdefault(patch_id, {})
        frecords = value.setdefault(ekey, [])
        frecords.append(frecord)

    def _post_process_loop(self, frecord, pimg, patch_id, dataset_tup, cls_idx, col_idx):
        """Post processing for main image iterating loop.

        Parameters
        ----------
        frecord : :obj:`list`
            list of values [file_path, file_position, class_label]

        pimg : `array_like`
            Color image patch or raw data item.

        patch_id : str
            :obj:`NNPatch` identification string. See also `NNPatch.id`.

        dataset_tup : list of `tuple`
            Each `tuple` has <dataset_key> and an indication for a new class.
            i.e [(Dataset.TR, True), (Dataset.TR, False), ...]

        cls_idx : int
            Class index. Belongs to `union_cls_range`.

        col_idx : int
            Column index. Belongs to `union_col_range`.
        """
        # PERF: Save the images if needed
        if (self._save_images):
            fpath, fpos = self._save_data(pimg, patch_id, cls_idx, col_idx)
            frecord[0] = fpath
            frecord[1] = fpos

        # Update the fregistry and auto assign class label
        for edataset, is_new_class in dataset_tup:

            # Calculate the new class label (continuous)
            tmp_dataset = self._dict_cls_lbl.setdefault(patch_id, {})
            cls_lbl = tmp_dataset.setdefault(edataset, -1)
            if (is_new_class):
                cls_lbl += 1
                tmp_dataset[edataset] = cls_lbl

            # Update frecord for new class label
            frecord[2] = cls_lbl

            # Add to file record to fregistry
            self._add_to_fregistry(patch_id, edataset, frecord)

    def _extract_impatch(self, cimg, nnpatch):
        """Extract the image patch from the nnpatch.

        Parameters
        ----------
        cimg : `array_like`
            Color image.
 
        nnpatch : :obj:`NNPatch`
            Information about the image patch. (dimension and offset).
        """
        if (nnpatch.is_holistic):
            return cimg
        
        # TODO extract the patch from cimg to pimg
        pimg = cimg
        return pimg

    def _save_data(self, pimg, patch_id, cls_idx, col_idx, 
                                        data_format=None, scale=False):
        """Save data to the disk.

        Parameters
        ----------
        pimg : `array_like`
            Color image patch or raw data item.

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
        fname = '{cls_idx}_{col_idx}_{patch_id}.{format}'.format(
                                                    cls_idx=cls_idx,
                                                    col_idx=col_idx,
                                                    patch_id=patch_id,
                                                    format='jpg')
        cls_dir = os.path.join(self._save_to_dir, str(cls_idx))
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

        fpath = os.path.join(cls_dir, fname)
        img = array_to_img(pimg, data_format='channels_last', scale=scale)
        img.save(fpath)
        return fpath, np.uint8(0)

    ##########################################################################
    # Special Interface
    ##########################################################################
    def __eq__(self, nndiskman):
        """Equality of two :obj:`NNDiskMan` instances.

        Parameters
        ----------
        nndiskman : :obj:`NNDiskMan`
            The instance to be compared against this instance.

        Returns
        -------
        bool
            True if both instances are the same. False otherwise.

        Notes
        -----
        This is used to compare the important fields of `NNDiskman` 
        after deserializing the object from the disk.
        """
        return (self.sel == nndiskman.sel) and\
                (self._db_pp_param == nndiskman._db_pp_param) and\
                (self._iter_param == nndiskman._iter_param)

    def __getstate__(self):
        """Serialization call back with Pickle. 

        Used to Remove the following fields from serialization.
        """
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['dskman_dataiter']        
        del odict['_nndb']
        del odict['_db_dir']
        del odict['_save_sub_dir']
        del odict['_save_images']
        del odict['_force_save_images']
        del odict['_update_dicts']
        del odict['_dict_cls_lbl']
        return odict

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __init_dataiter(self):
        """Initialize the diskman data iterator only if needed.

            Load any saved diskman instance and create disman iterators 
            only if necessary.
        """
        create_data_generator = True

        # PERF: Try loading the saved nndiskman from parent directory first.
        cur_directory = os.path.dirname(self._save_to_dir)
        dskman = self.load_nndiskman(cur_directory)
        if (dskman is None):
            # PERF: Next try to load from the spicified directory.
            dskman = self.load_nndiskman(self._save_to_dir)

        if (dskman is not None and self == dskman):  # will invoke __eq__()
            
            # PERF: Disable saving of images
            self._save_images = False

            # Set the data from the loaded nndiskman
            self._dict_fregistry = dskman._dict_fregistry
            self._dict_nb_class = dskman._dict_nb_class

            # PERF: Stop updating dictionaries if paths are the same
            if (self._save_to_dir == dskman._save_to_dir or
                cur_directory == dskman._save_to_dir):
                self._update_dicts = False
                create_data_generator = False

        elif (dskman is not None):  # and (self != dskman)
            # PERF: Forcefully save the images
            self._force_save_images = True

        # Instantiate an disman data iterator object
        if (self._nndb is not None):
            if (create_data_generator):
                self.dskman_dataiter = self._create_dskman_memdataiter(self._db_pp_param)
                self.dskman_dataiter.init_params(self._nndb, self._save_to_dir)

        elif (self._db_dir is not None):
            if (create_data_generator):
                self.dskman_dataiter = self._create_dskman_dskdataiter(self._db_pp_param)
                self.dskman_dataiter.init_params(self._db_dir, self._save_sub_dir)

        else:
            raise Exception("ARG_ERR: Database is not mentioned")
