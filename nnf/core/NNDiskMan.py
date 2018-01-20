# -*- coding: utf-8 -*-
"""
.. module:: NNDiskMan
   :platform: Unix, Windows
   :synopsis: Represent NNDiskMan class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import os
import pickle
from enum import Enum

# Local Imports
from nnf.db.Dataset import Dataset
from nnf.db.DbSlice import DbSlice
from nnf.utl.internals import array_to_img
from nnf.core.iters.disk.DskmanDskDataIterator import DskmanDskDataIterator
from nnf.core.iters.memory.DskmanMemDataIterator import DskmanMemDataIterator


class NNDiskManMode(Enum):
    """NNDiskManMode enumeration describes ....."""
    INVALID = -1,
    UNSUPPORTED = 0,
    MEM_TO_DSK = 1,     # Reads data in the memory, process it, write it on the disk
    DSK = 2,            # Reads data in the disk, process it, write it on the disk

class NNDiskMan(object):
    """Manage database in disk.

    Attributes
    ----------
    sel : :obj:`Selection`
        Information to split the dataset.

    _dskdb_param : :obj:`dict`, optional
        Iterator parameters and Pre-processing parameters (keras) for iterators.
        Iterator parameters are used in :obj:`NNBigDataDiskMan`
        to handle binary files. (Default value = None).
        See also :obj:`ImageDataPreProcessor`.

    _nndb : :obj:`NNdb`, optional
        Database to be processed against `sel` and `_dskman_param['pp']`.
        Either `nndb` or `_dskman_param['db_dir']` must be provided. (Default value = None).

    _save_dir : str, optional
        Path to save the processed data. (Default value = None).

    save_dir_abspath : str
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
        Iterator for the database.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, sel, dskdb_param, memdb_param=None, save_dir=None):
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

        save_dir : str, optional
            Path to save the processed data. (Default value = None).
        """
        self.sel = sel
        self.save_dir_abspath = os.path.join(dskdb_param['db_dir'], save_dir)
        self._dskdb_param = dskdb_param
        self._memdb_param = memdb_param
        self._save_dir = save_dir

        # Set defaults for configurations that will be utilized in `NNDiskMan`
        memdb_param.setdefault('pp', None)
        dskdb_param.setdefault('pp', None)

        # Create the save_dir_abspath if does not exist
        if not os.path.exists(self.save_dir_abspath):
            os.makedirs(self.save_dir_abspath)

        # Track the files for each dataset of each patch
        # Keyed by the patch_id, and [Dataset.TR|Dataset.VAL|Dataset.TE|...]
        # value = [file_path, fpos, cls_lbl] <= frecord
        self._dict_fregistry = {}

        # Track the class counts for each dataset
        # Keyed by [Dataset.TR|Dataset.VAL|Dataset.TE|...]
        # value = <int> - class count
        self._dict_nb_class = {}

        # Counter for auto assigned class label for each patch of each dataset
        # Keyed by the patch_id, and [Dataset.TR|Dataset.VAL|Dataset.TE|...]
        # value = <int> - counter for the class label
        self._dict_cls_lbl = {}

        # PERF: Required for `_dict_fpath_ref`
        # Image count for each dataset of each patch
        # Keyed by the patch_id, and [Dataset.TR|Dataset.VAL|Dataset.TE|...]
        # value = <int> - image count
        self._dict_imcount = {}

        # PERF: File path is saved only in first record of each class
        # Reference image index to fetch filepath for each class of each dataset of each patch
        # Keyed by the patch_id, and [Dataset.TR|Dataset.VAL|Dataset.TE|...], and class label
        # value = <int> - image index that contains the filepath for each class label
        self._dict_fpath_ref = {}

        # PERF: Whether to save the images that are processed via nndiskman
        self._save_images = True

        # PERF: Whether to save the images forcefully that are processed via nndiskman
        self._force_save_images = False

        # PERF: Whether to update the dictionaries 
        # (_dict_fregistry, _dict_nb_class, _dict_cls_lbl)
        self._update_dicts = True

        # Diskman data iterator
        self.dskman_dataiter = None

        # Datasets currently in use across all patches of this `nndiskman`
        # Used by NNFramework.
        self.datasets_in_use = set()

    def init(self):
        """Initialize the :obj:`NNDiskMan` instance."""

        # Init data iterator
        self.__init_dataiter()

        # PERF: Update the dictionaries if necessary
        if not self._update_dicts:
            return

        # Developer Note:
        # - DbSlice utilizes DskmanDskDataIterator to perform selection on nndb (without pp_param).
        # - NNDiskman utilizes DskmanDskDataIterator to perform selection on disk db (optionally with db_pp_param)
        #     to build the index for data items: ‘_dict_fregistry’,
        #     to save the processed patches of images on the disk.

        # Initialize class ranges and column ranges
        # DEPENDENCY -The order must be preserved as per the enum Dataset
        # REF_ORDER: [TR=0, VAL=1, TE=2, TR_OUT=3, VAL_OUT=4], Refer Dataset enum
        cls_ranges = [self.sel.class_range, self.sel.val_class_range, self.sel.te_class_range,
                      self.sel.class_range, self.sel.val_class_range, self.sel.te_class_range]
        col_ranges = [self.sel.tr_col_indices, self.sel.val_col_indices, self.sel.te_col_indices,
                      self.sel.tr_out_col_indices, self.sel.val_out_col_indices, self.sel.te_out_col_indices]

        # Set the default range if not specified
        # noinspection PyProtectedMember
        DbSlice._set_default_cls_range(0, cls_ranges, col_ranges)

        # Must have atleast 1 patch to represent the whole image
        nnpatch = self.sel.nnpatches[0]

        # PERF: Init dskman_dataiter (could be in memory iter or disk iter)
        need_patch_processing = not ((len(self.sel.nnpatches) == 1) and nnpatch.is_holistic)

        self._save_images = self._force_save_images or (self._save_images and
                                                        (self.sel.need_processing() or need_patch_processing or
                                                         isinstance(self.dskman_dataiter, DskmanMemDataIterator)))
        self.dskman_dataiter.init_ex(cls_ranges, col_ranges, self._save_images)

        # [PERF] Iterate through the chosen subset of the disk|nndb database
        for cimg, frecord, cls_idx, col_idx, dataset_tup in self.dskman_dataiter:

            # cimg: array_like data (maybe an image or raw data item) 
            # frecord: [fpath, fpos, cls_lbl], assert(cls_lbl == cls_idx)
            # cls_idx, col_idx: int
            # dataset_tup: list of tuples
            #       [(Dataset.TR, True), (Dataset.TR, False), (Dataset.VAL, True), ...]
    
            # SPECIAL NOTE:  cls_lbl, cls_idx, col_idx are expressed with respect to the
            # cls_ranges, col_ranges defined above and may not consist of continuous
            # indices. i.e cls_lbl=[0 4 6 3]

            # Perform image operations only if cimg is a image
            if cimg is not None and cimg.ndim >= 3:

                # For histogram equalization operation (canonical image)
                cann_cimg = None
                if self.sel.histmatch_col_index is not None:
                    cann_cimg, _ = self.dskman_dataiter.get_cimg_frecord(cls_idx, self.sel.histmatch_col_index)

                # Perform image pre-processing
                cimg = DbSlice.preprocess_im_with_sel(cimg, cann_cimg, self.sel, self.dskman_dataiter.get_im_ch_axis())

            # Update the class counts for each dataset  
            for edataset, is_new_class in dataset_tup:
                if is_new_class:
                    self._increment_nb_class(edataset)

            # PERF: If there is only 1 patch and it is the whole image
            if not need_patch_processing:
                pimg = self._extract_impatch(cimg, nnpatch)
                self._post_process_loop(frecord, pimg, self.sel.nnpatches[0].id, dataset_tup, cls_idx, col_idx)
            
            else:
                # Process the nnpatches against cimg
                for nnpatch in self.sel.nnpatches:
                    pimg = self._extract_impatch(cimg, nnpatch)
                    self._post_process_loop(frecord, pimg, nnpatch.id, dataset_tup, cls_idx, col_idx)    

        # Cleanup the iterator by release the internal resources
        self.dskman_dataiter.release()

        # PERF: Save this nndiskman object
        self.save(self.save_dir_abspath)

    def save(self, destination_dir):
        """Save this nndiskman to disk.
    
        Parameters
        ----------
        destination_dir : str
            Path to the folder that contains the `diskman.pkl` file.
        """
        pkl_fpath = os.path.join(destination_dir, 'diskman.pkl')

        # IMPORTANT: This will serialize this object including `sel` object with sel.nnpatches.
        # If extended `NNPatch` object contains a non-serializable attribute, this method will fail.
        try:
            # Pickle the self object using the highest protocol available.
            with open(pkl_fpath, "wb") as f:
                pickle.dump(self, f, -1)

        except Exception as e:
            # Unpickling failed
            print("Pickle failed. Possibly due to a non-serializable attribute in extended `NNPath` object.")
            raise pickle.PickleError(repr(e))

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

    @staticmethod
    def mode(dskdb_param, memdb_param):

        if not dskdb_param['force_load_db']:

            # Currently it only support `nndb` data traversal
            if memdb_param['nndb'] is not None:
                return NNDiskManMode.MEM_TO_DSK

            # FUTURE: Add support to handle `nndbs_tup` in `NNDiskMan`.
            elif memdb_param['nndbs_tup'] is not None:
                return NNDiskManMode.UNSUPPORTED

        if dskdb_param['db_dir'] is not None:
            return NNDiskManMode.DSK

        else:
            return NNDiskManMode.INVALID

    @staticmethod
    def load_nndiskman(destination_dir):
        """Load diskman from the disk.

        Parameters
        ----------
        destination_dir : str
            Path to the folder that contains the `diskman.pkl` file.

        Returns
        -------
        :obj:`NNDiskMan`
            :obj:`NNDiskMan` if successful, None otherwise.
        """
        pkl_fpath = os.path.join(destination_dir, 'diskman.pkl')

        if not os.path.isfile(pkl_fpath):
            return None

        try:
            with open(pkl_fpath, "rb") as f:
                nndskman = pickle.load(f)

        except pickle.UnpicklingError:
            raise

        except Exception as e:
            # Unpickling failed
            raise pickle.UnpicklingError(repr(e))

        return nndskman

    ##########################################################################
    # Protected Interface
    ##########################################################################
    # noinspection PyMethodMayBeStatic
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

        pimg : ndarray
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
        # PERF: Save the images only if needed
        if self._save_images:
            fpath_to_class, filename = self._save_data(pimg, patch_id, cls_idx, col_idx)
            frecord[0] = fpath_to_class
            frecord[1] = filename

        # Save the original
        frecord_ori = frecord

        # Update the fregistry and auto assign class label
        for edataset, is_new_class in dataset_tup:

            # Take a copy
            frecord = frecord_ori.copy()

            # Calculate the new class label (continuous)
            tmp_datasets = self._dict_cls_lbl.setdefault(patch_id, {})
            cls_lbl = tmp_datasets.setdefault(edataset, -1)

            if is_new_class:
                cls_lbl += 1
                tmp_datasets[edataset] = cls_lbl

            # Calculate image count
            tmp_datasets = self._dict_imcount.setdefault(patch_id, {})
            imcount = tmp_datasets.setdefault(edataset, 0)
            tmp_datasets[edataset] = imcount + 1

            # Update file path reference by using `imcount` as an index
            # PERF: Memory efficient implementation (only first frecord of the file has the path to the file)
            tmp_datasets = self._dict_fpath_ref.setdefault(patch_id, {})
            fpath_ref_by_cls = tmp_datasets.setdefault(edataset, {})

            if is_new_class:
                fpath_ref_by_cls[cls_lbl] = imcount  # index to start of the class
            else:
                frecord[0] = fpath_ref_by_cls[cls_lbl]

            # Update frecord for new class label
            frecord[2] = cls_lbl

            # Add to file record to fregistry
            self._add_to_fregistry(patch_id, edataset, frecord)

            # Update
            self.datasets_in_use.add(edataset)

    # noinspection PyMethodMayBeStatic
    def _extract_impatch(self, cimg, nnpatch):
        """Extract the image patch from the nnpatch.

        Parameters
        ----------
        cimg : ndarray
            Color image.
 
        nnpatch : :obj:`NNPatch`
            Information about the image patch. (dimension and offset).
        """
        if nnpatch.is_holistic:
            return cimg
        
        # BUG: extract the patch from cimg to pimg
        pimg = cimg
        return pimg

    def _save_data(self, pimg, patch_id, cls_idx, col_idx,
                   data_format=None, scale=False):
        """Save data to the disk.

        Parameters
        ----------
        pimg : ndarray
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
            File path the class directory.

        str | int :
            File name or file position where the data is written.
        """
        fname = '{cls_idx}_{col_idx}_{patch_id}.{format}'.format(
                                                    cls_idx=cls_idx,
                                                    col_idx=col_idx,
                                                    patch_id=patch_id,
                                                    format='jpg')
        cls_dir = os.path.join(self.save_dir_abspath, str(cls_idx))
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

        fpath = os.path.join(cls_dir, fname)
        img = array_to_img(pimg, data_format='channels_last', scale=scale)
        img.save(fpath)
        return cls_dir, fname

    ##########################################################################
    # Special Interface
    ##########################################################################
    # noinspection PyProtectedMember
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
        This is used to compare the mandatory fields of `NNDiskman`
        after deserializing the object from the disk.
        """
        # noinspection PyProtectedMember
        return (self.sel == nndiskman.sel) and\
               (self._dskdb_param['pp'] == nndiskman._dskdb_param['pp']) and\
               (self._memdb_param['pp'] == nndiskman._memdb_param['pp'])

    def __getstate__(self):
        """Serialization call back with Pickle. 

        Used to remove the following fields from serialization.
        """
        odict = self.__dict__.copy()  # copy the dict since we change it
        del odict['dskman_dataiter']
        tmp = odict['_dskdb_param']
        odict['_dskdb_param'] = {'pp': tmp['pp'], 'force_load_db': tmp['force_load_db']}
        tmp = odict['_memdb_param']
        odict['_memdb_param'] = {'pp': tmp['pp']}
        del odict['_save_dir']
        del odict['_save_images']
        del odict['_force_save_images']
        del odict['_update_dicts']
        del odict['_dict_cls_lbl']
        return odict

    ##########################################################################
    # Private Interface
    ##########################################################################
    # noinspection PyProtectedMember
    def __init_dataiter(self):
        """Initialize the diskman data iterator only if needed.

            Load any saved diskman instance and create diskman iterators
            only if necessary.
        """
        create_data_generator = True

        # PERF: Try loading the saved nndiskman from parent directory first.
        cur_directory = os.path.dirname(self.save_dir_abspath)
        dskman = self.load_nndiskman(cur_directory)
        if dskman is None:
            # PERF: Next try to load from the specified directory.
            dskman = self.load_nndiskman(self.save_dir_abspath)

        if dskman is not None and self == dskman:  # will invoke __eq__()
            # PERF: Disable saving of images
            self._save_images = False

            # Set the data from the loaded nndiskman
            self._dict_fregistry = dskman._dict_fregistry
            self._dict_nb_class = dskman._dict_nb_class
            self.datasets_in_use = dskman.datasets_in_use

            # PERF: Stop updating dictionaries if paths are the same
            if (self.save_dir_abspath == dskman.save_dir_abspath) or\
                    (cur_directory == dskman.save_dir_abspath):
                self._update_dicts = False
                create_data_generator = False

        elif dskman is not None:  # and (self != dskman)
            # PERF: Forcefully save the images
            self._force_save_images = True

        # Instantiate an diskman data iterator object depending on the operating mode
        mode = NNDiskMan.mode(self._dskdb_param, self._memdb_param)

        # nndb -> process sel -> write to disk
        if  mode == NNDiskManMode.MEM_TO_DSK:
            if create_data_generator:
                self.dskman_dataiter = self._create_dskman_memdataiter(self._memdb_param['pp'])
                self.dskman_dataiter.init_params(self._memdb_param['nndb'], self.save_dir_abspath)

        # db_dir -> process sel -> write to disk
        elif mode == NNDiskManMode.DSK:
            if create_data_generator:
                self.dskman_dataiter = self._create_dskman_dskdataiter(self._dskdb_param['pp'])
                self.dskman_dataiter.init_params(self._dskdb_param['db_dir'], self._save_dir)

        else:
            raise Exception("Invalid internal `NNDiskMan` mode")
