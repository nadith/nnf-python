"""DBSLICE Module to represent DbSlice class."""
# -*- coding: utf-8 -*-
# Global Imports
import scipy.misc
import numpy as np
from numpy.random import rand
from numpy.random import permutation
from collections import namedtuple

# Local Imports
from nnf.db.Format import Format
from nnf.db.NNdb import NNdb
from nnf.db.Noise import Noise
from nnf.pp.im_pre_process import im_pre_process
from nnf.utl.rgb2gray import rgb2gray
from nnf.db.Selection import *
from nnf.db.Dataset import Dataset
from nnf.core.iters.memory.DskmanMemDataIterator import DskmanMemDataIterator
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator

class DbSlice(object):
    """Perform slicing of nndb with the help of a selection structure.

    Selection Structure (with defaults)
    -----------------------------------
    sel.tr_col_indices      = None    # Training column indices
    sel.tr_noise_mask       = None    # Noisy tr. col indices (bit mask)
    sel.tr_noise_rate       = None    # Rate or noise types for the above field
    sel.tr_out_col_indices  = None    # Training target column indices
    sel.val_col_indices     = None    # Validation column indices
    sel.val_out_col_indices = None    # Validation target column indices
    sel.te_col_indices      = None    # Testing column indices
    sel.nnpatches           = None    # NNPatch object array
    sel.use_rgb             = True    # Use rgb or convert to grayscale
    sel.color_index         = None    # Specific color indices (set .use_rgb = false)
    sel.use_real            = False   # Use real valued database TODO: (if .normalize = true, Operations ends in real values)  # noqa E501
    sel.scale               = None    # Scaling factor (resize factor)
    sel.normalize           = False   # Normalize (0 mean, std = 1)
    sel.histeq              = False   # Histogram equalization
    sel.histmatch           = False   # Histogram match (ref. image: first image of the class)  # noqa E501
    sel.class_range         = None    # Class range for training database or all (tr, val, te)
    sel.val_class_range     = None    # Class range for validation database
    sel.te_class_range      = None    # Class range for testing database
    sel.pre_process_script  = None    # Custom preprocessing script

    i.e
    Pass a selection structure to split the nndb accordingly
    [nndb_tr, _, _, _] = DbSlice.slice(nndb, sel)

    Methods
    -------
    slice()
        Slice the database according to the selection structure.

    examples()
        Extensive set of examples.

    Examples
    --------
    Database with same no. of images per class, with build class idx
    >>> nndb = NNdb('any_name', imdb, 8, true)

    Notes
    -----
    All methods are implemented via static methods thus thread-safe.

    Copyright 2015-2016 Nadith Pathirage, Curtin University.
    (chathurdara@gmail.com).
    """

    _data_generator = None # PERF: repeated static calls DbSlice.slice()

    ##########################################################################
    # Public Interface
    ##########################################################################
    def safe_slice(nndb, sel=None):
        """Thread Safe"""
        self._data_generator = DskmanMemDataIterator(nndb)
        DbSlice.slice(nndb, sel, self._data_generator)

    @staticmethod
    def slice(nndb, sel=None, data_generator=None):
        """Slice the database according to the selection structure.

        Parameters
        ----------
        nndb : NNdb
            NNdb object that represents the dataset.

        sel : selection structure
            Information to split the dataset.

            i.e
            Selection Structure (with defaults)
            -----------------------------------
            sel.tr_col_indices      = None    # Training column indices
            sel.tr_noise_mask       = None    # Noisy tr. col indices (bit mask)
            sel.tr_noise_rate       = None    # Rate or noise types for the above field
            sel.tr_out_col_indices  = None    # Training target column indices
            sel.val_col_indices     = None    # Validation column indices
            sel.val_out_col_indices = None    # Validation target column indices
            sel.te_col_indices      = None    # Testing column indices
            sel.nnpatches           = None    # NNPatch object array
            sel.use_rgb             = True    # Use rgb or convert to grayscale
            sel.color_index         = None    # Specific color indices (set .use_rgb = false)
            sel.use_real            = False   # Use real valued database TODO: (if .normalize = true, Operations ends in real values)  # noqa E501
            sel.scale               = None    # Scaling factor (resize factor)
            sel.normalize           = False   # Normalize (0 mean, std = 1)
            sel.histeq              = False   # Histogram equalization
            sel.histmatch           = False   # Histogram match (ref. image: first image of the class)  # noqa E501
            sel.class_range         = None    # Class range for training database or all (tr, val, te)
            sel.val_class_range     = None    # Class range for validation database
            sel.te_class_range      = None    # Class range for testing database
            sel.pre_process_script  = None    # Custom preprocessing script

        Returns
        -------
        nndbs_tr : NNdb or list- NNdb
             Training NNdb object(s). (Incase patch division is required).

        nndb_tr_out : NNdb or list- NNdb
             Training target NNdb object(s). (Incase patch division is required).

        nndbs_val : NNdb or list- NNdb
             Validation NNdb object(s). (Incase patch division is required).

        nndbs_te : NNdb or list- NNdb
             Testing NNdb object(s). (Incase patch division is required).

        nndbs_tr_out : NNdb or cell- NNdb
            Training target NNdb object(s). (Incase patch division is required).
        
        nndbs_val_out : NNdb or cell- NNdb
            Validation target NNdb object(s). (Incase patch division is required).

        Notes
        -----
        The new dbs returned will contain img at consecutive locations for
        duplicate indices irrespective of the order that is mentioned.
        i.e Tr:[1 2 3 1], DB:[1 1 2 3]
        """
        # Set defaults for arguments
        if (sel is None):
            sel = Selection()
            sel.tr_col_indices = np.arange(nndb.n_per_class[0])
            sel.tr_noise_mask = None
            sel.tr_noise_rate = None
            sel.tr_noise_mask = np.array([0, 1, 0, 0, 0, 0, 0, 0])       # index is affected or not # noqa E501
            sel.tr_noise_rate = np.array([0, 0.5, 0, 0, 0, 0, 0, 0]) # percentage of corruption # noqa E501
            sel.tr_out_col_indices = None
            sel.val_col_indices = None
            sel.val_out_col_indices = None
            sel.te_col_indices = None
            sel.nnpatches = None
            sel.use_rgb = True
            sel.color_index = None
            sel.use_real = False
            sel.scale = None
            sel.normalize = False
            sel.histeq = False
            sel.histmatch = False
            sel.class_range = None
            sel.val_class_range = None
            sel.te_class_range = None
            sel.pre_process_script = None

        # Set defaults for selection fields, if the field does not exist
        if (not hasattr(sel, 'tr_noise_mask')): sel.tr_noise_mask = None  # noqa E701
        if (not hasattr(sel, 'tr_noise_rate')): sel.tr_noise_rate = None  # noqa E701
        if (not hasattr(sel, 'tr_out_col_indices')): sel.tr_out_col_indices = None  # noqa E501, E701
        if (not hasattr(sel, 'val_col_indices')): sel.val_col_indices = None  # noqa E501, E701
        if (not hasattr(sel, 'val_out_col_indices')): sel.val_out_col_indices = None  # noqa E501, E701
        if (not hasattr(sel, 'te_col_indices')): sel.te_col_indices = None  # noqa E701
        if (not hasattr(sel, 'nnpatches')): sel.nnpatches = None  # noqa E701
        if (not hasattr(sel, 'use_rgb')): sel.use_rgb = True  # noqa E701
        if (not hasattr(sel, 'color_index')): sel.color_index = None  # noqa E701
        if (not hasattr(sel, 'use_real')): sel.use_real = False  # noqa E701
        if (not hasattr(sel, 'resize')): sel.resize = None  # noqa E701
        if (not hasattr(sel, 'normalize')): sel.normalize = False  # noqa E701
        if (not hasattr(sel, 'scale')): sel.scale = None  # noqa E701
        if (not hasattr(sel, 'histeq')): sel.histeq = False  # noqa E701
        if (not hasattr(sel, 'histmatch')): sel.histmatch = False  # noqa E701
        if (not hasattr(sel, 'class_range')): sel.class_range = None  # noqa E701
        if (not hasattr(sel, 'val_class_range')): sel.val_class_range = None  # noqa E701
        if (not hasattr(sel, 'te_class_range')): sel.te_class_range = None  # noqa E701
        if (not hasattr(sel, 'pre_process_script')): sel.pre_process_script = None  # noqa E501, E701

        # Error handling for arguments
        if (sel.tr_col_indices is None and
           sel.tr_out_col_indices is None and
           sel.val_col_indices is None and
           sel.val_out_col_indices is None and
           sel.te_col_indices is None):
            raise Exception('ARG_ERR: [tr|tr_out|val|val_out|te]_col_indices: mandory field')  # noqa E501

        if (sel.use_rgb and
           sel.color_index is not None):
            raise Exception('ARG_CONFLICT: sel.use_rgb, sel.color_index')

        if (sel.tr_noise_mask is not None and
           sel.tr_noise_rate is None):
            raise Exception('ARG_MISSING: specify sel.tr_noise_rate field')

        # Set defaults for data generator   
        if (data_generator is None):
            # Safe iterator object for repeated static calls DbSlice.slice()
            DbSlice._data_generator  = DskmanMemDataIterator(nndb) if (DbSlice._data_generator is None) else DbSlice._data_generator
            data_generator =  DbSlice._data_generator

        # Set defaults for class range (tr|val|te|...)
        sel.class_range = np.arange(nndb.cls_n) if (sel.class_range is None) else sel.class_range

        # Initialize class ranges and column ranges
        # DEPENDANCY -The order must be preserved as per the enum Dataset 
        # REF_ORDER: [TR=0, VAL=1, TE=2, TR_OUT=3, VAL_OUT=4]
        cls_ranges = [sel.class_range, 
                        sel.val_class_range, 
                        sel.te_class_range, 
                        sel.class_range, 
                        sel.val_class_range]

        col_ranges = [sel.tr_col_indices, 
                        sel.val_col_indices, 
                        sel.te_col_indices, 
                        sel.tr_out_col_indices, 
                        sel.val_out_col_indices]

        # Set defaults for class ranges [val|te|tr_out|val_out]
        DbSlice._set_default_cls_range(0, cls_ranges, col_ranges)

        # NOTE: TODO: Whitening the root db did not perform well
        # (co-variance is indeed needed)

        # Initialize NNdb dictionary of lists
        # i.e dict_nndbs[Dataset.TR] => [NNdb_Patch_1, NNdb_Patch_2, ...]
        dict_nndbs = DbSlice._create_dict_nndbs(col_ranges, cls_ranges, nndb, sel)
        
        # Patch iterating loop's max count
        patch_loop_max_n = 1 if (sel.nnpatches is None) else len(sel.nnpatches)

        # Initialize the sample countss
        # Ordered by REF_ORDER defined above
        sample_counts = [np.zeros(patch_loop_max_n, 'uint16'), 
                        np.zeros(patch_loop_max_n, 'uint16'),
                        np.zeros(patch_loop_max_n, 'uint16'),
                        np.zeros(patch_loop_max_n, 'uint16'),
                        np.zeros(patch_loop_max_n, 'uint16')]

        # Noise required indices
        noise_req_indices = np.where(sel.tr_noise_mask == 1)[0]

        # Pre-processing params struct
        pp_params = namedtuple('pp_params', ['histeq', 'normalize', 'histmatch', 'cann_img'])  # noqa E501
        
        # Initialize the generator
        data_generator.init(cls_ranges, col_ranges)

        # [PERF] Iterate through the choset subset of the nndb database
        for cimg, cls_idx, col_idx, datasets in data_generator:

            # Iterate through image patches
            for pi in range(patch_loop_max_n):

                # Holistic image (by default)
                pimg = cimg

                # Generate the image patch
                if (sel.nnpatches is not None):
                    nnpatch = sel.nnpatches[pi];
                    x = nnpatch.offset[1];
                    y = nnpatch.offset[0];                                                                                
                    w = nnpatch.w;
                    h = nnpatch.h;
                        
                    # Extract the patch
                    pimg = cimg[y:y+h, x:x+w, :]

                # Peform image operations only if db format comply them
                # TODO: Compatibility with (nndb.format == Format.N_H_W_CH):
                if (nndb.format == Format.H_W_CH_N):

                    # Perform resize
                    if (sel.scale is not None):
                        pimg = scipy.misc.imresize(pimg, sel.scale)

                    # Perform histrogram matching against the cannonical image
                    cls_st_img = None
                    if (sel.histmatch):
                        if (sel.scale is not None):
                            cls_st = nndb.cls_st[cls_idx]
                            cls_st_img = scipy.misc.imresize(nndb.get_data_at(cls_st), sel.scale)  # noqa E501

                    # Color / Gray Scale Conversion (if required)
                    pimg = DbSlice._process_color(pimg, sel)
                    cls_st_img = DbSlice._process_color(cls_st_img, sel)

                    # Pre-Processing
                    pp_params.histeq = sel.histeq
                    pp_params.normalize = sel.normalize
                    pp_params.histmatch = sel.histmatch
                    pp_params.cann_img = cls_st_img
                    pimg = im_pre_process(pimg, pp_params)

                    # [CALLBACK] the specific pre-processing script
                    if (sel.pre_process_script is not None):
                        pimg = sel.pre_process_script(pimg)

                # Iterate through datasets
                for edataset, _ in datasets:

                    nndbs = dict_nndbs.setdefault(edataset, None)
                    samples = sample_counts[edataset.int()]

                    # Build Traiing DB
                    if (edataset == Dataset.TR):

                        # Check whether col_idx is a noise required index 
                        process_noise = False
                        noise_rate = None
                        for nf in nf_indices:
                            if (col_idx == sel.tr_col_indices[nf]):
                                process_noise = True
                                noise_rate = sel.tr_noise_rate[nf]
                                break

                        [nndbs, samples] =\
                            DbSlice._build_nndb_tr(nndbs, pi, samples, pimg, process_noise, noise_rate)  # noqa E501

                    # Build Training Output DB
                    elif(edataset == Dataset.TR_OUT):
                       
                        [nndbs, samples] =\
                        DbSlice._build_nndb_tr_out(nndbs, pi, samples, pimg)  # noqa E501

                    # Build Valdiation DB
                    elif(edataset == Dataset.VAL):                        
                        [nndbs, samples] =\
                            DbSlice._build_nndb_val(nndbs, pi, samples, pimg)  # noqa E501

                    # Build Valdiation Target DB
                    elif(edataset == Dataset.VAL_OUT):     
                        [nndbs, samples] =\
                            DbSlice._build_nndb_val_out(nndbs, pi, samples, pimg)  # noqa E501

                    # Build Testing DB
                    elif(edataset == Dataset.TE): 
                        [nndbs, samples] =\
                            DbSlice._build_nndb_te(nndbs, pi, samples, pimg)  # noqa E501

        # Returns NNdb object instead of list (when no patches)
        if (sel.nnpatches is None):
            return  dict_nndbs[Dataset.TR][0],\
                    dict_nndbs[Dataset.VAL][0],\
                    dict_nndbs[Dataset.TE][0],\
                    dict_nndbs[Dataset.TR_OUT][0],\
                    dict_nndbs[Dataset.VAL_OUT][0]

        return  dict_nndbs[Dataset.TR],\
                dict_nndbs[Dataset.VAL],\
                dict_nndbs[Dataset.TE],\
                dict_nndbs[Dataset.TR_OUT],\
                dict_nndbs[Dataset.VAL_OUT]

    @staticmethod
    def examples(imdb_8):
        """Extensive set of examples.

        # Full set of options
        nndb = NNdb('original', imdb_8, 8, true)
        sel.tr_col_indices        = [1:3 7:8] #[1 2 3 7 8]
        sel.tr_noise_mask         = None
        sel.tr_noise_rate         = None
        sel.tr_out_col_indices    = None
        sel.tr_cm_col_indices     = None
        sel.te_col_indices        = [4:6] #[4 5 6]
        sel.use_rgb               = False
        sel.color_index           = None
        sel.use_real              = False
        sel.scale                 = 0.5
        sel.normalize             = False
        sel.histeq                = True
        sel.histmatch             = False
        sel.class_range           = [1:36 61:76 78:100]
        #sel.pre_process_script   = @custom_pprocess
        sel.pre_process_script    = None
        [nndb_tr, _, nndb_te, _]  = DbSlice.slice(nndb, sel)

        Parameters
        ----------
        imdb_8 : NNdb
            NNdb object that represents the dataset. It should contain
            only 8 images per subject.

            Format: (Samples x H x W x CH).
        """
        #
        # Select 1st 2nd 4th images of each identity for training.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        [nndb_tr, _, _, _, _] = DbSlice.slice(nndb, sel) # nndb_tr = DbSlice.slice(nndb, sel) # noqa E501


        #
        # Select 1st 2nd 4th images of each identity for training.
        # Divide into patches
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        patch_gen = NNPatchGenerator(nndb.h, nndb.w, 33, 33, 33, 33)
        sel.nnpatches = patch_gen.generate_patches()

        # Cell arrays of NNdb objects for each patch
        [nndbs_tr, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndbs_tr[0].show()
        nndbs_tr[1].show()
        nndbs_tr[3].show()
        nndbs_tr[4].show()   


        #
        # Select 1st 2nd 4th images of each identity for training.
        # Select 3rd 5th images of each identity for testing.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        [nndb_tr, _, nndb_te, _, _] = DbSlice.slice(nndb, sel)


        #
        # Select 1st 2nd 4th images of identities denoted by class_range for training. # noqa E501
        # Select 3rd 5th images of identities denoted by class_range for testing. # noqa E501
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.class_range = np.arange(10)                         # First ten identities # noqa E501
        [nndb_tr, _, nndb_te, _, _] = DbSlice.slice(nndb, sel)


        # 
        # Select 1st 2nd 4th images of identities denoted by class_range for training.
        # Select 1st 2nd 4th images images of identities denoted by class_range for validation.   
        # Select 3rd 5th images of identities denoted by class_range for testing. 
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.val_col_indices= np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.class_range    = np.arange(10)
        [nndb_tr, nndb_val, nndb_te, _, _] = DbSlice.slice(nndb, sel);
            
            
        # 
        # Select 1st 2nd 4th images of identities denoted by class_range for training.
        # Select 1st 2nd 4th images images of identities denoted by val_class_range for validation.   
        # Select 3rd 5th images of identities denoted by te_class_range for testing. \
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.val_col_indices= np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.class_range    = np.arange(10)
        sel.val_class_range= np.arange(6,15)
        sel.te_class_range = np.arange(17, 20)
        [nndb_tr, nndb_val, nndb_te, _, _] = DbSlice.slice(nndb, sel)

   
        # 
        # Select 1st 2nd 4th images of identities denoted by class_range for training.
        # Select 3rd 4th images of identities denoted by val_class_range for validation.
        # Select 3rd 5th images of identities denoted by te_class_range for testing.
        # Select 1st 1st 1st images of identities denoted by class_range for training target.
        # Select 1st 1st images of identities denoted by val_class_range for validation target.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.val_col_indices= np.array([2, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.tr_out_col_indices  = np.zeros(3, dtype='uint8')  # [1, 1, 1]
        sel.val_out_col_indices = np.zeros(2, dtype='uint8')  # [1, 1]
        sel.class_range    = np.arange(9)
        sel.val_class_range= np.arange(6,15)
        sel.te_class_range = np.arange(17, 20)
        [nndb_tr, nndb_val, nndb_te, nndb_tr_out, nndb_val_out] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)
        nndb_val.show(10, 2)
        nndb_te.show(4, 2)
        nndb_tr_out.show(10, 3)
        nndb_val_out.show(10, 2)


        #
        # Select 1st 2nd 4th images of each identity for training +
        #               add various noise types @ random locations of varying degree. # noqa E501
        #               default noise type: random black and white dots.
        # Select 3rd 5th images of each identity for testing.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.tr_noise_mask = np.array([0, 1, 0, 1, 0, 1])       # index is affected or not # noqa E501
        sel.tr_noise_rate = np.array([0, 0.5, 0, 0.5, 0, 0.5]) # percentage of corruption # noqa E501
        # sel.tr_noise_rate  = [0 0.5 0 0.5 0 Noise.G]         # last index with Gauss noise # noqa E501
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        [nndb_tr, _, nndb_te, _, _] = DbSlice.slice(nndb, sel)


        #
        # To prepare regression datasets, training dataset and training target dataset # noqa E501
        # Select 1st 2nd 4th images of each identity for training.
        # Select 1st 1st 1st image of each identity for corresponding training target # noqa E501
        # Select 3rd 5th images of each identity for testing.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.tr_out_col_indices = np.array([1, 1, 1])  # [1 1 1] (Regression 1->1, 2->1, 4->1) # noqa E501
        sel.te_col_indices = np.array([2, 4])
        [nndb_tr, nndb_tr_out, nndb_te, _, _] = DbSlice.slice(nndb, sel)


        #
        # Resize images by 0.5 scale factor.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.scale = 0.5
        [nndb_tr, _, nndb_te, _, _] = DbSlice.slice(nndb, sel)


        #
        # Use gray scale images.
        # Perform histrogram equalization.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.histeq = True
        [nndb_tr, _, nndb_te, _, _] = DbSlice.slice(nndb, sel)


        #
        # Use gray scale images.
        # Perform histrogram match. This will be performed with the 1st image
        #  of each identity irrespective of the selection choice.
        # (refer code for more details)
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.histmatch = True
        [nndb_tr, _, nndb_te, _, _] = DbSlice.slice(nndb, sel)


        #
        # If imdb_8 supports many color channels
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.color_index = 5  # color channel denoted by 5th index
        [nndb_tr, _, nndb_te, _, _] = DbSlice.slice(nndb, sel)

    ##########################################################################
    # Private Interface
    ##########################################################################
    @staticmethod
    def _create_dict_nndbs(col_ranges, cls_ranges, nndb, sel):
        """Create a dictionary of nndb lists.

            i.e dict_nndbs[Dataset.TR] => [NNdb_Patch_1, NNdb_Patch_2, ...]

            IMPLEMENTATION NOTES:
                LIMITATION: if nndb has different n_per_class,
                            and Select.ALL is requested, need to decide on a scalar
                            n_per_class to pre-allocate arrays.
                TODO:       Allocate to the maximum value of n_per_class and reduce/np.resize
                            it later
        """
        dict_nndbs = {}

        # Iterate through ranges
        for rng_idx in range(len(col_ranges)):
            
            # Determine n_per_class
            n_per_class = 0
            col_range = col_ranges[rng_idx]  
            if ((col_range is not None) and 
                (isinstance(col_range, Enum) and  col_range == Select.ALL)):

                common_n_per_class = np.unique(nndb.n_per_class)
                if (common_n_per_class.size != 1):
                    raise Exception("nndb.n_per_class must be same for all classes when Select.ALL oepration is used")
                
                n_per_class = common_n_per_class

            elif (col_range is not None):
                n_per_class = col_range.size

            # Determine cls_range_size
            cls_range_size = 0     
            cls_range = cls_ranges[rng_idx]  
            if ((cls_range is not None) and 
                (isinstance(cls_range, Enum) and  cls_range == Select.ALL)):               
                cls_range_size = nndb.cls_n

            elif (col_ranges[rng_idx] is not None):
                cls_range_size = cls_range.size

            # Create nndb and store it at dict_nndbs[ekey]
            ekey = Dataset.enum(rng_idx)
            dict_nndbs.setdefault(ekey, None)
            dict_nndbs[ekey] = DbSlice._init_nndb(str(ekey), sel, nndb, n_per_class, cls_range_size, True)  # noqa E501

        return dict_nndbs

    @staticmethod
    def _set_default_cls_range(default_idx,  cls_ranges, col_ranges):
        """Set the value of the class range at default_idx 
            to the class ranges at other indices [val|te|tr_out|val_out]
        """

        for rng_idx in range(len(col_ranges)):
            if (col_ranges[rng_idx] is not None and 
                cls_ranges[rng_idx] is None):
                cls_ranges[rng_idx] = cls_ranges[default_idx]

    @staticmethod
    def _init_nndb(name, sel, nndb, n_per_class, cls_n, build_cls_idx):
        """Initialize a new NNdb object lists.
                    
        Returns
        -------
        new_nndbs : list- NNdb
            new NNdb object(s).
        """
        # n_per_class: must be a scalar
        if (n_per_class == 0): return  # noqa E701

        # Init Variables
        db = []

        # Peform image operations only if db format comply them
        if (nndb.format == Format.H_W_CH_N):

            # Dimensions for the new NNdb Objects
            nd1 = nndb.h
            nd2 = nndb.w

            if (sel.scale is not None):
                if (np.isscalar(sel.scale)):
                    nd1 = nd1 * sel.scale
                    nd2 = nd2 * sel.scale
                else:
                    nd1 = sel.scale[0]
                    nd2 = sel.scale[1]

            # Channels for the new NNdb objects
            ch = nndb.ch
            if (not sel.use_rgb):
                ch = 0 if (sel.color_index is None) else np.size(sel.color_index) # Selected color channels 
                if (ch == 0): ch = 1  # Grayscale  # noqa E701

            if (sel.nnpatches is not None):
                # Patch count
                patch_n = len(sel.nnpatches);                    
                    
                # Init db for each patch
                for i in range(patch_n):
                    nnpatch = sel.nnpatches[i]
                    db.append(np.zeros((nnpatch.h, nnpatch.w, ch, n_per_class*cls_n), dtype=nndb.db.dtype))
            else:
                db.append(np.zeros((nd1, nd2, ch, n_per_class*cls_n), dtype=nndb.db.dtype))

        elif (nndb.format == Format.H_N):
            nd1 = nndb.h
            db.append(np.zeros((nd1, n_per_class*cls_n), dtype=nndb.db.dtype))

        new_nndbs = []

        # Init nndb for each patch
        for i in range(len(db)):
            new_nndbs.append(NNdb(name + "_P" + str(i), db[i], n_per_class, build_cls_idx))
 
        return new_nndbs

    @staticmethod
    def _process_color(img, sel):
        """Perform color related functions.

        Returns
        -------
        img : 3D tensor -uint8
            Color processed image.
        """
        if (img is None): return  # noqa E701
        _, _, ch = img.shape

        # Color / Gray Scale Conversion (if required)
        if (not sel.use_rgb):

            # if image has more than 3 channels
            if (ch >= 3):
                if (sel.color_index is not None):
                    X = img[:, :, sel.color_index]
                else:
                    X = rgb2gray(img, keepDims=True)
            else:
                X = img[:, :, 1]

            img = X

        return img

    @staticmethod
    def _build_nndb_tr(nndbs, pi, samples, img, process_noise, noise_rate):
        """Build the nndb training database.

        Returns
        -------
        nndbs : cell- NNdb
            Updated NNdb objects.

        samples : vector -uint16
            Updated image count vector.
        """
        if (nndbs is None): return nndbs, samples  # noqa E701
        nndb = nndbs[pi]    

        if (process_noise):

            h, w, ch = img.shape

            # Add different noise depending on the type or rate
            # (ref. Enums/Noise)
            if (noise_rate == Noise.G):
                pass
                # img = imnoise(img, 'gaussian')
                # #img = imnoise(img, 'gaussian')
                # #img = imnoise(img, 'gaussian')
                # #img = imnoise(img, 'gaussian')

            else:
                # Perform random corruption
                # Corruption Size (H x W)
                cs = [np.uint16(h*noise_rate), np.uint16(w*noise_rate)]

                # Random location choice
                # Start of H, W (location)
                sh = np.uint8(1 + rand()*(h-cs[0]-1))
                sw = np.uint8(1 + rand()*(w-cs[1]-1))

                # Set the corruption
                cimg = np.uint8(DbSlice._rand_corrupt(cs[0], cs[1])).astype('uint8')  # noqa E501

                if (ch == 1):
                    img[sh:sh+cs[1], sw:sw+cs[2]] = cimg
                else:
                    for ich in range(ch):
                        img[sh:sh+cs[0], sw:sw+cs[1], ich] = cimg

            nndb.set_data_at(img, samples[pi])

        else:
            nndb.set_data_at(img, samples[pi])
    
        samples[pi] = samples[pi] + 1
        return nndbs, samples

    @staticmethod
    def _rand_corrupt(height, width):
        """Corrupt the image with a (height, width) block.

        Returns
        -------
        img : 3D tensor -uint8
            Corrupted image.
        """
        percentage_white = 50  # Alter this value as desired

        dot_pat = np.zeros((height, width), dtype='uint8')

        # Set the desired percentage of the elements in dotPattern to 1
        flat = dot_pat.flatten()
        flat[1:round(0.01 * percentage_white * np.size(dot_pat))] = 1

        # Seed the random number generator
        #             rand('twister',100*sum(clock))

        # Randomly permute the element order
        dot_pat = np.reshape(flat[permutation(np.size(flat))], (height, width))  # noqa E501
        img = dot_pat * 255

        return img

    @staticmethod
    def _build_nndb_tr_out(nndbs, pi, samples, img):
        """Build the nndb training target database.
                    
        Returns
        -------
        nndbs : cell- NNdb
            Updated NNdb objects.
        
        samples : vector -uint16
            Updated image count vector.
        """
        if (nndbs is None): return nndbs, samples
        nndb = nndbs[pi]

        nndb.set_data_at(img, samples[pi])
        samples[pi] = samples[pi] + 1

        return nndbs, samples

    @staticmethod
    def _build_nndb_val(nndbs, pi, samples, img):
        """"Build the nndb validation database.
                    
        Returns
        -------
        nndbs : cell- NNdb
            Updated NNdb objects.
        
        samples : vector -uint16
            Updated image count vector.
        """
        if (nndbs is None): return nndbs, samples
        nndb = nndbs[pi]

        nndb.set_data_at(img, samples[pi])
        samples[pi] = samples[pi] + 1

        return nndbs, samples

    @staticmethod
    def _build_nndb_val_out(nndbs, pi, samples, img):
        """"Build the nndb validation target database.
                    
        Returns
        -------
        nndbs : cell- NNdb
            Updated NNdb objects.
        
        samples : vector -uint16
            Updated image count vector.
        """
        if (nndbs is None): return nndbs, samples
        nndb = nndbs[pi]

        nndb.set_data_at(img, samples[pi])
        samples[pi] = samples[pi] + 1

        return nndbs, samples

    @staticmethod
    def _build_nndb_te(nndbs, pi, samples, img):
        """Build the testing database.
                    
        Returns
        -------
        nndbs : cell- NNdb
            Updated NNdb objects.
        
        samples : vector -uint16
            Updated image count vector.
        """
        if (nndbs is None): return nndbs, samples
        nndb = nndbs[pi]

        nndb.set_data_at(img, samples[pi])
        samples[pi] = samples[pi] + 1

        return nndbs, samples