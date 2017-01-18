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
from nnf.db.Selection import Selection

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

    ##########################################################################
    # Public Interface
    ##########################################################################
    @staticmethod
    def slice(nndb, sel=None):
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

        # Fetch the counts
        tr_n_per_class = 0 if (sel.tr_col_indices is None) else np.size(sel.tr_col_indices)
        tr_out_n_per_class = 0 if (sel.tr_out_col_indices is None) else np.size(sel.tr_out_col_indices)
        val_n_per_class = 0 if (sel.val_col_indices is None) else np.size(sel.val_col_indices)
        val_out_n_per_class = 0 if (sel.val_out_col_indices is None) else np.size(sel.val_out_col_indices)
        te_n_per_class = 0 if (sel.te_col_indices is None) else np.size(sel.te_col_indices)

        # Set defaults for class range (tr or tr, val, te)
        cls_range = np.arange(nndb.cls_n) if (sel.class_range is None) else sel.class_range

        # Set defaults for other class ranges (val, te)
        val_cls_range = cls_range if (sel.val_class_range is None) else sel.val_class_range
        te_cls_range =  cls_range if (sel.te_class_range is None) else sel.te_class_range

        # NOTE: TODO: Whitening the root db did not perform well
        # (co-variance is indeed needed)

        # Initialize NNdb lists
        nndbs_tr = DbSlice.init_nndb('Training', nndb.db.dtype, sel, nndb, tr_n_per_class, cls_range.size, True)  # noqa E501
        nndbs_tr_out = DbSlice.init_nndb('Cannonical', nndb.db.dtype, sel, nndb, tr_out_n_per_class, cls_range.size, False)  # noqa E501
        nndbs_val = DbSlice.init_nndb('Validation', nndb.db.dtype, sel, nndb, val_n_per_class, val_cls_range.size, True)  # noqa E501
        nndbs_val_out = DbSlice.init_nndb('ValCannonical', nndb.db.dtype, sel, nndb, val_out_n_per_class, val_cls_range.size, False)  # noqa E501
        nndbs_te = DbSlice.init_nndb('Testing', nndb.db.dtype, sel, nndb, te_n_per_class, te_cls_range.size, True)  # noqa E501
        
        # Fetch iterative range
        data_range = DbSlice.get_data_range(cls_range, val_cls_range, te_cls_range, 
                                                sel.tr_col_indices, sel.tr_out_col_indices,
                                                sel.val_col_indices, sel.val_out_col_indices,
                                                sel.te_col_indices, nndb)
           
        # Iterate over the cls_st indices
        i_cls = 1

        # Patch count
        patch_loop_max_n = 1 if (sel.nnpatches is None) else len(sel.nnpatches)

        # Initialize the indices
        tr_idxs = np.zeros(patch_loop_max_n, 'uint16')
        tr_out_idxs = np.zeros(patch_loop_max_n, 'uint16')
        val_idxs = np.zeros(patch_loop_max_n, 'uint16')
        val_out_idxs = np.zeros(patch_loop_max_n, 'uint16')
        te_idxs = np.zeros(patch_loop_max_n, 'uint16')

        # PERF: Noise required indices (avoid find in each iteration)
        # nf = find(sel.tr_noise_mask == 1)
        nf = np.where(sel.tr_noise_mask == 1)[0]

        # Pre-processing params struct
        pp_params = namedtuple('pp_params', ['histeq', 'normalize', 'histmatch', 'cann_img'])  # noqa E501


        # TODO: Use this iterator
        ## Initialize class ranges and column ranges
        ## DEPENDANCY -The order must be preserved as per the enum Dataset 
        ## REF_ORDER: [TR=0, VAL=1, TE=2, TR_OUT=3, VAL_OUT=4]
        #cls_ranges = [sel.cls_range, 
        #                sel.val_cls_range, 
        #                sel.te_cls_range, 
        #                sel.cls_range, 
        #                sel.val_cls_range]

        #col_ranges = [sel.tr_col_indices, 
        #                sel.val_col_indices, 
        #                sel.te_col_indices, 
        #                sel.tr_out_col_indices, 
        #                sel.val_out_col_indices]

        #DbSlice.set_default_ranges(0, cls_ranges, col_ranges)
        #self.data_generator.init(cls_ranges, col_ranges)

        ## [PERF] Iterate through the choset subset of the disk|nndb database
        #for cimg, has_cls_changed, cls_idx, col_idx, is_in_x_data in self.data_generator:

        for i in data_range:

            # Update current class index 'i_cls'
            # Since 'i' may not be consecutive
            while ((nndb.cls_st.size > i_cls+1) and
                    (i >= nndb.cls_st[i_cls+1])):
                i_cls = i_cls + 1

            # Handling the boundary condition and determine cls_boundary.
            # Refer Dbslice.dicard_needed() -> Dbslice.find() method for more details.
            # <number> % 0 => ZeroDivisionError: integer division or modulo by zero (unlike matlab).
            if (nndb.cls_st[i_cls] == 0):
                if (nndb.cls_st.size <= i_cls+1):
                    # When only 1 class is available in nndb
                    cls_boundary = nndb.cls_st[i_cls] + nndb.n_per_class[i_cls]
                else:
                    cls_boundary = nndb.cls_st[i_cls+1]
                    
            else:
                cls_boundary = nndb.cls_st[i_cls]

            # Image index 'i' in data_range may not belong to one of 'tr', 'val', 'te', etc when 
            # sel.tr_col_indices = [0 1]
            # sel.val_col_indices = [2 3]
            # sel.cls_range = [0 1]
            # sel.val_cls_range = [2 3]
            # TODO: detect this change in data_range() and remove the following code block
            p = not DbSlice.dicard_needed(nndbs_tr, i, i_cls, cls_range, cls_boundary, sel.tr_col_indices)[0]
            p = p | (not DbSlice.dicard_needed(nndbs_tr_out, i, i_cls, cls_range, cls_boundary, sel.tr_out_col_indices)[0])
            p = p | (not DbSlice.dicard_needed(nndbs_val, i, i_cls, val_cls_range, cls_boundary, sel.val_col_indices)[0])
            p = p | (not DbSlice.dicard_needed(nndbs_val_out, i, i_cls, val_cls_range, cls_boundary, sel.val_out_col_indices)[0])
            p = p | (not DbSlice.dicard_needed(nndbs_te, i, i_cls, te_cls_range, cls_boundary, sel.te_col_indices)[0])
            if (not p): continue  # noqa E701

            # Color image
            cimg = nndb.get_data_at(i)

            # Iterate through image patches
            for pI in range(patch_loop_max_n):

                # Holistic image (by default)
                img = cimg

                # Init variables
                if (sel.nnpatches is not None):
                    nnpatch = sel.nnpatches[pI];
                    x = nnpatch.offset[1];
                    y = nnpatch.offset[0];                                                                                
                    w = nnpatch.w;
                    h = nnpatch.h;
                        
                    # Extract the patch
                    img = cimg[y:y+h, x:x+w, :]

                # Peform image operations only if db format comply them
                if (nndb.format == Format.H_W_CH_N):

                    # Perform resize
                    if (sel.scale is not None):
                        img = scipy.misc.imresize(img, sel.scale)

                    # Perform histrogram matching against the cannonical image
                    cls_st_img = None
                    if (sel.histmatch):
                        if (sel.scale is not None):
                            cls_st = cls_boundary + 1
                            cls_st_img = scipy.misc.imresize(nndb.get_data_at(cls_st), sel.scale)  # noqa E501

                    # Color / Gray Scale Conversion (if required)
                    img = DbSlice.process_color(img, sel)
                    cls_st_img = DbSlice.process_color(cls_st_img, sel)

                    # Pre-Processing
                    pp_params.histeq = sel.histeq
                    pp_params.normalize = sel.normalize
                    pp_params.histmatch = sel.histmatch
                    pp_params.cann_img = cls_st_img
                    img = im_pre_process(img, pp_params)

                    # [CALLBACK] the specific pre-processing script
                    if (sel.pre_process_script is not None):
                        img = sel.pre_process_script(img)

                # Build Training DB
                [nndbs_tr, tr_idxs] =\
                    DbSlice.build_nndb_tr(nndbs_tr, pI, tr_idxs, i, i_cls, cls_range, cls_boundary, img, sel, nf)  # noqa E501

                # Build Training Output DB
                [nndbs_tr_out, tr_out_idxs] =\
                    DbSlice.build_nndb_tr_out(nndbs_tr_out, pI, tr_out_idxs, i, i_cls, cls_range, cls_boundary, img, sel)  # noqa E501

                # Build Valdiation DB
                [nndbs_val, val_idxs] =\
                    DbSlice.build_nndb_val(nndbs_val, pI, val_idxs, i, i_cls, val_cls_range, cls_boundary, img, sel)  # noqa E501

                # Build Valdiation Target DB
                [nndbs_val_out, val_out_idxs] =\
                    DbSlice.build_nndb_val_out(nndbs_val_out, pI, val_out_idxs, i, i_cls, val_cls_range, cls_boundary, img, sel)  # noqa E501

                # Build Testing DB
                [nndbs_te, te_idxs] =\
                    DbSlice.build_nndb_te(nndbs_te, pI, te_idxs, i, i_cls, te_cls_range, cls_boundary, img, sel)  # noqa E501

        # Returns NNdb object instead of cell array (non patch requirement)
        if (sel.nnpatches is None):
            nndbs_tr = None if (nndbs_tr is None) else nndbs_tr[0]  # noqa E701
            nndbs_val = None if (nndbs_val is None) else nndbs_val[0]  # noqa E701            
            nndbs_te = None if (nndbs_te is None) else nndbs_te[0]  # noqa E701
            nndbs_tr_out = None if (nndbs_tr_out is None) else nndbs_tr_out[0]  # noqa E701            
            nndbs_val_out = None if (nndbs_val_out is None) else nndbs_val_out[0]  # noqa E701

        return (nndbs_tr, nndbs_val, nndbs_te, nndbs_tr_out, nndbs_val_out)

    @staticmethod
    def set_default_ranges(default_idx,  cls_ranges, col_ranges):
        for rng_idx in range(len(col_ranges)):
            if (col_ranges[rng_idx] is not None and 
                cls_ranges[rng_idx] is None):
                cls_ranges[rng_idx] = cls_ranges[default_idx]


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
        [nndb_tr, _, _, _] = DbSlice.slice(nndb, sel) # nndb_tr = DbSlice.slice(nndb, sel) # noqa E501


        #
        # Select 1st 2nd 4th images of each identity for training.
        # Divide into patches
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.nnpatches = NNPatch.GeneratePatch(nndb, 33, 33, 33, 33)

        # Cell arrays of NNdb objects for each patch
        [nndbs_tr, _, _, _] = DbSlice.slice(nndb, sel)
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
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)


        #
        # Select 1st 2nd 4th images of identities denoted by class_range for training. # noqa E501
        # Select 3rd 5th images of identities denoted by class_range for testing. # noqa E501
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.class_range = np.arange(10)                         # First ten identities # noqa E501
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)

             
            
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
        [nndb_tr, nndb_val, nndb_te, _] = DbSlice.slice(nndb, sel);
            
            
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
        [nndb_tr, nndb_val, nndb_te, _] = DbSlice.slice(nndb, sel)
            
            
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
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)


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
        [nndb_tr, nndb_tr_out, nndb_te, _] = DbSlice.slice(nndb, sel)


        #
        # Resize images by 0.5 scale factor.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.scale = 0.5
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)


        #
        # Use gray scale images.
        # Perform histrogram equalization.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.histeq = True
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)


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
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)


        #
        # If imdb_8 supports many color channels
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.color_index = 5  # color channel denoted by 5th index
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)

    ##########################################################################
    # Private Interface
    ##########################################################################
    @staticmethod
    def init_nndb(name, type, sel, nndb, n_per_class, cls_n, build_cls_idx):
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
                    db.append(np.zeros((nnpatch.h, nnpatch.w, ch, n_per_class*cls_n), dtype=type))
            else:
                db.append(np.zeros((nd1, nd2, ch, n_per_class*cls_n), dtype=type))

        elif (nndb.format == Format.H_N):
            nd1 = nndb.h
            db.append(np.zeros((nd1, n_per_class*cls_n), dtype=type))

        new_nndbs = []

        # Init nndb for each patch
        for i in range(len(db)):
            new_nndbs.append(NNdb(name, db[i], n_per_class, build_cls_idx))
 
        return new_nndbs

    @staticmethod
    def get_data_range(cls_range, val_cls_range, te_cls_range, 
                        tr_col, tr_out_col, val_col, val_out_col, te_col, nndb):
        """Fetch the data_range (images indices).

        Returns
        -------
        img : vector -uint32
            data indicies.
        """
        union = np.union1d
        intersect = np.intersect1d

        # Union of all class ranges
        cls_range = union(union(cls_range, val_cls_range), te_cls_range)

        # Union of all col indices        
        col_indices = union(union(union(union(tr_col, tr_out_col), val_col), val_out_col), te_col)

        # Class count
        cls_n = np.size(cls_range)

        # *Ease of implementation
        # Allocate more memory, shrink it later
        data_range = np.zeros(cls_n * np.size(col_indices), dtype='uint32')

        st = 0
        for i in range(cls_n):
            ii = cls_range[i]
            cls_offset = nndb.cls_st[ii]
            data_range[st:st+np.size(col_indices)] = col_indices.astype('uint32') + dst;               
            st = st + np.size(col_indices) 

        # Shrink the vector (Can safely ignore this code)
        data_range = np.resize(data_range,(st))
        return data_range

    @staticmethod
    def process_color(img, sel):
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
    def build_nndb_tr(nndbs, pi, ni, i, i_cls, cls_range, prev_cls_en, img, sel, noise_found):
        """Build the nndb training database.

        Returns
        -------
        nndbs : cell- NNdb
            Updated NNdb objects.

        ni : vector -uint16
            Updated image count vector.
        """
        if (nndbs is None): return nndbs, ni  # noqa E701
        nndb = nndbs[pi]

        # Find whether 'i' is in required indices
        [discard, found] = DbSlice.dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, sel.tr_col_indices)
        if (discard): return nndbs, ni  # noqa E701       

        # Iterate over found indices
        for j in range(np.size(found)):

            # Check whether found contain a noise required index
            if ((np.where(noise_found == found[j])[0]).shape[0]):

                # Currently supports noise for images only
                if (nndb.format != Format.H_W_CH_N):
                    nndb.set_data_at(img, ni[pi])
                    continue

                h, w, ch = img.shape

                # Fetch noise rate
                rate = sel.tr_noise_rate[found[j]]

                # Add different noise depending on the type or rate
                # (ref. Enums/Noise)
                if (rate == Noise.G):
                    pass
                    # img = imnoise(img, 'gaussian')
                    # #img = imnoise(img, 'gaussian')
                    # #img = imnoise(img, 'gaussian')
                    # #img = imnoise(img, 'gaussian')

                else:
                    # Perform random corruption
                    # Corruption Size (H x W)
                    cs = [np.uint16(h*rate), np.uint16(w*rate)]

                    # Random location choice
                    # Start of H, W (location)
                    sh = np.uint8(1 + rand()*(h-cs[0]-1))
                    sw = np.uint8(1 + rand()*(w-cs[1]-1))

                    # Set the corruption
                    cimg = np.uint8(DbSlice.rand_corrupt(cs[0], cs[1])).astype('uint8')  # noqa E501

                    if (ch == 1):
                        img[sh:sh+cs[1], sw:sw+cs[2]] = cimg
                    else:
                        for ich in range(ch):
                            img[sh:sh+cs[0], sw:sw+cs[1], ich] = cimg

                nndb.set_data_at(img, ni[pi])
            else:
                nndb.set_data_at(img, ni[pi])
    
            ni[pi] = ni[pi] + 1

        return nndbs, ni

    @staticmethod
    def rand_corrupt(height, width):
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
    def build_nndb_tr_out(nndbs, pi, ni, i, i_cls, cls_range, prev_cls_en, img, sel):
        """Build the nndb training target database.
                    
        Returns
        -------
        nndbs : cell- NNdb
            Updated NNdb objects.
        
        ni : vector -uint16
            Updated image count vector.
        """
        if (nndbs is None): return nndbs, ni
        nndb = nndbs[pi]

        # Find whether 'i' is in required indices
        [discard, found] = DbSlice.dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, sel.tr_out_col_indices)
        if (discard): return nndbs, ni  # noqa E701   

        # Iterate over found indices
        for j in range(np.size(found)):
            nndb.set_data_at(img, ni[pi])
            ni[pi] = ni[pi] + 1

        return nndbs, ni

    @staticmethod
    def build_nndb_val(nndbs, pi, ni, i, i_cls, cls_range, prev_cls_en, img, sel):
        """"Build the nndb validation database.
                    
        Returns
        -------
        nndbs : cell- NNdb
            Updated NNdb objects.
        
        ni : vector -uint16
            Updated image count vector.
        """
        if (nndbs is None): return nndbs, ni
        nndb = nndbs[pi]

        # Find whether 'i' is in required indices
        [discard, found] = DbSlice.dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, sel.val_col_indices)
        if (discard): return nndbs, ni  # noqa E701 

        # Iterate over found indices
        for j in range(np.size(found)):
            nndb.set_data_at(img, ni[pi])
            ni[pi] = ni[pi] + 1

        return nndbs, ni

    @staticmethod
    def build_nndb_val_out(nndbs, pi, ni, i, i_cls, cls_range, prev_cls_en, img, sel):
        """"Build the nndb validation target database.
                    
        Returns
        -------
        nndbs : cell- NNdb
            Updated NNdb objects.
        
        ni : vector -uint16
            Updated image count vector.
        """
        if (nndbs is None): return nndbs, ni
        nndb = nndbs[pi]

        # Find whether 'i' is in required indices
        [discard, found] = DbSlice.dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, sel.val_out_col_indices)
        if (discard): return nndbs, ni  # noqa E701 

        # Iterate over found indices
        for j in range(np.size(found)):
            nndb.set_data_at(img, ni[pi])
            ni[pi] = ni[pi] + 1

        return nndbs, ni

    @staticmethod
    def build_nndb_te(nndbs, pi, ni, i, i_cls, cls_range, prev_cls_en, img, sel):
        """Build the testing database.
                    
        Returns
        -------
        nndbs : cell- NNdb
            Updated NNdb objects.
        
        ni : vector -uint16
            Updated image count vector.
        """
        if (nndbs is None): return nndbs, ni
        nndb = nndbs[pi]

        # Find whether 'i' is in required indices
        [discard, found] = DbSlice.dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, sel.te_col_indices)
        if (discard): return nndbs, ni  # noqa E701 

        # Iterate over found indices
        for j in range(np.size(found)):
            nndb.set_data_at(img, ni[pi])
            ni[pi] = ni[pi] + 1

        return nndbs, ni

    @staticmethod
    def find(i, prev_cls_en, col_indices):
        """Find whether 'i' is in required indices.

        Returns
        -------
        found : vector- uint16
            The index 'i' found positions in col_indices.
        """
        found = None
        if (col_indices is None): return found  # noqa E701

        # Find whether 'i' is in required indices
        found = np.where(((i-col_indices) % prev_cls_en == 0) == True)[0]  # noqa E712, E501
        found = None if (found.size == 0) else found
        return found

    @staticmethod
    def dicard_needed(nndbs, i, i_cls, cls_range, prev_cls_en, col_indices):
        """Find whether 'i' is in required indices.

        Returns
        -------
        discard : bool
            Boolean indicating the dicard.

        found : vector- uint16
            If discard=false, found denotes the index 'i' found positions in col_indices.
        """                        
        found = DbSlice.find(i, prev_cls_en, col_indices)
        discard = not ((nndbs is not None) and
                    (np.intersect1d(i_cls, cls_range).size != 0) and
                    (found is not None))

        return discard, found