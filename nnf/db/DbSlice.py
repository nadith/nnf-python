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


class Selection:
    """Denote the selection structure."""

    def __init__(self, **kwds):
        """Initialize a selection structure with given field-values."""
        self.__dict__.update(kwds)


class DbSlice(object):
    """Perform slicing of nndb with the help of a selection structure.

    Selection Structure (with defaults)
    -----------------------------------
    sel.tr_col_indices    = None    # Training column indices
    sel.tr_noise_mask     = None    # Noisy tr. col indices (bit mask)
    sel.tr_noise_rate     = None    # Rate or noise types for the above field
    sel.tr_out_col_indices= None    # Training target column indices
    sel.tr_cm_col_indices = None    # TODO: Document
    sel.te_col_indices    = None    # Testing column indices
    sel.use_rgb           = True    # Use rgb or convert to grayscale
    sel.color_index       = None    # Specific color indices (set .use_rgb = false)
    sel.use_real          = False   # Use real valued database TODO: (if .normalize = true, Operations ends in real values)  # noqa E501
    sel.scale             = None    # Scaling factor (resize factor)
    sel.normalize         = False   # Normalize (0 mean, std = 1)
    sel.histeq            = False   # Histogram equalization
    sel.histmatch         = False   # Histogram match (ref. image: first image of the class)  # noqa E501
    sel.class_range       = None    # Class range
    sel.pre_process_script= None    # Custom preprocessing script

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
            sel.tr_col_indices    = None    # Training column indices
            sel.tr_noise_mask     = None    # Noisy tr. col indices (bit mask)
            sel.tr_noise_rate     = None    # Rate or noise types for the above field
            sel.tr_out_col_indices= None    # Training target column indices
            sel.tr_cm_col_indices = None    # TODO: Document
            sel.te_col_indices    = None    # Testing column indices
            sel.use_rgb           = True    # Use rgb or convert to grayscale
            sel.color_index       = None    # Specific color indices (set .use_rgb = false)             
            sel.use_real          = False   # Use real valued database TODO: (if .normalize = true, Operations ends in real values)  # noqa E501
            sel.scale             = None    # Scaling factor (resize factor)
            sel.normalize         = False   # Normalize (0 mean, std = 1)
            sel.histeq            = False   # Histogram equalization
            sel.histmatch         = False   # Histogram match (ref. image: first image of the class)  # noqa E501
            sel.class_range       = None    # Class range
            sel.pre_process_script= None    # Custom preprocessing script

        Returns
        -------
        nndb_tr : NNdb
             Training dataset.

        nndb_tr_out : NNdb
             Training target dataset.

        nndb_te : NNdb
             Testing target dataset.

        nndb_tr_cm : NNdb
             TODO: Yet to be documented.

        Notes
        -----
        The new dbs returned will contain img at consecutive locations for
        duplicate indices irrespective of the order that is mentioned.
        i.e Tr:[1 2 3 1], DB:[1 1 2 3]

        Copyright 2015-2016 Nadith Pathirage, Curtin University.
        (chathurdara@gmail.com).

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
            sel.tr_cm_col_indices = None
            sel.te_col_indices = None
            sel.use_rgb = True
            sel.color_index = None
            sel.use_real = False
            sel.scale = None
            sel.normalize = False
            sel.histeq = False
            sel.histmatch = False
            sel.class_range = None
            sel.pre_process_script = None

        # Set defaults for selection fields, if the field does not exist
        if (not hasattr(sel, 'tr_noise_mask')): sel.tr_noise_mask = None  # noqa E701
        if (not hasattr(sel, 'tr_noise_rate')): sel.tr_noise_rate = None  # noqa E701
        if (not hasattr(sel, 'tr_out_col_indices')): sel.tr_out_col_indices = None  # noqa E501, E701
        if (not hasattr(sel, 'tr_cm_col_indices')): sel.tr_cm_col_indices = None  # noqa E501, E701
        if (not hasattr(sel, 'te_col_indices')): sel.te_col_indices = None  # noqa E701
        if (not hasattr(sel, 'use_rgb')): sel.use_rgb = True  # noqa E701
        if (not hasattr(sel, 'color_index')): sel.color_index = None  # noqa E701
        if (not hasattr(sel, 'use_real')): sel.use_real = False  # noqa E701
        if (not hasattr(sel, 'resize')): sel.resize = None  # noqa E701
        if (not hasattr(sel, 'normalize')): sel.normalize = False  # noqa E701
        if (not hasattr(sel, 'histeq')): sel.histeq = False  # noqa E701
        if (not hasattr(sel, 'histmatch')): sel.histmatch = False  # noqa E701
        if (not hasattr(sel, 'class_range')): sel.class_range = None  # noqa E701
        if (not hasattr(sel, 'pre_process_script')): sel.pre_process_script = None  # noqa E501, E701

        # Error handling for arguments
        if (sel.tr_col_indices is None and
           sel.tr_out_col_indices is None and
           sel.tr_cm_col_indices is None and
           sel.te_col_indices is None):
            raise Exception('ARG_ERR: [tr|tr_out|tr_cm|te]_col_indices: mandory field')  # noqa E501

        if (sel.use_rgb and
           sel.color_index is not None):
            raise Exception('ARG_CONFLICT: sel.use_rgb, sel.color_index')

        if (sel.tr_noise_mask is not None and
           sel.tr_noise_rate is None):
            raise Exception('ARG_MISSING: specify sel.tr_noise_rate field')

        # Fetch the counts
        tr_n_per_class = np.size(sel.tr_col_indices)
        tr_out_n_per_class = np.size(sel.tr_out_col_indices)
        tr_cm_n_per_class = np.size(sel.tr_cm_col_indices)
        te_n_per_class = np.size(sel.te_col_indices)

        cls_range = sel.class_range
        if (cls_range is None): cls_range = np.arange(nndb.cls_n)  # noqa E701

        # NOTE: TODO: Whitening the root db did not perform well
        # (co-variance is indeed needed)

        # Initialize NNdb Objects
        cls_n = np.size(cls_range)
        nndb_tr = DbSlice.init_nndb('Training', nndb.db.dtype, sel, nndb, tr_n_per_class, cls_n, True)  # noqa E501
        nndb_te = DbSlice.init_nndb('Testing', nndb.db.dtype, sel, nndb, te_n_per_class, cls_n, True)  # noqa E501
        nndb_tr_out = DbSlice.init_nndb('Cannonical', nndb.db.dtype, sel, nndb, tr_out_n_per_class, cls_n, False)  # noqa E501
        nndb_tr_cm = DbSlice.init_nndb('Cluster-Means', nndb.db.dtype, sel, nndb, tr_cm_n_per_class, cls_n, False)  # noqa E501

        # Fetch iterative range
        data_range = DbSlice.get_data_range(cls_range, nndb)

        # Iterate over the cls_st indices
        j = 1

        # Initialize the indices
        tr_idx = 0; tr_out_idx = 0; tr_cm_idx = 0  # noqa E702
        te_idx = 0  # noqa E702

        # PERF: Noise required indices (avoid find in each iteration)
        # nf = find(sel.tr_noise_mask == 1)
        nf = np.where(sel.tr_noise_mask == 1)[0]

        # Pre-processing params struct
        pp_params = namedtuple('pp_params', ['histeq', 'normalize', 'histmatch', 'cann_img'])  # noqa E501

        for i in data_range:
            img = nndb.get_data_at(i)

            # Update the current prev_cls_en
            # Since 'i' may not be consecutive
            while ((np.size(nndb.cls_st) >= j+2) and
                    (i >= nndb.cls_st[j+1])):
                j = j + 1
            prev_cls_en = nndb.cls_st[j]

            # Checks whether current 'img' needs processing
            f = DbSlice.find(nndb_tr, i, prev_cls_en, sel.tr_col_indices)
            f = (np.size(f)!=0) | (np.size(DbSlice.find(nndb_tr_out, i, prev_cls_en, sel.tr_out_col_indices)) != 0)  # noqa E501, E701
            f = (np.size(f)!=0) | (np.size(DbSlice.find(nndb_tr_cm, i, prev_cls_en, sel.tr_cm_col_indices)) != 0)  # noqa E501, E701
            f = (np.size(f)!=0) | (np.size(DbSlice.find(nndb_te, i, prev_cls_en, sel.te_col_indices)) != 0)  # noqa E501, E701
            if (not f): continue  # noqa E701

            # Peform image operations only if db format comply them
            if (nndb.format == Format.H_W_CH_N):

                # Perform resize
                if (sel.scale is not None):
                    img = scipy.misc.imresize(img, sel.scale)

                # Perform histrogram matching against the cannonical image
                cls_st_img = None
                if (sel.histmatch):
                    if (sel.scale is not None):
                        cls_st = prev_cls_en + 1
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
            [nndb_tr, tr_idx] =\
                DbSlice.build_nndb_tr(nndb_tr, tr_idx, i, prev_cls_en, img, sel, nf)  # noqa E501

            # Build Training Output DB
            [nndb_tr_out, tr_out_idx] =\
                DbSlice.build_nndb_tr_out(nndb_tr_out, tr_out_idx, i, prev_cls_en, img, sel)  # noqa E501

            # Build Training Cluster Centers (For Multi Label DDA) DB
            [nndb_tr_cm, tr_cm_idx] =\
                DbSlice.build_nndb_tr_cm(nndb_tr_cm, tr_cm_idx, i, prev_cls_en, img, sel)  # noqa E501

            # Build Testing DB
            [nndb_te, te_idx] =\
                DbSlice.build_nndb_te(nndb_te, te_idx, i, prev_cls_en, img, sel)  # noqa E501

        return (nndb_tr, nndb_tr_out, nndb_te, nndb_tr_cm)

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

        Notes
        ------
        Copyright 2015-2016 Nadith Pathirage, Curtin University.
        (chathurdara@gmail.com).
        """
        #
        # Select 1st 2nd 4th images of each identity for training.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([1, 2, 4])
        [nndb_tr, _, _, _] = DbSlice.slice(nndb, sel) # nndb_tr = DbSlice.slice(nndb, sel) # noqa E501

        #
        # Select 1st 2nd 4th images of each identity for training.
        # Select 3rd 5th images of each identity for testing.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([1, 2, 4])
        sel.te_col_indices = np.array([3, 5])
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)

        #
        # Select 1st 2nd 4th images of identities denoted by class_range for training. # noqa E501
        # Select 3rd 5th images of identities denoted by class_range for testing. # noqa E501
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([1, 2, 4])
        sel.te_col_indices = np.array([3, 5])
        sel.class_range = np.arange(10)                         # First then identities # noqa E501
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)

        #
        # Select 1st 2nd 4th images of each identity for training +
        #               add various noise types @ random locations of varying degree. # noqa E501
        #               default noise type: random black and white dots.
        # Select 3rd 5th images of each identity for testing.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([1, 2, 4])
        sel.tr_noise_mask = np.array([0, 1, 0, 1, 0, 1])       # index is affected or not # noqa E501
        sel.tr_noise_rate = np.array([0, 0.5, 0, 0.5, 0, 0.5]) # percentage of corruption # noqa E501
        # sel.tr_noise_rate  = [0 0.5 0 0.5 0 Noise.G]         # last index with Gauss noise # noqa E501
        sel.tr_col_indices = np.array([1, 2, 4])
        sel.te_col_indices = np.array([3, 5])
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)

        #
        # To prepare regression datasets, training dataset and training target dataset # noqa E501
        # Select 1st 2nd 4th images of each identity for training.
        # Select 1st 1st 1st image of each identity for corresponding training target # noqa E501
        # Select 3rd 5th images of each identity for testing.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([1, 2, 4])
        sel.tr_out_col_indices = np.array([1, 1, 1])  # [1 1 1] (Regression 1->1, 2->1, 4->1) # noqa E501
        sel.te_col_indices = np.array([3, 5])
        [nndb_tr, nndb_tr_out, nndb_te, _] = DbSlice.slice(nndb, sel)

        #
        # Resize images by 0.5 scale factor.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([1, 2, 4])
        sel.te_col_indices = np.array([3, 5])
        sel.scale = 0.5
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)

        #
        # Use gray scale images.
        # Perform histrogram equalization.
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([1, 2, 4])
        sel.te_col_indices = np.array([3, 5])
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
        sel.tr_col_indices = np.array([1, 2, 4])
        sel.te_col_indices = np.array([3, 5])
        sel.use_rgb = False
        sel.histmatch = True
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)

        #
        # If imdb_8 supports many color channels
        nndb = NNdb('original', imdb_8, 8, True)
        sel = Selection()
        sel.tr_col_indices = np.array([1, 2, 4])
        sel.te_col_indices = np.array([3, 5])
        sel.use_rgb = False
        sel.color_index = 5  # color channel denoted by 5th index
        [nndb_tr, _, nndb_te, _] = DbSlice.slice(nndb, sel)

    ##########################################################################
    # Private Interface
    ##########################################################################
    @staticmethod
    def init_nndb(name, type, sel, nndb, n_per_class, cls_n, build_cls_idx):
        """Initialize a new empty 4D tensor database with a nndb object."""
        # n_per_class: must be a scalar
        if (n_per_class == 0): return  # noqa E701

        # Peform image operations only if db format comply them
        if (nndb.format == Format.H_W_CH_N or
           nndb.format == Format.H_W_CH_N_NP):

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
                ch = np.size(sel.color_index)  # Selected color channels
                if (ch == 0): ch = 1  # Grayscale  # noqa E701

            if (nndb.format == Format.H_W_CH_N):
                db = np.zeros((nd1, nd2, ch, n_per_class*cls_n), dtype=type)

            else:  # (nndb.format == Format.H_W_CH_N_NP)
                assert(False)  # TODO: Implement
                db = np.zeros((nd1, nd2, ch, n_per_class*cls_n, nndb.p), dtype=type)  # noqa E501

        elif (nndb.format == Format.H_N):
            nd1 = nndb.h
            db = np.zeros((nd1, n_per_class*cls_n), dtype=type)

        elif (nndb.format == Format.H_N_NP):
            nd1 = nndb.h
            assert(False)  # TODO: Implement
            db = np.zeros((nd1, n_per_class*cls_n, nndb.p), dtype=type)

        return NNdb(name, db, n_per_class, build_cls_idx)

    @staticmethod
    def get_data_range(cls_range, nndb):
        """Fetch the data_range (images indices)."""
        # Class count
        cls_n = np.size(cls_range)

        # *Ease of implementation
        # Allocate more memory, shrink it later
        data_range = np.zeros(cls_n * max(nndb.n_per_class), dtype='uint32')

        # TODO: Compatibility for other NNdb Formats. (Hx W x CH x N x NP)
        st = 0
        for i in range(cls_n):
            ii = cls_range[i]
            dst = nndb.cls_st[ii]
            data_range[st:st+nndb.n_per_class[ii]] =\
                np.arange(dst, dst + nndb.n_per_class[ii]).astype('uint32')
            st = st + nndb.n_per_class[ii]

        # Shrink the vector
        data_range[st:] = []
        return data_range

    @staticmethod
    def process_color(img, sel):
        """Perform color related functions."""
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
    def build_nndb_tr(nndb, db_idx, i, prev_cls_en, img, sel, noise_found):
        """Build the nndb training database."""
        # Find whether 'i' is in required indices
        found = DbSlice.find(nndb, i, prev_cls_en, sel.tr_col_indices)
        if (found is None): return nndb, db_idx  # noqa E701

        # Iterate over found indices
        for j in range(np.size(found)):

            # Check whether found contain a noise required index
            if ((np.where(noise_found == found[j])[0]).shape[0]):

                # Currently supports noise for images only
                if (nndb.format != Format.H_W_CH_N):
                    nndb.set_data_at(img, db_idx)
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

                nndb.set_data_at(img, db_idx)
            else:
                nndb.set_data_at(img, db_idx)

            db_idx = db_idx + 1

        return nndb, db_idx

    @staticmethod
    def rand_corrupt(height, width):
        """Corrupt the image with a (height, width) block."""
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
    def build_nndb_tr_out(nndb, db_idx, i, prev_cls_en, img, sel):
        """Build the nndb training target database."""
        # Find whether 'i' is in required indices
        found = DbSlice.find(nndb, i, prev_cls_en, sel.tr_out_col_indices)
        if (found is None): return nndb, db_idx  # noqa E701

        # Iterate over found indices
        for j in range(np.size(found)):
            nndb.set_data_at(img, db_idx)
            db_idx = db_idx + 1

        return nndb, db_idx

    @staticmethod
    def build_nndb_tr_cm(nndb, db_idx, i, prev_cls_en, img, sel):
        """"Build the nndb training  mean centre database."""
        # Find whether 'i' is in required indices
        found = DbSlice.find(nndb, i, prev_cls_en, sel.tr_cm_col_indices)
        if (found is None): return nndb, db_idx  # noqa E701

        # Iterate over found indices
        for j in range(np.size(found)):
            nndb.set_data_at(img, db_idx)
            db_idx = db_idx + 1

        return nndb, db_idx

    @staticmethod
    def build_nndb_te(nndb, db_idx, i, prev_cls_en, img, sel):
        """Build the testing database."""
        # Find whether 'i' is in required indices
        found = DbSlice.find(nndb, i, prev_cls_en, sel.te_col_indices)
        if (found is None): return nndb, db_idx  # noqa E701

        # Iterate over found indices
        for j in range(np.size(found)):
            nndb.set_data_at(img, db_idx)
            db_idx = db_idx + 1

        return nndb, db_idx

    @staticmethod
    def find(nndb, i, prev_cls_en, col_indices):
        """Find whether 'i' is in required indices."""
        found = None
        if (nndb is None or col_indices is None): return found  # noqa E701

        # Find whether 'i' is in required indices
        found = np.where(((i-col_indices) % prev_cls_en == 0) == True)[0]  # noqa E712, E501

        return found
