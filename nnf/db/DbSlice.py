# -*- coding: utf-8 -*- TODO: Revisit the comments
"""
.. module:: DbSlice
   :platform: Unix, Windows
   :synopsis: Represent DbSlice class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import math
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
from nnf.db.Dataset import Dataset
from nnf.core.iters.memory.DskmanMemDataIterator import DskmanMemDataIterator
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator

class DbSlice(object):
    """Perform slicing of nndb with the help of a selection structure.

    Attributes
    ----------
    Selection Structure (with defaults)
    -----------------------------------
    sel.tr_col_indices      = None    # Training column indices
    sel.tr_noise_rate       = None    # Noise rate or Noise types for `tr_col_indices`
    sel.tr_occlusion_rate   = None    # Occlusion rate for `tr_col_indices`
    sel.tr_occlusion_type   = None    # Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right) for `tr_col_indices`
    sel.tr_occlusion_offset = None    # Occlusion start offset from top/bottom/left/right corner depending on `tr_occlusion_type`
    sel.tr_out_col_indices  = None    # Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right) for `tr_col_indices`
    sel.val_col_indices     = None    # Validation column indices
    sel.val_out_col_indices = None    # Validation target column indices
    sel.te_col_indices      = None    # Testing column indices
    sel.te_out_col_indices  = None    # Testing target column indices
    sel.nnpatches           = None    # NNPatch object array
    sel.use_rgb             = None    # Use rgb or convert to grayscale
    sel.color_indices       = None    # Specific color indices (set .use_rgb = false)
    sel.use_real            = False   # Use real valued database TODO: (if .normalize = true, Operations ends in real values)  # noqa E501
    sel.scale               = None    # Scaling factor (resize factor)
    sel.normalize           = False   # Normalize (0 mean, std = 1)
    sel.histeq              = False   # Histogram equalization
    sel.histmatch_col_index = None    # Histogram match reference column index
    sel.class_range         = None    # Class range for training database or all (tr, val, te)
    sel.val_class_range     = None    # Class range for validation database
    sel.te_class_range      = None    # Class range for testing database
    sel.pre_process_script  = None    # Custom preprocessing script

    i.e
    Pass a selection structure to split the nndb accordingly
    [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)

    Notes
    -----
    All methods are implemented via static methods thus thread-safe.
    """
    ##########################################################################
    # Public Interface
    ##########################################################################
    def safe_slice(nndb, sel=None):
        """Thread Safe

        Parameters
        ----------
        nndb : describe
            descriebe.

        sel : selection structure
            Information to split the dataset.

        Returns
        -------
        """
        DbSlice.slice(nndb, sel, DskmanMemDataIterator(nndb))

    @staticmethod
    def slice(nndb, sel=None, data_generator=None, pp_param=None):
        """Slice the database according to the selection structure.

        Parameters
        ----------
        nndb : NNdb
            NNdb object that represents the dataset.

        sel : selection structure
            Information to split the dataset. (Ref: class documentation)

        Returns
        -------
        :obj:`NNdb` or list of :obj:`NNdb`
             nndbs_tr: Training NNdb instance or instances for each nnpatch.

        :obj:`NNdb` or list of :obj:`NNdb`
             nndb_tr_out: Training target NNdb instance or instances for each nnpatch.

        :obj:`NNdb` or list of :obj:`NNdb`
             nndbs_val: Validation NNdb instance or instances for each nnpatch.

        :obj:`NNdb` or list of :obj:`NNdb`
             nndbs_te: Testing NNdb instance or instances for each nnpatch.

        :obj:`NNdb` or list of :obj:`NNdb`
            nndbs_tr_out: Training target NNdb instance or instances for each nnpatch.
        
        :obj:`NNdb` or list of :obj:`NNdb`
            nndbs_val_out: Validation target NNdb instance or instances for each nnpatch.

        :obj:`NNdb` or list of :obj:`NNdb`
            nndbs_te_out: Testing target NNdb instance or instances for each nnpatch.

        obj:`list` : NNdb or cell- NNdb
            List of dataset enums. Dataset order returned above.

        Notes
        -----
        The new dbs returned will contain img at consecutive locations for
        duplicate indices irrespective of the order that is mentioned.
        i.e Tr:[1 2 3 1], DB:[1 1 2 3]
        """
        # Set defaults for arguments
        if (sel is None):
            sel = Selection()
            unq_n_per_cls = np.unique(nndb.n_per_class)

            if (np.isscalar(unq_n_per_cls)):
                sel.tr_col_indices = np.arange(unq_n_per_cls)

            else:
                raise Exception('Selection is not provided.' + 
                                ' tr_col_indices = nndb.n_per_class' + 
                                    ' must be same for all classes')

        # Error handling for arguments
        if (sel.tr_col_indices is None and
           sel.tr_out_col_indices is None and
           sel.val_col_indices is None and
           sel.val_out_col_indices is None and
           sel.te_col_indices is None and
           sel.te_out_col_indices is None):
            raise Exception('ARG_ERR: [tr|tr_out|val|val_out|te]_col_indices: mandory field')  # noqa E501

        if ((sel.use_rgb is not None and sel.use_rgb) and
           sel.color_indices is not None):
            raise Exception('ARG_CONFLICT: sel.use_rgb, sel.color_indices')

        # Set defaults for data generator 
        if (data_generator is None):
            data_generator =  DskmanMemDataIterator(pp_param)
        data_generator.init_params(nndb)

        # Set default for sel.use_rgb
        if (sel.use_rgb is None):
            sel.use_rgb = True
            print('[DEFAULT] selection.use_rgb (None): True')
    
        # Set default for sel.class range
        sel.class_range = np.arange(nndb.cls_n) if (sel.class_range is None) else sel.class_range

        # Initialize class ranges and column ranges
        # DEPENDANCY -The order must be preserved as per the enum Dataset 
        # REF_ORDER: [TR=0, VAL=1, TE=2, TR_OUT=3, VAL_OUT=4]
        cls_ranges = [sel.class_range, 
                        sel.val_class_range, 
                        sel.te_class_range, 
                        sel.class_range, 
                        sel.val_class_range,
                        sel.te_class_range]

        col_ranges = [sel.tr_col_indices, 
                        sel.val_col_indices, 
                        sel.te_col_indices, 
                        sel.tr_out_col_indices, 
                        sel.val_out_col_indices,
                        sel.te_out_col_indices]

        # Set defaults for class ranges [val|te|tr_out|val_out]
        DbSlice._set_default_cls_range(0, cls_ranges, col_ranges)

        # NOTE: Whitening the root db did not perform well
        # (co-variance is indeed needed)

        # Initialize NNdb dictionary
        # i.e dict_nndbs[Dataset.TR] => [NNdb_Patch_1, NNdb_Patch_2, ...]
        dict_nndbs = DbSlice._init_dict_nndbs(col_ranges)
        
        # Set patch iterating loop's max count and nnpatch list
        patch_loop_max_n = 1
        nnpatch_list = None

        if (sel.nnpatches is not None):
            # Must have atleast 1 patch to represent the whole image
            nnpatch = sel.nnpatches[0]

            # PERF: If there is only 1 patch and it is the whole image
            if ((len(sel.nnpatches) == 1) and nnpatch.is_holistic):
                pass

            else:
                nnpatch_list = sel.nnpatches
                patch_loop_max_n = len(sel.nnpatches)

        # Initialize the generator
        data_generator.init(cls_ranges, col_ranges, True)

        # PERF: Iterate through the choset subset of the nndb database
        for cimg, _, cls_idx, col_idx, datasets in data_generator:

            # cimg: array_like data (maybe an image or raw data item) 
            # frecord: [fpath, fpos, cls_lbl], assert(cls_lbl == cls_idx)
            # cls_idx, col_idx: int
            # datasets: list of tuples
            #       [(Dataset.TR, True), (Dataset.TR, False), (Dataset.VAL, True), ...]
    
            # SPECIAL NOTE:  cls_lbl, cls_idx, col_idx are expressed with respect to the
            # cls_ranges, col_ranges defined above and may not consist of continuous
            # indices. i.e cls_lbl=[0 4 6 3]

            # Perform pre-processing before patch division
            # Peform image operations only if db format comply them
            if (nndb.format == Format.H_W_CH_N or nndb.format == Format.N_H_W_CH):
                    
                # For histogram equalizaion operation (cannonical image)
                cann_cimg = None                
                if (sel.histmatch_col_index is not None):
                    cann_cimg, _= data_generator._get_cimg_frecord_in_next(cls_idx, sel.histmatch_col_index)
                    # Alternative:
                    # cls_st = nndb.cls_st[sel.histmatch_col_index]
                    # cann_cimg = nndb.get_data_at(cls_st)    
                    
                # Peform image preocessing
                cimg = DbSlice.preprocess_im_with_sel(cimg, cann_cimg, sel, data_generator.get_im_ch_axis())

            # Iterate through image nnpatches
            for pi in range(patch_loop_max_n):

                # Holistic image (by default)
                pimg = cimg

                # Generate the image patch
                if (nnpatch_list is not None):
                    nnpatch = nnpatch_list[pi];
                    x = nnpatch.offset[1]
                    y = nnpatch.offset[0]
                    w = nnpatch.w
                    h = nnpatch.h
                        
                    # Extract the patch
                    if (nndb.format == Format.H_W_CH_N or nndb.format == Format.N_H_W_CH):
                        pimg = cimg[y:y+h, x:x+w, :]

                    elif (nndb.format == Format.H_N or nndb.format == Format.N_H):
                        # 1D axis is `h` if w > 1, otherwise 'w'
                        pimg = cimg[x:x+w] if (w > 1) else cimg[y:y+h]
                            

                # The offset/index for the col_index in the tr_col_indices vector
                tci_offsets = None

                # Iterate through datasets                
                for dsi, tup_dataset in enumerate(datasets):
                    edataset = tup_dataset[0]
                    is_new_class = tup_dataset[1]
                    # Fetch patch databases, if None, create
                    nndbs = dict_nndbs.setdefault(edataset, None)
                    if (nndbs is None):
                        dict_nndbs[edataset] = nndbs = []

                        # Add an empty NNdb for all `nnpatch` on first edataset entry
                        for pi_tmp in range(patch_loop_max_n):
                            nndbs.append(NNdb(str(edataset) + "_p" + str(pi_tmp), format=nndb.format))

                    # Build Traiing DB
                    if (edataset == Dataset.TR):

                         #If noise or occlusion is required
                        if (((sel.tr_noise_rate is not None) or 
                            (sel.tr_occlusion_rate is not None))
                            and (tci_offsets is None)):
                            tci_offsets = np.where(sel.tr_col_indices == col_idx)[0]

                        # Check whether col_idx is a noise required index 
                        noise_rate = None
                        if ((sel.tr_noise_rate is not None) and
                            (tci_offsets[dsi] < sel.tr_noise_rate.size) and
                            (0 != sel.tr_noise_rate[tci_offsets[dsi]])):
                            noise_rate = sel.tr_noise_rate[tci_offsets[dsi]]
              
                        # Check whether col_idx is a occlusion required index 
                        occl_rate = None
                        occl_type = None
                        occl_offset = None
                        if ((sel.tr_occlusion_rate is not None) and
                            (tci_offsets[dsi] <sel.tr_occlusion_rate.size) and
                            (0 != sel.tr_occlusion_rate[tci_offsets[dsi]])):
                            occl_rate = sel.tr_occlusion_rate[tci_offsets[dsi]]

                            if (sel.tr_occlusion_type is not None):
                                occl_type = sel.tr_occlusion_type[tci_offsets[dsi]]

                            if (sel.tr_occlusion_offset is not None):
                                occl_offset = sel.tr_occlusion_offset[tci_offsets[dsi]]

                        DbSlice._build_nndb_tr(nndbs, pi, is_new_class, pimg, noise_rate, occl_rate, occl_type, occl_offset)  # noqa E501

                    # Build Training Output DB
                    elif(edataset == Dataset.TR_OUT):
                        DbSlice._build_nndb_tr_out(nndbs, pi, is_new_class, pimg)  # noqa E501

                    # Build Valdiation DB
                    elif(edataset == Dataset.VAL):
                        DbSlice._build_nndb_val(nndbs, pi, is_new_class, pimg)  # noqa E501

                    # Build Valdiation Target DB
                    elif(edataset == Dataset.VAL_OUT):
                        DbSlice._build_nndb_val_out(nndbs, pi, is_new_class, pimg)  # noqa E501

                    # Build Testing DB
                    elif(edataset == Dataset.TE):
                        DbSlice._build_nndb_te(nndbs, pi, is_new_class, pimg)  # noqa E501

                    # Build Testing Target DB
                    elif(edataset == Dataset.TE_OUT):
                        DbSlice._build_nndb_te_out(nndbs, pi, is_new_class, pimg)  # noqa E501

        # Returns NNdb instance instead of a list 
        # (when no nnpatches are mentioned in selection structure)
        if (sel.nnpatches is None):
    
            def p0_nndbs(dict_nndbs, ekey):
                return None if (dict_nndbs[ekey] is None) else dict_nndbs[ekey][0]

            return  p0_nndbs(dict_nndbs, Dataset.TR),\
                    p0_nndbs(dict_nndbs, Dataset.VAL),\
                    p0_nndbs(dict_nndbs, Dataset.TE),\
                    p0_nndbs(dict_nndbs, Dataset.TR_OUT),\
                    p0_nndbs(dict_nndbs, Dataset.VAL_OUT),\
                    p0_nndbs(dict_nndbs, Dataset.TE_OUT),\
                    [Dataset.TR, Dataset.VAL, Dataset.TE, Dataset.TR_OUT, Dataset.VAL_OUT, Dataset.TE_OUT]

        return  dict_nndbs[Dataset.TR],\
                dict_nndbs[Dataset.VAL],\
                dict_nndbs[Dataset.TE],\
                dict_nndbs[Dataset.TR_OUT],\
                dict_nndbs[Dataset.VAL_OUT],\
                dict_nndbs[Dataset.TE_OUT],\
                [Dataset.TR, Dataset.VAL, Dataset.TE, Dataset.TR_OUT, Dataset.VAL_OUT, Dataset.TE_OUT]

    @staticmethod
    def preprocess_im_with_sel(cimg, cann_cimg, sel, ch_axis=None):
        """Peform image preocessing."""

        # Image resize
        cimg = DbSlice._perform_resize(cimg, sel.scale)
        if (cann_cimg is not None):
            cann_cimg = DbSlice._perform_resize(cann_cimg, sel.scale)

        # Color / Gray Scale Conversion (if required)
        cimg = DbSlice._process_color(cimg, sel)
        if (cann_cimg is not None):
            cann_cimg = DbSlice._process_color(cann_cimg, sel)

        # Image pre-processing parameters
        pp_params = {'histeq':sel.histeq, 
                        'normalize':sel.normalize, 
                        'histmatch':sel.histmatch_col_index is not None, 
                        'cann_img':cann_cimg, 
                        'ch_axis':ch_axis}
        cimg = im_pre_process(cimg, pp_params)

        # [CALLBACK] the specific pre-processing script
        if (sel.pre_process_script is not None):
            cimg = sel.pre_process_script(cimg)

        return cimg

    @staticmethod
    def examples(imdb,im_per_class):
        """Extensive set of examples.

        # Full set of options
        nndb = NNdb('original', imdb_8, 8, true)
        sel.tr_col_indices        = [1:3 7:8] #[1 2 3 7 8]
        sel.tr_noise_rate         = None
        sel.tr_occlusion_rate     = None
        sel.tr_occlusion_type     = None
        sel.tr_occlusion_offset   = None
        sel.tr_out_col_indices    = None
        sel.val_col_indices       = None
        sel.val_out_col_indices   = None
        sel.te_col_indices        = [4:6] #[4 5 6]
        sel.te_out_col_indices    = None
        sel.nnpatches             = None
        sel.use_rgb               = False
        sel.color_indices         = None
        sel.use_real              = False
        sel.scale                 = 0.5
        sel.normalize             = False
        sel.histeq                = True
        sel.histmatch_col_index   = None
        sel.class_range           = [1:36 61:76 78:100]
        sel.val_class_range       = None
        sel.te_class_range        = None
        #sel.pre_process_script   = fn_custom_pprocess
        sel.pre_process_script    = None
        [nndb_tr, _, nndb_te, _, _, _, _]  = DbSlice.slice(nndb, sel)

        Parameters
        ----------
        imdb_8 : NNdb
            NNdb object that represents the dataset. It should contain
            only 8 images per subject.

            Format: (Samples x H x W x CH).
        """
        #
        # Select 1st 2nd 4th images of each identity for training.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel) # nndb_tr = DbSlice.slice(nndb, sel) # noqa E501
        nndb_tr.show(8,3)

        #
        # Select 1st 2nd 4th images of each identity for training.
        # Divide into nnpatches
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        patch_gen = NNPatchGenerator(nndb.h, nndb.w, 33, 33, 33, 33)
        sel.nnpatches = patch_gen.generate_patches()

        # Cell arrays of NNdb objects for each patch
        [nndbs_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndbs_tr[0].show()
        nndbs_tr[1].show()
        nndbs_tr[3].show()
        nndbs_tr[4].show()   


        #
        # Select 1st 2nd 4th images of each identity for training.
        # Select 3rd 5th images of each identity for testing.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(8,3)
        nndb_te.show(8,3)
        #
        # Select 1st 2nd 4th images of identities denoted by class_range for training. # noqa E501
        # Select 3rd 5th images of identities denoted by class_range for testing. # noqa E501
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.class_range = np.arange(10)                         # First ten identities # noqa E501
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10,3)
        nndd_te.show(10,2)

        #
        # Select 1st and 2nd image from 1st class and 2nd, 3rd and 5th image from 2nd class for training 
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices      = [np.array([0, 1], dtype='uint8'), np.array([1, 2, 4], dtype='uint8')]
        sel.class_range         = np.uint8(np.arange(0, 2))
        [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)

        # 
        # Select 1st 2nd 4th images of identities denoted by class_range for training.
        # Select 1st 2nd 4th images images of identities denoted by class_range for validation.   
        # Select 3rd 5th images of identities denoted by class_range for testing. 
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.val_col_indices= np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.class_range    = np.arange(10)
        [nndb_tr, nndb_val, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel);
        nndb_tr.show(10,3)
        nndd_te.show(10,3)  
        nndd_te.show(10,2)    
            
        # 
        # Select 1st 2nd 4th images of identities denoted by class_range for training.
        # Select 1st 2nd 4th images images of identities denoted by val_class_range for validation.   
        # Select 3rd 5th images of identities denoted by te_class_range for testing. \
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.val_col_indices= np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.class_range    = np.arange(10)
        sel.val_class_range= np.arange(6,15)
        sel.te_class_range = np.arange(17, 20)
        [nndb_tr, nndb_val, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)

   
        # 
        # Select 1st 2nd 4th images of identities denoted by class_range for training.
        # Select 3rd 4th images of identities denoted by val_class_range for validation.
        # Select 3rd 5th images of identities denoted by te_class_range for testing.
        # Select 1st 1st 1st images of identities denoted by class_range for training target.
        # Select 1st 1st images of identities denoted by val_class_range for validation target.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.val_col_indices= np.array([2, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.tr_out_col_indices  = np.zeros(3, dtype='uint8')  # [0, 0, 0]
        sel.val_out_col_indices = np.zeros(2, dtype='uint8')  # [0, 0]
        sel.te_out_col_indices  = np.zeros(2, dtype='uint8')  # [0, 0]
        sel.class_range    = np.arange(9)
        sel.val_class_range= np.arange(6,15)
        sel.te_class_range = np.arange(17, 20)
        [nndb_tr, nndb_val, nndb_te, nndb_tr_out, nndb_val_out, nndb_te_out, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)
        nndb_val.show(10, 2)
        nndb_te.show(4, 2)
        nndb_tr_out.show(10, 3)
        nndb_val_out.show(10, 2)
        nndb_te_out.show(10, 2)

        #
        # Using special enumeration values
        # Training column will consists of first 60% of total columns avaiable
        # Testing column will consists of first 40% of total columns avaiable
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices      = Select.PERCENT_60
        sel.te_col_indices      = Select.PERCENT_40
        sel.class_range         = np.uint8(np.arange(0, 60))
        [nndb_tr, _, nndb_te, _, _, _, _] =\
                                DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 5)
        nndb_te.show(10, 4)


        #
        # Select 1st 2nd 4th images of each identity for training +
        #               add various noise types @ random locations of varying degree. # noqa E501
        #               default noise type: random black and white dots.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.tr_noise_rate = np.array([0, 0.5, 0.2]) # percentage of corruption # noqa E501
        # sel.tr_noise_rate  = [0 0.5 Noise.G]         # last index with Gauss noise # noqa E501
        [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10,3)

        #
        # Select 1st 2nd 4th images of each identity for training +
        #               add various occlusion types ('t':top, 'b':bottom, 'l':left, 'r':right, 'h':horizontal, 'v':vertical) of varying degree. # noqa E501
        #               default occlusion type: 'b'.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.tr_occlusion_rate = np.array([0, 0.5, 0.2]) # percentage of occlusion # noqa E501
        sel.tr_occlusion_type = 'ttt' # occlusion type: 't' for selected tr. indices [0, 1, 3]
        #sel.tr_occlusion_type = 'tbr' 
        #sel.tr_occlusion_type = 'lrb' 
        [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10,3)


        #
        # Select 1st 2nd 4th images of each identity for training +
        #               add occlusions in the middle (horizontal/vertical).
        #               default occlusion type: 'b'.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.tr_occlusion_rate = np.array([0, 0.5, 0.2])         # percentage of occlusion # noqa E501
        sel.tr_occlusion_type = 'ttt'                           # occlusion type: 't' for selected tr. indices [0, 1, 3]
        sel.tr_occlusion_offset = np.array([0, 0.25, 0.4])      # Offset from top since 'tr_occlusion_type = t'
        # sel.tr_occlusion_type = 'rrr'                         # occlusion type: 'r' for selected tr. indices [1, 2, 4]
        # sel.tr_occlusion_offset = np.array([0, 0.25, 0.4])    # Offset from right since 'tr_occlusion_type = r'
        [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10,3)

        #
        # To prepare regression datasets, training dataset and training target dataset # noqa E501
        # Select 1st 2nd 4th images of each identity for training.
        # Select 1st 1st 1st image of each identity for corresponding training target # noqa E501
        # Select 3rd 5th images of each identity for testing.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.tr_out_col_indices = np.array([1, 1, 1])  # [1 1 1] (Regression 1->1, 2->1, 4->1) # noqa E501
        sel.te_col_indices = np.array([2, 4])
        [nndb_tr, nndb_tr_out, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10,3)
        nndb_te.show(10,2)
        
        #
        # Resize images by 0.5 scale factor.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.scale = 0.5
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10,3)
        nndb_te.show(10,2)

        #
        # Use gray scale images.
        # Perform histrogram equalization.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.histeq = True
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10,3)
        nndb_te.show(10,2)
         
        # Use gray scale images.
        # Perform histrogram match. This will be performed with the 1st image
        #  of each identity irrespective of the selection choice.
        # (refer code for more details)
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.histmatch_col_index = 0
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10,3)
        nndb_te.show(10,2)

        #
        # If imdb_8 supports many color channels
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.color_indices = 5  # color channel denoted by 5th index
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10,3)
        nndb_te.show(10,2)

    @staticmethod
    def get_occlusion_patch(h, w, dtype, occl_type, occl_rate, occl_offset=None):

        # Set defaults for arguments
        if (occl_offset is None): occl_offset = 0

        filter = np.ones((h, w))
        filter = filter.astype(dtype)

        if ((occl_type is None) or (occl_type == 'b')):
            sh = math.ceil(occl_rate * h)
            en = math.floor((1-occl_offset) * h)
            st = en - sh; 
            if (st < 0): st = 1
            filter[st:en, 0:w] = 0

        elif (occl_type == 'r'):
            sh = math.ceil(occl_rate * w)
            en = math.floor((1-occl_offset) * w)
            st = en - sh; 
            if (st < 0): st = 1
            filter[0:h, st:en] = 0

        elif (occl_type == 't' or occl_type == 'v'):
            sh = math.floor(occl_rate * h)
            st = math.floor(occl_offset * h)
            en = st + sh
            if (en > h): en = h
            filter[st:en, 0:w] = 0

        elif (occl_type == 'l' or occl_type == 'h'):
            sh = math.floor(occl_rate * w)
            st = math.floor(occl_offset * w)
            en = st + sh
            if (en > w): en = w
            filter[0:h, st:en] = 0

        return filter   

    ##########################################################################
    # Protected Interface
    ##########################################################################
    @staticmethod
    def _set_default_cls_range(default_idx,  cls_ranges, col_ranges):
        """Set the value of the class range at default_idx 
            to the class ranges at other indices [val|te|tr_out|val_out]

        Parameters
        ----------
        default_idx : describe
            decribe.

        cls_ranges : describe
            describe.

        col_ranges : describe
            describe.

        Returns
        -------
        """
        for ri, col_range in enumerate(col_ranges):
            if (col_range is not None and 
                cls_ranges[ri] is None):
                print("[DEFAULT] " + str(Dataset.enum(ri)) + "_class_range (None): = class_range")
                cls_ranges[ri] = cls_ranges[default_idx]

    @staticmethod
    def _init_dict_nndbs(col_ranges):
        """Initialize a dictionary that tracks the `nndbs` for each dataset.

            i.e dict_nndbs[Dataset.TR] => [NNdb_Patch_1, NNdb_Patch_2, ...]

        Parameters
        ----------
        col_ranges : describe
            decribe.

        Returns
        -------
        dict_nndbs : describe
        """
        dict_nndbs = {}

        # Iterate through ranges
        for ri, col_range in enumerate(col_ranges):
            edataset = Dataset.enum(ri)
            dict_nndbs.setdefault(edataset, None)

        return dict_nndbs

    @staticmethod
    def _perform_resize(cimg, scale):
        if (scale is not None):
            # LIMITATION: Python scipy imresize, unlike matlab
            _, _, ch = cimg.shape
            if (ch == 1):
                cimg = scipy.misc.imresize(cimg[:, :, 0], scale)
                cimg = np.expand_dims(cimg, axis=2)
            else:
                cimg = scipy.misc.imresize(cimg, scale)

        return cimg

    @staticmethod
    def _process_color(img, sel):
        """Perform color related functions.

        Parameters
        ----------
        img : describe
            decribe.

        sel : describe
            describe.

        Returns
        -------
        img : array_like -uint8
            Color processed image.
        """
        if (img is None): return  # noqa E701
        _, _, ch = img.shape

        # Color / Gray Scale Conversion (if required)
        if (not sel.use_rgb):

            # if image has more than 3 channels
            if (ch >= 3):
                if (sel.color_indices is not None):
                    img = img[:, :, sel.color_indices]
                else:                    
                    img = rgb2gray(img, img.dtype, keepDims=True)

        return img

    @staticmethod
    def _build_nndb_tr(nndbs, pi, is_new_class, img, noise_rate, occl_rate, occl_type, occl_offset):
        """Build the nndb training database.

        Returns
        -------
        nndbs : :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if (nndbs is None): return nndbs  # noqa E701
        nndb = nndbs[pi]    

        if ((occl_rate is not None) or (noise_rate is not None)):

            h, w, ch = img.shape

            #Adding different occlusions depending on the precentage
            if (occl_rate is not None):                
                filter = DbSlice.get_occlusion_patch(h, w, img.dtype, occl_type, occl_rate, occl_offset);                    
                filter = np.expand_dims(filter, 2)

                # For grey scale
                if (ch == 1):                    
                    img = filter * img
                else:
                    # For colored
                    img = np.tile(filter, (1, 1, ch)) * img

            # Add different noise depending on the type
            # (ref. Enums/Noise)
            elif ((noise_rate is not None) and (noise_rate == Noise.G)):
                pass
                # img = imnoise(img, 'gaussian')
                # #img = imnoise(img, 'gaussian')
                # #img = imnoise(img, 'gaussian')
                # #img = imnoise(img, 'gaussian')

            # Perform random corruption with the rate
            elif (noise_rate is not None):
                img =  np.copy(img)
                
                # Corruption Size (H x W)
                cs = [np.uint16(h*noise_rate), np.uint16(w*noise_rate)]

                # Random location choice
                # Start of H, W (location)
                sh = np.uint8(1 + rand()*(h-cs[0]-1))
                sw = np.uint8(1 + rand()*(w-cs[1]-1))

                # Set the corruption
                corrupt_patch = np.uint8(DbSlice._rand_corrupt(cs[0], cs[1])).astype('uint8')  # noqa E501

                if (ch == 1):
                    img[sh:sh+cs[1], sw:sw+cs[2]] = corrupt_patch
                else:
                    for ich in range(ch):
                        img[sh:sh+cs[0], sw:sw+cs[1], ich] = corrupt_patch

        # Add data to nndb
        nndb.add_data(img)
    
        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs

    @staticmethod
    def _rand_corrupt(height, width):
        """Corrupt the image with a (height, width) block.

        Parameters
        ----------
        height : describe
            decribe.

        width : describe
            describe.

        Returns
        -------
        img : array_like -uint8
            3D-Data tensor that contains the corrupted image.
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
    def _build_nndb_tr_out(nndbs, pi, is_new_class, img):
        """Build the nndb training target database.

        Returns
        -------
        nndbs : :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if (nndbs is None): return nndbs
        nndb = nndbs[pi]
        nndb.add_data(img)

        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs

    @staticmethod
    def _build_nndb_val(nndbs, pi, is_new_class, img):
        """"Build the nndb validation database.

        Returns
        -------
        nndbs : :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if (nndbs is None): return nndbs
        nndb = nndbs[pi]
        nndb.add_data(img)

        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs

    @staticmethod
    def _build_nndb_val_out(nndbs, pi, is_new_class, img):
        """"Build the nndb validation target database.

        Returns
        -------
        nndbs : :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if (nndbs is None): return nndbs
        nndb = nndbs[pi]
        nndb.add_data(img)

        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs

    @staticmethod
    def _build_nndb_te(nndbs, pi, is_new_class, img):
        """Build the testing database.

        Returns
        -------
        nndbs : :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if (nndbs is None): return nndbs
        nndb = nndbs[pi]
        nndb.add_data(img)

        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs

    @staticmethod
    def _build_nndb_te_out(nndbs, pi, is_new_class, img):
        """"Build the nndb validation target database.

        Returns
        -------
        nndbs : :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if (nndbs is None): return nndbs
        nndb = nndbs[pi]
        nndb.add_data(img)

        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs