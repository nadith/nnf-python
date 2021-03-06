# -*- coding: utf-8 -*- TODO: Revisit the comments
"""
.. module:: DbSlice
   :platform: Unix, Windows
   :synopsis: Represent DbSlice class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import os
import math
import scipy.misc
import scipy.ndimage.interpolation
import numpy as np
from numpy.random import rand
from numpy.random import permutation
from warnings import warn as warning

# Local Imports
from nnf.db.NNdb import NNdb
from nnf.db.Noise import Noise
from nnf.db.Format import Format
from nnf.db.Dataset import Dataset
from nnf.db.Selection import Select
from nnf.utl.rgb2gray import rgb2gray
from nnf.db.Selection import Selection
from nnf.pp.im_pre_process import im_pre_process
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
from nnf.core.iters.memory.DskmanMemDataIterator import DskmanMemDataIterator


from nnf.utl.Benchmark import Benchmark

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
    @staticmethod
    def slice(nndb, sel=None, data_generator=None, pp_param=None, savepath=None, buffer_sizes=None):
        """Slice the database according to the selection structure.

        Parameters
        ----------
        nndb : NNdb
            NNdb object that represents the dataset.

        sel : :obj:`Selection`
            Information to split the dataset. (Ref: class documentation).

        data_generator : :obj:`DskmanMemDataIterator`
            NNDiskman iterator object for in memory databases. (Default value = None).

        pp_param : :obj:`dict`
            Dictionary of pre-processor parameters for `data_generator` passed above.
            (Default value = None).

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
        if sel is None:
            sel = Selection()

        # Error handling for arguments
        if ((sel.tr_col_indices is None) and
                (sel.tr_out_col_indices is None) and
                (sel.val_col_indices is None) and
                (sel.val_out_col_indices is None) and
                (sel.te_col_indices is None) and
                (sel.te_out_col_indices is None)):

            unq_n_per_cls = int(np.unique(nndb.n_per_class))
            if np.isscalar(unq_n_per_cls):
                sel.tr_col_indices = np.arange(unq_n_per_cls)
            else:
                raise Exception('Selection is not provided.' +
                                ' tr_col_indices = nndb.n_per_class' +
                                ' must be same for all classes')

        if sel.use_rgb is not None and sel.color_indices is not None:
            raise Exception('ARG_CONFLICT: sel.use_rgb, sel.color_indices')

        if sel.tr_occlusion_rate is not None and sel.tr_occlusion_filter is not None:
            warning(['`sel.tr_occlusion_filter` will be applied over all sel.tr_occlusion_xxx configurations'])

        # Set defaults for data generator 
        if data_generator is None:
            data_generator = DskmanMemDataIterator(pp_param)
        data_generator.init_params(nndb)

        # Set default for sel.class range
        sel.class_range = np.arange(nndb.cls_n) if (sel.class_range is None) else sel.class_range

        # Initialize class ranges and column ranges
        # DEPENDENCY -The order must be preserved as per the enum Dataset
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

        if sel.nnpatches is not None:
            # Must have atleast 1 patch to represent the whole image
            nnpatch = sel.nnpatches[0]

            # PERF: If there is only 1 patch and it is the whole image
            if (len(sel.nnpatches) == 1) and nnpatch.is_holistic:
                pass

            else:
                nnpatch_list = sel.nnpatches
                patch_loop_max_n = len(sel.nnpatches)

        # Initialize the generator
        data_generator.init_ex(cls_ranges, col_ranges, True)

        # PERF: Iterate through the chosen subset of the nndb database
        for cimg, _, cls_idx, col_idx, datasets in data_generator:

            #with Benchmark("DbSlice_"+str(col_idx)):
            # cimg: array_like data (maybe an image or raw data item)
            # frecord: [fpath, fpos, cls_lbl], assert(cls_lbl == cls_idx)
            # cls_idx, col_idx: int
            # datasets: list of tuples
            #       [(Dataset.TR, True), (Dataset.TR, False), (Dataset.VAL, True), ...]

            # SPECIAL NOTE:  cls_lbl, cls_idx, col_idx are expressed with respect to the
            # cls_ranges, col_ranges defined above and may not consist of continuous
            # indices. i.e cls_lbl=[0 4 6 3]

            # Perform pre-processing before patch division
            # Perform image operations only if db_format comply them
            if (nndb.db_format == Format.H_W_CH_N) or (nndb.db_format == Format.N_H_W_CH):

                # For histogram equalization operation (canonical image)
                cann_cimg = None
                if sel.histmatch_col_index is not None:
                    cann_cimg, _ = data_generator.get_cimg_frecord(cls_idx, sel.histmatch_col_index)
                    # Alternative:
                    # cls_st = nndb.cls_st[sel.histmatch_col_index]
                    # cann_cimg = nndb.get_data_at(cls_st)

                # Perform image pre-processing
                cimg = DbSlice.preprocess_im_with_sel(cimg, cann_cimg, sel, data_generator.get_im_ch_axis())

            # with Benchmark("DbSlice_" + str(col_idx)):
            # Iterate through image nnpatches
            for pi in range(patch_loop_max_n):

                # Holistic image (by default)
                pimg = cimg

                # Generate the image patch
                if nnpatch_list is not None:
                    nnpatch = nnpatch_list[pi]
                    x = nnpatch.offset[1]
                    y = nnpatch.offset[0]
                    w = nnpatch.w
                    h = nnpatch.h

                    # Extract the patch
                    if (nndb.db_format == Format.H_W_CH_N) or (nndb.db_format == Format.N_H_W_CH):
                        pimg = cimg[y:y+h, x:x+w, :]

                    elif (nndb.db_format == Format.H_N) or (nndb.db_format == Format.N_H):
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
                    if nndbs is None:
                        dict_nndbs[edataset] = nndbs = []

                        # Add an empty NNdb for all `nnpatch` on first edataset entry
                        for pi_tmp in range(patch_loop_max_n):
                            buff_size = buffer_sizes[edataset] if buffer_sizes is not None and edataset in buffer_sizes else None
                            nndbs.append(NNdb(str(edataset) + "_p" + str(pi_tmp), db_format=nndb.db_format, buffer_size=buff_size))

                    # Build Training DB
                    if edataset == Dataset.TR:

                        # If noise or occlusion is required
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
                                (tci_offsets[dsi] < sel.tr_occlusion_rate.size) and
                                (0 != sel.tr_occlusion_rate[tci_offsets[dsi]])):
                            occl_rate = sel.tr_occlusion_rate[tci_offsets[dsi]]

                            if sel.tr_occlusion_type is not None:
                                occl_type = sel.tr_occlusion_type[tci_offsets[dsi]]

                            if sel.tr_occlusion_offset is not None:
                                occl_offset = sel.tr_occlusion_offset[tci_offsets[dsi]]

                        #with Benchmark("DbSlice_" + str(col_idx)):
                        DbSlice._build_nndb_tr(nndbs, pi, is_new_class, pimg, noise_rate, occl_rate,
                                               occl_type, occl_offset, sel.tr_occlusion_filter)  # noqa E501

                    # Build Training Output DB
                    elif edataset == Dataset.TR_OUT:
                        DbSlice._build_nndb_tr_out(nndbs, pi, is_new_class, pimg)  # noqa E501

                    # Build Validation DB
                    elif edataset == Dataset.VAL:
                        DbSlice._build_nndb_val(nndbs, pi, is_new_class, pimg)  # noqa E501

                    # Build Validation Target DB
                    elif edataset == Dataset.VAL_OUT:
                        DbSlice._build_nndb_val_out(nndbs, pi, is_new_class, pimg)  # noqa E501

                    # Build Testing DB
                    elif edataset == Dataset.TE:
                        DbSlice._build_nndb_te(nndbs, pi, is_new_class, pimg)  # noqa E501

                    # Build Testing Target DB
                    elif edataset == Dataset.TE_OUT:
                        DbSlice._build_nndb_te_out(nndbs, pi, is_new_class, pimg)  # noqa E501

        # Returns NNdb instance instead of a list
        # (when no nnpatches are mentioned in selection structure)
        if sel.nnpatches is None:
    
            def p0_nndbs(dictionary, ekey):
                return None if (dictionary[ekey] is None) else dictionary[ekey][0]

            # Finalize the nndbs(PERF) and save the splits in the disk
            for dataset in Dataset.get_enum_list():
                tmp_nndb = p0_nndbs(dict_nndbs, dataset)
                if tmp_nndb is not None:
                    # with Benchmark("DbSlice"):
                    tmp_nndb.finalize()
                    if savepath is not None:
                        tmp_nndb.save(os.path.splitext(savepath)[0] + "_" + str.upper(str(dataset)) + ".mat")

            return (p0_nndbs(dict_nndbs, Dataset.TR),
                    p0_nndbs(dict_nndbs, Dataset.VAL),
                    p0_nndbs(dict_nndbs, Dataset.TE),
                    p0_nndbs(dict_nndbs, Dataset.TR_OUT),
                    p0_nndbs(dict_nndbs, Dataset.VAL_OUT),
                    p0_nndbs(dict_nndbs, Dataset.TE_OUT),
                    [Dataset.TR, Dataset.VAL, Dataset.TE, Dataset.TR_OUT, Dataset.VAL_OUT, Dataset.TE_OUT])

        # Finalize the nndbs(PERF) and save the splits in the disk
        for dataset in Dataset.get_enum_list():
            tmp_nndbs = None if (dict_nndbs[dataset] is None) else dict_nndbs[dataset]
            if tmp_nndbs is not None:
                for pi, tmp_nndb in enumerate(tmp_nndbs):
                    tmp_nndb.finalize()
                    if savepath is not None:
                        tmp_nndb.save(os.path.splitext(savepath)[0] + "_" + str.upper(str(dataset)) + "_" + str(pi) + ".mat")

        return (dict_nndbs[Dataset.TR],
                dict_nndbs[Dataset.VAL],
                dict_nndbs[Dataset.TE],
                dict_nndbs[Dataset.TR_OUT],
                dict_nndbs[Dataset.VAL_OUT],
                dict_nndbs[Dataset.TE_OUT],
                [Dataset.TR, Dataset.VAL, Dataset.TE, Dataset.TR_OUT, Dataset.VAL_OUT, Dataset.TE_OUT])

    @staticmethod
    def preprocess_im_with_sel(cimg, cann_cimg, sel, ch_axis=None):
        """Perform image preprocessing with compliance to selection object.

        Parameters
        ----------
        cimg : ndarray -uint8
            3D Data tensor to represent the color image.

        cann_cimg : ndarray
            Canonical/Target image (corresponds to sel.tr_out_indices).

        sel : :obj:`Selection`
            Information to pre-process the dataset. (Ref: class documentation).

        ch_axis : int
            Color axis of the image.

        Returns
        -------
        ndarray -uint8
            Pre-processed image.
        """

        # Image resize
        cimg = DbSlice._perform_resize(cimg, sel.scale)
        if cann_cimg is not None:
            cann_cimg = DbSlice._perform_resize(cann_cimg, sel.scale)

        # Color / Gray Scale Conversion (if required)
        cimg = DbSlice._process_color(cimg, sel)
        if cann_cimg is not None:
            cann_cimg = DbSlice._process_color(cann_cimg, sel)

        # Image pre-processing parameters
        pp_params = {'histeq': sel.histeq,
                     'normalize': sel.normalize,
                     'histmatch': sel.histmatch_col_index is not None,
                     'cann_img': cann_cimg,
                     'ch_axis': ch_axis}
        cimg = im_pre_process(cimg, pp_params)

        # [CALLBACK] the specific pre-processing script
        if sel.pre_process_script is not None:
            cimg = sel.pre_process_script(cimg)

        return cimg

    # noinspection PyPep8
    @staticmethod
    def examples(imdb, im_per_class):
        """Extensive set of examples.

        Parameters
        ----------
        imdb : ndarray
            NNdb object that represents the dataset. Assume it contain only 8 images per subject.

            Format: (Samples x H x W x CH).

        im_per_class : int
            Image per class.
        """
        #
        # Select 1st 2nd 4th images of each identity for training
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)  # nndb_tr = DbSlice.slice(nndb, sel) # noqa E501
        nndb_tr.show(8, 3)

        #
        # Select 1st 2nd 4th images of each identity for training
        # Divide into nnpatches
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        patch_gen = NNPatchGenerator(nndb.h, nndb.w, 33, 33, 33, 33)
        sel.nnpatches = patch_gen.generate_nnpatches()

        # Cell arrays of NNdb objects for each patch
        [nndbs_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndbs_tr[0].show()
        nndbs_tr[1].show()
        nndbs_tr[3].show()
        nndbs_tr[4].show()   

        #
        # Select 1st 2nd 4th images of each identity for training
        # Select 3rd 5th images of each identity for testing
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(8, 3)
        nndb_te.show(8, 3)
        #
        # Select 1st 2nd 4th images of identities denoted by class_range for training # noqa E501
        # Select 3rd 5th images of identities denoted by class_range for testing # noqa E501
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices  = np.array([0, 1, 3])
        sel.te_col_indices  = np.array([2, 4])
        sel.class_range     = np.arange(10)  # First ten identities # noqa E501
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)
        nndb_te.show(10, 2)

        #
        # Select 1st and 2nd image from 1st class and 2nd, 3rd and 5th image from 2nd class for training 
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices          = [np.array([0, 1], dtype='uint8'), np.array([1, 2, 4], dtype='uint8')]
        sel.class_range             = np.uint8(np.arange(0, 2))
        [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 2)

        # 
        # Select 1st 2nd 4th images of identities denoted by class_range for training
        # Select 1st 2nd 4th images images of identities denoted by class_range for validation
        # Select 3rd 5th images of identities denoted by class_range for testing
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices  = np.array([0, 1, 3])
        sel.val_col_indices = np.array([0, 1, 3])
        sel.te_col_indices  = np.array([2, 4])
        sel.class_range     = np.arange(10)
        [nndb_tr, nndb_val, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)
        nndb_val.show(10, 3)
        nndb_te.show(10, 2)
            
        # 
        # Select 1st 2nd 4th images of identities denoted by class_range for training
        # Select 1st 2nd 4th images images of identities denoted by val_class_range for validation
        # Select 3rd 5th images of identities denoted by te_class_range for testing
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices  = np.array([0, 1, 3])
        sel.val_col_indices = np.array([0, 1, 3])
        sel.te_col_indices  = np.array([2, 4])
        sel.class_range     = np.arange(10)
        sel.val_class_range = np.arange(6, 15)
        sel.te_class_range  = np.arange(17, 20)
        [nndb_tr, nndb_val, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)
        nndb_val.show(10, 3)
        nndb_te.show(10, 2)

        # 
        # Select 1st 2nd 4th images of identities denoted by class_range for training
        # Select 3rd 4th images of identities denoted by val_class_range for validation
        # Select 3rd 5th images of identities denoted by te_class_range for testing
        # Select 1st 1st 1st images of identities denoted by class_range for training target
        # Select 1st 1st images of identities denoted by val_class_range for validation target
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices      = np.array([0, 1, 3])
        sel.val_col_indices     = np.array([2, 3])
        sel.te_col_indices      = np.array([2, 4])
        sel.tr_out_col_indices  = np.zeros(3, dtype='uint8')  # [0, 0, 0]
        sel.val_out_col_indices = np.zeros(2, dtype='uint8')  # [0, 0]
        sel.te_out_col_indices  = np.zeros(2, dtype='uint8')  # [0, 0]
        sel.class_range         = np.arange(9)
        sel.val_class_range     = np.arange(6, 15)
        sel.te_class_range      = np.arange(17, 20)
        [nndb_tr, nndb_val, nndb_te, nndb_tr_out, nndb_val_out, nndb_te_out, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)
        nndb_val.show(10, 2)
        nndb_te.show(4, 2)
        nndb_tr_out.show(10, 3)
        nndb_val_out.show(10, 2)
        nndb_te_out.show(10, 2)

        #
        # Using special enumeration values
        # Training column will consists of first 60% of total columns available
        # Testing column will consists of first 40% of total columns available
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices      = Select.PERCENT_60
        sel.te_col_indices      = Select.PERCENT_40
        sel.class_range         = np.uint8(np.arange(0, 60))
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 5)
        nndb_te.show(10, 4)

        #
        # Select 1st 2nd 4th images of each identity for training +
        #               add various noise types @ random locations of varying degree # noqa E501
        #               default noise type: random black and white dots.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.tr_noise_rate = np.array([0, 0.5, 0.2])  # percentage of corruption # noqa E501
        # sel.tr_noise_rate  = [0 0.5 Noise.G]       # last index with Gauss noise # noqa E501
        [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)

        #
        # Select 1st 2nd 4th images of each identity for training +
        #               add various occlusion types ('t':top, 'b':bottom, 'l':left, 'r':right, 'h':horizontal, 'v':vertical) of varying degree. # noqa E501
        #               default occlusion type: 'b'.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.tr_occlusion_rate = np.array([0, 0.5, 0.2])  # percentage of occlusion # noqa E501
        sel.tr_occlusion_type = 'ttt'  # occlusion type: 't' for selected tr. indices [0, 1, 3]
        # sel.tr_occlusion_type = 'tbr'
        # sel.tr_occlusion_type = 'lrb'
        [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)

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
        nndb_tr.show(10, 3)

        #
        # Select 1st 2nd 4th images of each identity for training +
        #              use custom occlusion filter for 66x66 images.
        occl_filter = np.ones(66, 66)
        for i in range(0, 33):
            for j in range(0,33-i):
                occl_filter[i, j] = 0
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.tr_occlusion_filter = occl_filter
        [nndb_tr, _, _, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)

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
        nndb_tr.show(10, 3)
        nndb_tr_out.show(10, 3)
        nndb_te.show(10, 2)
        
        #
        # Resize images by 0.5 scale factor.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.scale = 0.5
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)
        nndb_te.show(10, 2)

        #
        # Use gray scale images.
        # Perform histogram equalization.
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.histeq = True
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)
        nndb_te.show(10, 2)
         
        # Use gray scale images.
        # Perform histogram match. This will be performed with the 1st image
        #  of each identity irrespective of the selection choice.
        # (refer code for more details)
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.histmatch_col_index = 0
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)
        nndb_te.show(10, 2)

        #
        # If imdb_8 supports many color channels
        nndb = NNdb('original', imdb, im_per_class, True)
        sel = Selection()
        sel.tr_col_indices = np.array([0, 1, 3])
        sel.te_col_indices = np.array([2, 4])
        sel.use_rgb = False
        sel.color_indices = 5  # color channel denoted by 5th index
        [nndb_tr, _, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)
        nndb_tr.show(10, 3)
        nndb_te.show(10, 2)

    @staticmethod
    def get_occlusion_patch(h, w, dtype, occl_type, occl_rate, occl_offset=None):
        """Get a occlusion patch to place on top of an image.

        Parameters
        ----------
        h : int
            Height of the occlusion patch.

        h : int
            Width of the occlusion patch.

        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        occl_type : char
            Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right).

        occl_rate : double
            Occlusion ratio.

        occl_offset : double
            Occlusion start offset (as a ratio) from top/bottom/left/right corner depending on `occl_type`.

        Returns
        -------
        ndarray -uint8
            Occlusion filter. (ones and zeros).
        """
        # Set defaults for arguments
        if occl_offset is None: occl_offset = 0

        oc_filter = np.ones((h, w))
        oc_filter = oc_filter.astype(dtype)

        if (occl_type is None) or (occl_type == 'b'):
            sh = math.ceil(occl_rate * h)
            en = math.floor((1-occl_offset) * h)
            st = en - sh
            if st < 0: st = 1
            oc_filter[st:en, 0:w] = 0

        elif occl_type == 'r':
            sh = math.ceil(occl_rate * w)
            en = math.floor((1-occl_offset) * w)
            st = en - sh
            if st < 0: st = 1
            oc_filter[0:h, st:en] = 0

        elif (occl_type == 't') or (occl_type == 'v'):
            sh = math.floor(occl_rate * h)
            st = math.floor(occl_offset * h)
            en = st + sh
            if en > h: en = h
            oc_filter[st:en, 0:w] = 0

        elif (occl_type == 'l') or (occl_type == 'h'):
            sh = math.floor(occl_rate * w)
            st = math.floor(occl_offset * w)
            en = st + sh
            if en > w: en = w
            oc_filter[0:h, st:en] = 0

        return oc_filter

    ##########################################################################
    # Protected Interface
    ##########################################################################
    @staticmethod
    def _set_default_cls_range(default_idx,  cls_ranges, col_ranges):
        """Set the value of the class range at default_idx 
            to undefined class ranges at other indices [val|te|tr_out|val_out].

        Parameters
        ----------
        default_idx : int
            Index for list of class ranges. Corresponds to the enumeration `Dataset`.

        cls_ranges : list of :obj:`list`
            Class range for each dataset. Indexed by enumeration `Dataset`.

        col_ranges : list of :obj:`list`
            Column range for each dataset. Indexed by enumeration `Dataset`.

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
        col_ranges : list of :obj:`list`
            Column range for each dataset. Indexed by enumeration `Dataset`.

        Returns
        -------
        :obj:`dict`
            Dictionary of nndbs. Keyed by enumeration `Dataset`.
        """
        dict_nndbs = {}

        # Iterate through ranges
        for ri, col_range in enumerate(col_ranges):
            edataset = Dataset.enum(ri)
            dict_nndbs.setdefault(edataset, None)

        return dict_nndbs

    @staticmethod
    def _perform_resize(cimg, scale):
        """Perform scaling related functions.

        Parameters
        ----------
        cimg : ndarray -uint8
            3D Data tensor. (format = H x W x CH)

        scale : float or tuple, optional
            Scale factor.
            * float - Fraction of current size.
            * tuple - Size of the output image.
             (Default value = None).

        Returns
        -------
        ndarray -uint8
            Scaled image.
        """
        if scale is not None:
            # LIMITATION: Python scipy imresize, unlike matlab
            _, _, ch = cimg.shape
            if ch == 1:
                cimg = scipy.misc.imresize(cimg[:, :, 0], scale)
                cimg = np.expand_dims(cimg, axis=2)
            else:
                # cimg = scipy.misc.imresize(cimg, scale, mode='F')
                if isinstance(scale, list) or isinstance(scale, tuple):
                    factor_x = scale[0] / cimg.shape[0]
                    factor_y = scale[1] / cimg.shape[1]
                    cimg = scipy.ndimage.interpolation.zoom(cimg, (factor_x, factor_y, 1.0))

                else:
                    cimg = scipy.ndimage.interpolation.zoom(cimg, (scale, scale, 1.0))

        return cimg

    @staticmethod
    def _process_color(img, sel):
        """Perform color related functions.

        Parameters
        ----------
        img : ndarray -uint8
            3D Data tensor. (format = H x W x CH)

        sel : :obj:`Selection`
            Selection object with the color processing fields.

        Returns
        -------
        ndarray -uint8
            Color processed image.
        """
        if img is None: return  # noqa E701
        _, _, ch = img.shape

        if sel.use_rgb is None:
            if sel.color_indices is not None:
                img = img[:, :, sel.color_indices]

        elif sel.use_rgb == False:
            if ch == 3:
                img = rgb2gray(img, img.dtype, keepDims=True)

            elif not ch == 1:
                raise Exception("nndb does not support grayscale conversion via `sel.use_rgb`.")

        elif sel.use_rgb == True:
            if ch == 1:
                raise Exception("nndb does not support color processing via `sel.use_rgb`. Reset sel.use_rgb=False")

        return img

    @staticmethod
    def _build_nndb_tr(nndbs, pi, is_new_class, img, noise_rate, occl_rate, occl_type, occl_offset, occl_filter):
        """Build the nndb training database.

        Parameters
        ----------
        nndbs : :obj:`list` of :obj:`NNdb`
            List of nndbs each corresponds to patches.

        pi : int
            Patch index that used to index `nndbs`.

        is_new_class : int
            Whether it's a new class or not.

        img : ndarray -uint8
            3D or 1D Data tensor. (format = H x W x CH or H)

        noise_rate : double
            Noise ratio.

        occl_rate : double
            Occlusion ratio.

        occl_type : char
            Occlusion type ('t':top, 'b':bottom, 'l':left, 'r':right).

        occl_offset : double
            Occlusion start offset (as a ratio) from top/bottom/left/right corner depending on `occl_type`.

        occl_filter : ndarray
            Occlusion filter for custom occlusion patterns. Format: H X W

        Returns
        -------
        :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if nndbs is None: return nndbs  # noqa E701
        nndb = nndbs[pi]

        # Adding user-defined occlusion filter or different occlusions depending on the percentage
        if occl_filter is not None or occl_rate is not None:

            # Information about image (img must be a 3D data tensor)
            h, w, ch = img.shape

            if (occl_filter is None):
                occl_filter = DbSlice.get_occlusion_patch(h, w, img.dtype, occl_type, occl_rate, occl_offset)

            occl_filter = np.expand_dims(occl_filter, 2)

            # For grey scale
            if ch == 1:
                img = occl_filter.astype(np.uint8) * img
            else:
                # For colored
                img = np.tile(occl_filter.astype(np.uint8), (1, 1, ch)) * img

        # Add different noise depending on the type
        # (ref. Enums/Noise)
        elif (noise_rate is not None) and (noise_rate == Noise.G):
            # Information about image (img must be a 3D data tensor)
            h, w, ch = img.shape
            pass
            # img = imnoise(img, 'gaussian')
            # #img = imnoise(img, 'gaussian')
            # #img = imnoise(img, 'gaussian')
            # #img = imnoise(img, 'gaussian')

        # Perform random corruption with the rate
        elif noise_rate is not None:
            # Information about image (img must be a 3D data tensor)
            h, w, ch = img.shape

            img = np.copy(img)

            # Corruption Size (H x W)
            cs = [np.uint16(h*noise_rate), np.uint16(w*noise_rate)]

            # Random location choice
            # Start of H, W (location)
            sh = np.uint8(1 + rand()*(h-cs[0]-1))
            sw = np.uint8(1 + rand()*(w-cs[1]-1))

            # Set the corruption
            corrupt_patch = np.uint8(DbSlice._rand_corrupt(cs[0], cs[1])).astype('uint8')  # noqa E501

            if ch == 1:
                img[sh:sh+cs[1], sw:sw+cs[2]] = corrupt_patch
            else:
                for ich in range(ch):
                    img[sh:sh+cs[0], sw:sw+cs[1], ich] = corrupt_patch

        # with Benchmark("DbSlice"):
        # Add data to nndb
        nndb.add_data(img)
    
        # Update the properties of nndb
        # with Benchmark("DbSlice"):
        nndb.update_attr(is_new_class)
        return nndbs

    @staticmethod
    def _rand_corrupt(height, width):
        """Corrupt the image with a (height, width) block.

        Parameters
        ----------
        height : int
            Height of the corruption block.

        width : int
            Width of the corruption block.

        Returns
        -------
        ndarray -uint8
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

        Parameters
        ----------
        nndbs : :obj:`list` of :obj:`NNdb`
            List of nndbs each corresponds to patches.

        pi : int
            Patch index that used to index `nndbs`.

        is_new_class : int
            Whether it's a new class or not.

        img : ndarray -uint8
            3D Data tensor. (format = H x W x CH)

        Returns
        -------
        :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if nndbs is None: return nndbs
        nndb = nndbs[pi]
        nndb.add_data(img)

        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs

    @staticmethod
    def _build_nndb_val(nndbs, pi, is_new_class, img):
        """"Build the nndb validation database.

        Parameters
        ----------
        nndbs : :obj:`list` of :obj:`NNdb`
            List of nndbs each corresponds to patches.

        pi : int
            Patch index that used to index `nndbs`.

        is_new_class : int
            Whether it's a new class or not.

        img : ndarray -uint8
            3D Data tensor. (format = H x W x CH)

        Returns
        -------
        :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if nndbs is None: return nndbs
        nndb = nndbs[pi]
        nndb.add_data(img)

        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs

    @staticmethod
    def _build_nndb_val_out(nndbs, pi, is_new_class, img):
        """"Build the nndb validation target database.

        Parameters
        ----------
        nndbs : :obj:`list` of :obj:`NNdb`
            List of nndbs each corresponds to patches.

        pi : int
            Patch index that used to index `nndbs`.

        is_new_class : int
            Whether it's a new class or not.

        img : ndarray -uint8
            3D Data tensor. (format = H x W x CH)

        Returns
        -------
        :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if nndbs is None: return nndbs
        nndb = nndbs[pi]
        nndb.add_data(img)

        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs

    @staticmethod
    def _build_nndb_te(nndbs, pi, is_new_class, img):
        """Build the testing database.

        Parameters
        ----------
        nndbs : :obj:`list` of :obj:`NNdb`
            List of nndbs each corresponds to patches.

        pi : int
            Patch index that used to index `nndbs`.

        is_new_class : int
            Whether it's a new class or not.

        img : ndarray -uint8
            3D Data tensor. (format = H x W x CH)

        Returns
        -------
        :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if nndbs is None: return nndbs
        nndb = nndbs[pi]
        nndb.add_data(img)

        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs

    @staticmethod
    def _build_nndb_te_out(nndbs, pi, is_new_class, img):
        """"Build the nndb validation target database.

        Parameters
        ----------
        nndbs : :obj:`list` of :obj:`NNdb`
            List of nndbs each corresponds to patches.

        pi : int
            Patch index that used to index `nndbs`.

        is_new_class : int
            Whether it's a new class or not.

        img : ndarray -uint8
            3D Data tensor. (format = H x W x CH)

        Returns
        -------
        :obj:`list` of :obj:`NNdb`
            NNdb objects for each patch.
        """
        if nndbs is None: return nndbs
        nndb = nndbs[pi]
        nndb.add_data(img)

        # Update the properties of nndb
        nndb.update_attr(is_new_class)
        return nndbs
