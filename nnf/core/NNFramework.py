# -*- coding: utf-8 -*-
"""
.. module:: NNFramework
   :platform: Unix, Windows
   :synopsis: Represent NNFramework, NNPatchMan, NNModelMan classes.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from warnings import warn as warning
from abc import ABCMeta, abstractmethod

# Local Imports
from nnf.db.NNdb import NNdb
from nnf.db.Dataset import Dataset
from nnf.db.DbSlice import DbSlice
from nnf.db.Selection import Selection
from nnf.core.NNDiskMan import NNDiskMan
from nnf.core.NNDiskMan import NNDiskManMode
from nnf.core.models.NNModel import NNModel
from nnf.core.models.NNModelPhase import NNModelPhase
from nnf.core.iters.disk.DskDataIterator import DskDataIterator
from nnf.core.iters.memory.MemDataIterator import MemDataIterator
from nnf.core.iters.memory.MemRTStreamIterator import MemRTStreamIterator
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator

# noinspection PyUnresolvedReferences
class NNFramework(object):
    """NNFramework represents the base class of the Neural Network Framework.

    .. warning:: abstract class and must not be instantiated.

        Process parameters from the user and create necessary objects
        to be utilized in the framework for further processing.

    Attributes
    ----------
    _SAVE_DIR_PREFIX : str
        [CONSTANT][INTERNAL_USE] Directory to save the processed data.

    __dict_diskman : dict of :obj:`NNDiskMan`
        :obj:`NNDiskMan` for each of the user `dbparam`.
    """
    __metaclass__ = ABCMeta

    # [CONSTANT][INTERNAL_USE] Directory to save the processed data
    _SAVE_DIR_PREFIX = "_processed"

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self):
        """Constructor of the abstract class :obj:`NNFramework`."""
        # Keep track of :obj:`NNDiskMan` objects for each `dbparam`
        self.__dict_diskman = {}

        # Keep track of the datasets used in each `dbparam`
        # for building the in memory or disk iterators
        # List, indexed by dbparam index
        self.datasets_in_use = []

        # Reset the uid at the entry of the test case
        NNModel.reset_uid()

    @staticmethod
    def init(generator, dbparams=None):
        """Initiate core nnf framework object depending on the :obj:`generator` object type.

        Parameters
        ----------
        generator : :obj:`NNPatchGenerator` | :obj:`NNModelGenerator`
            Neural Network patch generator to generate list of nnpatches.

        dbparams : list of :obj:`dict`
            List of user dbparams, each describing a database.
        """
        if isinstance(generator, NNPatchGenerator):
            core = NNPatchMan(generator, dbparams)
        else:
            core = NNModelMan(generator, dbparams)

        return core

    @abstractmethod
    def pre_train(self, precfgs=None, cfg=None):
        """Initiate pre-training in the framework.

        Parameters
        ----------
        precfgs : list of :obj:`NNCfg`
            List of Neural Network configurations. Useful for layer-wise pre-training.

        cfg : :obj:`NNCfg`
            Neural Network configuration that will be used in training. 
            Useful to build the deep stacked network after layer-wise pre-training.
        """
        pass

    @abstractmethod
    def train(self, cfg=None):
        """Initiate training in the framework.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.        
        """ 
        pass

    @abstractmethod
    def test(self, cfg=None):
        """Initiate testing in the framework.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.        
        """ 
        pass

    @staticmethod
    def get_param_map(param):
        """Fetch parameter set for each dataset.

        Parameters
        ----------
        param : :obj:`list` or :obj:`dict`
            List of `dict` element, `tuple` element in the following format.
            [{default_param},
            (Dataset.TR, {params}),
            (Dataset.VAL, {params}),
            ((Dataset.TE, Dataset.TE_OUT), {params})]

            OR
            `dict` indicating the {default_params}

        Returns
        -------
        :obj:`dict`
            Dictionary of pre-processor parameters keyed by Dataset
            enumeration. (i.e Dataset.TR|VAL|...).
        """
        param_map = {}
        default_param = None

        if isinstance(param, dict):  # {}
            default_param = param.copy()

        elif isinstance(param, list):  # [{}, (..., {}), (..., {}), ((..., ...), {})]
            for entry in param:  # (..., ...) or ((..., ...), {})
                if isinstance(entry, tuple):
                    edatasets, param = entry

                    if isinstance(edatasets, Dataset):
                        edatasets = [edatasets]

                    for edataset in edatasets:
                        ptmp = param.copy()
                        param_map[edataset] = {**default_param, **ptmp} if default_param is not None else {**ptmp}

                elif isinstance(entry, dict):
                    default_param = entry.copy()

                else:
                    raise Exception("Unsupported Format")

        edatasets = Dataset.get_enum_list()
        for edataset in edatasets:
            if edataset not in param_map:
                param_map.setdefault(edataset, None if default_param is None else default_param.copy())

        return param_map

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def __update_nnpatches_with_rl_stream(self, idbp, rt_stream, nnpatches):

        for pi, nnpatch in enumerate(nnpatches):
            nnpatch = nnpatches[pi]

            # Initialize an empty set, add `edataset` to the set
            if (idbp + 1) > len(self.datasets_in_use):
                self.datasets_in_use.append(set())

            # Set rt_stream for nnpatch by dbparam and Dataset.TR|VAL|TE|...
            # nnpatch -> i.e [dbparam][Dataset.TR] = [rlstream_nnpatch_dbparam_TR]
            dict_rtstream = nnpatch.setdefault_udata(idbp, {})

            # TODO: Clone rt_stream, change rt_stream.element_indices depending on nnpatch
            if rt_stream.tr_col_indices is not None:
                dict_rtstream[Dataset.TR] = rt_stream
                self.datasets_in_use[idbp].add(Dataset.TR)

            if rt_stream.val_col_indices is not None:
                dict_rtstream[Dataset.VAL] = rt_stream
                self.datasets_in_use[idbp].add(Dataset.VAL)

            if rt_stream.te_col_indices is not None:
                dict_rtstream[Dataset.TE] = rt_stream
                self.datasets_in_use[idbp].add(Dataset.TE)

    def __update_nnpatches_with_nndb_patch(self, idbp, nndbs_tup, nnpatches):

        # Iterate through nnpatches
        for pi, nnpatch in enumerate(nnpatches):
            nnpatch = nnpatches[pi]
            edatasets = nndbs_tup[-1]

            # Iterate through the results of DbSlice above
            for ti in range(len(nndbs_tup) - 1):
                nndbs = nndbs_tup[ti]

                # Set nndb for nnpatch by dbparam and Dataset.TR|VAL|TE|...
                # nnpatch -> i.e [dbparam][Dataset.TR] = [nndb_patch_dbparam_TR]
                if nndbs is not None:
                    dict_nndb = nnpatch.setdefault_udata(idbp, {})
                    edataset = edatasets[ti]
                    dict_nndb[edataset] = nndbs[pi] if (not isinstance(nndbs, NNdb)) else nndbs

                    # Initialize an empty set, add `edataset` to the set
                    if (idbp + 1) > len(self.datasets_in_use):
                        self.datasets_in_use.append(set())
                    self.datasets_in_use[idbp].add(edataset)

    def __create_and_init_nndskman_for_dpparam(self, idbp, fn_nndiskman, sel, dskdb_param, memdb_param, save_dir):

        dskman = fn_nndiskman(sel_dskdb, dskdb_param, memdb_param, save_dir) \
            if (fn_nndiskman is not None) else NNDiskMan(sel, dskdb_param, memdb_param, save_dir)

        # init() will utilzie sel, sel.nnpatches for further processing
        dskman.init()

        # Update datasets_in_use with `dskman.datasets_in_use` list
        # Union for sets
        if (idbp + 1) > len(self.datasets_in_use):
            self.datasets_in_use.append(set())
            self.datasets_in_use[idbp] = dskman.datasets_in_use
        else:
            self.datasets_in_use[idbp] = self.datasets_in_use[idbp] | dskman.datasets_in_use

        return dskman

    def _process_db_params(self, nnpatches, dbparams):
        """Process user dbparams (database related parameters)
           
            Process the database and attach slices to corresponding nnpatches
            if the database is in memory. Otherwise initialize nndiskman to process
            disk database.

        Parameters
        ----------
        nnpatches : list of :obj:`NNPatch`
            List of :obj:`NNPatch` instances.

        dbparams : list of :obj:`dict`
            List of user dbparams, each describing a database.
        """
        # When a pre-loaded database are in use
        if dbparams is None:
            return

        # Iterate through dbparams
        for idbp, dbparam in enumerate(dbparams):

            # Read fields from dbparam. Also set default values for undefined keys
            alias = dbparam.setdefault('alias', None)
            memdb_param = dbparam.setdefault('memdb_param', None)
            sel = dbparam.setdefault('selection', None)
            rt_stream = dbparam.setdefault('rt_stream', None)
            dskdb_param = dbparam.setdefault('dskdb_param', None)
            fn_nndiskman = dbparam.setdefault('fn_nndiskman', None)


            # Database directory for disk data feeding
            db_dir = dskdb_param['db_dir'] if (dskdb_param is not None) else None

            # Fetch memory database related parameters
            nndb = None; memdb_pp = None; nndbs_tup = None
            if memdb_param is not None:
                nndb = memdb_param['nndb'] if 'nndb' in memdb_param else None
                memdb_pp = memdb_param['pp'] if 'pp' in memdb_param else None
                nndbs_tup = memdb_param['nndbs_tup'] if 'nndbs_tup' in memdb_param else None

            # Whether databases are available or not
            data_in_mem = nndb is not None or nndbs_tup is not None
            data_in_dsk = db_dir is not None
            data_in_rts = rt_stream is not None

            # Init dskman, sel_dskdb
            dskman = None
            sel_dskdb = None

            # For real time data stream that generates synthetic data on the go
            if data_in_rts:

                # Check whether data_in_mem and data_in_dsk (db_dir is not None) is not used when rt_stream is utilized
                if data_in_mem or data_in_dsk:
                    raise Exception("Using real-time stream does not support dskdb_param['db_dir'] or memdb_param['nndb'] or memdb_param['nndbs_tup'].")

                # Selection structure should not utilized in real time data stream
                if sel is not None:
                    warning("dbparam['selection'] will be ignored for real-time data stream.")

                # Set rt_stream @ each nnpatch
                self.__update_nnpatches_with_rl_stream(idbp, rt_stream, nnpatches)

            else:
                # Error handling for invalid configuration setups
                if not data_in_mem and not data_in_dsk:
                    raise Exception("dskdb_param['db_dir'] or memdb_param['nndb'] or memdb_param['nndbs_tup'] unspecified.")

                if nndb is not None and nndbs_tup is not None:
                    assert(data_in_mem)
                    warning("memdb_param['nndb'] and memdb_param['nndbs_tup'] both are provided. "
                            "Discarding memdb_param['nndb'] for DbSlice.")

                # When data in disk
                if data_in_dsk:

                    # Used in `NNDiskMan`, also below
                    dskdb_param.setdefault('force_load_db', False)

                    # Initialize selection for disk database
                    selection_choice = None
                    if dskdb_param['force_load_db']:
                        if 'selection' in dskdb_param:
                            selection_choice = "dskdb_param['selection']"
                            sel_dskdb = dskdb_param['selection']
                        else:
                            warning("dskdb_param['selection'] is empty. Utilizing the global `selection` specified.")
                            selection_choice = "selection"
                            sel_dskdb = sel
                    else:
                        selection_choice = "selection"
                        sel_dskdb = sel

                    # Check `NNDiskman` compatibility with the user provided configurations
                    if memdb_param is None: memdb_param = {}
                    memdb_param.setdefault('nndb', None)
                    memdb_param.setdefault('nndbs_tup', None)

                    nndskman_mode = NNDiskMan.mode(dskdb_param, memdb_param)
                    assert nndskman_mode is not NNDiskManMode.INVALID  # data atleast should be in the disk

                    # nndb -> process sel -> write to disk
                    if nndskman_mode == NNDiskManMode.MEM_TO_DSK:
                        if nndbs_tup is not None:
                            warning("memdb_param['nndb'] and memdb_param['nndbs_tup'] both are provided. "
                                    "Discarding memdb_param['nndbs_tup'] for `NNDiskMan`.")

                    elif nndskman_mode == NNDiskManMode.UNSUPPORTED:
                        raise Exception(
                            """memdb_param['nndb'] is not given for in memory data traversal of `NNDiskMan`.""")

                    # db_dir -> process sel -> write to disk
                    elif nndskman_mode == NNDiskManMode.DSK:
                        if sel.scale is not None:
                            warning(selection_choice + ".scale will not be used in processing disk data. "
                                                       "Use dskdb_param['target_size'] and iter_param['input_shape'] instead.")

                # Set defaults for sel
                if sel is None:
                    dbparam['selection'] = sel = Selection()

                # Detecting conflicts with sel.nnpatches
                if sel.nnpatches is not None:
                    warning('ARG_CONFLICT: selection.nnpatches is already set. '
                            'Discarding selection.nnpatches...')
                sel.nnpatches = nnpatches

            # If iterators are needed for in memory databases
            if data_in_mem:

                # Split the database according to the `sel` or use nndbs from nndbs_tup
                if nndb is not None and nndbs_tup is None:
                    nndbs_tup = DbSlice.slice(nndb, sel, pp_param=memdb_pp)

                # Set nndb_patch @ each nnpatch
                self.__update_nnpatches_with_nndb_patch(idbp, nndbs_tup, nnpatches)

            # If iterators are needed for in disk databases, initialize a nndiskman for each dbparam
            if data_in_dsk:

                # Create diskman for each idbp and init()
                save_dir = NNFramework._SAVE_DIR_PREFIX + "_" + (str(idbp) if (alias is None) else alias)
                dskman = self.__create_and_init_nndskman_for_dpparam(idbp,
                                                                     fn_nndiskman,
                                                                     sel_dskdb,
                                                                     dskdb_param,
                                                                     memdb_param,
                                                                     save_dir)

            # Add dskman to dictionary of dskmans (add `None` if unavailable)
            self.__dict_diskman.setdefault(idbp, dskman)

    def _init_model_params(self, nnpatches, dbparams, nnmodel=None):
        """Initialize `nnmodels` at `nnpatches` along with iterstores.

        Parameters
        ----------
        nnpatches : list of :obj:`NNPatch`
            List of :obj:`NNPatch` instances.

        dbparams : list of :obj:`dict`
            List of user dbparams, each describing a database.

        nnmodel : :obj:`NNModel`, optional
            Neural Network model instance to update. (Default = None)
        """
        # Precondition assert: index for dbparams is used 
        # assuming the order of the items in the list
        if dbparams is not None:
            assert(isinstance(dbparams, list))

        # Iterate the patch list
        for nnpatch in nnpatches:

            # Construct `nnmodels` for `nnpatch` (if required)
            nnmodels = nnpatch._generate_nnmodels(nnpatch) if nnmodel is None else nnmodel
            if isinstance(nnmodels, NNModel):
                nnmodels = [nnmodels]

            # Iterate the nnmodels under 'nnpatch'
            for nnmodel in nnmodels:

                # Register `nnmodel` to `nnpatch`
                nnpatch.add_nnmodels(nnmodel)

                # Register `nnpatch` to `nnmodel`
                nnmodel.add_nnpatches(nnpatch)

                # Initialize dict|list of iterstore for nnmodel @ nnpatch
                # dict_iterstore => {alias1:[iter_TR, iter_VAL, ...], alias2:[iterstore_for_param2_db], ...}
                dict_iterstore = None

                # list_iterstore => [[iter_TR, iter_VAL, ...], [iterstore_for_dbparam2_db], ...]
                list_iterstore = None

                # save_dir_abspaths => [folder_for_dbparam_1_db, folder_for_dbparam_2_db] for each patch
                save_dir_abspaths = None  # PERF

                # IMPORTANT: (dbparams is None) when using pre-loaded database
                if dbparams is not None:

                    # Iterate through dbparams
                    for idbp, dbparam in enumerate(dbparams):

                        # Read fields from dbparam
                        alias = dbparam['alias']
                        sel = dbparam['selection']
                        rt_stream = dbparam.setdefault('rt_stream', None)
                        dskdb_param = dbparam['dskdb_param']

                        # Get iterator parameters via nnmodel
                        iter_params = {}

                        if nnmodel.iter_params is not None:
                            if isinstance(nnmodel.iter_params, list):

                                if isinstance(nnmodel.iter_params[0], list):
                                    iter_params = nnmodel.iter_params[idbp]
                                else:
                                    iter_params = nnmodel.iter_params

                            elif isinstance(nnmodel.iter_params, dict):
                                iter_params = nnmodel.iter_params[alias] if alias in nnmodel.iter_params else nnmodel.iter_params

                        # Get iterator preprocessor parameters via nnmodel
                        iter_pp_params = {}
                        if nnmodel.iter_pp_params is not None:
                            if isinstance(nnmodel.iter_pp_params, list):

                                if isinstance(nnmodel.iter_pp_params[0], list):
                                    iter_pp_params = nnmodel.iter_pp_params[idbp]
                                else:
                                    iter_pp_params = nnmodel.iter_pp_params

                            elif isinstance(nnmodel.iter_pp_params, dict):
                                iter_pp_params = nnmodel.iter_pp_params[alias] if alias in nnmodel.iter_pp_params else nnmodel.iter_pp_params

                        # Iterator pre-processing params for each dataset
                        iter_pp_param_map = self.get_param_map(iter_pp_params)

                        # Iterator params for each dataset
                        iter_param_map = self.get_param_map(iter_params)

                        # Resolve conflicts with selection structure against other parameters defined in iter_param
                        self.__resolve_conflicts(sel, iter_param_map)

                        # Iterator store for this dbparam
                        iterstore = {}

                        # PERF: Create the dictionary list in 1st iteration, if only necessary
                        if idbp == 0:
                            list_iterstore = []
                            if alias is not None:
                                dict_iterstore = {}

                        # Get nndb or rt_stream dictionary for this dbparam
                        dict_data = nnpatch.setdefault_udata(idbp, {})

                        # Diskman for this dbparam
                        diskman = self.__dict_diskman[idbp]

                        # Create data iterators (memory/disk) for each dataset (Dataset.TR|VAL|TE|...)
                        edatasets = Dataset.get_enum_list()
                        for edataset in edatasets:

                            # If `edataset` is not used in this dbparam
                            if not (edataset in self.datasets_in_use[idbp]):
                                continue

                            # Get iter_param, iter_pp_param for `edataset`
                            ds_iter_param = iter_param_map[edataset]
                            ds_iter_pp_param = iter_pp_param_map[edataset]
                            iter_in_mem = ds_iter_param['in_mem'] if ('in_mem' in ds_iter_param) else True
                            fn_coreiter = ds_iter_param['fn_coreiter'] if ('fn_coreiter' in ds_iter_param) else None

                            # If in-memory iterators are required and in-memory stream info is available
                            if iter_in_mem and (rt_stream is not None) and (edataset in dict_data):
                                rt_stream = dict_data[edataset]
                                memiter = MemRTStreamIterator(edataset,
                                                                rt_stream,
                                                                ds_iter_pp_param,
                                                                fn_coreiter)
                                memiter.init_ex(ds_iter_param)

                                # Add memory iterator created above
                                iterstore.setdefault(edataset, memiter)

                            # If in-memory iterators are required and in-memory data is available
                            elif iter_in_mem and (edataset in dict_data):

                                nndb_dataset = dict_data[edataset]
                                memiter = MemDataIterator(edataset,
                                                          nndb_dataset,
                                                          ds_iter_pp_param,
                                                          fn_coreiter)

                                # PP setting for training dataset
                                tr_setting = None

                                # Fetch TR pre-process setting and apply it on other
                                if edataset == Dataset.TR:
                                    tr_setting = memiter.init_ex(ds_iter_param)
                                else:
                                    memiter.init_ex(ds_iter_param, setting=tr_setting)

                                # Add memory iterator created above
                                iterstore.setdefault(edataset, memiter)

                            # If disk iterators are required and disk data is available
                            elif not iter_in_mem and diskman is not None:

                                # Update save_dir_abspaths
                                save_dir_abspaths = [] if save_dir_abspaths is None else save_dir_abspaths  # PERF
                                save_dir_abspaths.append(diskman.save_dir_abspath)

                                dskiter = None

                                # PERF: Create iterator only if necessary
                                if diskman.get_nb_class(edataset) > 0:
                                    frecords = diskman.get_frecords(nnpatch.id, edataset)
                                    nb_class = diskman.get_nb_class(edataset)
                                    dskiter = DskDataIterator(edataset,
                                                              frecords,
                                                              nb_class,
                                                              ds_iter_pp_param,
                                                              fn_coreiter)

                                    # Assign common dskiter params from dskdb_param
                                    ext_ds_iter_param = ds_iter_param.copy()
                                    ext_ds_iter_param['target_size'] = dskdb_param['target_size'] if ('target_size' in dskdb_param) else None
                                    ext_ds_iter_param['file_format'] = dskdb_param['file_format'] if ('file_format' in dskdb_param) else None
                                    ext_ds_iter_param['data_field'] = dskdb_param['data_field'] if ('data_field' in dskdb_param) else None
                                    dskiter.init_ex(params=ext_ds_iter_param)

                                # Add disk iterator created above (add `None` if unavailable)
                                iterstore.setdefault(edataset, dskiter)

                            elif not iter_in_mem and (edataset in dict_data):
                                raise Exception("Disk data iterators cannot read data from memory: iter_params['in_mem']=False")

                            elif iter_in_mem and diskman is not None:
                                raise Exception("In memory data iterators cannot read data from disk: iter_params['in_mem']=True")

                            else:
                                # Add `None` for edataset, otherwise
                                iterstore.setdefault(edataset, None)

                        # Update dict_iterstore and list_iterstore
                        if alias is not None: dict_iterstore.setdefault(alias, iterstore)
                        list_iterstore.append(iterstore)

                # Add the iterstores and save directory paths, indexed by `nnpatch` index
                nnmodel.add_iterstores(list_iterstore, dict_iterstore)
                nnmodel.add_save_dirs(save_dir_abspaths)

    def release(self):
        """Release internal resources used by NNFramework."""
        del self.__dict_diskman
        self.__dict_diskman = None

    ##########################################################################
    # Private Interface
    ##########################################################################
    @staticmethod
    def __resolve_conflicts(sel, iter_param_map):
        """Resolve conflicts with selection structure against other iterator parameters

           Note: Selection structure has high priority than iterator parameters
        """
        edatasets = Dataset.get_enum_list()

        for edataset in edatasets:
            iter_param = iter_param_map[edataset]

            if 'color_mode' in iter_param:
                assert(False)  # Support suspended, moved to internal config '_use_rgb'

            if sel is not None:
                # This parameter is not used in memory data iterators, but disk data iterators
                iter_param['_use_rgb'] = sel.use_rgb

            # Remove: legacy code
            # # Detecting conflicts with sel.use_rgb and iter_param['color_mode']
            # if sel.use_rgb is not None:
            #
            #     # Issue a warning if iter_param also has a color_mode that differ from sel.use_rgb
            #     if 'color_mode' in iter_param:
            #         assert
            #         cmode = iter_param['color_mode']
            #         if (cmode != 'rgb' and sel.use_rgb) or (cmode != 'grayscale' and not sel.use_rgb):
            #             warning("[ARG_CONFLICT] iter_param['color_mode']:" + str(cmode) +
            #                     " with selection.use_rgb: " + str(sel.use_rgb) +
            #                     ". Hence selection.use_rgb is prioritized.")
            #
            #     # Assign the color_mode to the iter_param dictionary
            #     # This parameter is not used in memory data iterators, but disk data iterators
            #     iter_param['color_mode'] = 'rgb' if sel.use_rgb else 'grayscale'

class NNPatchMan(NNFramework):
    """`NNPatchMan` represents patch based sub framework in `NNFramework'.

    Attributes
    ----------
    nnpatches : list of :obj:`NNPatch`
        List of :obj:`NNPatch` instances.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, generator, dbparams=None):
        """Constructs :obj:`NNPatchMan` instance.

        Parameters
        ----------
        generator : :obj:`NNPatchGenerator`
            Neural Network patch generator to generate list of nnpatches.

        dbparams : list of :obj:`dict`
            List of user dbparams, each describing a database.
        """
        super().__init__()

        if (isinstance(dbparams, dict)):
            dbparams = [dbparams]

        # Initialize instance variables
        self.nnpatches = []

        # Generate the nnpatches
        self.nnpatches = generator.generate_nnpatches()

        # Process dbparams and attach dbs to nnpatches
        self._process_db_params(self.nnpatches, dbparams)

        # Process dbparams against the nnmodels
        self._init_model_params(self.nnpatches, dbparams)

    def pre_train(self, prenncfgs=None, nncfg=None):
        """Initiate pre-training in the framework.

        Parameters
        ----------
        prenncfgs : list of :obj:`NNCfg`
            List of Neural Network configurations. Useful for layer-wise pre-training.

        nncfg : :obj:`NNCfg`
            Neural Network configuration that will be used in training.
            Useful to build the deep stacked network after layer-wise pre-training.

        Notes
        -----
        Some of the layers may not be pre-trained. Hence precfgs itself is
        not sufficient to determine the architecture of the final
        stacked network.

        FUTURE: Parallel processing (1 level - model level) or (2 level - patch level)
        """
        # Parallelization Level-1: patch level
        # Parallelization Level-2: each patch's model level
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:

                # If `prenncfgs` or `nncfg` is already provided at NNModel creation, prioritize !
                if (nnmodel.nncfgs is not None):
                    if NNModelPhase.PRE_TRAIN in nnmodel.nncfgs:
                        tmp = nnmodel.nncfgs[NNModelPhase.PRE_TRAIN]
                        if tmp is not None and prenncfgs is not None:
                            warning('NNModel already contains pre-training model configurations `prenncfgs`. '
                                    'Discarding `prenncfgs` provided to pre_train(...) method')
                        prenncfgs = tmp

                    if NNModelPhase.TRAIN in nnmodel.nncfgs:
                        tmp = nnmodel.nncfgs[NNModelPhase.TRAIN]
                        if tmp is not None and nncfg is not None:
                            warning('NNModel already contains a model configuration `nncfg`. '
                                    'Discarding `nncfg` provided to pre_train(...) method')
                        nncfg = tmp

                nnmodel.pre_train(prenncfgs, nncfg)

    def train(self, nncfg=None):
        """Initiate training in the framework.

        Parameters
        ----------
        nncfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        Notes
        -----
        FUTURE: Parallelize processing (1 level - model level) or (2 level - patch level)
        """
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:

                # If `nncfg` is already provided at NNModel creation, prioritize !
                if ((nnmodel.nncfgs is not None) and
                    (NNModelPhase.TRAIN in nnmodel.nncfgs)):
                        tmp = nnmodel.nncfgs[NNModelPhase.TRAIN]
                        if tmp is not None and nncfg is not None:
                            warning('NNModel already contains a model configuration `nncfg`. '
                                    'Discarding `nncfg` provided to train(...) method')
                        nncfg = tmp

                nnmodel.train(nncfg)

    def test(self, nncfg=None):
        """Initiate testing in the framework.

        Parameters
        ----------
        nncfg : :obj:`NNCfg`
            Neural Network configuration used in training.
        """
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:

                # If `nncfg` is already provided at NNModel creation, prioritize !
                if ((nnmodel.nncfgs is not None) and
                    (NNModelPhase.TEST in nnmodel.nncfgs)):
                        tmp = nnmodel.nncfgs[NNModelPhase.TEST]
                        if tmp is not None and nncfg is not None:
                            warning('NNModel already contains a model configuration `nncfg`. '
                                    'Discarding `nncfg` provided to test(...) method')
                        nncfg = tmp

                nnmodel.test(nncfg)

    def predict(self, nncfg=None):
        """Initiate predict in the framework.

        Parameters
        ----------
        nncfg : :obj:`NNCfg`
            Neural Network configuration used in training.
        """
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:

                # If `nncfg` is already provided at NNModel creation, prioritize !
                if ((nnmodel.nncfgs is not None) and
                    (NNModelPhase.PREDICT in nnmodel.nncfgs)):
                        tmp = nnmodel.nncfgs[NNModelPhase.PREDICT]
                        if tmp is not None and nncfg is not None:
                            warning('NNModel already contains a model configuration `nncfg`. '
                                    'Discarding `nncfg` provided to predict(...) method')
                        nncfg = tmp

                nnmodel.predict(nncfg)

    def release(self):
        """Release internal resources used by NNFramework."""
        super().release()
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:
                nnmodel.release_iterstores()

class NNModelMan(NNFramework):
    """`NNModelMan` represents model based sub framework in `NNFramework'.

    Attributes
    ----------
    nnmodels : list of :obj:`NNModel`
        List of Neural Network model instances.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, generator, dbparams=None):
        """Constructs :obj:`NNModelMan` instance.

        Parameters
        ----------
        generator : :obj:`NNModelGenerator`
            Neural Network model generator to generate list of models.

        dbparams : list of :obj:`dict`
            List of user dbparams, each describing a database.
        """
        super().__init__()

        if (isinstance(dbparams, dict)):
            dbparams = [dbparams]

        # Generate the nnmodels
        self.nnmodels = generator.generate_nnmodels()

        # Iterate through nnmodels
        for nnmodel in self.nnmodels:
            # Generate nnpatches
            nnpatches = nnmodel._generate_nnpatches()
            # nnmodel.init_nnpatches()

            # Process dbparams and attach dbs to nnpatches
            self._process_db_params(nnpatches, dbparams)

            # Process dbparams against the nnmodels
            self._init_model_params(nnpatches, dbparams, nnmodel)

    def pre_train(self, precfgs=None, cfg=None):
        """Initiate pre-training in the framework.

        Parameters
        ----------
        precfgs : list of :obj:`NNCfg`
            List of Neural Network configurations. Useful for layer-wise pre-training.

        cfg : :obj:`NNCfg`
            Neural Network configuration that will be used in training.
            Useful to build the deep stacked network after layer-wise pre-trianing.

        Notes
        -----
        Some of the layers may not be pre-trianed. Hence precfgs itself is
        not sufficient to determine the architecture of the final
        stacked network.

        FUTURE: Parallelize processing (1 level - model level) or (2 level - patch level)
        """
        # Parallelization Level-1: model level
        for nnmodel in self.nnmodels:
            nnmodel.pre_train(precfgs, cfg)

        # Parallelization Level-2: each model's patch level
        # for nnmodel in self.nnmodels:
        #    for nnpatch in nnmodel.nnpatches:
        #        patch_idx = 0
        #        nnmodel.pre_train(precfgs, cfg, patch_idx)
        #        patch_idx += 1

    def train(self, cfg=None):
        """Initiate training in the framework.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.

        Notes
        -----
        FUTURE: Parallelize processing (1 level - model level) or (2 level - patch level)
        """
        # Parallelization Level-1: model level
        for nnmodel in self.nnmodels:
            nnmodel.train(cfg)

        # Parallelization Level-2: each model's patch level
        # for nnmodel in self.nnmodels:
        #    for nnpatch in nnmodel.nnpatches:
        #        patch_idx = 0
        #        nnmodel.train(cfg, patch_idx)
        #        patch_idx += 1

    def test(self, cfg=None):
        """Initiate testing in the framework.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.
        """
        # Parallelization Level-1: model level
        for nnmodel in self.nnmodels:
            nnmodel.test(cfg)

        # Parallelization Level-2: each model's patch level
        # for nnmodel in self.nnmodels:
        #    for nnpatch in nnmodel.nnpatches:
        #        patch_idx = 0
        #        nnmodel.test(cfg, patch_idx)
        #        patch_idx += 1

    def release(self):
        """Release internal resources used by NNFramework."""
        super().release()
        for nnmodel in self.nnmodels:
            nnmodel.release_iterstores()