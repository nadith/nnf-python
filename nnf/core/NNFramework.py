# -*- coding: utf-8 -*-
"""
.. module:: NNFramework
   :platform: Unix, Windows
   :synopsis: Represent NNFramework class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from abc import ABCMeta, abstractmethod
from warnings import warn as warning


# Local Imports
from nnf.db.NNdb import NNdb
from nnf.core.NNDiskMan import NNDiskMan
from nnf.core.iters.disk.DskDataIterator import DskDataIterator
from nnf.core.iters.memory.MemDataIterator import MemDataIterator
from nnf.core.models.NNModel import NNModel
from nnf.db.Dataset import Dataset
from nnf.db.DbSlice import DbSlice
from nnf.db.Selection import Selection


# noinspection PyUnresolvedReferences
class NNFramework(object):
    """`NNFramework` represents the base class of the Neural Network Framework.

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

        # Reset the uid at the entry of the test case
        NNModel.reset_uid()

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

    @abstractmethod
    def pre_train_with(self, generator):
        """Initiate pre-training in the framework.

        Parameters
        ----------
        generator : :obj:`NNCfgGenerator`
            Neural Network configuration generator for each model.
        """
        pass

    @abstractmethod
    def train_with(self, generator):
        """Initiate training in the framework.

        Parameters
        ----------
        generator : :obj:`NNCfgGenerator`
            Neural Network configuration generator for each model.
        """ 
        pass

    @abstractmethod
    def test_with(self, generator):
        """Initiate testing in the framework.

        Parameters
        ----------
        generator : :obj:`NNCfgGenerator`
            Neural Network configuration generator for each model.
        """ 
        pass

    ##########################################################################
    # Protected Interface
    ##########################################################################
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
        # When pre-loaded database are in use
        if dbparams is None:
            return

        # Iterate through dbparams
        for idbp, dbparam in enumerate(dbparams):

            # Read fields from dbparam. Also set default values for undefined keys
            alias = dbparam.setdefault('alias', None)
            nndb = dbparam.setdefault('nndb', None)
            sel = dbparam.setdefault('selection', None)
            dskman_param = dbparam.setdefault('dskman_param', None)
            iter_pp_param = dbparam.setdefault('iter_pp_param', None)
            dbparam.setdefault('iter_param', None)
            iter_in_mem = dbparam.setdefault('iter_in_mem', True)
            fn_nndiskman = dbparam.setdefault('fn_nndiskman', None)
            dbparam.setdefault('fn_coreiter', None)

            # Database directory
            db_dir = dskman_param['db_dir'] if (dskman_param is not None) else None

            # Diskman iterator pre-processing parameters
            dskman_pp = None
            if dskman_param is not None:
                if 'pp' in dskman_param:
                    dskman_pp = dskman_param['pp']

            # Conflict resolution
            if (nndb is not None) and (dskman_pp is not None) and (iter_pp_param is not None):
                warning("dskman_pp: ignored, iter_pp_param is used")

            # Set defaults for sel
            if sel is None:
                dbparam['selection'] = sel = Selection()

            # Detecting conflicts with sel.nnpatches
            if sel.nnpatches is not None:
                warning('ARG_CONFLICT: sel.nnpatches is already set. '
                        'Discarding sel.nnpatches...')
            sel.nnpatches = nnpatches

            # If iterators are needed for in memory databases
            if iter_in_mem:

                # Warning
                if nndb is None:
                    warning("""nndb: is not given. 
                                Hence data iterators will not be created.
                                Make sure to use an external DB via NNCfg for training, testing, etc""")
                    continue

                # Split the database according to the `sel`
                nndbs_tup = DbSlice.slice(nndb, sel, pp_param=dskman_pp)

                # Iterate through nnpatches
                for pi, nnpatch in enumerate(nnpatches):
                    nnpatch = nnpatches[pi]                    
                    edatasets = nndbs_tup[-1]

                    # Iterate through the results of DbSlice above
                    for ti in range(len(nndbs_tup)-1):                        
                        nndbs = nndbs_tup[ti]

                        # Set nndb for nnpatch by dbparam and Dataset.TR|VAL|TE|...
                        # nnpatch -> i.e [dbparam][Dataset.TR] = [nndb_patch_dbparam_TR]
                        if nndbs is not None:
                            dict_nndb = nnpatch.setdefault_udata(idbp, {})
                            edataset = edatasets[ti]                            
                            dict_nndb[edataset] = nndbs[pi] if (not isinstance(nndbs, NNdb)) else nndbs

            # Initialize NNDiskman
            if db_dir is None and not iter_in_mem:
                warning("""image directory: is not given. 
                    Hence data iterators will not be created.
                    Make sure to use an external DB via NNCfg for training, testing, etc""")
                dskman = None

            elif db_dir is None:
                dskman = None

            else:
                # Set defaults
                dskman_param.setdefault('pp', None)

                # Create diskman and init()
                # The init() will use sel, sel.nnpatches for further processing
                save_dir = NNFramework._SAVE_DIR_PREFIX + "_" + (str(idbp) if (alias is None) else alias)
                dskman = fn_nndiskman(sel, dskman_param, nndb, save_dir)\
                    if (fn_nndiskman is not None) else NNDiskMan(sel, dskman_param, nndb, save_dir)
                dskman.init()   

            # Add dskman to dictionary of dskmans
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

            # Initialize dict|list of iterstore for nnmodel @ nnpatch
            # dict_iterstore => {alias1:[iter_TR, iter_VAL, ...], alias2:[iterstore_for_param2_db], ...}
            dict_iterstore = None 

            # list_iterstore => [[iter_TR, iter_VAL, ...], [iterstore_for_dbparam2_db], ...]
            list_iterstore = None

            # save_dir_abspaths => [folder_for_dbparam_1_db, folder_for_dbparam_2_db]
            save_dir_abspaths = None  # PERF

            # When pre-loaded database are in use
            if dbparams is not None:

                # Iterate through dbparams
                for idbp, dbparam in enumerate(dbparams):

                    # Read fields from dbparam
                    alias = dbparam['alias']
                    # Following fields are utilized in `_process_db_params(...) method
                    # nndb = dbparam['nndb']
                    sel = dbparam['selection']
                    dskman_param = dbparam['dskman_param']
                    iter_pp_param = dbparam['iter_pp_param']
                    iter_param = dbparam['iter_param']
                    iter_in_mem = dbparam['iter_in_mem']
                    fn_coreiter = dbparam['fn_coreiter']

                    # Iterator pre-processing params for each dataset
                    iter_pp_param_map = self.__get_param_map(iter_pp_param)

                    # Iterator params for each dataset
                    iter_param_map = self.__get_param_map(iter_param)

                    # Resolve conflicts with selection structure against other parameters defined in iter_param
                    self.__resolve_conflicts(sel, iter_param_map)

                    # Iterator store for this dbparam
                    iterstore = {}

                    # PERF: Create the dictionary list in 1st iteration, if only necessary
                    if idbp == 0:
                        list_iterstore = []
                        if alias is not None:
                            dict_iterstore = {}

                    # If iterators are need for in memory databases
                    if iter_in_mem:

                        # Get nndb dictionary for this dbparam
                        dict_nndb = nnpatch.setdefault_udata(idbp, {})

                        # Create iterators for each dataset in nndb of this dbparam
                        edatasets = Dataset.get_enum_list()

                        # PP setting for training dataset
                        tr_setting = None

                        for edataset in edatasets:
                            memiter = None

                            if edataset in dict_nndb:
                                nndb_dataset = dict_nndb[edataset]
                                iter_param = iter_param_map[edataset]
                                iter_pp_param = iter_pp_param_map[edataset]
                                memiter = MemDataIterator(edataset,
                                                          nndb_dataset,
                                                          nndb_dataset.cls_n,
                                                          iter_pp_param,
                                                          fn_coreiter)

                                # Fetch TR pre-process setting and apply it on other
                                if edataset == Dataset.TR:
                                    tr_setting = memiter.init_ex(iter_param)
                                else:
                                    memiter.init_ex(iter_param, setting=tr_setting)

                            # Add None or the iterator created above
                            iterstore.setdefault(edataset, memiter)

                    else:
                        # Diskman for `idbp` parameter
                        diskman = self.__dict_diskman[idbp]

                        # Update save_dir_abspaths
                        if diskman is not None:
                            save_dir_abspaths = [] if save_dir_abspaths is None else save_dir_abspaths  # PERF
                            save_dir_abspaths.append(diskman.save_dir_abspath)

                        # Create iterators for the nndb of this dbparam
                        edatasets = Dataset.get_enum_list()
                        for edataset in edatasets:
                            dskiter = None

                            # Create iterator if necessary
                            if (diskman is not None) and \
                                    (diskman.get_nb_class(edataset) > 0):

                                iter_param = iter_param_map[edataset]
                                iter_param = {**iter_param, **dskman_param}  # Merge two dictionaries
                                frecords = diskman.get_frecords(nnpatch.id, edataset)
                                nb_class = diskman.get_nb_class(edataset)
                                iter_pp_param = iter_pp_param_map[edataset]

                                dskiter = DskDataIterator(edataset,
                                                          frecords,
                                                          nb_class,
                                                          iter_pp_param,
                                                          fn_coreiter)
                                dskiter.init_ex(params=iter_param)

                            # Add None or the iterator created above
                            iterstore.setdefault(edataset, dskiter)

                    # Update dict_iterstore and list_iterstore
                    if alias is not None: dict_iterstore.setdefault(alias, iterstore)
                    list_iterstore.append(iterstore)

            # Set the params on the nnmodel (if provided)
            if nnmodel is not None:
                # Per patch addition
                nnmodel.add_iterstores(list_iterstore, dict_iterstore)
                nnmodel.add_save_dirs(save_dir_abspaths)

            else:
                # Initialize NN nnmodels for this patch
                nnpatch.init_models(dict_iterstore, list_iterstore, save_dir_abspaths)

        # Release internal resources used by NNFramework.
        self._release()

    def _release(self):
        """Release internal resources used by NNFramework."""
        del self.__dict_diskman

    ##########################################################################
    # Private Interface
    ##########################################################################
    @staticmethod
    def __get_param_map(param):
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
        
        elif isinstance(param, list):  # [{}, (..., {}), (..., {}), ((..., {}), .{})]
            for entry in param:        # (..., ...) or ((..., ...), ...)
                if isinstance(entry, tuple):
                    edatasets, param = entry

                    if isinstance(edatasets, tuple):
                        for edataset in edatasets:
                            param_map[edataset] = param.copy()
                    else:  # edatasets is a scalar
                        param_map[edatasets] = param.copy()

                elif isinstance(entry, dict):
                    default_param = entry.copy()

                else:
                    raise Exception("Unsupported Format")

        edatasets = Dataset.get_enum_list()
        for edataset in edatasets:
            if edataset not in param_map:
                param_map.setdefault(edataset, None if default_param is None else default_param.copy())

        return param_map

    @staticmethod
    def __resolve_conflicts(sel, iter_param_map):
        """Resolve conflicts with selection structure against other iterator parameters

           Note: Selection structure has high priority than iterator parameters
        """
        edatasets = Dataset.get_enum_list()

        for edataset in edatasets:
            iter_param = iter_param_map[edataset]

            # Detecting conflicts with sel.use_rgb and iter_param['color_mode']
            if sel.use_rgb is not None:

                # Issue a warning if iter_param also has a color_mode that differ from sel.use_rgb
                if 'color_mode' in iter_param:
                    cmode = iter_param['color_mode']
                    if (cmode != 'rgb' and sel.use_rgb) or (cmode != 'grayscale' and not sel.use_rgb):
                        warning("[ARG_CONFLICT] iter_param['color_mode']:" + str(cmode) +
                                " with selection.use_rgb: " + str(sel.use_rgb) +
                                ". Hence selection.use_rgb is prioritized.")

                # Assign the color_mode to the iter_param dictionary
                iter_param['color_mode'] = 'rgb' if sel.use_rgb else 'grayscale'
