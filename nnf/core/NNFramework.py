# -*- coding: utf-8 -*-
"""
.. module:: NNFrmework
   :platform: Unix, Windows
   :synopsis: Represent NNFrmework class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from abc import ABCMeta, abstractmethod
from warnings import warn as warning
import numpy as np

# Local Imports
from nnf.db.DbSlice import DbSlice
from nnf.core.NNDiskMan import NNDiskMan
from nnf.core.iters.memory.MemDataIterator import MemDataIterator
from nnf.core.iters.memory.NumpyArrayIterator import NumpyArrayIterator
from nnf.core.iters.disk.DskDataIterator import DskDataIterator
from nnf.core.iters.disk.DirectoryIterator import DirectoryIterator
from nnf.db.Dataset import Dataset
from nnf.db.Selection import Selection
from nnf.core.models.NNModel import NNModel

class NNFramework(object):
    """`NNFramework` represents the base class of the Neural Network Framework.

    .. warning:: abstract class and must not be instantiated.

        Process paramters from the user and create necessary objects
        to be utilized in the frameowrk for further processing.

    Attributes
    ----------
    _dict_diskman : dict of :obj:`NNDiskMan`
        :obj:`NNDiskMan` for each of the user `dbparam`.

	_SAVE_TO_DIR : str
		[CONSTANT][INTERNAL_USE] Directory to save the processed data.
    """
    __metaclass__ = ABCMeta

    # [CONSTANT][INTERNAL_USE] Directory to save the processed data
    _SAVE_TO_DIR = "_processed"

    #################################################################
    # Public Interface
    #################################################################
    def __init__(self, dbparams):
        """Constructor of the abstract class :obj:`NNFramework`.

        Parameters
        ----------
        dbparams : list of :obj:`dict`
            List of user dbparams, each describing a database.
        """
        self._dict_diskman = {}

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
            Useful to build the deep stacked network after layer-wise pre-trianing.
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

    @abstractmethod
    def test(self):
        """Initiate training in the framework."""
        pass

    ##########################################################################
    # Protected Interface
    ##########################################################################
    def _process_db_params(self, nnpatches, dbparams):
        """Process user dbparams (database related parameters)
           
            Process the database and attach slices to corresponding nnpatches
            if the database is in memory. Otherwise init nndiskman to process
            disk database.

        Parameters
        ----------
        nnpatches : list of :obj:`NNPatch`
            List of :obj:`NNPatch` instances.

        dbparams : list of :obj:`dict`
            List of user dbparams, each describing a database.
        """
        # Iterate through dbparams
        for idbp, dbparam in enumerate(dbparams):

            # Read fields from dbparam
            alias = dbparam.setdefault('alias', None)
            nndb = dbparam.setdefault('nndb', None)
            db_dir = dbparam.setdefault('db_dir', None)
            sel = dbparam.setdefault('selection', None)
            db_pp_param = dbparam.setdefault('db_pp_param', None)
            iter_pp_param = dbparam.setdefault('iter_pp_param', None)
            iter_param = dbparam.setdefault('iter_param', None)
            iter_in_mem = dbparam.setdefault('iter_in_mem', True)
            fn_nndiskman = dbparam.setdefault('fn_nndiskman', None)
            dbparam.setdefault('fn_coreiter', None)

            # Conflict resolution
            if ((nndb is not None) and (db_pp_param is not None) and (iter_pp_param is not None)):
                warning("db_pp_param: ignored, iter_pp_param is used")

            # Set defaults for sel
            if (sel is None):
                dbparam['selection'] = sel = Selection()
    
            # Resolve conflicts with selection structure against other pre-processing parameters
            self.__resolve_conflicts(sel, iter_param, nndb, db_dir, nnpatches)

            # If iterators are need for in memory databases
            if (iter_in_mem):

                # Warning
                if (nndb is None):
                    warning("""nndb: is not given. 
                                Hence data iterators will not be created.
                                Makesure to use an external DB via NNCfg for training, testing, etc""")
                    continue

                # Split the database according to the sel
                nndbs_tup = DbSlice.slice(nndb, sel, pp_param=db_pp_param)

                # Itearte through nnpatches
                for pi, nnpatch in enumerate(nnpatches):
                    nnpatch = nnpatches[pi]                    
                    edatasets = nndbs_tup[-1]

                    # Itearate through the results of DbSlice above
                    for ti in range(len(nndbs_tup)-1):                        
                        nndbs = nndbs_tup[ti];

                        # Set nndb for nnpatch by dbparam and Dataset.TR|VAL|TE|...
                        # nnpatch -> i.e [dbparam][Dataset.TR] = [nndb_patch_dbparam_TR]
                        if (nndbs is not None):                            
                            dict_nndb = nnpatch._setdefault_udata(idbp, {})
                            edataset = edatasets[ti]
                            dict_nndb[edataset] = nndbs[pi]

            # Initialzie NNDiskman
            if (db_dir is None and not iter_in_mem):
                warning("""image directory: is not given. 
                    Hence data iterators will not be created.
                    Makesure to use an external DB via NNCfg for training, testing, etc""")
                dskman = None

            elif (db_dir is None):
                dskman = None

            else:
                # Create diskman and init()
                # The init() will use sel, sel.nnpatches for further processing
                save_dir = NNFramework._SAVE_TO_DIR + "_" + (str(idbp) if (alias is None) else alias)
                dskman = fn_nndiskman(sel, db_pp_param, nndb, db_dir, save_dir, iter_param) \
                            if (fn_nndiskman is not None) else \
                            NNDiskMan(sel, db_pp_param, nndb, db_dir, save_dir, iter_param)
                dskman.init()   

            # Add diskman to _diskmams
            self._dict_diskman.setdefault(idbp, dskman)

    def _init_model_params(self, nnpatches, dbparams, nnmodel=None):
        """Initialize `nnmodels` at `nnpatches` along with iteratorstores.

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
        assert(isinstance(dbparams, list))

        # Iterate the patch list
        for nnpatch in nnpatches:

            # Initialize dict|list of iteratorstore for nnmodel @ nnpatch
            # dict_iterstore => {alias1:[iter_TR, iter_VAL, ...], alias2:[iterstore_for_param2_db], ...}
            dict_iterstore = None 

            # list_iterstore => [[iter_TR, iter_VAL, ...], [iterstore_for_dbparam2_db], ...]
            list_iterstore = None

            # save_to_dirs => [folder_for_dbparam_1_db, folder_for_dbparam_2_db]             
            save_to_dirs = None  # PERF  

            # Iterate through dbparams
            for idbp, dbparam in enumerate(dbparams):

                # Read fields from dbparam
                alias = dbparam['alias']
                nndb = dbparam['nndb']
                db_dir = dbparam ['db_dir']
                sel = dbparam ['selection']
                db_pp_param = dbparam ['db_pp_param']
                iter_pp_param = dbparam ['iter_pp_param']
                iter_param = dbparam ['iter_param']
                iter_in_mem = dbparam ['iter_in_mem']
                fn_coreiter = dbparam ['fn_coreiter']

                # Iterator pre-processing params for each dataset
                iter_pp_param_map = self.__get_pp_params_map(iter_pp_param)

                # Iterator store for this dbparam
                iterstore = {}

                # PERF: Create the dictionary list in 1st iteration, if only necessary
                if (idbp==0):
                    list_iterstore = []
                    if (alias is not None):
                        dict_iterstore = {}

                # If iterators are need for in memory databases
                if (iter_in_mem):

                    # Get nndb dictionary for this dbparam
                    dict_nndb = nnpatch._setdefault_udata(idbp, {})

                    # Create iteartors for each dataset in nndb of this dbparam
                    edatasets = Dataset.get_enum_list()
                    for edataset in edatasets:
                        memiter = None

                        if (edataset in dict_nndb):
                            nndb_dataset = dict_nndb[edataset]
                            iter_pp_param = iter_pp_param_map[edataset]
                            memiter =  MemDataIterator(edataset, nndb_dataset, nndb_dataset.cls_n, iter_pp_param, fn_coreiter)
                            memiter.init(iter_param)                   

                        # Add None or the iterator created above
                        iterstore.setdefault(edataset, memiter)

                else:
                    # Diskman for `idbp` paramter
                    diskman = self._dict_diskman[idbp]

                    # Update save_to_dirs
                    if (diskman is not None):
                        save_to_dirs = [] if (save_to_dirs == None) else save_to_dirs # PERF
                        save_to_dirs.append(diskman._save_to_dir)

                    # Create iteartors for the nndb of this dbparam
                    edatasets = Dataset.get_enum_list()
                    for edataset in edatasets:
                        dskiter = None
                
                        # Create iterator if necessary 
                        if ((diskman is not None) and 
                                (diskman.get_nb_class(edataset) > 0)):
                            frecords = diskman.get_frecords(nnpatch.id, edataset)
                            nb_class = diskman.get_nb_class(edataset)     
                            iter_pp_param = iter_pp_param_map[edataset]                       
                            dskiter =  DskDataIterator(edataset, frecords, nb_class, iter_pp_param, fn_coreiter)
                            dskiter.init(iter_param)
                            
                        # Add None or the iterator created above
                        iterstore.setdefault(edataset, dskiter)

                # Update dict_iterstore and list_iterstore
                if (alias is not None): dict_iterstore.setdefault(alias, iterstore)
                list_iterstore.append(iterstore)

            # Set the params on the nnmodel (if provided)
            if (nnmodel is not None):
                # Per patch addition
                nnmodel._add_iterstores(list_iterstore, dict_iterstore)
                nnmodel._add_save_to_dirs(save_to_dirs)

            else:
                # Initialize NN nnmodels for this patch
                nnpatch._init_models(dict_iterstore, list_iterstore, save_to_dirs)

        # Release internal resources used by NNFramework.
        self._release()

    def _release(self):
        """Release internal resources used by NNFramework."""
        del  self._dict_diskman
    ##########################################################################
    # Private Interface
    ##########################################################################
    def __get_pp_params_map(self, iter_pp_param):
        """Fetch iterator pre-processor parameter set for each dataset.

        Parameters
        ----------
        iter_pp_param : :obj:`list` or :obj:`dict`
            List of `dict` element, `tuple` element in the following format.
            [{default_pp_params}, 
            (Dataset.TR, {pp_params}), 
            (Dataset.VAL, {pp_params}), 
            ((Dataset.TE, Dataset.TE_OUT), {pp_params})]

            OR
            `dict` indicating the {default_pp_params}

        Returns
        -------
        :obj:`dict`
            Dictonary of pre-processor parameters keyed by Dataset 
            enumeration. (i.e Dataset.TR|VAL|...).
        """
        map = {}
        default_pp_params = None
        if (isinstance(iter_pp_param, list)):  # [{}, (..., ...), (..., ...), ((..., ...), ...)] 
            for iter_pp in iter_pp_param:       # (..., ...) 
                if (isinstance(iter_pp, tuple)):
                    edataset, pp_params = iter_pp

                    if (isinstance(edataset, tuple)):                                
                        for edataset in iter_pp[0]:
                            map[edataset] = pp_params.copy()
                    else:
                        map[edataset] = pp_params.copy()

                elif (isinstance(iter_pp, dict)):
                    default_pp_params = iter_pp.copy()

                else:
                    raise Exception("Unsupported Format")

        elif (isinstance(iter_pp_param, dict)):
            default_pp_params = iter_pp_param.copy()

        edatasets = Dataset.get_enum_list()
        for edataset in edatasets:
            if (edataset not in map):
                map.setdefault(edataset, None if (default_pp_params is None) \
                                            else default_pp_params.copy())

        return map

    def __resolve_conflicts(self, sel, iter_param, nndb, db_dir, nnpatches):
        """Resolve conflics with selection structure against other parameters

           Note: Selection structure has high priority than iterator params
        """
        # Detecting conflics with sel.nnpatches
        if (sel.nnpatches is not None): warning('ARG_CONFLICT: '
                                                'selection.nnpatches is already set. '
                                                'Discarding the current value...')
        sel.nnpatches = nnpatches

        if (iter_param is None):
            return;

        # Detecting conflics with sel.use_rgb and iter_param['color_mode']
        if (sel.use_rgb is not None):

            # Issue a warning if iter_param also has a color_mode that differ from sel.use_rgb
            if ('color_mode' in iter_param):
                cmode = iter_param['color_mode']                                        
                if ((cmode != 'rgb' and sel.use_rgb) or
                    (cmode != 'grayscale' and not sel.use_rgb)):
                    warning("[ARG_CONFLICT] iter_param['color_mode']:" + str(cmode) + 
                            " with selection.use_rgb: "+ str(sel.use_rgb) + 
                            ". Hence selection.use_rgb is prioratized.")
        
            # Assign the color_mode to the iter_param dictionary
            iter_param['color_mode'] = 'rgb' if (sel.use_rgb) else 'grayscale'