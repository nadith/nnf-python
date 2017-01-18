"""NNFrmework to represent NNFrmework class."""
# -*- coding: utf-8 -*-
# Global Imports
from abc import ABCMeta, abstractmethod
from warnings import warn as warning

# Local Imports
from nnf.db.DbSlice import DbSlice
from nnf.core.NNDiskMan import NNDiskMan
from nnf.core.iters.memory.MemTrIterator import MemTrIterator
from nnf.core.iters.memory.MemValIterator import MemValIterator
from nnf.core.iters.memory.MemTeIterator import MemTeIterator
from nnf.core.iters.disk.DskTrIterator import DskTrIterator
from nnf.core.iters.disk.DskValIterator import DskValIterator
from nnf.core.iters.disk.DskTeIterator import DskTeIterator

class NNFramework(object):
    """ABSTRACT CLASS (SHOULD NOT BE INISTANTIATED)"""
    __metaclass__ = ABCMeta

    def __init__(self, params): 
        pass

    def _process_db_params(self, nnpatches, params):
        """Process db related parameters and attach dbs to corresponding nnpatches."""
        
        # TODO: PERF: Can move this to a one loop
        # Initializing _user_data
        for nnpatch in nnpatches:
            nnpatch._set_udata('_nndbs_tr', [])
            nnpatch._set_udata('_nndbs_val', [])
            nnpatch._set_udata('_nndbs_te', [])
            
        # Iterate through params
        self._diskman = []
        for param in params:            
            nndb = param[0]
            sel = param[1]            
            db_in_mem = param[2]

            # Set sel.nnpatches
            if (hasattr(sel, 'nnpatches') and
                sel.nnpatches is not None): warning('ARG_CONFLICT: '
                                                    'sel.nnpatches is already set. '
                                                    'Discarding the current value...')
            sel.nnpatches = nnpatches
            
            # Slicing the database
            if (db_in_mem):

                # Warning
                if (nndb is None):
                    warning("""nndb: is not given. 
                                Hence data iterators will not be created.
                                Makesure to use an external DB via NNCfg for training, testing, etc""")
                    continue

                # Split the database according to the selection
                trs, vals, tes, _, _ = DbSlice.slice(nndb, sel)

                # Update dbs @ nnpatches
                for i in range(len(nnpatches)):
                    nnpatch = nnpatches[i]
                    if (trs is not None): nnpatch._get_udata('_nndbs_tr').append(trs[i])
                    if (vals is not None): nnpatch._get_udata('_nndbs_val').append(vals[i])
                    if (tes is not None): nnpatch._get_udata('_nndbs_te').append(tes[i])

            else:
                directory = None
                if (isinstance(nndb, str)):
                    directory = nndb
                    nndb = None

                if (directory is None):
                    warning("""image directory: is not given. 
                        Hence data iterators will not be created.
                        Makesure to use an external DB via NNCfg for training, testing, etc""")
                    continue

                # Initialzie NNDiskman
                dskman = NNDiskMan(sel, nndb, directory)       
                self._diskman.append(dskman)

                # Update dbs @ diskman
                dskman.process(nnpatches)                    

    def _init_model_params(self, nnpatches, params, nnmodel=None):
        """Initializes models at nnpatches along with iterator stores for 
            the databases stored at nnpatches.
            
            assume self._diskman is populated.
        """
        # Iterate the patch list
        for nnpatch in nnpatches:

            # Initialize iterators for each model initialized above
            iterstore = []
            for i in range(len(params)):

                TrItertor = ValItertor = TeIterator = None
                db_in_mem = params[i][2]

                if (db_in_mem):
                    # Create a TrIterator
                    dbs = nnpatch._get_udata('_nndbs_tr')
                    iter_tr = None if (len(dbs) == 0) else MemTrIterator(dbs[i])
            
                    # Create a ValIterator
                    dbs = nnpatch._get_udata('_nndbs_val')
                    iter_val = None if (len(dbs) == 0) else MemValIterator(dbs[i])
                    
                    # Create a TeIterator
                    dbs = nnpatch._get_udata('_nndbs_te')
                    iter_te = None if (len(dbs) == 0) else MemTeIterator(dbs[i])

                else:
                    if (self._diskman[i] is None):
                        iter_tr = None
                        iter_val = None
                        iter_te = None
                        iter_tr_out = None
                        iter_val_out = None

                    else:                    
                        # Create a TrIterator, ValIterator, TeIterator
                        iter_tr = DskTrIterator(nnpatch, self._diskman[i])
                        iter_val = DskValIterator(nnpatch, self._diskman[i])
                        iter_te =  DskTeIterator(nnpatch, self._diskman[i])
                    
                iterstore.append((iter_tr, iter_val, iter_te))

            # Set the params on the nnmodel (if provided)
            if (nnmodel is not None):
                nnmodel.set_iterstore(iterstore)
                nnpatch.add_model(nnmodel)

            else:
                # Initialize NN models for this patch
                nnpatch._init_models(iterstore)
