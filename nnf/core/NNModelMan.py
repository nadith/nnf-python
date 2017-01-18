"""NNModelMan Module to represent NNModelMan class."""
# -*- coding: utf-8 -*-
# Global Imports

# Local Imports
from nnf.core.NNFramework import NNFramework

class NNModelMan(NNFramework):
    """NNPatchMan represents patch manager for NNFramework.

    Attributes
    ----------
    patches : list -NNPatch
        List of NNPatch objects.
    """
    def __init__(self, generator, params=[(None, None, True)]):
        # params is a list of tuples
        # [(NNdb, Selection, db_in_mem:bool)]       

        super().__init__(params)        

        # Init variables
        self.patches = []
        self._diskman = []

        # Generate the models 
        self.models = generator.generate_models()

        # Iterate through models
        for nnmodel in self.models:

            # Generate patches
            nnmodel._init_patches()

            # Process params and attach dbs to patches
            self._process_db_params(nnmodel.patches, params)

            # Process params against the models
            self._init_model_params(nnmodel.patches, params, nnmodel)

    # Pretrain layers and build the stacked network for training
    def pre_train(self, precfgs=None, cfg=None):
        # precfgs => list of objects
        # cfg => cfg object
        # TODO: Parallelize processing (1 level - patch level)
        for nnmodel in self.models:    
            for nnpatch in nnmodel.patches:
                assert(len(nnpatch.models) == 1)
                nnpatch.models[0].pre_train(precfgs, cfg) 
                 
    def train(self, cfg=None):
        # TODO: Parallelize processing (1 level - patch level)
        for nnmodel in self.models:    
            for nnpatch in nnmodel.patches:
                assert(len(nnpatch.models) == 1)
                nnpatch.models[0].train(precfg)  

    def test(self):
        for nnmodel in self.models:    
            for nnpatch in nnmodel.patches:
                assert(len(nnpatch.models) == 1)
                nnpatch.models[0].test(precfg)

    def get_stats():
        pass

