"""NNPatchMan Module to represent NNPatchMan class."""
# -*- coding: utf-8 -*-
# Global Imports

# Local Imports
from nnf.core.NNFramework import NNFramework

class NNPatchMan(NNFramework):
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

        # Generate the patches 
        self.patches = generator.generate_patches()

        # Process params and attach dbs to patches
        self._process_db_params(self.patches, params)

        # Process params against the models
        self._init_model_params(self.patches, params)
       
    # Pretrain layers and build the stacked network for training
    def pre_train(self, precfgs=None, cfg=None):
        # precfgs => list of objects
        # cfg => cfg object
        # TODO: Parallelize processing (2 level - patch level or model level)
        for nnpatch in self.patches:
            for nnmodel in nnpatch.models:
                nnmodel.pre_train(precfgs, cfg) 
                 
    def train(self, cfg=None):
        # TODO: Parallelize processing (2 level - patch level or model level)
        for nnpatch in self.patches:
            for nnmodel in nnpatch.models:
                nnmodel.train(precfg)  

    def test(self):
        for nnpatch in self.patches:
            for nnmodel in nnpatch.models:
                nnmodel.test(precfg)

    def get_stats():
        pass

