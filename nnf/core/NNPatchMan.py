# -*- coding: utf-8 -*-
"""
.. module:: NNPatchMan
   :platform: Unix, Windows
   :synopsis: Represent NNPatchMan class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports

# Local Imports
from nnf.core.NNFramework import NNFramework

class NNPatchMan(NNFramework):
    """`NNModelMan` represents model based sub framework in `NNFramework'.

    Attributes
    ----------
    nnpatches : list of :obj:`NNPatch`
        List of :obj:`NNPatch` instances.
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    def __init__(self, generator, dbparams=[]):
        """Constructs :obj:`NNPatchMan` instance.

        Parameters
        ----------
        generator : :obj:`NNPatchGenerator`
            Neural Network patch generator to generate list of nnpatches.

        dbparams : list of :obj:`dict`
            List of user dbparams, each describing a database. 
        """ 
        super().__init__(dbparams) 

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
        TODO: Parallelize processing (1 level - model level) or (2 level - patch level)
        """
        # Parallization Level-1: patch level
        # Parallization Level-2: each patch's model level
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:
                nnmodel.pre_train(precfgs, cfg) 
                 
    def train(self, cfg=None):
        """Initiate training in the framework.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.     

        Notes
        -----
        TODO: Parallelize processing (1 level - model level) or (2 level - patch level)
        """
        # TODO: Parallelize processing (2 level - patch level or model level)
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:
                nnmodel.train(cfg)

    def test(self, cfg=None):
        """Initiate testing in the framework.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.
        """ 
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:
                nnmodel.test(cfg)

    def predict(self, cfg=None):
        """Initiate predict in the framework.

        Parameters
        ----------
        cfg : :obj:`NNCfg`
            Neural Network configuration used in training.
        """
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:
                nnmodel.predict(cfg)

    def pre_train_with(self, generator):
        """Initiate pre-training in the framework.

        Parameters
        ----------
        generator : :obj:`NNCfgGenerator`
            Neural Network configuration generator for each model.

        Notes
        -----
        TODO: Parallelize processing (1 level - model level) or (2 level - patch level)        
        """
        # Parallization Level-1: patch level
        # Parallization Level-2: each patch's model level
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:
                precfgs = generator.generate_nnprecfgs(nnmodel, nnpatch)
                cfg = generator.generate_nncfg(nnmodel, nnpatch)
                nnmodel.pre_train(precfgs, cfg)
                 
    def train_with(self, generator):
        """Initiate training in the framework.

        Parameters
        ----------
        generator : :obj:`NNCfgGenerator`
            Neural Network configuration generator for each model.

        Notes
        -----
        TODO: Parallelize processing (1 level - model level) or (2 level - patch level)   
        """ 
        # Parallelize processing (2 level - patch level or model level)
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:
                cfg = generator.generate_nncfg(nnmodel, nnpatch)
                nnmodel.train(cfg)

    def test_with(self, generator):
        """Initiate testing in the framework.

        Parameters
        ----------
        generator : :obj:`NNCfgGenerator`
            Neural Network configuration generator for each model.
        """ 
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:
                cfg = generator.generate_nncfg(nnmodel, nnpatch)
                nnmodel.test(cfg)

    def predict_with(self, generator):
        """Initiate predict in the framework.

        Parameters
        ----------
        generator : :obj:`NNCfgGenerator`
            Neural Network configuration generator for each model.
        """ 
        for nnpatch in self.nnpatches:
            for nnmodel in nnpatch.nnmodels:
                cfg = generator.generate_nncfg(nnmodel, nnpatch)
                nnmodel.predict(cfg)


