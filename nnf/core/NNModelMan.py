# -*- coding: utf-8 -*-
"""
.. module:: NNModelMan
   :platform: Unix, Windows
   :synopsis: Represent NNModelMan class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports

# Local Imports
from nnf.core.NNFramework import NNFramework

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
    def __init__(self, generator, dbparams=[]):
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
            nnmodel.init_nnpatches()

            # Process dbparams and attach dbs to nnpatches
            self._process_db_params(nnmodel.nnpatches, dbparams)

            # Process dbparams against the nnmodels
            self._init_model_params(nnmodel.nnpatches, dbparams, nnmodel)

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
        # Parallization Level-1: model level
        for nnmodel in self.nnmodels:
            nnmodel.pre_train(precfgs, cfg)

        # Parallization Level-2: each model's patch level
        #for nnmodel in self.nnmodels:
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
        TODO: Parallelize processing (1 level - model level) or (2 level - patch level)
        """
        # Parallization Level-1: model level
        for nnmodel in self.nnmodels:
            nnmodel.train(cfg)

        # Parallization Level-2: each model's patch level
        #for nnmodel in self.nnmodels:
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
        # Parallization Level-1: model level
        for nnmodel in self.nnmodels:
            nnmodel.test(cfg)

        # Parallization Level-2: each model's patch level
        #for nnmodel in self.nnmodels:
        #    for nnpatch in nnmodel.nnpatches:
        #        patch_idx = 0
        #        nnmodel.test(cfg, patch_idx)
        #        patch_idx += 1


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
        # Parallization Level-1: model level
        for nnmodel in self.nnmodels:
            precfgs = generator.generate_nnprecfgs(nnmodel)
            nnmodel.pre_train(precfgs, cfg)

        # Parallization Level-2: each model's patch level
        #for nnmodel in self.nnmodels:
        #    precfgs = generator.generate_nnprecfgs(nnmodel)
        #    for nnpatch in nnmodel.nnpatches:
        #        nnmodel.pre_train(precfgs, cfg, nnpatch.uid)

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
        # Parallization Level-1: model level
        for nnmodel in self.nnmodels:
            cfg = generator.generate_nncfg(nnmodel)
            nnmodel.train(cfg)

        # Parallization Level-2: each model's patch level
        #for nnmodel in self.nnmodels:
        #    cfg = generator.generate_nncfg(nnmodel)
        #    for nnpatch in nnmodel.nnpatches:
        #        nnmodel.train(cfg, nnpatch.uid)

    def test_with(self, generator):
        """Initiate testing in the framework.

        Parameters
        ----------
        generator : :obj:`NNCfgGenerator`
            Neural Network configuration generator for each model.
        """ 
        # Parallization Level-1: model level
        for nnmodel in self.nnmodels:
            cfg = generator.generate_nncfg(nnmodel)
            nnmodel.test(cfg)

        # Parallization Level-2: each model's patch level
        #for nnmodel in self.nnmodels:
        #    cfg = generator.generate_nncfg(nnmodel)
        #    for nnpatch in nnmodel.nnpatches:
        #        nnmodel.test(cfg, nnpatch.uid)




