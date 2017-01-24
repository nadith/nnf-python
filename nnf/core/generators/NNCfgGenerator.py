# -*- coding: utf-8 -*-
"""
.. module:: NNCfgGenerator
   :platform: Unix, Windows
   :synopsis: Represent NNCfgGenerator class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from abc import ABCMeta, abstractmethod

# Local Imports
from nnf.core.NNCfg import DAEPreCfg
from nnf.core.NNCfg import DAECfg

class NNCfgGenerator:
    """Generator describes the generator for Neural Network Configuration.

        See also :obj:`NNCfg`.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """Constructs :obj:`NNCfgGenerator` instance."""
        pass

    ##########################################################################
    # Public Interface
    ##########################################################################
    @abstractmethod
    def generate_nnprecfgs(self, nnmodel, nnpatch=None):
        """Generate Neural Network configuration(s) for pre-training of the framework.

        Parameters
        ----------
        nnmodel : :obj:`NNModel'
            The Neural Network Model.

        nnpatch : :obj:`NNPatch'
            The `NNPatch` instance. For Model based frameworks, `nnpatch`=None

        Returns
        -------
        :obj:`NNCfg` or list of :obj:`NNCfg`
            Neural Network configuration(s) for the provided :obj:`NNModel' instance.

        Note
        ----
        Extend this method to construct set of pre-training configurations.
        """
        daepcfgs = []
        daepcfgs.append(DAEPreCfg([1089, 800, 1089]))
        daepcfgs.append(DAEPreCfg([800, 500, 800]))
        return daepcfgs

    @abstractmethod
    def generate_nncfg(self, nnmodel, nnpatch=None):
        """Generate Neural Network configuration for training of the framework.

        Parameters
        ----------
        nnmodel : :obj:`NNModel'
            The Neural Network Model.

        nnpatch : :obj:`NNPatch'
            The `NNPatch` instance. For Model based frameworks, `nnpatch`=None

        Returns
        -------
        :obj:`NNCfg`
            Neural Network configuration(s) for the provided :obj:`NNModel' instance.

        Note
        ----
        Extend this method to construct set of training configuration.
        """
        return DAECfg([1089, 800, 500, 800, 1089])