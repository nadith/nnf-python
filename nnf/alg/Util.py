# -*- coding: utf-8 -*-
"""
.. module:: Util
   :platform: Unix, Windows
   :synopsis: Represent Util class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
from sklearn.metrics import accuracy_score

# Local Imports

class Util(object):
    """Util provide common utility functions for algorithms.
    
        Refer method specific help for more details. 
    
        Currently Support:
        ------------------
        - test (evaluates classification accurary in the given subspace)
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    @staticmethod
    def test(W, nndb_g, nndb_p, info={}):
        """TEST: evaluates classification accurary in the given subspace. 
            
        Parameters
        ----------
        nndb_g : :obj:`nnf.db.NNdb`
            Gallery database object.
             
        nndb_p : :obj:`nnf.db.NNdb`
            Probe database object.
             
        info : :obj:`dict`, optional
            Provide additional information to perform test.
            (Default value = {}).
                    
            Info Params (Defaults)
            ----------------------
            - info['dist'] = False          # Calculate distance matrix. TODO: implement
            - info['dcc_norm'] = False      # Distance metric norm to be used. (False => Use L2 norm).
            - info['dcc_sigma'] = eye(n, n) # Sigma Matrix that describes kernel bandwidth for each class. 

        Returns
        -------
        double
            Classification accuracy.

        `array_like`
            2D Distance matrix. (Tr.Samples x Te.Samples)
                        
        Examples
        --------
        >>> import nnf.alg.LDA
        >>> import nnf.alg.Util
        >>> W, info = LDA.l2(nndb_tr)
        >>> accurary = Util.test(W, nndb_tr, nndb_te, info)
        """
        # Initialize the variables
        accuracy = None
        clf = info['clf']

        # Use classifier object to predict
        if (clf is not None):
            pred_cls_lbl = clf.predict(nndb_p.features_scipy)
            accuracy = accuracy_score(nndb_p.cls_lbl, pred_cls_lbl)

        return accuracy