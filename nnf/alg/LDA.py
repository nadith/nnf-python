# -*- coding: utf-8 -*-
"""
.. module:: LDA
   :platform: Unix, Windows
   :synopsis: Represent LDA class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Local Imports
from nnf.utl.immap import immap
from nnf.pp.contrast_strech import contrast_strech

class LDA(object):
    """LDA represents Linear Discriminant Analysis algorithm and varients.

        Refer method specific help for more details.

        Currently Support:
        ------------------
        - LDA.fl2 (fisher L2)
        - LDA.dl2 (direct lda - l2)
        - LDA.l2  (regularized L2)
        - LDA.r1  (rotational invariant - l1)
    """

    ##########################################################################
    # Public Interface
    ##########################################################################
    @staticmethod
    def l2(nndb, info={}):
        """Learn the regularized LDA subspace with l2-norm.

         Parameters
         ----------
         nndb : :obj:`nnf.db.NNdb`
            Data object that contains the database.

         info : :obj:`dict`, optional
            Provide additional information to perform LDA.
            (Default value = {}).

            Info Params (Defaults)
            ----------------------
            - info['visualize'] = False  # Visualize LDA faces

            For other info params, 
            refer: sklearn.discriminant_analysis.LinearDiscriminantAnalysis

         Returns
         -------
         `array_like`
            Weight matrix.
        
         :obj:`dict`
            Info dictionary. Provide additional information to perform test.
            (Default value = {}).

         Examples
         --------
         >>> import nnf.alg.LDA
         >>> W, info = LDA.l2(nndb_tr)        
        """        
        # svd solver issues:
        # Ref: http://stats.stackexchange.com/questions/29385/collinear-variables-in-multiclass-lda-training
        # Ref: https://github.com/scikit-learn/scikit-learn/issues/1649

        # Set defaults
        solver = info['solver'] if ('solver' in info) else 'eigen' #'svd' #'eigen' #'lsqr'
        shrinkage = info['shrinkage'] if ('shrinkage' in info) else 'auto' #None #'auto' #'auto'
        priors = info['priors'] if ('priors' in info) else None
        n_components = info['n_components'] if ('n_components' in info) else None
        store_covariance = info['store_covariance'] if ('store_covariance' in info) else False
        tol = info['tol'] if ('tol' in info) else 1e-4

        # Instantiate scikit-learn LDA object
        info['clf'] = clf = LinearDiscriminantAnalysis(
                                            solver=solver, 
                                            shrinkage=shrinkage, 
                                            priors=priors,
                                            n_components=n_components, 
                                            store_covariance=store_covariance,
                                            tol=tol)
        # Fit LDA l2
        clf.fit(nndb.features_scipy, nndb.cls_lbl)           

        # Solvers like 'lsqr' has no weight matrix avialable.
        # Refer LinearDiscriminantAnalysis.transform(...)        
        W = clf.scalings_ if (hasattr(clf, 'scalings_')) else None

        # Visualize lda faces if required
        if (W is not None and 'visualize' in info and info['visualize']):
            imdb = np.uint8(np.reshape(W*255 + np.amin(W), 
                                            (nndb.h, nndb.w, nndb.ch, -1)))
            # Scipy format
            imdb = np.transpose(imdb, (3, 0, 1, 2))
            for i in range(len(imdb)):
                imdb[i] = contrast_strech(imdb[i])

            # Visualize the lda face grid
            immap(imdb, 3, 5)

        return W, info