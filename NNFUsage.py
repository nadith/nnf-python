import numpy as np
import scipy.io
matStruct = scipy.io.loadmat(r'F:\#Research Data\FaceDB\IMDB_66_66_AR_8.mat',
                            struct_as_record=False, squeeze_me=True)
imdb_obj = matStruct['imdb_obj']
db = np.rollaxis(imdb_obj.db, 3)

## Test image map (visualizations) ################################
#from nnf.utl.immap import *
#immap(db, rows=5, cols=8)
#from nnf.db.NNdb import NNdb
#nndb = NNdb('Original', imdb_obj.db, 8, True)
#nndb.show(5, 8)

## Test patch division ###########################################
#from nnf.db.NNdb import NNdb
#from nnf.db.DbSlice import *
#nndb = NNdb('Original', imdb_obj.db, 8, True)
#sel = Selection()
#sel.tr_col_indices = np.array([0, 1, 3])

#from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
#patch_gen = NNPatchGenerator(nndb.h, nndb.w, 33, 33, 33, 33)
#sel.nnpatches = patch_gen.generate_patches()

## Cell arrays of NNdb objects for each patch
#[nndbs_tr, _, _, _, _] = DbSlice.slice(nndb, sel)
#nndbs_tr[0].show()
#nndbs_tr[1].show()
#nndbs_tr[2].show()
#nndbs_tr[3].show()

## Test db slicing ################################################
#from nnf.db.NNdb import NNdb
#from nnf.db.DbSlice import DbSlice
#from nnf.db.Selection import Selection

#nndb = NNdb('original', imdb_obj.db, 8, True)
#sel = Selection()
#sel.tr_col_indices = np.array([0, 1, 3])
#sel.val_col_indices= np.array([2, 3])
#sel.te_col_indices = np.array([2, 4])
#sel.tr_out_col_indices  = np.zeros(3, dtype='uint8')  # [1, 1, 1]
#sel.val_out_col_indices = np.zeros(2, dtype='uint8')  # [1, 1]
#sel.class_range    = np.arange(9)
#sel.val_class_range= np.arange(6,15)
#sel.te_class_range = np.arange(17, 20)
#[nndb_tr, nndb_val, nndb_te, nndb_tr_out, nndb_val_out] = DbSlice.slice(nndb, sel)
#nndb_tr.show(10, 3)
#nndb_val.show(10, 2)
#nndb_te.show(4, 2)
#nndb_tr_out.show(10, 3)
#nndb_val_out.show(10, 2)


 # <<<<<<<<<<<<<<<<<<<<  DEEP LEARNING FRAMEWORK >>>>>>>>>>>>>>>>>>>>
# -------------------------------------------------------------------
# Patch Based Approach
# -------------------------------------------------------------------
from nnf.db.NNdb import NNdb
nndb = NNdb('Original', imdb_obj.db, 8, True)

from nnf.db.NNPatch import NNPatch
from nnf.core.models.DAEModel import DAEModel
class AEPatch(NNPatch):
    def generate_nnmodels(self, iterstore):
        """overiddable"""
        # Can use interal patch information to generate patch specific models

        # Model I for 'self' patch
        nnmodels = []
        nnmodels.append(DAEModel(self, iterstore))        

        # Model II for 'self' patch
        # nnmodels.append(Autoencoder())

        # Model III for 'self' patch
        # nnmodels.append(DAEModel())
        
        return nnmodels

from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
class AEPG(NNPatchGenerator):
    def __init__(self, im_h, im_w, pat_h, pat_w, xstrip, ystrip):
        super().__init__(im_h, im_w, pat_h, pat_w, xstrip, ystrip)

    def new_nnpatch(self, h, w, offset): 
        return AEPatch(h, w, offset)

from nnf.db.Selection import Selection
from nnf.db.Selection import Select
sel = Selection()
sel.use_rgb = False
sel.tr_col_indices = np.array([0, 1, 1])
#sel.tr_col_indices = Select.ALL
sel.tr_out_col_indices = np.array([])
sel.val_col_indices = np.array([1, 4])
sel.val_out_col_indices = np.array([])
sel.te_col_indices = np.array([])
sel.class_range = np.uint8(np.array([0, 4]))
#sel.class_range = np.uint8(np.array([0, 200]))
#sel.class_range = Select.ALL
sel.val_class_range = np.uint8(np.array([0, 4]))
sel.te_class_range = np.uint8(np.array([]))

from nnf.core.NNPatchMan import NNPatchMan
#nnpatchman = NNPatchMan(AEPG(nndb.h, nndb.w, 33, 33, 33, 33), [(nndb, sel, False)])
nnpatchman = NNPatchMan(AEPG(nndb.h, nndb.w, 33, 33, 33, 33), [("D:/TestImageFolder", sel, False)])
#nnpatchman = NNPatchMan(AEPG(nndb.h, nndb.w, 33, 33, 33, 33), [(None, sel, True)])
#====================================================================
## TESTED TILL THIS POINT ##







# Pre-Training 
#from nnf.core.NNCfg import *
#daepcfgs = []
#daepcfgs.append(DAEPreCfg([1089, 800, 1089]))
#daepcfgs.append(DAEPreCfg([800, 500, 800]))
#daecfg = DAECfg([1089, 800, 500, 800, 1089])
#nnpatchman.pre_train(daepcfgs, daecfg)


# -------------------------------------------------------------------
# Model Based Approach
# -------------------------------------------------------------------
from nnf.core.NNModelMan import NNModelMan
from nnf.core.generators.NNModelGenerator import NNModelGenerator

#class AEMG(NNModelGenerator):
#    def __init__(self):
#        super().__init__()

#    def generate_models(self):
#        """Generates list of NNModels.

#        Overridable

#        Returns
#        -------
#        nnpatches : list- NNModel
#            NNModel objects.
#        """
#        nnmodels = []
#        nnmodels.append(DAEModel(None, None))  
#        return nnmodels


from nnf.db.Selection import Selection
from nnf.db.Selection import Select
#sel = Selection()
#sel.use_rgb = False
#sel.tr_col_indices = np.array([1, 2, 3])
#sel.te_col_indices = np.array([4, 5])
#nnmodelman = NNModelMan(AEMG(), [(nndb, sel, True)])
#====================================================================

# Pre-Training 
#from nnf.core.NNCfg import *
#daepcfgs = []
#daepcfgs.append(DAEPreCfg([1089, 800, 1089]))
#daepcfgs.append(DAEPreCfg([800, 500, 800]))
#daecfg = DAECfg([1089, 800, 500, 800, 1089])
#nnmodelman.pre_train(daepcfgs, daecfg)


