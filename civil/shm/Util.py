# Global Imports
import numpy as np
import scipy.io

# Local Imports


#import numpy, scipy.io
#arr = numpy.arange(9)
#arr = arr.reshape((3, 3))  # 2d array of 3x3
#scipy.io.savemat('F:/#TO_DELETE/arrdata.mat', mdict={'arr': arr})


class Util(object):
    """Utilities for civil health monitoring problems."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    @staticmethod
    def load_from_txt(fpath, save=False, save_fpath=None, save_var_name=None):
        if (save):
            assert(save_fpath is not None and save_var_name is not None)
        
        data = np.loadtxt(fpath)

        if (save):
            scipy.io.savemat(save_fpath, mdict={save_var_name: data})

        return data
