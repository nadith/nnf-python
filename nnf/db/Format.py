# Global Imports
from enum import Enum

# Local Imports

class Format(Enum):
    # NNDBFORMAT Enumeration describes the format of the NNdb database.    
    H_W_CH_N = 1,     # =0 Height x Width x Channels x Samples (image db format) 
    H_W_CH_N_NP = 2,  # Height x Width x Channels x Samples x PatchCount
    H_N = 3,          # Height x Samples
    H_N_NP = 4        # Height x Samples x PatchCount