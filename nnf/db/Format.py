# Global Imports
from enum import Enum

# Local Imports

class Format(Enum):
    # NNDBFORMAT Enumeration describes the format of the NNdb database.    
    H_W_CH_N = 1,     # =1 Height x Width x Channels x Samples (image db format)     
    H_N = 2,          # Height x Samples
    N_H_W_CH = 3,     # Samples x Height x Width x Channels x Samples (image db format) 
    N_H = 4,          # Samples x Height