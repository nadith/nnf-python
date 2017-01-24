# Global Imports
from enum import Enum

# Local Imports

class NNModelPhase(Enum):
    # NNModelPhase Enumeration describes phases of NNModel.
    PRE_TRAIN = 0,  # Pre-training phase
    TRAIN = 1,      # Training phase
    TEST = 2,       # Testing phase
    PREDICT = 3,    # Prediction phase
