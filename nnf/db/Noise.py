# Global Imports
from enum import Enum

# Local Imports

class Noise(Enum):
    # NOISE Enumeration describes the different noise types.
    G = -1,  # Gaussian noise
    L = -2,  # Laplacian noise
