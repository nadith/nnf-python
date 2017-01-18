"""NNPreCfg to represent NNPreCfg class."""
# -*- coding: utf-8 -*-
# Global Imports
from enum import Enum

# Local Imports

# TODO: Divide the following classes to different class files
class NNPreCfg(object):
    def __init__(self):
        self.opt_fn = 'adadelta'
        self.lr = 0
        self.mr = 0  
        self.non_sparse_penalty = 0
        self.weight_decay_L2 = 0
        self.batch_size = 1
        self.numepochs = 1
        self.loss_fn = 'mean_squared_error'
        self.use_db  = 'generator' #'mnist'

class NNCfg(NNPreCfg):
    def __init__(self):
        pass

# DAEPreCfg is to pretraining each layer
# DAECfg is to build the stacked network.
# some of the layers may not be pre-trianed. Hence DAEPreCfg 
# itself is not enough to determine the architecture of the final
# stacked network.
class DAEPreCfg(NNPreCfg):
    def __init__(self, arch=[784, 128, 784]):
        super().__init__()
        self.arch = arch
        self.enc_idx = 1
        self.act_fns = ['input', 'sigmoid', 'sigmoid']

    def validate():
        # TODO: validate the lengths of arch and act_fns
        pass
        
# Can describe many different architectures
# Generic DAEmodel:
# Multiple encoder, single decoder model: [784, 128, 64, 32, 16, 784]
class DAECfg(DAEPreCfg):
    def __init__(self, arch=[784, 128, 64, 32, 16, 784]):
        super().__init__(arch)
        #self.enc_end_idx= 3
        self.act_fns = ['input', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']        

    def validate():
        # TODO: validate the lengths of arch and act_fns
        pass