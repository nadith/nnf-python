"""NNPatch to represent NNPatch class."""
# -*- coding: utf-8 -*-
# Global Imports

# Local Imports
from nnf.core.models.NNModel import NNModel
from nnf.core.models.Autoencoder import Autoencoder


class NNPatch(object):
    """NNPatch describes the patch information.

    Attributes
    ----------
    h : int
        Height (Y dimension).

    w : int
        Width (X dimension).

    offset : (int, int)
        Position of the patch.

    _user_data : dict -obj
        Hold _nndbtr, _nndbval, _nndbte patch databases in DL framework.
    """
    def __init__(self, width, height, offset=(None, None)):
        self.w = width
        self.h = height
        self.offset = offset
        self._user_data = {}
        self.models = []

    def add_model(self, models):
        if (isinstance(models, list)):
            self.models = self.models + models
        else:
            self.models.append(models)

    def _init_models(self, iterstore):
        """Patch Based Framewowk"""
        self.add_model(self.generate_nnmodels(iterstore))

        # Assign this patch to the model
        for model in self.models:
            model.add_patches(self)

    def _set_udata(self, key, value):
       self._user_data[key] = value

    def _get_udata(self, key):
        return self._user_data[key]

    def generate_nnmodels(self, iterstore):
        """overiddable"""
        print("TODO: generate_nnmodels(iterstore): Override and implement")


    #################################################################
    # Dependant property Implementations
    #################################################################
    @property
    def id(self):
        """Unique patch id"""
        patch_id = '{offset_x}_{offset_y}_{height}_{width}'.format(offset_x=self.offset[0],
                                                                    offset_y=self.offset[1],
                                                                    height=self.h,
                                                                    width=self.w)
        return patch_id